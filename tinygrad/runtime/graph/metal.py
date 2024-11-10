from typing import List, Any, Dict, cast, Optional
import ctypes
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, getenv
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.ops import Variable
from tinygrad.runtime.ops_metal import new_var

class MTLIndirectCommandType:
  MTLIndirectCommandTypeConcurrentDispatch = (1 << 5)

class MTLResourceUsage:
  MTLResourceUsageRead = 0b01
  MTLResourceUsageWrite = 0b10

class MetalGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    # create metal batch exec
    icb_descriptor = self.device.msg("MTLIndirectCommandBufferDescriptor", "new", res=new_var())
    self.device.msg(icb_descriptor,"setCommandTypes:","MTLIndirectCommandTypeConcurrentDispatch")
    self.device.msg(icb_descriptor,"setInheritBuffers:","false")
    self.device.msg(icb_descriptor,"setInheritPipelineState:","false")
    self.device.msg(icb_descriptor,"setMaxKernelBufferBindCount:",31)

    self.icb = self.device.msg("d","newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",icb_descriptor,len(self.jit_cache),
      "MTLResourceCPUCacheModeDefaultCache",res=new_var())
    #if self.icb.value is None: raise GraphException("create indirect command buffer failed, does your system support this?") works for iphone 13 apple 8
    #https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    self.needs_icb_fix = True #todo, assuming true for my iphone

    if len(self.vars): 
      self.int_buf = self.device.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    all_resources = [self.int_buf.buf] if len(self.vars) else []
    all_pipelines = []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = self.device.msg(self.icb,"indirectComputeCommandAtIndex:",j,res=new_var())
      all_pipelines.append(prg.clprg.pipeline_state)
      self.device.msg(icb_command, "setComputePipelineState:", prg.clprg.pipeline_state)
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          self.device.msg(icb_command,"setKernelBuffer:offset:atIndex:",b._buf.buf,b._buf.offset,i)
          all_resources.append(b._buf.buf)
      for i,v in enumerate(prg.p.vars):
        self.device.msg(icb_command,"setKernelBuffer:offset:atIndex:",self.int_buf.buf,self.vars.index(v)*4, len(ji.bufs)+i)

      global_size, local_size = prg.p.launch_dims(var_vals)
      self.device.msg(icb_command,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
      self.device.msg(icb_command,"setBarrier")

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    self.command_buffer: Any = None
    if len(self.vars):
      self.int_buf_view = self.device.allocator.as_buffer(self.int_buf).cast('i')

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    if self.command_buffer is not None and self.command_buffer in self.device.mtl_buffers_in_flight:
      self.device.msg(self.command_buffer,"waitUntilCompleted")
    all_resources = dedup(self.all_resources + [x._buf.buf for x in input_rawbuffers])
    
    for (j,i),input_idx in self.input_replace.items():
      computeCommand = self.device.msg(self.icb, "indirectComputeCommandAtIndex:", j, res=new_var())
      self.device.msg(computeCommand, "setKernelBuffer:offset:atIndex:", input_rawbuffers[input_idx]._buf.buf,
                                                                                 input_rawbuffers[input_idx]._buf.offset, i)

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      prg = cast(CompiledRunner, self.jit_cache[j].prg)
      global_size, local_size = global_dims or prg.p.global_size, local_dims or prg.p.local_size
      computeCommand = self.device.msg(self.icb, "indirectComputeCommandAtIndex:", j,res=new_var())
      self.device.msg(computeCommand,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
    
    for j, var in enumerate(self.vars):
      self.int_buf_view[j] = var_vals[var]


    
    if len(self.vars) > 0:
      # in gpt2, int_buf_view = 8 bytes, start_pos and prev token
      formatted_hex = ' '.join(f'{b:02x}' for b in self.int_buf_view.tobytes())
      self.device.msg("copyin",formatted_hex,self.int_buf.buf)
    # has to do this?

    command_buffer = self.device.msg(self.device.mtl_queue,"commandBuffer",res=new_var())
    encoder = self.device.msg(command_buffer,"computeCommandEncoder",res=new_var())
    self.device.msg(encoder,"useResources:count:usage:",*all_resources,
            "MTLResourceUsage.MTLResourceUsageRead | MTLResourceUsage.MTLResourceUsageWrite") #can infer len in objc

    self.device.msg(encoder,"executeCommandsInBuffer:withRange:",self.icb,len(self.jit_cache)) #range is 0-len(jit_cache)
    self.device.msg(encoder,"endEncoding")
    self.device.msg(command_buffer,"commit")
    self.command_buffer = command_buffer

    self.device.mtl_buffers_in_flight.append(command_buffer)
    return None