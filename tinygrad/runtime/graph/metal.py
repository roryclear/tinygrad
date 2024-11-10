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
    icb_descriptor_ios = self.device.msg_ios("MTLIndirectCommandBufferDescriptor", "new", res=new_var())
    self.device.msg_ios(icb_descriptor_ios,"setCommandTypes:","MTLIndirectCommandTypeConcurrentDispatch")
    self.device.msg_ios(icb_descriptor_ios,"setInheritBuffers:","false")
    self.device.msg_ios(icb_descriptor_ios,"setInheritPipelineState:","false")
    self.device.msg_ios(icb_descriptor_ios,"setMaxKernelBufferBindCount:",31)

    self.icb_ios = self.device.msg_ios("d","newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",icb_descriptor_ios,len(self.jit_cache),
      "MTLResourceCPUCacheModeDefaultCache",res=new_var())
    #if self.icb.value is None: raise GraphException("create indirect command buffer failed, does your system support this?") works for iphone 13 apple 8
    #https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    self.needs_icb_fix = True #todo, assuming true for my iphone

    if len(self.vars): 
      self.int_buf = self.device.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    all_resources = [self.int_buf] if len(self.vars) else []
    all_pipelines_ios = []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command_ios = self.device.msg_ios(self.icb_ios,"indirectComputeCommandAtIndex:",j,res=new_var())
      all_pipelines_ios.append(prg.clprg.pipeline_state_ios)
      self.device.msg_ios(icb_command_ios, "setComputePipelineState:", prg.clprg.pipeline_state_ios)
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          self.device.msg_ios(icb_command_ios,"setKernelBuffer:offset:atIndex:",b._buf.buf_ios,b._buf.offset,i)
          all_resources.append(b._buf)
      for i,v in enumerate(prg.p.vars):
        self.device.msg_ios(icb_command_ios,"setKernelBuffer:offset:atIndex:",self.int_buf.buf_ios,self.vars.index(v)*4, len(ji.bufs)+i)

      global_size, local_size = prg.p.launch_dims(var_vals)
      self.device.msg_ios(icb_command_ios,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
      self.device.msg_ios(icb_command_ios,"setBarrier")

    self.all_resources = dedup(all_resources)
    self.all_pipelines_ios = dedup(all_pipelines_ios) #ns what this does but metal does it 
    self.command_buffer: Any = None
    self.command_buffer_ios: Any = None
    if len(self.vars):
      self.int_buf_view_ios = self.device.allocator.as_buffer_ios(self.int_buf).cast('i')

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    if self.command_buffer is not None and [self.command_buffer,self.command_buffer_ios] in self.device.mtl_buffers_in_flight:
      self.device.msg_ios(self.command_buffer_ios,"waitUntilCompleted")
    all_resources = dedup(self.all_resources + [x._buf for x in input_rawbuffers])
    
    for (j,i),input_idx in self.input_replace.items():
      computeCommand_ios = self.device.msg_ios(self.icb_ios, "indirectComputeCommandAtIndex:", j, res=new_var())
      self.device.msg_ios(computeCommand_ios, "setKernelBuffer:offset:atIndex:", input_rawbuffers[input_idx]._buf.buf_ios,
                                                                                 input_rawbuffers[input_idx]._buf.offset, i)

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      prg = cast(CompiledRunner, self.jit_cache[j].prg)
      global_size, local_size = global_dims or prg.p.global_size, local_dims or prg.p.local_size
      computeCommand_ios = self.device.msg_ios(self.icb_ios, "indirectComputeCommandAtIndex:", j,res=new_var())
      self.device.msg_ios(computeCommand_ios,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
    
    for j, var in enumerate(self.vars):
      self.int_buf_view_ios[j] = var_vals[var]


    
    if len(self.vars) > 0:
      # in gpt2, int_buf_view = 8 bytes, start_pos and prev token
      formatted_hex = ' '.join(f'{b:02x}' for b in self.int_buf_view_ios.tobytes())
      self.device.msg_ios("copyin",formatted_hex,self.int_buf.buf_ios)
    # has to do this?

    command_buffer_ios = self.device.msg_ios(self.device.mtl_queue_ios,"commandBuffer",res=new_var())
    encoder_ios = self.device.msg_ios(command_buffer_ios,"computeCommandEncoder",res=new_var())
    ios_res = [x.buf_ios for x in all_resources]
    self.device.msg_ios(encoder_ios,"useResources:count:usage:",*ios_res,
            "MTLResourceUsage.MTLResourceUsageRead | MTLResourceUsage.MTLResourceUsageWrite") #can infer len in objc

    self.device.msg_ios(encoder_ios,"executeCommandsInBuffer:withRange:",self.icb_ios,len(self.jit_cache)) #range is 0-len(jit_cache)
    self.device.msg_ios(encoder_ios,"endEncoding")
    self.device.msg_ios(command_buffer_ios,"commit")
    self.command_buffer_ios = command_buffer_ios

    self.device.mtl_buffers_in_flight.append([None,command_buffer_ios])
    return None