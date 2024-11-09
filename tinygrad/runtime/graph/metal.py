from typing import List, Any, Dict, cast, Optional
import ctypes
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, getenv
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.ops import Variable
from tinygrad.runtime.ops_metal import msg, libobjc, to_struct, objc_instance,\
  MTLResourceOptions, objc_id, msg_ios,new_var, MetalAllocator, MetalDevice

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
    icb_descriptor = msg(libobjc.objc_getClass(b"MTLIndirectCommandBufferDescriptor"), "new", restype=objc_instance)
    icb_descriptor_ios = msg_ios("MTLIndirectCommandBufferDescriptor", "new", res=new_var())
    msg(icb_descriptor, "setCommandTypes:", MTLIndirectCommandType.MTLIndirectCommandTypeConcurrentDispatch)
    msg_ios(icb_descriptor_ios,"setCommandTypes:","MTLIndirectCommandTypeConcurrentDispatch")
    msg(icb_descriptor, "setInheritBuffers:", False)
    msg_ios(icb_descriptor_ios,"setInheritBuffers:","false")
    msg(icb_descriptor, "setInheritPipelineState:", False)
    msg_ios(icb_descriptor_ios,"setInheritPipelineState:","false")
    msg(icb_descriptor, "setMaxKernelBufferBindCount:", 31)
    msg_ios(icb_descriptor_ios,"setMaxKernelBufferBindCount:",31)

    self.icb = msg(self.device.device, "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
      icb_descriptor, len(self.jit_cache), MTLResourceOptions.MTLResourceCPUCacheModeDefaultCache, restype=objc_instance)
    self.icb_ios = msg_ios("d","newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",icb_descriptor_ios,len(self.jit_cache),
      "MTLResourceCPUCacheModeDefaultCache",res=new_var())
    #if self.icb.value is None: raise GraphException("create indirect command buffer failed, does your system support this?") works for iphone 13 apple 8
    #https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    self.needs_icb_fix = True #todo, assuming true for my iphone

    if len(self.vars): 
      self.int_buf = self.device.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    all_resources = [self.int_buf] if len(self.vars) else []
    all_pipelines = []
    all_pipelines_ios = []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = msg(self.icb, "indirectComputeCommandAtIndex:", j, restype=objc_instance)
      icb_command_ios = msg_ios(self.icb_ios,"indirectComputeCommandAtIndex:",j,res=new_var())
      all_pipelines.append(prg.clprg.pipeline_state)
      all_pipelines_ios.append(prg.clprg.pipeline_state_ios)
      msg(icb_command, "setComputePipelineState:", prg.clprg.pipeline_state)
      msg_ios(icb_command_ios, "setComputePipelineState:", prg.clprg.pipeline_state_ios)
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          msg(icb_command, "setKernelBuffer:offset:atIndex:", b._buf.buf, b._buf.offset, i)
          msg_ios(icb_command_ios,"setKernelBuffer:offset:atIndex:",b._buf.buf_ios,b._buf.offset,i)
          all_resources.append(b._buf)
      for i,v in enumerate(prg.p.vars):
        msg(icb_command, "setKernelBuffer:offset:atIndex:", self.int_buf.buf, self.vars.index(v)*4, len(ji.bufs)+i)
        msg_ios(icb_command_ios,"setKernelBuffer:offset:atIndex:",self.int_buf.buf_ios,self.vars.index(v)*4, len(ji.bufs)+i)

      global_size, local_size = prg.p.launch_dims(var_vals)
      msg(icb_command, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", to_struct(*global_size), to_struct(*local_size))
      msg_ios(icb_command_ios,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
      msg(icb_command, "setBarrier")
      msg_ios(icb_command_ios,"setBarrier")

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    self.all_pipelines_ios = dedup(all_pipelines_ios) #ns what this does but metal does it 
    self.command_buffer: Any = None
    self.command_buffer_ios: Any = None
    if len(self.vars):
      self.int_buf_view = self.device.allocator.as_buffer_metal(self.int_buf).cast('i')
    self.range = to_struct(0, len(self.jit_cache))

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    if self.command_buffer is not None and [self.command_buffer,self.command_buffer_ios] in self.device.mtl_buffers_in_flight:
      msg(self.command_buffer, "waitUntilCompleted")
      msg_ios(self.command_buffer_ios,"waitUntilCompleted")
    print("rory res")
    print(self.all_pipelines)
    all_resources = dedup(self.all_resources + [x._buf for x in input_rawbuffers])
    
    #for x in all_resources:
    #  print("value for resource",x,x.buf_ios,"size =",x.size)
    #  MetalAllocator.as_buffer(MetalAllocator(MetalDevice("a")),x)


    for (j,i),input_idx in self.input_replace.items():
      computeCommand = msg(self.icb, "indirectComputeCommandAtIndex:", j, restype=objc_id)
      computeCommand_ios = msg_ios(self.icb_ios, "indirectComputeCommandAtIndex:", j, res=new_var())
      print("rory offset =",input_rawbuffers[input_idx]._buf.offset)
      msg(computeCommand, "setKernelBuffer:offset:atIndex:", input_rawbuffers[input_idx]._buf.buf,
                                                                                 input_rawbuffers[input_idx]._buf.offset, i)
      msg_ios(computeCommand_ios, "setKernelBuffer:offset:atIndex:", input_rawbuffers[input_idx]._buf.buf_ios,
                                                                                 input_rawbuffers[input_idx]._buf.offset, i)

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      print("RORY VAR_VALS")
      prg = cast(CompiledRunner, self.jit_cache[j].prg)
      global_size, local_size = global_dims or prg.p.global_size, local_dims or prg.p.local_size
      computeCommand = msg(self.icb, "indirectComputeCommandAtIndex:", j)
      computeCommand_ios = msg_ios(self.icb_ios, "indirectComputeCommandAtIndex:", j,res=new_var())
      msg(computeCommand, "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
                  to_struct(*cast(tuple, global_size)), to_struct(*cast(tuple, local_size)))
      msg_ios(computeCommand_ios,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
    
    for j, var in enumerate(self.vars):
      print("rory j var var_vals[var]? =",j,var,var_vals[var])
      self.int_buf_view[j] = var_vals[var]
    print("int_buf_view =",self.int_buf_view,self.int_buf_view.tobytes())
    print("int buf =",self.int_buf)
    # in gpt2, int_buf_view = 8 bytes, start_pos and prev token

    
    if len(self.vars) > 0:
      formatted_hex = ' '.join(f'{b:02x}' for b in self.int_buf_view.tobytes())
      print("int_buf_view hex =",formatted_hex)
      msg_ios("copyin",formatted_hex,self.int_buf.buf_ios)
    # has to do this?

    command_buffer = msg(self.device.mtl_queue, "commandBuffer", restype=objc_instance)
    command_buffer_ios = msg_ios(self.device.mtl_queue_ios,"commandBuffer",res=new_var())
    encoder = msg(command_buffer, "computeCommandEncoder", restype=objc_instance)
    encoder_ios = msg_ios(command_buffer_ios,"computeCommandEncoder",res=new_var())
    metal_res = [x.buf for x in all_resources]
    ios_res = [x.buf_ios for x in all_resources]
    msg(encoder, "useResources:count:usage:", (objc_id * len(metal_res))(*metal_res), len(metal_res),
        MTLResourceUsage.MTLResourceUsageRead | MTLResourceUsage.MTLResourceUsageWrite)
    msg_ios(encoder_ios,"useResources:count:usage:",*ios_res,
            "MTLResourceUsage.MTLResourceUsageRead | MTLResourceUsage.MTLResourceUsageWrite") #can infer len in objc

    # NOTE: the pipelines likely need to be added to the used resources to fix the crash on M1/M2, but I haven't figured out how
    # this is a O(n) hack to get them used. what should work is:
    #encoder.useResources_count_usage_(self.all_pipelines, len(self.all_pipelines), Metal.MTLResourceUsageRead)
    # but it fails with "Invalid Resource (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)"
    # to repro the crash (which can also crash other running GPU apps), run with FIX_METAL_ICB=0
    if getenv("FIX_METAL_ICB", self.needs_icb_fix):
      for ps in self.all_pipelines:
        msg(encoder, "setComputePipelineState:", ps)
        msg(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", to_struct(0,0,0), to_struct(0,0,0))
      #for ps in self.all_pipelines_ios:
        #msg_ios(encoder_ios,"setComputePipelineState:", ps)
        #msg_ios(encoder_ios, "dispatchThreadgroups:threadsPerThreadgroup:", 0,0,0,0,0,0) this crashes ios

    msg(encoder, "executeCommandsInBuffer:withRange:", self.icb, self.range)
    msg_ios(encoder_ios,"executeCommandsInBuffer:withRange:",self.icb_ios,len(self.jit_cache)) #range is 0-len(jit_cache)
    msg(encoder, "endEncoding")
    msg_ios(encoder_ios,"endEncoding")
    msg(command_buffer, "commit")
    msg_ios(command_buffer_ios,"commit")
    self.command_buffer = command_buffer
    self.command_buffer_ios = command_buffer_ios

    self.device.mtl_buffers_in_flight.append([command_buffer,command_buffer_ios])
    return None