from typing import List, Any, Dict, cast, Optional
import ctypes
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, getenv
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.ops import Variable
from tinygrad.runtime.ops_metal import wait_check, msg, msg_ios, libobjc, to_struct, objc_instance,\
  MTLResourceOptions, objc_id

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
    icb_descriptor = msg_ios(b"MTLIndirectCommandBufferDescriptor", "new", restype=objc_instance)
    msg_ios(icb_descriptor, "setCommandTypes:", MTLIndirectCommandType.MTLIndirectCommandTypeConcurrentDispatch)
    msg_ios(icb_descriptor, "setInheritBuffers:", False)
    msg_ios(icb_descriptor, "setInheritPipelineState:", False)
    msg_ios(icb_descriptor, "setMaxKernelBufferBindCount:", 31)

    self.icb = msg_ios(self.device.device, "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
      icb_descriptor, len(self.jit_cache), MTLResourceOptions.MTLResourceCPUCacheModeDefaultCache, restype=objc_instance)

    if len(self.vars): self.int_buf = self.device.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    all_resources = [self.int_buf.buf] if len(self.vars) else []
    all_pipelines = []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = msg_ios(self.icb, "indirectComputeCommandAtIndex:", j, restype=objc_instance)
      all_pipelines.append(prg.clprg.pipeline_state)
      msg_ios(icb_command, "setComputePipelineState:", prg.clprg.pipeline_state)
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          msg_ios(icb_command, "setKernelBuffer:offset:atIndex:", b._buf.buf, b._buf.offset, i)

      global_size, local_size = prg.p.launch_dims(var_vals)
      msg_ios(icb_command, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", tuple(global_size), tuple(local_size))
      msg_ios(icb_command, "setBarrier")

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    self.command_buffer: Any = None
    if len(self.vars): self.int_buf_view = self.device.allocator.as_buffer(self.int_buf).cast('i')
    self.range = tuple([0, len(self.jit_cache)])

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:

    if self.command_buffer is not None and self.command_buffer in self.device.mtl_buffers_in_flight: wait_check(self.command_buffer)
    all_resources = dedup(self.all_resources + [x._buf.buf for x in input_rawbuffers])

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      prg = cast(CompiledRunner, self.jit_cache[j].prg)
      global_size, local_size = global_dims or prg.p.global_size, local_dims or prg.p.local_size
      computeCommand = msg_ios(self.icb, "indirectComputeCommandAtIndex:", j,restype=objc_id)
      msg_ios(computeCommand, "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
                  tuple(global_size), tuple(local_size))
    for j, var in enumerate(self.vars): self.int_buf_view[j] = var_vals[var]

    command_buffer = msg_ios(self.device.mtl_queue, "commandBuffer", restype=objc_instance)
    encoder = msg_ios(command_buffer, "computeCommandEncoder", restype=objc_instance)
    msg_ios(encoder, "useResources:count:usage:", all_resources, len(all_resources),
      MTLResourceUsage.MTLResourceUsageRead | MTLResourceUsage.MTLResourceUsageWrite)

    msg_ios(encoder, "executeCommandsInBuffer:withRange:", self.icb, self.range)
    msg_ios(encoder, "endEncoding")
    msg_ios(command_buffer, "commit")
    return None
