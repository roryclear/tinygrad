from __future__ import annotations
import functools
from typing import List, Any, Tuple, Optional, cast, Dict
from tinygrad.device import Compiled, Compiler, LRUAllocator, Buffer
from tinygrad.renderer.cstyle import iOSRenderer
import json, gzip, requests, time
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, prod, IP
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.ops import Variable


var_num = -1
def new_var():
  global var_num
  return "v" + str(var_num:=var_num+1)

class IOSProgram:
  def __init__(self, dev:IOSDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib
    self.dev.msg("delete_files","0") #delete weights after copying to metal buffer, 
    options = self.dev.msg("MTLCompileOptions","new",res=new_var())
    code_ns_str = lib.decode()
    self.library = self.dev.msg("d","newLibraryWithSource:options:error:",code_ns_str,options,"error",res=new_var())
    self.fxn = self.dev.msg(self.library, "newFunctionWithName:", name, res=new_var())
    descriptor = self.dev.msg("MTLComputePipelineDescriptor", "new", res=new_var())
    self.dev.msg(descriptor, "setComputeFunction:", self.fxn)
    self.dev.msg(descriptor, "setSupportIndirectCommandBuffers:", "true")
    self.pipeline_state = self.dev.msg("d","newComputePipelineStateWithDescriptor:options:reflection:error:",
    descriptor,0,"none",res=new_var())

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if wait: #todo, can potentially crash iOS
      max_total_threads = self.dev.msg(self.pipeline_state, "maxTotalThreadsPerThreadgroup")
      if prod(local_size) > int(max_total_threads):
        exec_width = 0#todo self.dev.msg(self.pipeline_state, "threadExecutionWidth")
        memory_length = 0#todo self.dev.msg(self.pipeline_state, "staticThreadgroupMemoryLength")
        raise RuntimeError(f"local size {local_size} bigger than {max_total_threads} with exec width {exec_width} memory length {memory_length}")
    command_buffer = self.dev.msg(self.dev.mtl_queue,"commandBuffer",res=new_var())
    encoder = self.dev.msg(command_buffer,"computeCommandEncoder",res=new_var())
    self.dev.msg(encoder,"setComputePipelineState:",self.pipeline_state)
    for i,a in enumerate(bufs):
      self.dev.msg(encoder,"setBuffer:offset:atIndex:",a.buf,a.offset,i)
    for i,a in enumerate(vals,start=len(bufs)): 
      self.dev.msg(encoder,"setBytes:length:atIndex:",' '.join(f"{(a >> (i * 8)) & 0xff:02x}" for i in range(4)),4,i)
    self.dev.msg(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
    self.dev.msg(encoder,"endEncoding")
    self.dev.msg(command_buffer,"commit")
    self.dev.mtl_buffers_in_flight.append(command_buffer)
    if wait:
      for cbuf in self.dev.mtl_buffers_in_flight: self.dev.msg(cbuf,"waitUntilCompleted")
      self.dev.mtl_buffers_in_flight.clear()
      return float(self.dev.msg(command_buffer,"elapsed_time"))

class IOSBuffer:
  def __init__(self, buf:Any, size:int, offset=0): 
    self.buf, self.size, self.offset = buf, size, offset
    if buf == None: 
      self.buf = new_var()

class IOSAllocator(LRUAllocator):
  def __init__(self, dev:IOSDevice):
    self.dev:IOSDevice = dev
    super().__init__()
  def _alloc(self, size:int, options) -> IOSBuffer:
    buf = self.dev.msg("d","newBufferWithLength:options:",size,0,res=new_var())
    return IOSBuffer(buf, size)
  def as_buffer(self, src:IOSBuffer) -> memoryview:
    for cbuf in self.dev.mtl_buffers_in_flight: self.dev.msg(cbuf,"waitUntilCompleted")
    self.dev.mtl_buffers_in_flight.clear()
    byte_str = self.dev.msg(src.buf,"copyout")
    byte_values = bytearray(int(b, 16) for b in byte_str.split())
    return memoryview(byte_values[:src.size]) 
  
  def copy_from_disk(self,dest,src,nbytes):
    self.dev.send_bytes(str(dest.buf),src._buf().tobytes(),nbytes)
  
  def _copyin(self, dest:IOSBuffer, src:memoryview):
    for cbuf in self.dev.mtl_buffers_in_flight: self.dev.msg(cbuf,"waitUntilCompleted")
    self.dev.mtl_buffers_in_flight.clear()

    formatted_hex = ' '.join(f'{b:02x}' for b in src)
    self.dev.msg("copyin",formatted_hex,dest.buf)

  def _copyout(self, dest:memoryview, src:IOSBuffer): 
    dest[:] = self.as_buffer(src)
  def _offset(self, buf:IOSBuffer, size:int, offset:int):
    return IOSBuffer(buf.buf, size, offset)

class IOSGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    # create IOS batch exec
    icb_descriptor = self.dev.msg("MTLIndirectCommandBufferDescriptor", "new", res=new_var())
    self.dev.msg(icb_descriptor,"setCommandTypes:","MTLIndirectCommandTypeConcurrentDispatch")
    self.dev.msg(icb_descriptor,"setInheritBuffers:","false")
    self.dev.msg(icb_descriptor,"setInheritPipelineState:","false")
    self.dev.msg(icb_descriptor,"setMaxKernelBufferBindCount:",31)

    self.icb = self.dev.msg("d","newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",icb_descriptor,len(self.jit_cache),
      "MTLResourceCPUCacheModeDefaultCache",res=new_var())

    if len(self.vars): 
      self.int_buf = self.dev.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    all_resources = [self.int_buf.buf] if len(self.vars) else []
    all_pipelines = []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = self.dev.msg(self.icb,"indirectComputeCommandAtIndex:",j,res=new_var())
      all_pipelines.append(prg._prg.pipeline_state)
      self.dev.msg(icb_command, "setComputePipelineState:", prg._prg.pipeline_state)
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          self.dev.msg(icb_command,"setKernelBuffer:offset:atIndex:",b._buf.buf,b._buf.offset,i)
          all_resources.append(b._buf.buf)
      for i,v in enumerate(prg.p.vars):
        self.dev.msg(icb_command,"setKernelBuffer:offset:atIndex:",self.int_buf.buf,self.vars.index(v)*4, len(ji.bufs)+i)

      global_size, local_size = prg.p.launch_dims(var_vals)
      self.dev.msg(icb_command,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
      self.dev.msg(icb_command,"setBarrier")

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    self.command_buffer: Any = None
    if len(self.vars):
      self.int_buf_view = self.dev.allocator.as_buffer(self.int_buf).cast('i')

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    if self.command_buffer is not None and self.command_buffer in self.dev.mtl_buffers_in_flight:
      self.dev.msg(self.command_buffer,"waitUntilCompleted")
    all_resources = dedup(self.all_resources + [x._buf.buf for x in input_rawbuffers])
    
    for (j,i),input_idx in self.input_replace.items():
      computeCommand = self.dev.msg(self.icb, "indirectComputeCommandAtIndex:", j, res=new_var())
      self.dev.msg(computeCommand, "setKernelBuffer:offset:atIndex:", input_rawbuffers[input_idx]._buf.buf,
                                                                                 input_rawbuffers[input_idx]._buf.offset, i)

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      prg = cast(CompiledRunner, self.jit_cache[j].prg)
      global_size, local_size = global_dims or prg.p.global_size, local_dims or prg.p.local_size
      computeCommand = self.dev.msg(self.icb, "indirectComputeCommandAtIndex:", j,res=new_var())
      self.dev.msg(computeCommand,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
    
    for j, var in enumerate(self.vars):
      self.int_buf_view[j] = var_vals[var]

    if len(self.vars) > 0:
      formatted_hex = ' '.join(f'{b:02x}' for b in self.int_buf_view.tobytes())
      self.dev.msg("copyin",formatted_hex,self.int_buf.buf)

    command_buffer = self.dev.msg(self.dev.mtl_queue,"commandBuffer",res=new_var())
    encoder = self.dev.msg(command_buffer,"computeCommandEncoder",res=new_var())
    self.dev.msg(encoder,"useResources:count:usage:",*all_resources,
            "MTLResourceUsage.MTLResourceUsageRead | MTLResourceUsage.MTLResourceUsageWrite")

    self.dev.msg(encoder,"executeCommandsInBuffer:withRange:",self.icb,len(self.jit_cache))
    self.dev.msg(encoder,"endEncoding")
    self.dev.msg(command_buffer,"commit")
    self.command_buffer = command_buffer

    self.dev.mtl_buffers_in_flight.append(command_buffer)
    if wait:
      for cbuf in self.dev.mtl_buffers_in_flight: self.dev.msg(cbuf,"waitUntilCompleted")
      self.dev.mtl_buffers_in_flight.clear()
      return float(self.dev.msg(command_buffer,"elapsed_time"))

class IOSDevice(Compiled):
  def __init__(self, device:str):
    self.sysdevice = "d"
    self.ip = "http://" + IP.value + ":8081"
    self.queue = []
    self.files = set()
    self.msg("delete","x")
    self.mtl_queue = self.msg("d","newCommandQueueWithMaxCommandBufferCount:",1024,res=new_var())
    self.mtl_buffers_in_flight: List[Any] = []

    super().__init__(device, IOSAllocator(self), iOSRenderer(), Compiler(),
                     functools.partial(IOSProgram, self), IOSGraph)
    

  def msg(self,ptr,selector,*args,res=None):
    req = [ptr,selector]
    req.append(len(args))
    for x in args: req.append(x)
    if res != None: req.append(res)
    self.queue.append(req)
    if selector in ["copyout","maxTotalThreadsPerThreadgroup","elapsed_time","file_exists"]:
      return self.send_queue() or res
    return res

  def send_bytes(self,dest,data,size):
    self.send_queue()
    url = self.ip + "/bytes/" + str(size) + "/" + dest
    return self.send_req(url,data)

  def send_queue(self):
    payload = json.dumps(self.queue)
    compressed_payload = gzip.compress(payload.encode('utf-8'))
    size = len(compressed_payload)
    url = self.ip + "/" + str(size)
    self.queue = []
    return self.send_req(url,compressed_payload)

  def send_req(self,ip,data):
    retries = 0
    while retries < 20:
      try:
        headers = {}
        response = requests.post(ip, data=data,headers=headers,timeout=3600)
        if response.status_code == 200:
            if len(response.text) > 0:
              return response.text
            return
        else:
            time.sleep(0.1)
            retries +=1
      except requests.exceptions.RequestException:
        time.sleep(0.1)
        retries += 1
    raise Exception("Maximum retries reached.")
      


