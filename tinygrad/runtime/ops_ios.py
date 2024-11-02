from __future__ import annotations
import os, functools
from typing import List, Any, Tuple, Optional
from tinygrad.helpers import getenv
from tinygrad.device import Compiled, Compiler, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.helpers import dedup, getenv
from tinygrad.dtype import dtypes
from tinygrad.device import Buffer
from typing import List, Any, Tuple, Optional, cast, TypeVar, Dict
from tinygrad.ops import Variable
import requests
import time
import json
import gzip
import ctypes

var_num = -1
def new_var():
  global var_num
  return "v" + str(var_num:=var_num+1)

class iosCompiler(Compiler):
  def __init__(self, device:Optional[iosDevice]):
    self.device = device
    if hasattr(self.device,"queue") == False:
      self.device.queue = {"queue":[]}
    super().__init__("compile_ios")
  def compile(self, src:str) -> bytes:
    name = src[src.index("void")+5:]
    name = name[:name.index("(")]
    self.device.queue["queue"].append(["new_library",src,name])
    #self.device.send_queue()
    return

class iosProgram:
  def __init__(self, device:iosDevice, name:str, lib:bytes):  
    self.device, self.name, self.lib = device, name, lib
    self.fxn = new_var()
    self.device.queue["queue"].append(["new_function",name,self.fxn])
    self.pipeline_state = new_var()
    self.device.queue["queue"].append(["new_pipeline_state",self.fxn,self.pipeline_state])

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    command_buffer = new_var()
    self.device.queue["queue"].append(["new_command_buffer",command_buffer])
    self.device.queue["queue"].append(["set_encoder",command_buffer])
    self.device.queue["queue"].append(["set_pipeline_state",self.pipeline_state])
    for i,a in enumerate(bufs): 
      self.device.queue["queue"].append(["set_buffer",a.buf,a.offset,i])
    for i,a in enumerate(vals,start=len(bufs)):
      self.device.queue["queue"].append(["set_bytes",' '.join(f"{(a >> (i * 8)) & 0xff:02x}" for i in range(4)),4,i]) #TODO
    self.device.queue["queue"].append(["dispatch",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2]])
    self.device.queue["queue"].append(["commit",command_buffer])
    self.device.mtl_buffers_in_flight.append(command_buffer)

class iosBuffer:
  def __init__(self, buf:Any, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class iosAllocator(LRUAllocator):
  def __init__(self, device:iosDevice):
    self.device:iosDevice = device
    super().__init__()
  def _alloc(self, size:int, options) -> iosBuffer:
    self.device.queue["queue"].append(["new_buffer",ret:=new_var(),size])
    if len(self.device.queue["queue"]) > 1000: #TODO remove this, check nbytes instead?
      self.device.send_queue()
    return iosBuffer(ret, size)
  def as_buffer(self, src:iosBuffer) -> memoryview:
    self.device.synchronize()
    var = new_var()
    self.device.queue["queue"].append(["copyout",str(src.buf)])
    byte_str = self.device.send_queue()
    byte_values = bytearray(int(b, 16) for b in byte_str.split())
    return memoryview(byte_values)
  def copy_from_disk(self,dest,src):
    file_name = src.device[::-1]
    file_name = file_name[:file_name.index("/")]
    file_name = file_name[::-1]
    buf_name = str(dest._buf.buf)
    self.device.queue["queue"].append(["memcpy",buf_name,file_name,src.offset,src.nbytes])
  def copyin(self, dest:iosBuffer, src:memoryview):
    formatted_hex = ' '.join(f'{b:02x}' for b in src)
    self.device.queue["queue"].append(["copy_in",formatted_hex,dest.buf])
  def copyout(self, dest:memoryview, src:iosBuffer):
    self.device.synchronize()
    self.device.queue["queue"].append(["copyout",str(src._buf.buf)])
    byte_str = self.device.send_queue()
    byte_values = bytes(int(b, 16) for b in byte_str.split())
    dest[:] = memoryview(byte_values)
  def offset(self, buf:iosBuffer, size:int, offset:int): return iosBuffer(buf.buf, size, offset)

def msg(*args,restype=None):
  #print("msg",args)
  return None

class MTLResourceOptions:
  MTLResourceCPUCacheModeDefaultCache = 0
  MTLResourceStorageModeShared = 0 << 4

class MTLPipelineOption:
  MTLPipelineOptionNone = 0

class MTLIndirectCommandType:
  MTLIndirectCommandTypeConcurrentDispatch = (1 << 5)

class MTLResourceUsage:
  MTLResourceUsageRead = 0b01
  MTLResourceUsageWrite = 0b10


def to_struct(*t: int, _type: type = ctypes.c_ulong):
  class Struct(ctypes.Structure): pass
  Struct._fields_ = [(f"field{i}", _type) for i in range(len(t))]
  return Struct(*t)

class objc_id(ctypes.c_void_p): # This prevents ctypes from converting response to plain int, and dict.fromkeys() can use it to dedup
  def __hash__(self): return hash(self.value)
  def __eq__(self, other): return self.value == other.value

class objc_instance(objc_id): # method with name "new", "alloc" should be freed after use
  def __del__(self): msg(self, "release")

def wait_check(cbuf: Any):
  msg(cbuf, "waitUntilCompleted")
  #error_check(msg(cbuf, "error", restype=objc_instance))

def error_check(error: objc_instance, error_constructor: type[Exception] = RuntimeError):
  if error.value is None: return None
  raise error_constructor(bytes(msg(msg(error, "localizedDescription", restype=objc_instance), "UTF8String", restype=ctypes.c_char_p)).decode())

def elapsed_time(cbuf: objc_id):
  return cast(float, msg(cbuf, "GPUEndTime", restype=ctypes.c_double)) - cast(float, msg(cbuf, "GPUStartTime", restype=ctypes.c_double))

class iosGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    libobjc = ctypes.CDLL("/usr/lib/libobjc.dylib")
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    # create metal batch exec
    icb_descriptor = msg(libobjc.objc_getClass(b"MTLIndirectCommandBufferDescriptor"), "new", restype=objc_instance)
    msg(icb_descriptor, "setCommandTypes:", MTLIndirectCommandType.MTLIndirectCommandTypeConcurrentDispatch)
    msg(icb_descriptor, "setInheritBuffers:", False)
    msg(icb_descriptor, "setInheritPipelineState:", False)
    msg(icb_descriptor, "setMaxKernelBufferBindCount:", 31)
    self.icb = msg(self.device.device, "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
      icb_descriptor, len(self.jit_cache), MTLResourceOptions.MTLResourceCPUCacheModeDefaultCache, restype=objc_instance)
    
    self.icb_ios = new_var()
    self.device.queue["queue"].append(["new_icb",len(self.jit_cache),self.icb_ios])


    if len(self.vars): 
      self.int_buf = self.device.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    all_resources = [self.int_buf.buf] if len(self.vars) else []
    all_pipelines = []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = msg(self.icb, "indirectComputeCommandAtIndex:", j, restype=objc_instance)
      icb_command_ios = new_var()
      self.device.queue["queue"].append(["icb_command",self.icb_ios,j,icb_command_ios])


      all_pipelines.append(prg.clprg.pipeline_state)
      msg(icb_command, "setComputePipelineState:", prg.clprg.pipeline_state)
      self.device.queue["queue"].append(["set_icb_pipeline_state",icb_command_ios,prg.clprg.pipeline_state])
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          msg(icb_command, "setKernelBuffer:offset:atIndex:", b._buf.buf, b._buf.offset, i)
          self.device.queue["queue"].append(["set_kernel_buffer",icb_command_ios,b._buf.buf, b._buf.offset, i])
          all_resources.append(b._buf.buf)
      for i,v in enumerate(prg.p.vars):
        #print("i =",i,"\t",self.int_buf.buf, self.vars.index(v)*4, len(ji.bufs)+i)
        msg(icb_command, "setKernelBuffer:offset:atIndex:", self.int_buf.buf, self.vars.index(v)*4, len(ji.bufs)+i)
        self.device.queue["queue"].append(["set_kernel_buffer_int",icb_command_ios,self.int_buf.buf, self.vars.index(v)*4, len(ji.bufs)+i])

      global_size, local_size = prg.p.launch_dims(var_vals)
      msg(icb_command, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", to_struct(*global_size), to_struct(*local_size))
      msg(icb_command, "setBarrier")
      self.device.queue["queue"].append(["concurrent_dispatch_and_barrier",icb_command_ios,global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2]])

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    if len(self.vars): self.int_buf_view = self.device.allocator.as_buffer(self.int_buf).cast('i')
    self.range = to_struct(0, len(self.jit_cache))

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    all_resources = dedup(self.all_resources + [x._buf.buf for x in input_rawbuffers])

    for (j,i),input_idx in self.input_replace.items():
      computeCommand = msg(self.icb, "indirectComputeCommandAtIndex:", j, restype=objc_id)
      msg(computeCommand, "setKernelBuffer:offset:atIndex:",input_rawbuffers[input_idx]._buf.buf,input_rawbuffers[input_idx]._buf.offset, i)
      self.device.queue["queue"].append(["input_replace",self.icb_ios,j,input_rawbuffers[input_idx]._buf.buf,input_rawbuffers[input_idx]._buf.offset, i])

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      prg = cast(CompiledRunner, self.jit_cache[j].prg)
      global_size, local_size = global_dims or prg.p.global_size, local_dims or prg.p.local_size
      computeCommand = msg(self.icb, "indirectComputeCommandAtIndex:", j)
      msg(computeCommand, "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
                  to_struct(*cast(tuple, global_size)), to_struct(*cast(tuple, local_size)))
      self.device.queue["queue"].append(["concurrent_dispatch",self.icb_ios,j,global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2]])
    for j, var in enumerate(self.vars): 
      self.int_buf_view[j] = var_vals[var]

    #command_buffer = msg(self.device.mtl_queue, "commandBuffer", restype=objc_instance)
    #encoder = msg(command_buffer, "computeCommandEncoder", restype=objc_instance)
    command_buffer = new_var()
    #encoder = new_var()
    #print(all_resources)
    #msg(encoder, " ", (objc_id * len(all_resources))(*all_resources), len(all_resources), #still add this !!
    #    MTLResourceUsage.MTLResourceUsageRead | MTLResourceUsage.MTLResourceUsageWrite)

    #msg(encoder, "executeCommandsInBuffer:withRange:", self.icb, self.range)
    #msg(encoder, "endEncoding")
    msg(command_buffer, "commit")
    self.device.queue["queue"].append(["commit_2",command_buffer,len(all_resources),*all_resources,self.icb_ios,len(self.jit_cache)])
    wait_check(command_buffer)

    if wait:
      wait_check(command_buffer)
      return elapsed_time(command_buffer)
    self.device.mtl_buffers_in_flight.append(command_buffer)
    return None

class iosDevice(Compiled):
  def __init__(self, device:str):
    self.queue = {"queue":[]} #todo
    self.device = new_var()
    #self.mtl_queue = new_var()
    self.mtl_queue = None
    self.mtl_buffers_in_flight: List[Any] = []

    super().__init__(device, iosAllocator(self), MetalRenderer(), iosCompiler(self if getenv("METAL_XCODE") else self),
                     functools.partial(iosProgram, self), iosGraph)
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight:
      self.queue["queue"].append(["wait",cbuf])
    self.mtl_buffers_in_flight.clear()

  def send_queue(self):
    #url = "http://192.168.1.105:8081" #your iOS device's local IP
    url = "http://192.168.1.12:8081"
    #payload = self.queue
    payload = json.dumps(self.queue) # Compress the JSON string 
    compressed_payload = gzip.compress(payload.encode('utf-8'))
    status = 400
    while status != 200:
      try:
        headers = {'Content-Encoding': 'gzip', 'Content-Type': 'application/json'}
        response = requests.post(url, compressed_payload,headers=headers,timeout=3600)
        self.queue = {"queue":[]} #TODO: hack to not crash iOS
        if response.status_code == 200:
            status = 200
            if len(response.text) > 0:
              return response.text
        else:
            time.sleep(0.1)
      except requests.exceptions.RequestException as e:
        #print("An error occurred:", e)
        time.sleep(0.2)