from __future__ import annotations
import os, subprocess, pathlib, ctypes, tempfile, functools
from typing import List, Any, Tuple, Optional, cast
from tinygrad.helpers import prod, getenv, T
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer
import json, gzip, requests, time

class objc_id(ctypes.c_void_p): # This prevents ctypes from converting response to plain int, and dict.fromkeys() can use it to dedup
  def __hash__(self): return hash(self.value)
  def __eq__(self, other): return self.value == other.value

class objc_instance(objc_id): # method with name "new", "alloc" should be freed after use
  def __del__(self):
    msg(self, "release")

@functools.lru_cache(None)
def sel(name: str):
  return libobjc.sel_registerName(name.encode())

class MTLResourceOptions:
  MTLResourceCPUCacheModeDefaultCache = 0
  MTLResourceStorageModeShared = 0 << 4

class MTLPipelineOption:
  MTLPipelineOptionNone = 0

libobjc = ctypes.CDLL("/usr/lib/libobjc.dylib")
libmetal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
libdispatch = ctypes.CDLL("/usr/lib/libSystem.dylib") # libdispatch is part of libSystem on mac
libobjc.objc_getClass.restype = objc_id
libobjc.sel_registerName.restype = objc_id
libmetal.MTLCreateSystemDefaultDevice.restype = objc_instance
libdispatch.dispatch_data_create.restype = objc_instance

# Ignore mypy error reporting incompatible default, because typevar default only works on python 3.12
def msg(ptr: objc_id, selector: str, /, *args: Any, restype: type[T] = objc_id) -> T: # type: ignore [assignment]
  #print(ptr,selector,args)
  sender = libobjc["objc_msgSend"] # Using attribute access returns a new reference so setting restype is safe
  sender.restype = restype
  return sender(ptr, sel(selector), *args)

def to_struct(*t: int, _type: type = ctypes.c_ulong):
  class Struct(ctypes.Structure): pass
  Struct._fields_ = [(f"field{i}", _type) for i in range(len(t))]
  return Struct(*t)

class MetalProgram:
  def __init__(self, device:MetalDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    options = msg(libobjc.objc_getClass(b"MTLCompileOptions"), "new", restype=objc_instance)
    code_ns_str = msg(libobjc.objc_getClass(b"NSString"), "stringWithUTF8String:", lib, restype=objc_instance)
    self.library = msg(self.device.device, "newLibraryWithSource:options:error:", code_ns_str, options,
                  ctypes.byref(compileError:=objc_instance()), restype=objc_instance)
    name_ns_str = msg(libobjc.objc_getClass(b"NSString"), "stringWithUTF8String:", name.encode(), restype=objc_instance)
    self.fxn = msg(self.library, "newFunctionWithName:", name_ns_str, restype=objc_instance)
    descriptor = msg(libobjc.objc_getClass(b"MTLComputePipelineDescriptor"), "new", restype=objc_instance)
    msg(descriptor, "setComputeFunction:", self.fxn)
    msg(descriptor, "setSupportIndirectCommandBuffers:", True)
    self.pipeline_state = msg(self.device.device, "newComputePipelineStateWithDescriptor:options:reflection:error:",
      descriptor, MTLPipelineOption.MTLPipelineOptionNone, None, ctypes.byref(error_pipeline_creation:=objc_instance()), restype=objc_instance)

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    max_total_threads = msg(self.pipeline_state, "maxTotalThreadsPerThreadgroup", restype=ctypes.c_ulong)
    if prod(local_size) > cast(int, max_total_threads):
      exec_width = msg(self.pipeline_state, "threadExecutionWidth", restype=ctypes.c_ulong)
      memory_length = msg(self.pipeline_state, "staticThreadgroupMemoryLength", restype=ctypes.c_ulong)
      raise RuntimeError(f"local size {local_size} bigger than {max_total_threads} with exec width {exec_width} memory length {memory_length}")
    command_buffer = msg(self.device.mtl_queue, "commandBuffer", restype=objc_instance)
    encoder = msg(command_buffer, "computeCommandEncoder", restype=objc_instance)
    msg(encoder, "setComputePipelineState:", self.pipeline_state)
    for i,a in enumerate(bufs): msg(encoder, "setBuffer:offset:atIndex:", a.buf, a.offset, i)
    for i,a in enumerate(vals,start=len(bufs)): msg(encoder, "setBytes:length:atIndex:", bytes(ctypes.c_int(a)), 4, i)
    msg(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", to_struct(*global_size), to_struct(*local_size))
    msg(encoder, "endEncoding")
    msg(command_buffer, "commit")
    self.device.mtl_buffers_in_flight.append(command_buffer)

class MetalBuffer:
  def __init__(self, buf:Any, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class MetalAllocator(LRUAllocator):
  def __init__(self, device:MetalDevice):
    self.device:MetalDevice = device
    super().__init__()
  def _alloc(self, size:int, options) -> MetalBuffer:
    # Buffer is explicitly released in _free() rather than garbage collected via reference count
    ret = msg(self.device.device, "newBufferWithLength:options:", size, MTLResourceOptions.MTLResourceStorageModeShared, restype=objc_id)
    if ret.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    return MetalBuffer(ret, size)
  def _free(self, opaque:MetalBuffer, options): msg(opaque.buf, "release")
  def transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice): exit() #TODO
  def from_buffer(self, src:memoryview) -> Optional[Any]: exit() #TODO
  def as_buffer(self, src:MetalBuffer) -> memoryview:
    for cbuf in self.device.mtl_buffers_in_flight:
      msg(cbuf, "waitUntilCompleted")
    self.device.mtl_buffers_in_flight.clear()
    ptr = msg(src.buf, "contents", restype=objc_id) # Shared memory, do not release here
    array = (ctypes.c_char * (src.offset + src.size)).from_address(ptr.value)
    return memoryview(array).cast("B")[src.offset:]
  def copy_from_disk(self,dest,src):
    file_name = src.device[::-1]
    file_name = file_name[:file_name.index("/")]
    file_name = file_name[::-1]
    buf_name = str(dest._buf.buf)
    self.device.queue["queue"].append(["memcpy",buf_name,file_name,src.offset,src.nbytes])
  def copyin(self, dest:MetalBuffer, src:memoryview): self.as_buffer(dest)[:] = src
  def copyout(self, dest:memoryview, src:MetalBuffer): dest[:] = self.as_buffer(src)
  def offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, size, offset)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.queue = {"queue":[]}
    self.device = libmetal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = msg(self.device, "newCommandQueueWithMaxCommandBufferCount:", 1024, restype=objc_instance)
    if self.mtl_queue is None: raise RuntimeError("Cannot allocate a new command queue")
    self.mtl_buffers_in_flight: List[Any] = []

    from tinygrad.runtime.graph.metal import MetalGraph
    super().__init__(device, MetalAllocator(self), MetalRenderer(), Compiler(),
                     functools.partial(MetalProgram, self), MetalGraph)

  def send_queue(self):
    url = "http://192.168.1.113:8081" #your iOS device's local IP
    #url = "http://192.168.1.12:8081"
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
