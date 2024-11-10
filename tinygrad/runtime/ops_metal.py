from __future__ import annotations
import os, subprocess, pathlib, ctypes, tempfile, functools
from typing import List, Any, Tuple, Optional, cast
from tinygrad.helpers import prod, getenv, T
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer
import json, gzip, requests, time

var_num = -1
def new_var():
  global var_num
  return "v" + str(var_num:=var_num+1)

def send_queue(queue):
  url = "http://192.168.1.113:8081" #your iOS device's local IP
  #url = "http://192.168.1.1:8081"
  #payload = self.queue
  payload = json.dumps(queue) # Compress the JSON string 
  compressed_payload = gzip.compress(payload.encode('utf-8'))
  status = 400
  while status != 200:
    try:
      headers = {'Content-Encoding': 'gzip', 'Content-Type': 'application/json'}
      response = requests.post(url, compressed_payload,headers=headers,timeout=3600)
      queue = {"queue":[]} #TODO: hack to not crash iOS
      if response.status_code == 200:
          status = 200
          if len(response.text) > 0:
            return response.text
      else:
          time.sleep(0.1)
    except requests.exceptions.RequestException as e:
      #print("An error occurred:", e)
      time.sleep(0.2)

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
  print("msg called")
  return None

def msg_ios(ptr,selector,*args,res=None):
  req = [ptr,selector]
  req.append(len(args))
  for x in args: req.append(x)
  if res != None: req.append(res)
  print(req)
  res2 = send_queue({"queue":[req]})
  if res2 != None: return res2
  return res

def to_struct(*t: int, _type: type = ctypes.c_ulong):
  class Struct(ctypes.Structure): pass
  Struct._fields_ = [(f"field{i}", _type) for i in range(len(t))]
  return Struct(*t)

class MetalProgram:
  def __init__(self, device:MetalDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    options_ios = msg_ios("MTLCompileOptions","new",res=new_var())
    code_ns_str_ios = lib.decode() #Don't need above in iOS
    print(code_ns_str_ios)
    self.library_ios = msg_ios("d","newLibraryWithSource:options:error:",code_ns_str_ios,options_ios,"error",res=new_var())
    name_ns_str_ios = name #Don't need above in iOS
    self.fxn_ios = msg_ios(self.library_ios, "newFunctionWithName:", name_ns_str_ios, res=new_var())
    descriptor_ios = msg_ios("MTLComputePipelineDescriptor", "new", res=new_var())
    msg_ios(descriptor_ios, "setComputeFunction:", self.fxn_ios)
    msg_ios(descriptor_ios, "setSupportIndirectCommandBuffers:", "true")
    self.pipeline_state_ios = msg_ios(self.device.device_ios,"newComputePipelineStateWithDescriptor:options:reflection:error:",
    descriptor_ios,0,"none",res=new_var())

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    command_buffer_ios = msg_ios(self.device.mtl_queue_ios,"commandBuffer",res=new_var())
    encoder_ios = msg_ios(command_buffer_ios,"computeCommandEncoder",res=new_var())
    msg_ios(encoder_ios,"setComputePipelineState:",self.pipeline_state_ios)
    for i,a in enumerate(bufs):
      msg_ios(encoder_ios,"setBuffer:offset:atIndex:",a.buf_ios,a.offset,i)
    for i,a in enumerate(vals,start=len(bufs)): 
      msg_ios(encoder_ios,"setBytes:length:atIndex:",' '.join(f"{(a >> (i * 8)) & 0xff:02x}" for i in range(4)),4,i)
    msg_ios(encoder_ios,"dispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
    msg_ios(encoder_ios,"endEncoding")
    msg_ios(command_buffer_ios,"commit")
    self.device.mtl_buffers_in_flight.append([None,command_buffer_ios])

class MetalBuffer:
  def __init__(self, buf:Any, size:int, offset=0,buf_ios=None): 
    self.buf, self.size, self.offset,self.buf_ios = buf, size, offset,buf_ios
    if buf_ios == None:
      res = new_var()
      self.buf_ios = res

class MetalAllocator(LRUAllocator):
  def __init__(self, device:MetalDevice):
    self.device:MetalDevice = device
    super().__init__()
  def _alloc(self, size:int, options) -> MetalBuffer:
    # Buffer is explicitly released in _free() rather than garbage collected via reference count
    ret_ios = msg_ios(self.device.device_ios,"newBufferWithLength:options:",size,0,res=new_var())
    return MetalBuffer("no buffer remove this", size,buf_ios=ret_ios)
  def _free(self, opaque:MetalBuffer, options): return #todo?
  def transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice): exit() #TODO
  def from_buffer(self, src:memoryview) -> Optional[Any]: exit() #TODO
  def as_buffer(self, src:MetalBuffer) -> memoryview:
    for cbuf in self.device.mtl_buffers_in_flight:
      if len(cbuf) > 1: msg_ios(cbuf[1],"waitUntilCompleted")
    self.device.mtl_buffers_in_flight.clear()
    
    #TODO below gets called when copying weights from disk to metal, so have noted out
    print("copying out",src.buf)
    byte_str = msg_ios("copyout",src.buf_ios)
    byte_values = bytearray(int(b, 16) for b in byte_str.split())
    ret_ios = memoryview(byte_values[:src.size]) 
    print(src.buf_ios,src.buf)
    #print("ios buffer\t",ret_ios.tobytes())
    
    return ret_ios
  
  def as_buffer_ios(self, src:MetalBuffer) -> memoryview:
    for cbuf in self.device.mtl_buffers_in_flight:
      if len(cbuf) > 1: msg_ios(cbuf[1],"waitUntilCompleted")
    self.device.mtl_buffers_in_flight.clear()
    
    #TODO below gets called when copying weights from disk to metal, so have noted out
    byte_str = msg_ios("copyout",src.buf_ios)
    byte_values = bytearray(int(b, 16) for b in byte_str.split())
    ret_ios = memoryview(byte_values)
    print(src.buf_ios,src.buf)
    #print("ios buffer\t",ret_ios.tobytes())
    #print("metal buffer\t",ret.tobytes())   
    
    return ret_ios
  
  def copy_from_disk(self,dest,src):
    file_name = src.device[::-1]
    file_name = file_name[:file_name.index("/")]
    file_name = file_name[::-1]
    buf_name = str(dest._buf.buf_ios)
    msg_ios("memcpy",buf_name,file_name,src.offset,src.nbytes)
    #self.device.queue["queue"].append(["memcpy",buf_name,file_name,src.offset,src.nbytes])
  
  def copyin(self, dest:MetalBuffer, src:memoryview):
    for cbuf in self.device.mtl_buffers_in_flight:
      if len(cbuf) > 1: msg_ios(cbuf[1],"waitUntilCompleted")
    self.device.mtl_buffers_in_flight.clear()

    formatted_hex = ' '.join(f'{b:02x}' for b in src)
    msg_ios("copyin",formatted_hex,dest.buf_ios)

  def copyout(self, dest:memoryview, src:MetalBuffer): 
    exit() #TODO
    dest[:] = self.as_buffer(src)
  def offset(self, buf:MetalBuffer, size:int, offset:int):
    return MetalBuffer(buf.buf, size, offset,buf_ios=buf.buf_ios)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.queue = {"queue":[]}
    self.device = libmetal.MTLCreateSystemDefaultDevice()
    self.device_ios = "d"
    self.mtl_queue_ios = msg_ios(self.device_ios,"newCommandQueueWithMaxCommandBufferCount:",1024,res=new_var())
    self.mtl_buffers_in_flight: List[Any] = []

    from tinygrad.runtime.graph.metal import MetalGraph
    super().__init__(device, MetalAllocator(self), MetalRenderer(), Compiler(),
                     functools.partial(MetalProgram, self), MetalGraph)


