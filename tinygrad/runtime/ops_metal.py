from __future__ import annotations
import ctypes, functools
from typing import List, Any, Tuple, Optional, cast
from tinygrad.helpers import prod, getenv, T
from tinygrad.device import Compiled, Compiler, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer
import json, gzip, requests, time

var_num = -1
def new_var():
  global var_num
  return "v" + str(var_num:=var_num+1)

class MetalProgram:
  def __init__(self, device:MetalDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    options_ios = self.device.msg_ios("MTLCompileOptions","new",res=new_var())
    code_ns_str_ios = lib.decode()
    print(code_ns_str_ios)
    self.library_ios = self.device.msg_ios("d","newLibraryWithSource:options:error:",code_ns_str_ios,options_ios,"error",res=new_var())
    self.fxn_ios = self.device.msg_ios(self.library_ios, "newFunctionWithName:", name, res=new_var())
    descriptor_ios = self.device.msg_ios("MTLComputePipelineDescriptor", "new", res=new_var())
    self.device.msg_ios(descriptor_ios, "setComputeFunction:", self.fxn_ios)
    self.device.msg_ios(descriptor_ios, "setSupportIndirectCommandBuffers:", "true")
    self.pipeline_state_ios = self.device.msg_ios(self.device.device_ios,"newComputePipelineStateWithDescriptor:options:reflection:error:",
    descriptor_ios,0,"none",res=new_var())

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    command_buffer_ios = self.device.msg_ios(self.device.mtl_queue_ios,"commandBuffer",res=new_var())
    encoder_ios = self.device.msg_ios(command_buffer_ios,"computeCommandEncoder",res=new_var())
    self.device.msg_ios(encoder_ios,"setComputePipelineState:",self.pipeline_state_ios)
    for i,a in enumerate(bufs):
      self.device.msg_ios(encoder_ios,"setBuffer:offset:atIndex:",a.buf,a.offset,i)
    for i,a in enumerate(vals,start=len(bufs)): 
      self.device.msg_ios(encoder_ios,"setBytes:length:atIndex:",' '.join(f"{(a >> (i * 8)) & 0xff:02x}" for i in range(4)),4,i)
    self.device.msg_ios(encoder_ios,"dispatchThreadgroups:threadsPerThreadgroup:",global_size[0],global_size[1],global_size[2],local_size[0],local_size[1],local_size[2])
    self.device.msg_ios(encoder_ios,"endEncoding")
    self.device.msg_ios(command_buffer_ios,"commit")
    self.device.mtl_buffers_in_flight.append(command_buffer_ios)

class MetalBuffer:
  def __init__(self, buf:Any, size:int, offset=0): 
    self.buf, self.size, self.offset = buf, size, offset
    if buf == None: 
      self.buf = new_var()

class MetalAllocator(LRUAllocator):
  def __init__(self, device:MetalDevice):
    self.device:MetalDevice = device
    super().__init__()
  def _alloc(self, size:int, options) -> MetalBuffer:
    # Buffer is explicitly released in _free() rather than garbage collected via reference count
    buf = self.device.msg_ios(self.device.device_ios,"newBufferWithLength:options:",size,0,res=new_var())
    return MetalBuffer(buf, size)
  def _free(self, opaque:MetalBuffer, options): return #todo?
  def transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice): exit() #TODO
  def from_buffer(self, src:memoryview) -> Optional[Any]: exit() #TODO
  def as_buffer(self, src:MetalBuffer) -> memoryview:
    for cbuf in self.device.mtl_buffers_in_flight: self.device.msg_ios(cbuf,"waitUntilCompleted")
    self.device.mtl_buffers_in_flight.clear()
    byte_str = self.device.msg_ios("copyout",src.buf)
    byte_values = bytearray(int(b, 16) for b in byte_str.split())
    return memoryview(byte_values[:src.size]) 
  
  def as_buffer_ios(self, src:MetalBuffer) -> memoryview:
    for cbuf in self.device.mtl_buffers_in_flight: self.device.msg_ios(cbuf,"waitUntilCompleted")
    self.device.mtl_buffers_in_flight.clear()
    byte_str = self.device.msg_ios("copyout",src.buf)
    byte_values = bytearray(int(b, 16) for b in byte_str.split())
    return memoryview(byte_values)    
  
  def copy_from_disk(self,dest,src):
    file_name = src.device[::-1]
    file_name = file_name[:file_name.index("/")]
    file_name = file_name[::-1]
    buf_name = str(dest._buf.buf)
    self.device.msg_ios("memcpy",buf_name,file_name,src.offset,src.nbytes)
    #self.device.queue["queue"].append(["memcpy",buf_name,file_name,src.offset,src.nbytes])
  
  def copyin(self, dest:MetalBuffer, src:memoryview):
    for cbuf in self.device.mtl_buffers_in_flight: self.device.msg_ios(cbuf,"waitUntilCompleted")
    self.device.mtl_buffers_in_flight.clear()

    formatted_hex = ' '.join(f'{b:02x}' for b in src)
    self.device.msg_ios("copyin",formatted_hex,dest.buf)

  def copyout(self, dest:memoryview, src:MetalBuffer): 
    exit() #TODO
    dest[:] = self.as_buffer(src)
  def offset(self, buf:MetalBuffer, size:int, offset:int):
    return MetalBuffer(buf.buf, size, offset)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.device_ios = "d"
    self.queue = {"queue":[]}
    self.mtl_queue_ios = self.msg_ios(self.device_ios,"newCommandQueueWithMaxCommandBufferCount:",1024,res=new_var())
    self.mtl_buffers_in_flight: List[Any] = []

    from tinygrad.runtime.graph.metal import MetalGraph
    super().__init__(device, MetalAllocator(self), MetalRenderer(), Compiler(),
                     functools.partial(MetalProgram, self), MetalGraph)
    

  def msg_ios(self,ptr,selector,*args,res=None):
    req = [ptr,selector]
    req.append(len(args))
    for x in args: req.append(x)
    if res != None: req.append(res)
    self.queue["queue"].append(req)
    print(req)
    if ptr == "copyout" or len(self.queue["queue"]) > 100: #todo
      res2 = self.send_queue()
      self.queue = {"queue":[]}
      if res2 != None: return res2
    return res

  def send_queue(self):
    url = "http://192.168.1.113:8081" #your iOS device's local IP
    #url = "http://192.168.1.1:8081"
    #payload = self.queue
    payload = json.dumps(self.queue) # Compress the JSON string 
    compressed_payload = gzip.compress(payload.encode('utf-8'))
    status = 400
    while status != 200:
      try:
        headers = {'Content-Encoding': 'gzip', 'Content-Type': 'application/json'}
        response = requests.post(url, compressed_payload,headers=headers,timeout=3600)
        if response.status_code == 200:
            status = 200
            if len(response.text) > 0:
              return response.text
        else:
            print(response)
            time.sleep(0.1)
      except requests.exceptions.RequestException as e:
        #print("An error occurred:", e)
        time.sleep(0.2)
    return