from __future__ import annotations
import os, functools
from typing import List, Any, Tuple, Optional
from tinygrad.helpers import getenv
from tinygrad.device import Compiled, Compiler, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer

class MTLResourceOptions:
  MTLResourceCPUCacheModeDefaultCache = 0
  MTLResourceStorageModeShared = 0 << 4

class MTLPipelineOption:
  MTLPipelineOptionNone = 0

def objc_name(x):
  if x == None: return "Nil"
  if type(x) == bool: return str(x).lower()
  if type(x) == str: return x
  if type(x) == int: return str(x)
  if type(x) == tuple:
    return "MTLSizeMake" + str(x)
  return str(x).replace("b", "").replace("'", "\"") #bytes

objc_types = {"newCommandQueueWithMaxCommandBufferCount:":"id<MTLCommandQueue> ","newBufferWithLength:options:":"id<MTLBuffer> ",
              "newComputePipelineStateWithDescriptor:options:reflection:error:":"id<MTLComputePipelineState> ",
              "commandBuffer":"id<MTLCommandBuffer> ","computeCommandEncoder":"id<MTLComputeCommandEncoder> "}

def add_to_objc(line):
  with open("tinygrad-objc-ios/tinygrad-objc-ios/ViewController.m", "r") as file:
    lines = file.readlines()
    for i,l in enumerate(lines):
      if "//END" in l:
        lines.insert(i, line + "\n")
        with open("tinygrad-objc-ios/tinygrad-objc-ios/ViewController.m", "w") as file:
          file.writelines(lines)
          break

var_num = -1
def new_var():
  global var_num
  return "v" + str(var_num:=var_num+1)

def msg(ptr, selector: str, /, *args: Any, res=False):
  ret = None
  if selector == "new":
    add_to_objc(objc_name(ptr) + " *"+(ret:=new_var())+" = ["+objc_name(ptr)+" new];")
    return ret
  
  line = ""
  labels = selector.split(":") if ":" in selector else [selector]
  if res: line += objc_types[selector] + (ret := new_var()) + " = "
  line += "[" + objc_name(ptr) + " "
  for i,a in enumerate(labels):
    line += a
    if i < len(args): line += ": " + objc_name(args[i]) + " "
  line += "];"
  add_to_objc(line)
  return ret

def wait_check(cbuf: Any):
  msg(cbuf, "waitUntilCompleted")

class iosCompiler(Compiler):
  def __init__(self, device:Optional[iosDevice]):
    if os.path.exists("tinygrad-objc-ios/f.metal"): os.remove("tinygrad-objc-ios/f.metal")
    self.device = device
    super().__init__("compile_ios")
  def compile(self, src:str) -> bytes:
    file = open("tinygrad-objc-ios/f.metal", "a")
    file.write(src+"\n")
    file.close()
    return

class iosProgram:
  def __init__(self, device:iosDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    self.fxn = new_var()
    add_to_objc("id<MTLFunction> "+self.fxn+" = [library newFunctionWithName: @\""+name+"\" ];")
    descriptor = msg("MTLComputePipelineDescriptor", "new", res=True)
    msg(descriptor, "setComputeFunction:", self.fxn)
    msg(descriptor, "setSupportIndirectCommandBuffers:", True)
    self.pipeline_state = msg(self.device.device, "newComputePipelineStateWithDescriptor:options:reflection:error:",
      descriptor, MTLPipelineOption.MTLPipelineOptionNone, None, "&error", res=True)

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    command_buffer = msg(self.device.mtl_queue, "commandBuffer", res=True)
    encoder = msg(command_buffer, "computeCommandEncoder", res=True)
    msg(encoder, "setComputePipelineState:", self.pipeline_state)
    for i,a in enumerate(bufs): msg(encoder, "setBuffer:offset:atIndex:", a.buf, a.offset, i)
    for i,a in enumerate(vals,start=len(bufs)): msg(encoder, "setBytes:length:atIndex:", a.to_bytes(4, byteorder='little'), 4, i)
    msg(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", global_size, local_size)
    msg(encoder, "endEncoding")
    msg(command_buffer, "commit")
    self.device.mtl_buffers_in_flight.append(command_buffer)

class iosBuffer:
  def __init__(self, buf:Any, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class iosAllocator(LRUAllocator):
  def __init__(self, device:iosDevice):
    self.device:iosDevice = device
    super().__init__()
  def _alloc(self, size:int, options) -> iosBuffer:
    ret = msg(self.device.device, "newBufferWithLength:options:", size, MTLResourceOptions.MTLResourceStorageModeShared, res=True)
    return iosBuffer(ret, size)
  def as_buffer(self, src:iosBuffer) -> memoryview:
    self.device.synchronize()
    assert False, "cannot copy from ios (iOS)"
  def copy_from_disk(self,dest,src):
    file_name = src.device[::-1]
    file_name = file_name[:file_name.index("/")]
    file_name = file_name[::-1]
    buf_name = str(dest._buf.buf)
    line = ""
    if file_name not in open("tinygrad-objc-ios/tinygrad-objc-ios/ViewController.m").read(): line += "NSData *f"+file_name+" = [NSData dataWithContentsOfURL:\
[[NSBundle mainBundle] URLForResource:@\""+file_name+"\" withExtension:nil]];\n"
    line += "memcpy(["+buf_name+" contents] + "+str(dest.offset)+", [f"+file_name+" bytes] + "+str(src.offset)+", "+str(src.nbytes)+");"
    add_to_objc(line)

  def copyin(self, dest:iosBuffer, src:memoryview):
    formatted_bytes = ("{"+ ", ".join([f"0x{byte:02x}" for byte in src.tobytes()])+ "}")
    add_to_objc("memcpy(["+objc_name(dest.buf)+" contents], (uint8_t[])"+formatted_bytes+", "+str(src.nbytes)+");")
  def offset(self, buf:iosBuffer, size:int, offset:int): return iosBuffer(buf.buf, size, offset)

class iosDevice(Compiled):
  def __init__(self, device:str):
    self.device = new_var()
    with open('tinygrad-objc-ios/tinygrad-objc-ios/ViewController.m', 'w') as dest,\
    open('tinygrad-objc-ios/tinygrad-objc-ios/templateViewController.m', 'r') as src: dest.write(src.read())
    add_to_objc("id<MTLDevice> "+objc_name(self.device)+" = MTLCreateSystemDefaultDevice();")
    self.mtl_queue = msg(self.device, "newCommandQueueWithMaxCommandBufferCount:", 1024, res=True)
    self.mtl_buffers_in_flight: List[Any] = []

    super().__init__(device, iosAllocator(self), MetalRenderer(), iosCompiler(None if getenv("METAL_XCODE") else self),
                     functools.partial(iosProgram, self), None)
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: wait_check(cbuf)
    self.mtl_buffers_in_flight.clear()
