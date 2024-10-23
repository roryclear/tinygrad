from __future__ import annotations
import os, ctypes, functools
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
    ret = new_var()
    add_to_objc(objc_name(ptr) + " *"+ret+" = ["+objc_name(ptr)+" new];")
    return ret
  
  line = ""
  if ":" in selector:
    labels = selector.split(":")
  else:
    labels = [selector]
  if res:
    line += objc_types[selector] + (ret := new_var()) + " = "
  line += "[" + objc_name(ptr) + " "
  for i,a in enumerate(labels):
    line += a
    if i < len(args): line += ": " + objc_name(args[i]) + " "
  line += "];"
  add_to_objc(line)
  return ret

def wait_check(cbuf: Any):
  msg(cbuf, "waitUntilCompleted")

class MetalCompiler(Compiler):
  def __init__(self, device:Optional[MetalDevice]):
    if os.path.exists("tinygrad-objc-ios/f.metal"): os.remove("tinygrad-objc-ios/f.metal")
    self.device = device
    super().__init__("compile_metal")
  def compile(self, src:str) -> bytes:
    file = open("tinygrad-objc-ios/f.metal", "a")
    file.write(src+"\n")
    file.close()
    return

class MetalProgram:
  def __init__(self, device:MetalDevice, name:str, lib:bytes):
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
    for i,a in enumerate(vals,start=len(bufs)): msg(encoder, "setBytes:length:atIndex:", bytes(ctypes.c_int(a)), 4, i)
    msg(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", global_size, local_size)
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
    ret = msg(self.device.device, "newBufferWithLength:options:", size, MTLResourceOptions.MTLResourceStorageModeShared, res=True)
    return MetalBuffer(ret, size)
  def as_buffer(self, src:MetalBuffer) -> memoryview:
    self.device.synchronize()
    assert False, "cannot copy from metal (iOS)"
  def copyin(self, dest:MetalBuffer, src:memoryview):
    formatted_bytes = ("{"+ ", ".join([f"0x{byte:02x}" for byte in src.tobytes()])+ "}")
    add_to_objc("memcpy(["+objc_name(dest.buf)+" contents], (uint8_t[])"+formatted_bytes+", "+str(src.nbytes)+");")
  def offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, size, offset)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.device = new_var()
    with open('tinygrad-objc-ios/tinygrad-objc-ios/ViewController.m', 'w') as dest,\
    open('tinygrad-objc-ios/tinygrad-objc-ios/templateViewController.m', 'r') as src: dest.write(src.read())
    add_to_objc("id<MTLDevice> "+objc_name(self.device)+" = MTLCreateSystemDefaultDevice();")
    self.mtl_queue = msg(self.device, "newCommandQueueWithMaxCommandBufferCount:", 1024, res=True)
    self.mtl_buffers_in_flight: List[Any] = []

    super().__init__(device, MetalAllocator(self), MetalRenderer(), MetalCompiler(None if getenv("METAL_XCODE") else self),
                     functools.partial(MetalProgram, self), None)
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: wait_check(cbuf)
    self.mtl_buffers_in_flight.clear()
