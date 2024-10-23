from __future__ import annotations
import os, subprocess, pathlib, ctypes, tempfile, functools
from typing import List, Any, Tuple, Optional, cast, TypeVar
from tinygrad.helpers import prod, getenv, DEBUG, IOS
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer

class objc_id(ctypes.c_void_p): # This prevents ctypes from converting response to plain int, and dict.fromkeys() can use it to dedup
  def __hash__(self): return hash(self.value)
  def __eq__(self, other): return self.value == other.value

class objc_instance(objc_id): # method with name "new", "alloc" should be freed after use
  def __del__(self): return

@functools.lru_cache(None)
def sel(name: str): return libobjc.sel_registerName(name.encode())

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

objc_names = {}
def objc_name(x,selector=None,og=True):
  if type(x) == str: return x # should always use this!
  if type(x) == list:
    return ("(id<MTLResource> []){"+str([objc_name(i) for i in x])[1:-1]+"}").replace("'","")
  if type(x) == tuple:
    if selector == "executeCommandsInBuffer:withRange:": return "NSMakeRange" + str(x)
    return "MTLSizeMake" + str(x)
  x_s = str(x)
  if x_s == "True": return "true"
  if x_s == "False": return "false"
  if ")" in x_s: return "&error"
  if "0x" not in x_s: 
    ret = x_s.replace("b'", "").replace("'", "")
    if ret == "None": return "Nil"
    return ret
  ret = x_s[x_s.index("0x")+1:x_s.index(">")]
  if ret in objc_names:
    if og == False: objc_names[ret] += 1
  else:
    objc_names[ret] = 0
  return ret + "_" + str(objc_names[ret])

objc_types = {"newCommandQueueWithMaxCommandBufferCount:":"id<MTLCommandQueue> ","newSharedEvent":"id<MTLSharedEvent> ",
              "stringWithUTF8String:":"NSString *","newBufferWithLength:options:":"id<MTLBuffer> ",
              "contents":"void *","newLibraryWithData:error:":"id<MTLLibrary> ","newFunctionWithName:":"id<MTLFunction> ",
              "newComputePipelineStateWithDescriptor:options:reflection:error:":"id<MTLComputePipelineState> ",
              "commandBuffer":"id<MTLCommandBuffer> ","computeCommandEncoder":"id<MTLComputeCommandEncoder> ",
              "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:":"id<MTLIndirectCommandBuffer> ",
              "indirectComputeCommandAtIndex:":"id<MTLIndirectComputeCommand> "}

quotes = {"stringWithUTF8String","setBytes"}

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
  var_num+=1
  return "v" + str(var_num)


T = TypeVar("T")
# Ignore mypy error reporting incompatible default, because typevar default only works on python 3.12
def msg(ptr: objc_id, selector: str, /, *args: Any, restype: type[T] = objc_id) -> T: # type: ignore [assignment]
  sender = libobjc["objc_msgSend"] # Using attribute access returns a new reference so setting restype is safe
  sender.restype = restype
  return sender(ptr, sel(selector), *args)

T = TypeVar("T")
# Ignore mypy error reporting incompatible default, because typevar default only works on python 3.12
def msg_ios2(ptr, selector: str, /, *args: Any, restype: type[T] = None) -> T: # type: ignore [assignment]
  ret = new_var()
  if selector == "new":
    add_to_objc(objc_name(ptr) + " *"+ret+" = ["+objc_name(ptr)+" new];")
    return ret

  line = ""
  selector_in = selector
  if ":" in selector:
    labels = [selector[:selector.index(":")]]
  else:
    labels = [selector]
  if restype != None:
    line +=  objc_types[selector] + ret + " = "
  if ":" in selector:
    selector = selector[selector.index(":")+1:]
  while ":" in selector: #TODO
    labels.append(selector[:selector.index(":")])
    selector = selector[selector.index(":")+1:]
  line += "[" + objc_name(ptr) + " "
  for i,a in enumerate(labels):
    line += a
    if i < len(args):
      line += ": "
      if a in quotes: line += "\""
      line += objc_name(args[i],selector=selector_in)
      if a in quotes: line += "\""
      line += " "
  line += "];"
  add_to_objc(line)
  return ret

T = TypeVar("T")
# Ignore mypy error reporting incompatible default, because typevar default only works on python 3.12
def msg_ios(ptr: objc_id, selector: str, /, *args: Any, restype: type[T] = None) -> T: # type: ignore [assignment]
  sender = libobjc["objc_msgSend"] # Using attribute access returns a new reference so setting restype is safe
  sender.restype = restype
  args_copy = []
  for i,x in enumerate(args):
    if type(x) == tuple:
      args_copy.append(to_struct(*x))
    elif type(x) == list:
      args_copy.append((objc_id * len(x))(*x))
    else:
      args_copy.append(x)
  if type(ptr) == bytes:
    ret = sender(libobjc.objc_getClass(ptr), sel(selector), *args_copy)
  else:
    ret = sender(ptr, sel(selector), *args_copy)
  if selector == "new":
    if IOS>0: add_to_objc(objc_name(ptr) + " *"+objc_name(ret,og=False)+" = ["+objc_name(ptr)+" new];")
    return ret
  if IOS<1: return ret

  line = ""
  selector_in = selector
  if ":" in selector:
    labels = [selector[:selector.index(":")]]
  else:
    labels = [selector]
  if restype != None:
    line +=  objc_types[selector] + objc_name(ret,og=False) + " = "
  if ":" in selector:
    selector = selector[selector.index(":")+1:]
  while ":" in selector: #TODO
    labels.append(selector[:selector.index(":")])
    selector = selector[selector.index(":")+1:]
  line += "[" + objc_name(ptr) + " "
  for i,a in enumerate(labels):
    line += a
    if i < len(args):
      line += ": "
      if a in quotes: line += "\""
      line += objc_name(args[i],selector=selector_in)
      if a in quotes: line += "\""
      line += " "
  line += "];"
  add_to_objc(line)
  return ret

def to_struct(*t: int, _type: type = ctypes.c_ulong):
  class Struct(ctypes.Structure): pass
  Struct._fields_ = [(f"field{i}", _type) for i in range(len(t))]
  return Struct(*t)

def wait_check(cbuf: Any):
  msg_ios2(cbuf, "waitUntilCompleted")

class MetalCompiler(Compiler):
  def __init__(self, device:Optional[MetalDevice]):
    if os.path.exists("tinygrad-objc-ios/f.metal"): os.remove("tinygrad-objc-ios/f.metal")
    self.device = device
    super().__init__("compile_metal")
  def compile(self, src:str) -> bytes:
    file = open("tinygrad-objc-ios/f.metal", "a")
    file.write(src+"\n")
    file.close()
    if self.device is None:
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=src.encode('utf-8'))
      return subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
    return

class MetalProgram:
  def __init__(self, device:MetalDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    self.fxn = new_var()
    if IOS>0: add_to_objc("id<MTLFunction> "+self.fxn+" = [library newFunctionWithName: @\""+name+"\" ];")
    descriptor = msg_ios2(b"MTLComputePipelineDescriptor", "new", restype=objc_instance)
    msg_ios2(descriptor, "setComputeFunction:", self.fxn)
    msg_ios2(descriptor, "setSupportIndirectCommandBuffers:", True)
    self.pipeline_state = msg_ios2(self.device.device, "newComputePipelineStateWithDescriptor:options:reflection:error:",
      descriptor, MTLPipelineOption.MTLPipelineOptionNone, None, ctypes.byref(objc_instance()), restype=objc_instance)

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    command_buffer = msg_ios2(self.device.mtl_queue, "commandBuffer", restype=objc_instance)
    encoder = msg_ios2(command_buffer, "computeCommandEncoder", restype=objc_instance)
    msg_ios2(encoder, "setComputePipelineState:", self.pipeline_state)
    for i,a in enumerate(bufs): msg_ios2(encoder, "setBuffer:offset:atIndex:", a.buf, a.offset, i)
    for i,a in enumerate(vals,start=len(bufs)): msg_ios2(encoder, "setBytes:length:atIndex:", bytes(ctypes.c_int(a)), 4, i)
    msg_ios2(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", global_size, local_size)
    msg_ios2(encoder, "endEncoding")
    msg_ios2(command_buffer, "commit")
    self.device.mtl_buffers_in_flight.append(command_buffer)

class MetalBuffer:
  def __init__(self, buf:Any, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class MetalAllocator(LRUAllocator):
  def __init__(self, device:MetalDevice):
    self.device:MetalDevice = device
    super().__init__()
  def _alloc(self, size:int, options) -> MetalBuffer:
    # Buffer is explicitly released in _free() rather than garbage collected via reference count
    ret = msg_ios2(self.device.device, "newBufferWithLength:options:", size, MTLResourceOptions.MTLResourceStorageModeShared, restype=objc_id)
    return MetalBuffer(ret, size)
  def _free(self, opaque:MetalBuffer, options): return
  def transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice): return
  def from_buffer(self, src:memoryview) -> Optional[Any]:
    return
  def as_buffer(self, src:MetalBuffer) -> memoryview:
    self.device.synchronize()
    assert False, "cannot copy from metal (IOS)"
  def copyin(self, dest:MetalBuffer, src:memoryview):
    if IOS>0:
      formatted_bytes = ("{"+ ", ".join([f"0x{byte:02x}" for byte in src.tobytes()])+ "}")
      add_to_objc("memcpy(["+objc_name(dest.buf)+" contents], (uint8_t[])"+formatted_bytes+", "+str(src.nbytes)+");")
  def offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, size, offset)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.device = libmetal.MTLCreateSystemDefaultDevice()
    if IOS>0:
      with open('tinygrad-objc-ios/tinygrad-objc-ios/ViewController.m', 'w') as dest,\
      open('tinygrad-objc-ios/tinygrad-objc-ios/templateViewController.m', 'r') as src: dest.write(src.read())
      add_to_objc("id<MTLDevice> "+objc_name(self.device)+" = MTLCreateSystemDefaultDevice();")
    self.mtl_queue = msg_ios2(self.device, "newCommandQueueWithMaxCommandBufferCount:", 1024, restype=objc_instance)
    self.mtl_buffers_in_flight: List[Any] = []

    from tinygrad.runtime.graph.metal import MetalGraph
    super().__init__(device, MetalAllocator(self), MetalRenderer(), MetalCompiler(None if getenv("METAL_XCODE") else self),
                     functools.partial(MetalProgram, self), MetalGraph)
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: wait_check(cbuf)
    self.mtl_buffers_in_flight.clear()
