import  struct, functools, os
from tinygrad.helpers import cpu_profile
from tinygrad.device import Compiled, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer
import urllib, json, base64, urllib.request
from tinygrad.helpers import diskcache_get

class IOSDevice(Compiled):
  def __init__(self, device:str):
    self.buf_num = 0
    self.q = []
    self.first_call = True
    super().__init__(device, IOSAllocator(self), [MetalRenderer], functools.partial(MetalProgram, self), None)

  def send_q(self):
    metas, blobs, off = [], [], 0
    for op in self.q:
      if "copyin" in op:
        d = op["copyin"]; b = bytes(d.pop("data"))
        metas.append({"copyin": {**d, "off": off}}); blobs.append(b); off += len(b)
      else:
        metas.append(op)
    meta = json.dumps(metas).encode()
    body = struct.pack("<I", len(meta)) + meta + b"".join(blobs)
    self.q = []
    self.first_call = True
    assert os.environ.get("IP") is not None, "no IP address provided, use IP=(iOS IP)."
    ip = os.environ.get("IP")
    req = urllib.request.Request(f"http://{ip}/batch", data=body,
                                headers={"Content-Type": "application/octet-stream"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp: return resp.read()

class MetalProgram:
  def __init__(self, dev:IOSDevice, name:str, lib:bytes, **kwargs):
    self.dev, self.name, self.lib, self.src = dev, name, lib, diskcache_get("compile_metal_direct", key=str(lib))
    self.dev.q.append({"program":{"name": name, "lib":base64.b64encode(bytes(lib)).decode("ascii"), "src": self.src}})

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    self.dev.q.append({"call":{"name":self.name, "buffers":[b.num for b in bufs], "buffer_offsets":[b.offset for b in bufs],
                       "vals":vals, "local_size":local_size, "global_size":global_size, "wait": wait, "benchmark_start": self.dev.first_call and os.environ.get("BENCHMARK") == "1", "benchmark_end": False}})
    self.dev.first_call = False
    if wait: return float(self.dev.send_q().decode('ascii'))

class IOSBuffer:
  def __init__(self, size:int, offset=0, num=0): self.size, self.offset, self.num = size, offset, num

class IOSAllocator(LRUAllocator[IOSDevice]):
  def _alloc(self, size:int, options) -> IOSBuffer:
    self.dev.buf_num+=1
    self.dev.q.append({"buff_alloc":{"num":self.dev.buf_num, "size":size}})
    if options.external_ptr: return IOSBuffer(size, num=self.dev.buf_num)
    return IOSBuffer(size, num=self.dev.buf_num)
  def _cp_mv(self, dst, src, prof_desc):
    with cpu_profile(prof_desc, f"{self.dev.device}:COPY"): dst[:] = src
  def _copyin(self, dest:IOSBuffer, src:memoryview):
    self.dev.q.append({"copyin": {"dest": dest.num, "len": len(src), "data": memoryview(src)}})
    if os.environ.get("LAZY_COPYIN") != "1": self.dev.send_q()
  def _copyout(self, dest:memoryview, src:IOSBuffer):
    if os.environ.get("BENCHMARK") == "1": # measure time to run a graph
      self.dev.q[-1]["call"]["benchmark_end"] = True
      print(f"time: {float( self.dev.send_q().decode('ascii'))}")

    self.dev.q.append({"copyout": src.num})
    data = self.dev.send_q()
    self._cp_mv(dest, memoryview(data), "METAL -> TINY")
