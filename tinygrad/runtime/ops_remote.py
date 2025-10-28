# the REMOTE=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with REMOTE tinygrad is  frontend <-> middleware <-> RemoteDevice ///HTTP/// remote_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Callable, Iterator, Any, cast
from collections import defaultdict
from dataclasses import dataclass, field, replace
import multiprocessing, threading, functools, itertools, asyncio, http, http.client, hashlib, time, os, binascii, struct, ast, contextlib, weakref
import traceback, builtins
from tinygrad.renderer import Renderer, ProgramSpec
from tinygrad.dtype import DTYPES_DICT, dtypes
from tinygrad.uop.ops import UOp, Ops, Variable, sint
from tinygrad.helpers import getenv, DEBUG, fromimport, unwrap, LazySeq, Timing
from tinygrad.engine.jit import GraphRunner, MultiGraphRunner, ExecItem, graph_class
from tinygrad.engine.realize import CompiledRunner, BufferXfer
from tinygrad.device import Compiled, Buffer, Allocator, Compiler, Device, BufferSpec
from tinygrad.runtime.support.ib import IBCtx, IBConn, SGE
import subprocess
from pathlib import Path
import re

# ***** API *****

@dataclass(frozen=True)
class SessionKey: host: str; idx: int; nonce: str # noqa: E702

@dataclass(frozen=True)
class RemoteRequest: session: SessionKey|None = field(default=None, kw_only=True)

@dataclass(frozen=True)
class SessionFree(RemoteRequest): pass

@dataclass(frozen=True)
class RemoteProperties:
  real_device: str
  renderer: tuple[str, str, tuple[Any, ...]]
  offset_supported: bool
  graph_supported: bool
  graph_supports_multi: bool
  ib_gid: bytes|None

@dataclass(frozen=True)
class RemoteException:
  exc: Exception
  trace: str = ""

@dataclass(frozen=True)
class GetProperties(RemoteRequest): pass

@dataclass(frozen=True)
class Event(RemoteRequest): event_session: SessionKey; event: int # noqa: E702

@dataclass(frozen=True)
class Wait(RemoteRequest): event: int

@dataclass(frozen=True)
class IBConnect(RemoteRequest): host: str; gid: bytes; qp_num: int # noqa: E702

@dataclass(frozen=True)
class BufferAlloc(RemoteRequest): buffer_num: int; size: int; options: BufferSpec # noqa: E702

@dataclass(frozen=True)
class BufferOffset(RemoteRequest): buffer_num: int; size: int; offset: int; sbuffer_num: int # noqa: E702

@dataclass(frozen=True)
class BufferIOVAS(RemoteRequest): buffer_nums: list[tuple[SessionKey, int]] # noqa: E702

@dataclass(frozen=True)
class BufferFree(RemoteRequest): buffer_num: int # noqa: E702

@dataclass(frozen=True)
class CopyIn(RemoteRequest): buffer_num: int; datahash: str # noqa: E702

@dataclass(frozen=True)
class CopyOut(RemoteRequest): buffer_num: int

@dataclass(frozen=True)
class Transfer(RemoteRequest): buffer_num: int; dsession: SessionKey; dbuffer_num: int # noqa: E702

@dataclass(frozen=True)
class BatchTransfer(RemoteRequest):
  sbuffer_nums: list[tuple[SessionKey, int]]
  dbuffer_nums: list[tuple[SessionKey, int]]

@dataclass(frozen=True)
class ProgramAlloc(RemoteRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramFree(RemoteRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramExec(RemoteRequest):
  name: str; datahash: str; bufs: tuple[int, ...]; vals: tuple[int, ...] # noqa: E702
  global_size: tuple[int, ...]|None; local_size: tuple[int, ...]|None; wait: bool # noqa: E702

@dataclass(frozen=True)
class GraphComputeItem:
  session: SessionKey
  name: str
  datahash: str
  bufs: tuple[int, ...]
  vars: tuple[Variable, ...]
  fixedvars: dict[str, int]
  ins: tuple[int, ...]
  outs: tuple[int, ...]
  global_size: tuple[sint, ...]|None
  local_size: tuple[sint, ...]|None

@dataclass(frozen=True)
class GraphAlloc(RemoteRequest):
  graph_num: int
  jit_cache: tuple[GraphComputeItem|Transfer, ...]
  bufs: tuple[tuple[SessionKey, int], ...]
  var_vals: dict[str, int]

@dataclass(frozen=True)
class GraphFree(RemoteRequest):
  graph_num: int

@dataclass(frozen=True)
class GraphExec(RemoteRequest):
  graph_num: int
  bufs: tuple[tuple[SessionKey, int], ...]
  var_vals: dict[str, int]
  wait: bool

# for safe deserialization
eval_excs = [v for k,v in builtins.__dict__.items() if isinstance(v, type) and issubclass(v, Exception) and not k.endswith("Warning")]
eval_globals = {x.__name__:x for x in [SessionKey, SessionFree, RemoteProperties, GetProperties, Event, Wait, BufferAlloc, BufferOffset, BufferIOVAS,
                                       BufferFree, CopyIn, CopyOut, Transfer, BatchTransfer, IBConnect, ProgramAlloc, ProgramFree, ProgramExec,
                                       GraphComputeItem, GraphAlloc, GraphFree, GraphExec, BufferSpec, UOp, Ops, dtypes, RemoteException] + eval_excs}
attribute_whitelist: dict[Any, set[str]] = {dtypes: {*DTYPES_DICT.keys(), 'imagef', 'imageh'}, Ops: {x.name for x in Ops}}
eval_fxns = {ast.Constant: lambda x: x.value, ast.Tuple: lambda x: tuple(map(safe_eval, x.elts)), ast.List: lambda x: list(map(safe_eval, x.elts)),
  ast.Dict: lambda x: {safe_eval(k):safe_eval(v) for k,v in zip(x.keys, x.values)},
  ast.Call: lambda x: safe_eval(x.func)(*[safe_eval(arg) for arg in x.args], **{kwarg.arg: safe_eval(kwarg.value) for kwarg in x.keywords}),
  ast.Name: lambda x: eval_globals[x.id], ast.Attribute: lambda x: safe_getattr(safe_eval(x.value), x.attr)}
def safe_getattr(value, attr):
  assert attr in attribute_whitelist.get(value, set()), f'getattr({value}, {repr(attr)}) is not whitelisted'
  return getattr(value, attr)
def safe_eval(node): return eval_fxns[node.__class__](node)

class BatchRequest:
  def __init__(self):
    self._q: list[RemoteRequest] = []
    self._h: dict[str, bytes] = {}
  def h(self, d:bytes|memoryview) -> str:
    datahash = hashlib.sha256(d).hexdigest() # NOTE: this is very slow, should use blake3 on gpu instead
    if datahash not in self._h:
      self._h[datahash] = bytes.fromhex(datahash)+struct.pack("<Q", len(d))+bytes(d)
    return datahash
  def q(self, x:RemoteRequest): self._q.append(x)
  def serialize(self) -> bytes:
    self.h(repr(self._q).encode())
    return b''.join(self._h.values())
  def deserialize(self, dat:bytes) -> BatchRequest:
    ptr = 0
    while ptr < len(dat):
      datahash, datalen = binascii.hexlify(dat[ptr:ptr+0x20]).decode(), struct.unpack("<Q", dat[ptr+0x20:ptr+0x28])[0]
      self._h[datahash] = dat[ptr+0x28:ptr+0x28+datalen]
      ptr += 0x28+datalen
    self._q = safe_eval(ast.parse(self._h[datahash], mode="eval").body)
    return self

# ***** backend *****

@dataclass
class RemoteSession:
  programs: dict[tuple[str, str], Any] = field(default_factory=dict)
  graphs: dict[int, GraphRunner] = field(default_factory=dict)
  buffers: dict[int, Buffer] = field(default_factory=dict)
  events: defaultdict[int, asyncio.Event] = field(default_factory=functools.partial(defaultdict, asyncio.Event))

class RemoteHandler:
  def __init__(self, base_device: str): return

  async def __call__(self, reader:asyncio.StreamReader, writer:asyncio.StreamWriter): return
  async def ib_connect(self, ssession:SessionKey, dsession:SessionKey) -> IBConn|None: return
  async def handle(self, method:str, path:str, body:bytes) -> tuple[http.HTTPStatus, bytes]: return http.HTTPStatus.OK, b""

def remote_server(port:int):
  device = getenv("REMOTEDEV", next(Device.get_available_devices()) if Device.DEFAULT == "REMOTE" else Device.DEFAULT)
  async def _inner_async(port:int, device:str):
    print(f"start remote server on {port} with device {device}")
    await (await asyncio.start_server(RemoteHandler(device), host='', port=port)).serve_forever()
  asyncio.run(_inner_async(port, device))

# ***** frontend *****

class RemoteAllocator(Allocator['RemoteDevice']):
  def __init__(self, dev:RemoteDevice):
    self.numbes_lines = []
    super().__init__(dev)
  # TODO: ideally we shouldn't have to deal with images here
  def _alloc(self, size:int, itemsize:int, options:BufferSpec) -> int:
    numbers_size = size // itemsize
    self.dev.buffer_num += numbers_size
    return self.dev.buffer_num - numbers_size
  def _free(self, opaque:int, options): return
  def _copyin(self, dest:int, src:memoryview, dtype:dtypes):
    chunks = [bytes(src)[i:i+4] for i in range(0, len(bytes(src)), dtype.itemsize)]
    if dtype == dtypes.uint:
      for i in range(len(chunks)): chunks[i] = int.from_bytes(chunks[i], byteorder='little', signed=False)
    if dtype == dtypes.int:
      for i in range(len(chunks)): chunks[i] = int.from_bytes(chunks[i], byteorder='little', signed=True)
    if dtype == dtypes.float:
      for i in range(len(chunks)): chunks[i] = struct.unpack('<f', chunks[i])[0]
    for i in range(len(chunks)): self.copyin_numbers(chunks[i], (dest+i))
    inner = "\n                    ".join(self.numbes_lines)
    self.script = f"""
    tell application "Numbers"
        activate
        tell document 1
            tell sheet 1
                tell table 1
                        {inner}
                end tell
            end tell
        end tell
    end tell
    """
    print(self.script)
    subprocess.run(['osascript', '-e', self.script], capture_output=True, text=True)
  def _copyout(self, dest:memoryview, src:int, dtype:dtypes):
    ncells = int(len(dest) // dtype.itemsize)
    cells = []
    for i in range(ncells): cells.append(get_cell(src+i))
    cell_refs = ", ".join([f'"{cell}"' for cell in cells])
    self.script = f'''
    tell application "Numbers"
        activate
        tell document 1
            tell sheet 1
                tell table 1
                    set valueList to {{}}
                    repeat with cellRef in {{{cell_refs}}}
                        set end of valueList to value of cell cellRef
                    end repeat
                    set output to ""
                    repeat with i from 1 to count of valueList
                        set output to output & (item i of valueList) as text
                        if i < count of valueList then set output to output & ", "
                    end repeat
                    return output
                end tell
            end tell
        end tell
    end tell
    '''
    print("AppleScript:")
    print(self.script)
    result = subprocess.run(['osascript', '-e', self.script], capture_output=True, text=True)
    result = result.stdout.replace(" ","").replace("\n","").split(",")
    result = [float(x) for x in result]
    byte_data_32 = b''.join(struct.pack('f', x) for x in result)
    dest[:] = byte_data_32
  def _transfer(self, dest, src, sz, src_dev, dest_dev): return
  def _dyn_offset(self, opaque:int, size:int, offset:int) -> int: return
  def copyin_numbers(self, x, cell):
    cell = get_cell(cell)
    self.numbes_lines.append(f'set value of cell "{cell}" to {x}')

def get_cell(n, max_cols=1000):  # max_cols is how many columns per row
    def number_to_column(num):
        column = ""
        num = num + 1
        while num > 0:
            num, remainder = divmod(num - 1, 26)
            column = chr(65 + remainder) + column
        return column

    col_index = n % max_cols
    row_index = n // max_cols
    col = number_to_column(col_index)
    row = row_index + 1
    return f"{col}{row}"

def get_temp_cell(n, max_cols=1000): return get_cell(999_999_999 - n)


class RemoteProgram:
  def __init__(self, dev:RemoteDevice, name:str, lib:bytes):
    self.dev, self.name = dev, name
    self.lib = lib
    super().__init__()

  @staticmethod
  def _fini(dev:RemoteDevice, name:str, datahash:str): dev.q(ProgramFree(name, datahash))

  def __call__(self, *bufs, global_size=None, local_size=None, vals:tuple[int, ...]=(), wait=False):
    cell_bufs = list(bufs)
    script = self.lib.decode('utf-8')
    script = re.sub(r'\b\w+\s+(\w+)\s*=', r'set \1 to', script) # assign
    script = re.sub(r'\*\(([^)]+)\)\s*=', r'set *(\1) to', script) #pointer assign
    script = script.replace("}","")
    script = script.replace(";","")
    script = script[script.index("{")+1:]
    def replace_data_expr(match):
        a = int(match.group(1))
        c = int(match.group(2))
        base_cell = cell_bufs[a]
        final_cell = get_cell(base_cell + c)
        return final_cell
    script = re.sub(r"data(\d+)_[0-9]+\+(\d+)", replace_data_expr, script)
    script = re.sub(r'\(\*\(([A-Z]+[0-9]+)\)\)', r'value of cell "\1"', script)
    script = re.sub(r'set \*\(([A-Z]+[0-9]+)\) to', r'set value of cell "\1" to', script)
    def replace_val(match):
      n = int(match.group(1))
      cell = get_temp_cell(n)
      return f'{cell}' # todo add back the quotes ?
    script = re.sub(r'\bval(\d+)\b', replace_val, script)

    script = re.sub(r'set\s+([A-Z]+\d+)\s+to\s+value of cell\s+"([A-Z]+\d+)"', r'set value of cell "\1" to value of cell "\2"', script)
    script = re.sub(r'set value of cell "([A-Z]+\d+)" to \(([^)]+)\)', lambda m: f'set value of cell "{m.group(1)}" to "={m.group(2)}"', script)

    def add_static_freeze(match):
      cell = match.group(1)
      assignment = match.group(0)
      freeze_line = f'set value of cell "{cell}" to value of cell "{cell}"'
      return assignment + "\n" + freeze_line
    script = re.sub(r'set value of cell "([A-Z]+\d+)" to [^\n]+', add_static_freeze, script)
    script = re.sub(
        r'(set value of cell "[A-Z]+1" to "=\([^"]*)"(\+.*?)(?=\s*set value of cell "[A-Z]+1" to value of cell)',
        r'\1\2"',
        script,
        flags=re.DOTALL
    )
    script = f"""tell application "Numbers"
        activate
        tell document 1
            tell sheet 1
                tell table 1
                    {script}
                end tell
            end tell
        end tell
    end tell"""
    print(script)
    subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    return None

@functools.cache
class RemoteConnection:
  q_lock = threading.Lock()
  all: dict[RemoteConnection, None] = {} # dict instead of set for deterministic ordering

  def __init__(self, host:str):
    if DEBUG >= 1: print(f"remote with host {host}")
    while 1:
      try:
        self.conn = http.client.HTTPConnection(host, timeout=getenv("REMOTE_TIMEOUT", 300.0))
        self.conn.connect()
        break
      except Exception as e:
        print(e)
        time.sleep(0.1)
    self.req: BatchRequest = BatchRequest()
    RemoteConnection.all[self] = None

  def q(self, x:RemoteRequest, wait:bool=False):
    with RemoteConnection.q_lock:
      self.req.q(x)
      if wait: return self.batch_submit(take_q=False)

  async def aq(self, x:RemoteRequest, wait:bool=False): return await asyncio.to_thread(self.q, x, wait=wait)

  def batch_submit(self, take_q:bool=True):
    if take_q: RemoteConnection.q_lock.acquire()
    conns = RemoteConnection.all.keys()
    datas = {conn: conn.req.serialize() for conn in conns}
    reqs, hashes, hash_datas = sum(len(c.req._q) for c in conns), sum(len(c.req._h) for c in conns), sum(len(data) for data in datas.values())
    ret, resps = None, []
    with Timing(f"*** send {reqs:-3d} requests {hashes:-3d} hashes with len {hash_datas/1024:.2f} kB in ", enabled=DEBUG>=3):
      for conn,data in datas.items(): conn.conn.request("POST", "/batch", data)
      for conn in datas.keys():
        resp = conn.conn.getresponse()
        body = resp.read()
        resps.append((conn, resp, body))
        conn.req = BatchRequest()
    if take_q: RemoteConnection.q_lock.release()
    for conn,resp,body in resps:
      match resp.status:
        case http.HTTPStatus.OK: pass
        case http.HTTPStatus.INTERNAL_SERVER_ERROR:
          #exc_wrapper = safe_eval(ast.parse(body.decode(), mode="eval").body)
          #exc_wrapper.exc.add_note(exc_wrapper.trace)
          #raise exc_wrapper.exc
          return 0.0
        case code: raise RuntimeError(f"POST /batch failed with {code}: {body.decode()}")
      if conn == self: ret = body
    return ret

def parse_hosts(hs:str) -> list[tuple[str, int]]|LazySeq[tuple[str, int]]:
  hosts = [(unwrap(h), int(c) if c is not None else c) for h,c in ((h.split("*", maxsplit=1)+[None,])[:2] for h in hs.split(","))]
  if len(hosts) == 1 and hosts[0][1] is None: return LazySeq(lambda idx: (hosts[0][0], idx))
  return [(h, i) for h,c in hosts for i in range(unwrap(c))]

class RemoteDevice(Compiled):
  devices = parse_hosts(getenv("HOST", ""))

  def __init__(self, device:str):
    host, idx = RemoteDevice.devices[int(device.split(":")[1]) if ":" in device else 0]

    # connection is shared between sessions on the same host
    self.session: SessionKey = SessionKey(host or RemoteDevice.local_server(), idx, binascii.hexlify(os.urandom(0x10)).decode())
    self.conn: RemoteConnection = RemoteConnection(self.session.host)

    # state for the session
    self.buffer_num: int = 0
    self.graph_num: Iterator[int] = itertools.count(0)
    self.event_num: Iterator[int] = itertools.count(0)

    self.properties = RemoteProperties(real_device='METAL', renderer=('tinygrad.renderer.cstyle', 'MetalRenderer', ()), offset_supported=True, graph_supported=True, graph_supports_multi=False, ib_gid=None)
    renderer = self.properties.renderer
    renderer_class = fromimport(renderer[0], renderer[1])  # TODO: is this secure?
    graph = fromimport('tinygrad.runtime.graph.remote', "RemoteGraph") if self.properties.graph_supported else None
    compilers = [(functools.partial(renderer_class, *renderer[2]), Compiler)]
    super().__init__(device, RemoteAllocator(self), compilers, functools.partial(RemoteProgram, self), graph, id(self.conn))
    self.renderer.device = device

  def finalize(self):
    with contextlib.suppress(ConnectionError, http.client.HTTPException): self.q(SessionFree(), wait=True)

  def q(self, x:RemoteRequest, wait:bool=False): return self.conn.q(replace(x, session=self.session), wait=wait)

  @functools.cache
  @staticmethod
  def local_server():
    multiprocessing.Process(target=remote_server, args=(6667,), name="MainProcess", daemon=True).start()
    return "127.0.0.1:6667"

if __name__ == "__main__": remote_server(getenv("PORT", 6667))
