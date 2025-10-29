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

# ***** frontend *****

class SheetAllocator(Allocator['SheetDevice']):
  def __init__(self, dev:SheetDevice):
    self.numbes_lines = []
    self.numbes_lines_csv = []
    Path("tiny.csv").touch()
    file = Path(__file__).parent / "tiny.numbers"
    script = f'''
    tell application "Numbers"
        activate
        set newDoc to make new document
        tell newDoc
            tell sheet 1
                tell table 1
                    set column count to 1000
                    set row count to 1000000
                end tell
            end tell
        end tell
        save newDoc in POSIX file "{file}"
    end tell
    '''
    subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    super().__init__(dev)
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
    self.numbes_lines_csv.append([cell, x])
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


class SheetProgram:
  def __init__(self, dev:SheetDevice, name:str, lib:bytes):
    self.dev, self.name = dev, name
    self.lib = lib
    super().__init__()
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
      cell = get_cell(self.dev.buffer_num+1+n)
      return f'{cell}' # todo add back the quotes ?
    script = re.sub(r'\bval(\d+)\b', replace_val, script)
    self.dev.buffer_num += len(cell_bufs) - 1

    script = re.sub(r'set\s+([A-Z]+\d+)\s+to\s+value of cell\s+"([A-Z]+\d+)"', r'set value of cell "\1" to value of cell \2', script)
    script = re.sub(r'value of cell ([A-Z]+\d+)', r'value of cell "\1"', script)

    def wrap_complex_expressions(match):
        assignment = match.group(0)
        if re.search(r'[+*\-/()]', assignment) and not re.search(r'to value of cell "[A-Z]+\d+"$', assignment):
            cell_match = re.search(r'set value of cell "([A-Z]+\d+)" to (.*)', assignment)
            if cell_match:
                cell = cell_match.group(1)
                expr = cell_match.group(2)
                if not re.match(r'value of cell "[A-Z]+\d+"$', expr.strip()):
                    return f'set value of cell "{cell}" to "={expr}"'
        return assignment

    script = re.sub(r'set value of cell "[A-Z]+\d+" to .*', wrap_complex_expressions, script)


    def add_static_freeze(match):
      cell = match.group(1)
      assignment = match.group(0)
      freeze_line = f'set value of cell "{cell}" to value of cell "{cell}"'
      return assignment + "\n" + freeze_line
    script = re.sub(r'set value of cell "([A-Z]+\d+)" to [^\n]+', add_static_freeze, script)
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

class SheetDevice(Compiled):
  def __init__(self, device:str):
    self.buffer_num: int = 0
    from tinygrad.renderer.cstyle import SheetRenderer
    super().__init__(device, SheetAllocator(self), [(SheetRenderer, Compiler)], functools.partial(SheetProgram, self))
    self.renderer.device = device
