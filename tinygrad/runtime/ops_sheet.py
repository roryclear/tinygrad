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

COLS = 28
ROWS = 10000

# ***** frontend *****

class SheetAllocator(Allocator['SheetDevice']):
  def __init__(self, dev:SheetDevice):
    self.numbes_lines = []
    file = Path(__file__).parent / "tiny.numbers"
    script = f'''
    tell application "Numbers"
        activate
        set newDoc to make new document
        tell newDoc
            tell sheet 1
                tell table 1
                    set column count to {COLS}
                    set row count to {ROWS}
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
    chunks = [bytes(src)[i:i+dtype.itemsize] for i in range(0, len(bytes(src)), dtype.itemsize)]
    if dtype == dtypes.uint:
      for i in range(len(chunks)): chunks[i] = int.from_bytes(chunks[i], byteorder='little', signed=False)
    if dtype == dtypes.uint8:
      for i in range(len(chunks)): chunks[i] = chunks[i][0]
    elif dtype == dtypes.int:
      for i in range(len(chunks)): chunks[i] = int.from_bytes(chunks[i], byteorder='little', signed=True)
    elif dtype == dtypes.float:
      for i in range(len(chunks)): chunks[i] = struct.unpack('<f', chunks[i])[0]
    elif dtype == dtypes.int64:
      for i in range(len(chunks)): chunks[i] = struct.unpack('<q', chunks[i])[0]
    elif dtype == dtypes.float64:
      for i in range(len(chunks)): chunks[i] = struct.unpack('<d', chunks[i])[0]
       
    else:
      print("dtype not supported:",dtype)
      exit()
    for i in range(len(chunks)): self.copyin_numbers(chunks[i], (dest+i))
    script = "\n                    ".join(self.numbes_lines)
    # todo same as exec
    batch_size = 100_000
    script_lines = script.strip().split('\n')

    batches = []
    current_batch = []
    current_length = 0

    for line in script_lines:
        line_length = len(line) + 1
        if current_length + line_length > batch_size and current_batch:
            batches.append('\n'.join(current_batch))
            current_batch = [line]
            current_length = line_length
        else:
            current_batch.append(line)
            current_length += line_length

    if current_batch: batches.append('\n'.join(current_batch))
    for i, batch_script in enumerate(batches, start=1):
        full_script = f"""tell application "Numbers"
            activate
            tell document 1
                tell sheet 1
                    tell table 1
                        {batch_script}
                    end tell
                end tell
            end tell
        end tell"""
        print(full_script)
        subprocess.run(['osascript', '-e', full_script], capture_output=False, text=True)

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
    if type(x) == float: x = round(x, 8)
    self.numbes_lines.append(f'set value of cell "{cell}" to {x}')

def get_cell(n, max_cols=COLS):  # max_cols is how many columns per row
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

def get_temp_cell(n): return get_cell((COLS*ROWS) - 1 - n)


class SheetProgram:
  def __init__(self, dev:SheetDevice, name:str, lib:bytes):
    self.dev, self.name = dev, name
    self.lib = lib
    self.alu_num = 0
    super().__init__()
  def __call__(self, *bufs, global_size=None, local_size=None, vals:tuple[int, ...]=(), wait=False):
    cell_bufs = list(bufs)

    # remove metal stuff
    script = self.lib.decode('utf-8')
    script = script[script.index("{")+1:]
    script = script[:script.index("}")]
    script = script.replace(";","")

    # pointer -> cell
    pattern = re.compile(r'\*\(data(\d+)_(\d+)\+(\d+)\)')
    def replacer(match):
        a = int(match.group(1))
        b = int(match.group(2))  # not used, but available
        c = int(match.group(3))
        result = get_cell(cell_bufs[a] + c)
        return result
    script = pattern.sub(replacer, script)

    # val -> cell
    pattern = re.compile(r'\bval(\d+)\b')
    def replacer(match):
      n = int(match.group(1))
      self.alu_num = max(n, self.alu_num) + 1
      return get_cell(self.dev.buffer_num+1+n)
    script = pattern.sub(replacer, script)
    # alu -> cell # todo, this doesn't work properly?
    pattern = re.compile(r'\balu(\d+)\b')
    def replacer(match):
      n = int(match.group(1))
      self.alu_num = max(n, self.alu_num)
      return get_cell(self.dev.buffer_num+self.alu_num+1+n)
    script = pattern.sub(replacer, script)

    # remove datatype
    script = re.sub(r'^\s*unsigned\s+int\s+(?=\w+\s*=)', '', script, flags=re.MULTILINE)
    script = re.sub(r'^\s*unsigned\s+char\s+(?=\w+\s*=)', '', script, flags=re.MULTILINE)
    script = re.sub(r'^\s*\w+\s+(?=\w+\s*=)', '', script, flags=re.MULTILINE)
    script = re.sub(r'^\s+', '', script, flags=re.MULTILINE)

    # EXP2 -> 2^
    script = script.replace("exp2", "2^")

    # N^X to (N^X)
    script = re.sub(r'(N\^[A-Z]+\d+)', r'(\1)', script)

    # XXN = -> "set value of cell "XXN" to "
    script = re.sub(r'^([A-Z]+\d+)\s*=\s*', r'set value of cell "\1" to ', script, flags=re.MULTILINE)

    # to XXN -> "to value of "XXN""
    script = re.sub(r'(set value of cell "[^"]+" to )([A-Z]+\d+)', r'\1value of cell "\2"', script)

    # to (XXN*XYN) -> to "=(XXN*XYN)"
    pattern = r'(set value of cell "[^"]+" to )\((.*)\)'
    replacement = r'\1"=(\2)"'
    script = re.sub(pattern, replacement, script)
    
    # to 2^() to "=2^()"
    script = re.sub(r'(set value of cell "[^"]+" to )(\d+)\^\(([^)]+)\)$', r'\1"=\2^(\3)"', script, flags=re.MULTILINE)
    script = re.sub(r'(set value of cell "[^"]+" to )([^"\n]+[+*^/-][^"\n]*)$', r'\1"=\2"', script, flags=re.MULTILINE)

    # log2(N) to LOG(N, 2)
    script = re.sub(r'(?i)\blog2\s*\(([^()]*?(?:\([^()]*\)[^()]*)*)\)', r'LOG(\1, 2)', script)

    # remove formula after each set
    script = re.sub(r'(set value of cell "([^"]+)" to [^\n]+)', r'\1\nset value of cell "\2" to value of cell "\2"', script)

    # remove f for floats
    script = re.sub(r'(\d+)f', r'\1', script)

    # caps sqrt
    script = script.replace("sqrt","SQRT")

    # remove casts
    pattern = re.compile(r'^\s*(\w+)\s*=\s*\(\(float\)\((\w+)\)\)\s*\n?', flags=re.MULTILINE)
    aliases = {}
    for alias, original in re.findall(pattern, script):
        aliases[alias] = original
    script = re.sub(pattern, '', script)
    for alias, original in aliases.items(): script = re.sub(rf'\b{re.escape(alias)}\b', original, script)

    # 1e8 -> (1*10^8)
    script = re.sub(r'(-?\d*\.?\d+)e(-?\d+)', r'(\1*10^(\2))', script)

    # ((T2<T1)?T1:T2)
    script = re.sub(r'\(\(([^?]+)<([^?]+)\)\?([^:]+):([^)]+)\)', r'IF(\1<\2, \3, \4)', script)

    batch_size = 100_000
    script_lines = script.strip().split('\n')

    batches = []
    current_batch = []
    current_length = 0

    for line in script_lines:
        line_length = len(line) + 1
        if current_length + line_length > batch_size and current_batch:
            batches.append('\n'.join(current_batch))
            current_batch = [line]
            current_length = line_length
        else:
            current_batch.append(line)
            current_length += line_length

    if current_batch: batches.append('\n'.join(current_batch))
    for i, batch_script in enumerate(batches, start=1):
        full_script = f"""tell application "Numbers"
            activate
            tell document 1
                tell sheet 1
                    tell table 1
                        {batch_script}
                    end tell
                end tell
            end tell
        end tell"""
        print(full_script)
        subprocess.run(['osascript', '-e', full_script], capture_output=False, text=True)
      
class SheetDevice(Compiled):
  def __init__(self, device:str):
    self.buffer_num: int = 0
    from tinygrad.renderer.cstyle import SheetRenderer
    super().__init__(device, SheetAllocator(self), [(SheetRenderer, Compiler)], functools.partial(SheetProgram, self))
    self.renderer.device = device


