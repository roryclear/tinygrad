from tinygrad import Tensor
from tinygrad.dtype import dtypes
import subprocess
import os
from pathlib import Path


file = Path(__file__).parent / "tiny.numbers"
script = f'''
tell application "Numbers"
    activate
    set newDoc to make new document
    save newDoc in POSIX file "{file}"
end tell
'''
result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)

x = Tensor([1,2,3])
y = Tensor([4,5,6])
print(x.numpy())
print(y.numpy())
#x = Tensor([10.0,20.0])
#x = x.cast(dtype=dtypes.float)
#print(x.numpy())