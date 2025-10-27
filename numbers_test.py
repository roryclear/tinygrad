from tinygrad import Tensor
from tinygrad.dtype import dtypes
import subprocess
import os
from pathlib import Path


file = Path(__file__).parent / "tinygrad" / "runtime" / "tiny.numbers"
script = f'''
tell application "Numbers"
    activate
    set newDoc to make new document
    tell newDoc
        tell sheet 1
            tell table 1
                set column count to 1000
                set row count to 1000
            end tell
        end tell
    end tell
    save newDoc in POSIX file "{file}"
end tell
'''
result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
#exit()

#x = Tensor([1,2,3])
#y = Tensor([4,5,6])
#print((x+y).numpy())
x = Tensor.arange(1000)
print(x.numpy())
#x = Tensor([10.0,20.0])
#x = x.cast(dtype=dtypes.float)
#print(x.numpy())