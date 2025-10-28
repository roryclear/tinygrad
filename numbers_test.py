from tinygrad import Tensor
from tinygrad.dtype import dtypes
import subprocess
import os
from pathlib import Path
import numpy as np


file = Path(__file__).parent / "tinygrad" / "runtime" / "tiny.numbers"
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
result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
#exit()
np.random.seed(1)
x = np.random.random((4, 4)).astype(np.float32)
y = np.random.random((4, 4)).astype(np.float32)
x = Tensor(x)
y = Tensor(y)
print((x.matmul(y)).numpy())
#x = Tensor([10.0,20.0])
#x = x.cast(dtype=dtypes.float)
#print(x.numpy())