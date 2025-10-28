from tinygrad import Tensor
from tinygrad.dtype import dtypes
import subprocess
import os
from pathlib import Path
import numpy as np

np.random.seed(1)
x = np.random.random((4, 4)).astype(np.float32)
y = np.random.random((4, 4)).astype(np.float32)
x = Tensor(x)
y = Tensor(y)
print((x.matmul(y)).numpy())
#x = Tensor([10.0,20.0])
#x = x.cast(dtype=dtypes.float)
#print(x.numpy())