from tinygrad import Tensor
from tinygrad.dtype import dtypes
x = Tensor([10,20])
print(x.numpy())
x = Tensor([10.0,20.0])
x = x.cast(dtype=dtypes.float)
print(x.numpy())