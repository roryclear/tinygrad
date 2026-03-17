from tinygrad import Tensor
import numpy as np
n = 10
a_np = np.random.rand(n, n).astype(np.float32)
b_np = np.random.rand(n, n).astype(np.float32)
a = Tensor(a_np)
b = Tensor(b_np)
x = Tensor.matmul(a,b)
x_np = np.matmul(a_np,b_np)
np.testing.assert_allclose(x.numpy(), x_np, rtol=1e-6)
x = x.sum()
x_np = x_np.sum()
np.testing.assert_allclose(x.numpy(), x_np, rtol=1e-6)