from tinygrad import Tensor
x = Tensor.arange(1024)
y = Tensor.arange(4).repeat(256)
x *= y
x = x.sum() + y.sum()
print(x.numpy())