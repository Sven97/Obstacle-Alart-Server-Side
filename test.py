import numpy as np

a = np.arange(12).reshape(3, 4)
print(a)

b, c, d = np.vsplit(a, 3)

print(b)
