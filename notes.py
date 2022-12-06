a = [1, 2]
b = [3, 4]
c = a + b
# print(c)
# >>> [1, 2, 3, 4]

c = [val_a + val_b for val_a, val_b in zip(a, b)]
# print(c)
# >>> [4, 6]

import numpy as np
a = np.array(a)
b = np.array(b)
c = a + b
print(c)
# >>> [4 6]

# universale (vector) functions (.map-like) are optimised to perform on numpy arrays
np.sin(c)
# >>> array([-0.7568025 ,  0.2794155 ])

# numpy arrays are also optimised for memory usage, enforcing a single data type

# numpy arrays are of type ndarray, for n dimensional array

array2d = np.array([[1, 2, 3], [4, 5, 6]])
print(array2d)
# >>> [[1 2 3]
#      [4 5 6]]
array2d.shape
# >>> (2, 3)
array2d.size
# >>> 6
array2d.ndim
# >>> 2

# numpy is what allows the pandas df indexing, vs list[0][1] indexing
# well pandas is built on numpy

# numerical computing with numpy

# numerical type heirarchy: bool -> int -> float -> complex
a = np.array([1, 2, 3.0, 4+1j]) # j is the imaginary number
print(a.dtype)
# >>> complex128

x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])
xy = x * y
print(xy)
# >>> [ 1  4  9 16]
x *= y
print(x)
# >>> [ 1  4  9 16]

np.sin(x)
# >>> array([ 0.84147098, -0.7568025 ,  0.41211849, -0.28790332])

a = np.array([1, 2, 3, 4])
b = np.asarray(a, dtype=np.float) #float32
print(b.dtype)
# >>> float64

a.shape
# >>> (4,)
# a's shape (1 row, 4 columns), different from an array with 4 elements
a.reshape(2, 2)
# >>> array([[1, 2],
#            [3, 4]])


a = np.array([x*np.pi for x in range(1, 3, 0.5)])
print(a)
# >>> [ 3.14159265  4.71238898  6.28318531  7.85398163  9.42477796 10.99557429]
np.get_printoptions()
# >>> {'edgeitems': 3, 'threshold': 1000, 'floatmode': 'maxprec', 'precision': 8, 'suppress': False, 'linewidth': 75, 'nanstr': 'nan', 'infstr': 'inf', 'formatter': None}

np.set_printoptions(precision=3)
print(a)
# >>> [ 3.142  4.712  6.283  7.854  9.425 10.996]

# using a context manager approach
with np.printoptions(precision=3):
    print(a)
    # >>> [ 3.142  4.712  6.283  7.854  9.425 10.996]

# slicing defaults: [start:stop:step] -> [0:array_length:1]
# slices create a view of the original array, not a copy

# generate a 2d array of size 6x6 with random integers between 0 and 10
a = np.random.randint(0, 10, (6, 6))
print(a)
# >>> [[9 2 2 4 7 6]
#      [8 8 1 6 7 7]
#      [9 9 8 9 9 9]
#      [4 4 0 3 3 0]
#      [2 4 4 1 9 9]
#      [0 1 4 4 2 1]]

print(a[:, 3]) 
# >>> [4 6 9 3 1 4]
print(a(2::2, 2::2)) # every other row, every other column both starting at 2
# >>> [[8 9 9]
#      [0 3 0]
#      [4 1 1]]









