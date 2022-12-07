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


a = np.array([x*np.pi for x in range(1, 6, 1)])
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
print(a[2::2, 2::2]) # every other row, every other column both starting at 2
# >>> [[8 9 9]
#      [0 3 0]
#      [4 1 1]]


b = a[1]
np.shares_memory(a, b)
# >>> True
b = a[1].copy()
np.shares_memory(a, b)
# >>> False

# Fancy indexing (indexing with an array of integers) creates a copy of the data unlike slicing
a = np.arange(0, 100, 10)
indices = [1, 5, -1]
b = a[indices]
np.shares_memory(a, b)
# >>> False

# Masking allows you to select elements that satisfy a condition, also creating a copy of the data
a = np.arange(0, 100, 10)
print(a)
# >>> [ 0 10 20 30 40 50 60 70 80 90]
# create a mask which will be used to select elements from a
mask = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
# select elements that satisfy the mask
b = a[mask]
print(b)
# >>> [10 50 90]
mask = a <= 50
b = a[mask]
print(b)
# >>> [ 0 10 20 30 40 50]
np.shares_memory(a, b)
# >>> False

# For a 2d array you can apply fancy indexing max to a specific column or row
# create random 2d array
a = np.random.randint(0, 10, (6, 6))
print(a)
# >>> [[9 2 2 4 7 6]
#      [8 8 1 6 7 7]
#      [9 9 8 9 9 9]
#      [4 4 0 3 3 0]
#      [2 4 4 1 9 9]
#      [0 1 4 4 2 1]]
mask = np.array([1, 0, 1, 0, 0, 1], dtype=bool)
b = a[mask, 2]
print(b)
# >>> [2 8 4]
b = a[2, mask]
print(b)
# >>> [9 8 9]

np.linspace(0, 10, 5)
# >>> array([ 0. ,  2.5,  5. ,  7.5, 10. ])
np.logspace(0, 10, 5)
# >>> array([1.e+00, 1.e+02, 1.e+04, 1.e+06, 1.e+08, 1.e+10])

# loadtxt() and savetxt() are useful for loading and saving data from/to text files
# they can also be used to load/save data from a URL

# Multi-dimensional array axis:
# 2D: 0 = rows, 1 = columns
# 3D: 0 = depth, 1 = rows, 2 = columns
# 4D: 0 = depth, 1 = rows, 2 = columns, 3 = channels
# Negative axis indexing: -1 = last axis, -2 = second last axis, etc. This is useful when you don't know the dimensionality of the array as it stays consistent with all dimensions

a = np.random.randint(0, 10, (6, 6))
print(a)
# >>> [[0 2 2 8 1 0]
#      [6 5 5 7 2 2]
#      [4 6 1 4 1 2]
#      [5 1 5 3 9 8]
#      [4 6 3 2 4 4]
#      [2 4 3 4 0 5]]
a.max()
# >>> 9
a.argmax()
# >>> 23
# argmax() returns the index of the maximum value but works on a flattened array
np.unravel_index(a.argmax(), a.shape)
# >>> (3, 4)
# unravel_index() returns the index of the maximum value after unflattening the array

mask = a > 8
print(np.where(mask))
# >>> (array([3], dtype=int64), array([4], dtype=int64))
# where() returns the indices of the elements that satisfy the condition, with the first array containing the row indices and the second array containing the column indices

# Transpose in numpy is a very cheap operation as it does not change the data in memoery itself, only the metadata for the array.
# This is in contrast to the transpose() method in pandas which creates a copy of the data.



