from control.matlab import *

# Parameters defining the system
m = 250.0  # system mass
k = 40.0  # spring constant
b = 60.0  # damping constant


# System matrices
A = [[0, 1.0], [-k / m, -b / m]]
B = [[0], [1 / m]]
C = [[1.0, 0]]

sys = ss(A, B, C, 0)
print(c2d(sys, 0.01, "bilinear", prewarp_frequency = 100))
