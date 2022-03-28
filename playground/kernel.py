import numpy as np
import numpy.linalg
import scipy as sp
import scipy.linalg

# Kernel (aka nullspace) of a matrix
# Construct an n-by-n matrix with a kernel of dimension r (r<n)
n = 7
r = 3
A = np.random.randint(-30, 31, size=[n - r, n])
A = np.vstack((A, np.random.randint(-5, 5, size=[r, n - r]) @ A))

# Determine the kernel of A (spanned by the columns of N)
N = sp.linalg.null_space(A)

# This means that all vectors of the form x = N*z are in the kernel of A
# Conversely, for any vector x in the kernel of A, there is a z such that
# x = N*z. For example:
z0 = np.random.randn(r, 1)
x0 = N @ z0
if not np.linalg.norm(A @ x0, np.inf) < 1e-10:
    print('x0 error')

# Projection on kernel
# Now we want to project a given vector x onto the kernel of A
# The projection is given by


def proj_ker(x_):
    return N @ numpy.linalg.lstsq(N, x_, rcond=None)[0]


# So, for example:
x = np.random.randn(n, 1)
x_ker = proj_ker(x)

# This means we can write any x in R^n as x = x_ker + x_im, where x_ker is
# in the kernel of A and x_im is in its image. Additionally, x_ker will be
# perpendicular to x_im:
x_im = x - x_ker
if not abs(x_im.T @ x_ker) < 1e-10:
    print('x_im error')
