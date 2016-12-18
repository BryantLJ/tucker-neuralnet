import numpy as np
import scipy.linalg as lin

"""
returns X, U3, U4. kernel dimensions are not decomposed.
with input dimension of S, output Dimension of T, sizes of U3 and U4 are
S x r3, T x r4 respectively, resulting in core tensor size of which is D x D x r3 x r4,
where 
"""
def tucker2(tensor, r3, r4):
  pass

"""
first two dimensions of tensor corresponds to T, S and reduced to r4, r3
tensor[T,S,D,D]

"""
def GLRAM(tensor, r3, r4, num_iter=500, error_print=False):
  """
  L[r4, T]
  R[r3, S]
  """
  orig_dims = tensor.shape
  mat = tensor.reshape(orig_dims[0], orig_dims[1], -1)
  dims = mat.shape

  T = dims[0]
  S = dims[1]

  L = np.zeros((T, r4)) #replace with proper init code
  L[:r4,:r4] = np.eye(r4, r4)
  R = np.zeros((S, r3))
  R[:r3,:r3] = np.eye(r3, r3)
  ML = np.zeros((T, T))
  MR = np.zeros((S, S))
  prev_err = None
  for _ in range(num_iter):
    if r3 != S:
      for i in range(dims[2]):
        tmp = mat[:,:,i].T.dot(L)
        MR += tmp.dot(tmp.T)
      eig, eigv = lin.eig(MR)
      idx = np.argsort(eig)[::-1]
      R = eigv[:,idx][:,:r3]

    if r4 != T:
      for i in range(dims[2]):
        tmp = mat[:,:,i].dot(R)
        ML += tmp.dot(tmp.T)
      eig, eigv = lin.eig(ML)
      idx = np.argsort(eig)[::-1]
      L = eigv[:,idx][:,:r4]

    if error_print:
      M = np.zeros((r4, r3, dims[2]))
      for i in range(dims[2]):
        M[:,:,i] = L.T.dot(mat[:,:,i].dot(R))
      #error check
      sum = 0
      for k in range(dims[2]):
        sum += mat[:,:,i].dot(mat[:,:,i].T).trace()
        sum += M[:,:,i].dot(M[:,:,i].T).trace()
        sum += -2 * L.dot(M[:,:,i]).dot(R.T.dot(mat[:,:,i].T)).trace()
      sum = np.sqrt(sum / dims[2])
      if prev_err is None:
        print("Iter %d, Error = %f" % (_, sum))
      else:
        diff = (prev_err-sum)/prev_err
        print("Iter %d, Error = %f, eta = %f" % (_, sum, diff))
        if abs(diff) < 1e-6:
          break
      if sum < 1e-9:
        break
      prev_err = sum

  M = np.zeros((r4, r3, dims[2]))
  for i in range(dims[2]):
    M[:,:,i] = L.T.dot(mat[:,:,i].dot(R))

  M = M.reshape(r4, r3, *orig_dims[2:])
  print ('M', M.shape, 'L', L.shape, 'R', R.shape)
  return (M, L, R)