"""
Supporting functions
"""
import numpy as np


                                                  
def clipValues(data, maxValue=15):
  # Clip the values  
  return (np.minimum(maxValue,
                     np.maximum(-maxValue, 
                                data)
                    ))

def oneHotKey(y, numOutputs):
  # y is a vector of integers and numOutputs is the number of possible integers
  # Return Y as vector with a 1 on the y index 
  Y = np.zeros((len(y), numOutputs))
  Y[np.arange(len(y)), y] = 1
  return (Y)

def add1s_right(matrix, coeff=1):
  return (np.concatenate((matrix,
                          coeff * np.ones((matrix.shape[0], 1))),
                         axis=1))
                                                  
def add1s_down(matrix, coeff=1):
  return (np.concatenate((matrix,
                          coeff * np.ones((1, matrix.shape[1]))),
                         axis=0))
                         
                         
""" 
im2col functions are taken from CS231n classes
https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
"""
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width = int((W + 2 * padding - field_width) / stride + 1)

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)

def im2col(x, field_height, field_width, padding=1, stride=1):
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols

def col2im(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  N, C, H, W = np.array(x_shape).astype(int)
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]
"""
End of im2col package
"""


      
