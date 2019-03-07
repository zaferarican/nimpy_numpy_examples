import nimpy
import nimpy/raw_buffers
import arraymancer


proc `+`[T](p: ptr T, val: int) : ptr T {.inline.}=
  cast[ptr T](cast[uint](p) + cast[uint](val * sizeof(T)))

# Operator [] overloading for type int32 (cint) to use for numpy buffer

proc `[]`(p:RawPyBuffer, x:uint32):cint {.inline.} =
  cast[ptr UncheckedArray[cint]](p.buf)[x]

proc `[]=`(p: RawPyBuffer, x:uint32, val:cint) {.inline.} =
  cast[ptr UncheckedArray[cint]](p.buf)[x] = val


proc toTensor*(aBuf: RawPyBuffer):Tensor[cint] =
  let rows = aBuf.shape[].int32
  let columns = (aBuf.shape + 1)[].int32
  var seq_xy = newSeq[cint](rows * columns)
  for i in 0..<(rows * columns):
    seq_xy[i] = aBuf[uint32(i)]
  result = seq_xy.toTensor.reshape(int(rows), int(columns)) 

proc toPyObject*(t: Tensor[cint], aBuf: RawPyBuffer) = 
  let rows = aBuf.shape[].uint32
  let columns = (aBuf.shape + 1)[].uint32
  for coord,v in t:
      aBuf[uint32(coord[0]) * columns + uint32(coord[1])] = v
#[
  Computation of in-place integral image using Viola Recursion Method:
    Viola, P.; Jones, M. Rapid Object Detection using a Boosted Cascade of Simple Features. 
  In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 
  Kauai, HI, USA, 8–14 December 2001; pp. 511–518.

  It assumes that the numpy array has dtype in np.int32
]#

proc integral(o: PyObject) {.exportpy.} =
  var aBuf: RawPyBuffer
  o.getBuffer(aBuf, PyBUF_WRITABLE or PyBUF_ND)
  let rows = aBuf.shape[].int
  let columns = (aBuf.shape + 1)[].int

  var integral_tensor = newTensor[cint](rows,columns)
  var bufPtr = cast[ptr UncheckedArray[cint]](aBuf.buf)
  var tensorDataPtr = cast[ptr UncheckedArray[cint]](integral_tensor.get_data_ptr)
  copyMem(tensorDataPtr, bufPtr, rows*columns*sizeof(cint))

  var s_o = integral_tensor[0,0]
  var s_c = s_o

  for row in 1..<rows:
    s_o = integral_tensor[row, 0]
    integral_tensor[row, 0] = integral_tensor[(row-1), 0] + s_o

  for column in 1..<columns:
    s_o = integral_tensor[0, column] + s_c
    s_c = s_o
    integral_tensor[0, column] = s_o

  for row in 1..<rows:
    s_c = integral_tensor[row, 0] - integral_tensor[(row-1), 0]
    for column in 1..<columns:
      s_o = integral_tensor[row, column] + s_c
      integral_tensor[row, column] = integral_tensor[(row-1), column] + s_o
      s_c = s_o
  copyMem(bufPtr, tensorDataPtr, rows * columns * sizeof(cint))
  aBuf.release()
