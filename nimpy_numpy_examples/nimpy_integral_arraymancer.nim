import nimpy
import nimpy/raw_buffers
import arraymancer


proc `+`*[T](p: ptr T, val: int) : ptr T {.inline.}=
  cast[ptr T](cast[uint](p) + cast[uint](val * sizeof(T)))

# Operator [] overloading for type int32 (cint) to use for numpy buffer
proc `[]`*(p:RawPyBuffer, y:uint32, x:uint32):cint {.inline.} =
  cast[ptr UncheckedArray[cint]](p.buf)[y * (p.shape + 1)[].uint32 + x]

proc `[]=`*(p: RawPyBuffer, y:uint32, x:uint32, val:cint) {.inline.} =
  cast[ptr UncheckedArray[cint]](p.buf)[y * (p.shape + 1)[].uint32 + x] = val

proc toTensor*(o: PyObject):Tensor[cint] =
  var aBuf: RawPyBuffer
  o.getBuffer(aBuf, PyBUF_WRITABLE or PyBUF_ND)
  let rows = aBuf.shape[].uint32
  let columns = (aBuf.shape + 1)[].uint32
  var 
    seq_xy : seq[seq[cint]]
    row_seq: seq[cint]

  seq_xy=newSeq[seq[cint]]()
  for row in 0..<rows:
    row_seq = newSeq[cint]()
    for column in 0..<columns:
      row_seq.add(aBuf[row,column])
    seq_xy.add(row_seq)
  let converted = seq_xy.toTensor()
  aBuf.release()
  return converted

proc toPyObject*(t: Tensor[cint], o: PyObject) =
  var aBuf: RawPyBuffer
  o.getBuffer(aBuf, PyBUF_WRITABLE or PyBUF_ND)
  let rows = aBuf.shape[].uint32
  let columns = (aBuf.shape + 1)[].uint32
  for coord,v in t:
      aBuf[uint32(coord[0]),uint32(coord[1])] = v
  aBuf.release()
#[
  Computation of in-place integral image using Viola Recursion Method:
    Viola, P.; Jones, M. Rapid Object Detection using a Boosted Cascade of Simple Features. 
  In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 
  Kauai, HI, USA, 8–14 December 2001; pp. 511–518.

  It assumes that the numpy array has dtype in np.int32
]#

proc integral(o: PyObject) {.exportpy.} =
  var integral_tensor = o.toTensor

  let rows = integral_tensor.shape[0]
  let columns = integral_tensor.shape[1]

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
  integral_tensor.toPyObject(o)
