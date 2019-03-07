import nimpy
import nimpy/raw_buffers
import arraymancer


proc `+`[T](p: ptr T, val: int) : ptr T {.inline.}=
  cast[ptr T](cast[uint](p) + cast[uint](val * sizeof(T)))

proc toTensor*(aBuf: RawPyBuffer):Tensor[cint] =
  let rows = aBuf.shape[].int
  let columns = (aBuf.shape + 1)[].int
  result = newTensor[cint](rows,columns)
  var bufPtr = cast[ptr UncheckedArray[cint]](aBuf.buf)
  var tensorDataPtr = cast[ptr UncheckedArray[cint]](result.get_data_ptr)
  copyMem(tensorDataPtr, bufPtr, rows*columns*sizeof(cint))

proc toPyObject*(t: Tensor[cint], aBuf: RawPyBuffer) =
  let rows = aBuf.shape[].int
  let columns = (aBuf.shape + 1)[].int
  var bufPtr = cast[ptr UncheckedArray[cint]](aBuf.buf)
  var tensorDataPtr = cast[ptr UncheckedArray[cint]](t.get_data_ptr)
  copyMem(bufPtr, tensorDataPtr, rows*columns*sizeof(cint))
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

  var integral_tensor = aBuf.toTensor

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
  integral_tensor.toPyObject(aBuf)
  aBuf.release()
