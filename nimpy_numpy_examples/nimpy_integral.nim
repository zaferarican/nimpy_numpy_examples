import nimpy
import nimpy/raw_buffers

proc `+`[T](p: ptr T, val: int) : ptr T {.inline.} =
  cast[ptr T](cast[uint](p) + cast[uint](val * sizeof(T)))

# Operator [] overloading for type int32 (cint) to use for numpy buffer
proc `[]`(p:RawPyBuffer, y:uint32, x:uint32):cint {.inline.} =
  cast[ptr UncheckedArray[cint]](p.buf)[y * (p.shape + 1)[].uint32 + x]

proc `[]=`(p: RawPyBuffer, y:uint32, x:uint32, val:cint) {.inline.} = 
  cast[ptr UncheckedArray[cint]](p.buf)[y * (p.shape + 1)[].uint32 + x] = val


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

  let rows = aBuf.shape[].uint32
  let columns = (aBuf.shape + 1)[].uint32

  var s_o = aBuf[0,0]
  var s_c = s_o

  for row in 1..<rows:
    s_o = aBuf[row, 0]
    aBuf[row, 0] = aBuf[(row-1), 0] + s_o

  for column in 1..<columns:
    s_o = aBuf[0, column] + s_c
    s_c = s_o
    aBuf[0, column] = s_o

  for row in 1..<rows:
    s_c = aBuf[row, 0] - aBuf[(row-1), 0]
    for column in 1..<columns:
       s_o = aBuf[row, column] + s_c
       aBuf[row, column] = aBuf[(row-1), column] + s_o
       s_c = s_o

  aBuf.release()
