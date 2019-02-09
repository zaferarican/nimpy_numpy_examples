import nimpy
import nimpy/raw_buffers
import nimpy/[py_types, py_utils]

#[
  Display NumPy array info using Nimpy .to conversion function
]#

proc getinfo(o: PyObject) {.exportpy.} =
  var ndim = o.ndim.to(int)
  echo ndim
  
  echo o.dtype
  var dt : string = o.dtype.name.to(string)
  echo dt
  
  var shape = o.shape.to(seq[int])
  echo "[",shape[0],",",shape[1],"]"  
