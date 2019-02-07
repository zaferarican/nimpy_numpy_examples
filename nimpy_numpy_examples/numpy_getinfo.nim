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
  
  let py = pyBuiltinsModule()
  var shape = py.list(o.shape)
  echo "[",shape[0],",",shape[1],"]"  
