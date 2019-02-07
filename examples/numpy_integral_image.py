import cv2
import numpy as np
import time
# Generated module using nimpy
import nimpy_integral as ni 

print("""

Integral image example and comparison to OpenCV integral image function

""")
a = np.random.randint(0,high=255, size=(400,100), dtype=np.uint8)

print("**********************************")
print("OpenCV integral image")
start_time = time.time()
sum_val = np.zeros(a.shape, np.int32)
a_int = cv2.integral(a, sum_val, -1)
a_int = a_int[1:,1:]
print(time.time()-start_time)
print(a_int[0:5,0:5])
print(" ")
print("**********************************")
print("Custom integral image using nimpy")
start_time = time.time()
b = a[:].astype(np.int32)
ni.integral(b)
print(time.time()-start_time)
print(b[0:5,0:5])

