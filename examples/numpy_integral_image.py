import cv2
import numpy as np
import time
# Generated module using nimpy
import nimpy_integral as ni
import nimpy_integral_arraymancer as na
print("""

Integral image example and comparison to OpenCV integral image function

""")
duration_cv2 = 0
duration_nim = 0
duration_arma = 0
num_of_trials = 100

for _ in range(num_of_trials):
    a = np.random.randint(0,high=255, size=(400,100), dtype=np.uint8)

    start_time = time.time()
    sum_val = np.zeros(a.shape, np.int32)
    a_int = cv2.integral(a, sum_val, -1)
    a_int = a_int[1:,1:]
    duration_cv2 += time.time() - start_time

    start_time = time.time()
    b = a[:].astype(np.int32)
    ni.integral(b)
    duration_nim += time.time()-start_time

    start_time = time.time()
    b = a[:].astype(np.int32)
    na.integral(b)
    duration_arma += time.time()-start_time

print("Num Of Trials: " + str(num_of_trials))
print("OpenCV - Average time: " + str(duration_cv2 / num_of_trials))
print("Nim    - Average time: " + str(duration_nim / num_of_trials))
print("Amancer- Average time: " + str(duration_arma / num_of_trials))
print(" ")

print("**********************************")
print("OpenCV integral image")
sum_val = np.zeros(a.shape, np.int32)
a_int = cv2.integral(a, sum_val, -1)
a_int = a_int[1:,1:]
print(a_int[0:5,0:5])
print(" ")
print("**********************************")
print("Custom integral image using nimpy")
b = a[:].astype(np.int32)
ni.integral(b)
print(b[0:5,0:5])

print("**********************************")
print("Custom integral image using nimpy and arraymancer")
b = a[:].astype(np.int32)
na.integral(b)
print(b[0:5,0:5])

