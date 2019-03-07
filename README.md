# nimpy_numpy_examples
Examples to access numpy array data using nimpy
- numpy_getinfo: This example gets info such as dtype, ndim, and shape from numpy array
- nimpy_integral: This example gets numpy data using buffer protocol in nimpy and implements integral image (summed area table)

Python examples in examples directory call functions generated with nim and nimpy

It uses [nimpy](https://github.com/yglukhov/nimpy) and of course [nim](https://nim-lang.org)

Python examples require numpy and opencv-python

It uses clang as the default compiler (--cc:clang flag). If you want to change it to another, please change in .nimble file.

## How To Run

```sh
$ nimble run
```

License
----

MIT

