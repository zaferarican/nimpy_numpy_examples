version       = "0.1.0"
author        = "Zafer ARICAN"
description   = "Examples to access Numpy array data using Nimpy"
license       = "MIT"

# Dependencies

requires "nim >= 0.17.0"
requires "nimpy"

import oswalkdir, ospaths, strutils

task run, "Run examples":
    let pluginExtension = when defined(windows): "pyd" else: "so"

    for f in walkDir("nimpy_numpy_examples"):
        let sf = f.path.splitFile()
        let folder = sf.dir.parentDir
        let plugin_path = folder.joinPath("examples").joinPath(sf.name.changeFileExt(pluginExtension))
        if sf.ext == ".nim":
          exec "nim c --cc:clang -d:release  --app:lib --out:" & plugin_path & " " & f.path

    for f in walkDir("examples"):
        let sf = f.path.splitFile()
        if sf.ext == ".py":
            exec "python3 " & f.path
