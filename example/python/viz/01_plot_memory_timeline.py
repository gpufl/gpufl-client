import gpufl.viz as viz

# 1. Show single run (Kernels, Host, GPU)
viz.init("*.log")

viz.show(app="PythonDemo")

viz.show(tag="io-bound")

viz.compare(group_by="app", metric="cpu", app=["PythonDemo", "PythonDemo2"])
