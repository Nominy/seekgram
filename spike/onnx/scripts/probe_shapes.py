"""Runs a graph on the CPU with extra outputs to print runtime shapes of the inputs of every
MatMul node with rank > 4 (DirectML's MatMul rejects them). Debugging aid for the DML patch.
Usage: uv run --no-project --with onnx --with onnxruntime --with numpy python scripts/probe_shapes.py <graph.onnx>"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

path = sys.argv[1]
model = onnx.load(path, load_external_data=False)
names = []
for node in model.graph.node:
    if node.op_type == "MatMul":
        names += [n for n in node.input if n not in {i.name for i in model.graph.initializer}]
for name in dict.fromkeys(names):
    model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
probe = os.path.join(os.path.dirname(path), "_probe.onnx")
onnx.save(model, probe)
session = ort.InferenceSession(probe, providers=["CPUExecutionProvider"])
frames = 2999
outputs = session.run(None, {"input_features": np.zeros((1, frames, 128), np.float32), "input_features_mask": np.ones((1, frames), bool)})
shapes = {o.name: v.shape for o, v in zip(session.get_outputs(), outputs)}
os.remove(probe)
count = 0
for node in model.graph.node:
    if node.op_type == "MatMul":
        dims = [shapes.get(n, "init") for n in node.input]
        if any(isinstance(d, tuple) and len(d) > 4 for d in dims):
            count += 1
            if count <= 3:
                print(node.name, dims)
print("MatMul nodes with rank > 4:", count)
