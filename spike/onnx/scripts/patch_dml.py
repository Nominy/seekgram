"""Makes the onnx-community graphs run on DirectML. Two rewrites, both numerically exact:

1. Reshape allowzero=1 -> 0 (DirectML rejects allowzero=1 with E_INVALIDARG, "The parameter is
   incorrect"). Identical whenever no target dimension is 0, which holds here: every target shape
   is assembled from dimensions of non-empty tensors.
2. MatMul with rank-5 operands (the audio encoder's chunked attention, 24 nodes,
   [b, heads, chunks, 12, d] x [b, heads, chunks, d, 24]) -> Reshape both to rank 3 by merging the
   three leading dims, MatMul, Reshape back. DirectML's MatMul accepts at most rank 4. Valid
   because both operands have the same leading dims (no broadcasting; checked with
   scripts/probe_shapes.py).

Writes models/embeddinggemma-2-ONNX-dml/onnx/<graph><suffix>.onnx and hard-links the weights.
Usage: uv run --no-project --with onnx python scripts/patch_dml.py fp16 fp32
"""
import os
import shutil
import sys

import onnx
from onnx import helper, TensorProto

src = "models/embeddinggemma-2-ONNX"
dst = "models/embeddinggemma-2-ONNX-dml"
os.makedirs(os.path.join(dst, "onnx"), exist_ok=True)
for name in os.listdir(src):
    if os.path.isfile(os.path.join(src, name)) and not os.path.exists(os.path.join(dst, name)):
        shutil.copy(os.path.join(src, name), os.path.join(dst, name))


def rank5_matmuls(model):
    """Names of MatMul nodes whose first input comes from a rank-5 Transpose (perm of length 5)."""
    producers = {output: node for node in model.graph.node for output in node.output}
    found = set()
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        ranks = []
        for name in node.input:
            producer = producers.get(name)
            perm = next((list(a.ints) for a in producer.attribute if a.name == "perm"), None) if producer is not None and producer.op_type == "Transpose" else None
            ranks.append(len(perm) if perm else None)
        if any(rank == 5 for rank in ranks):
            found.add(node.name)
    return found


def rewrite_matmul(model, node, counter):
    a, b = node.input
    (out,) = node.output
    p = f"dmlfix{counter}_"
    const = lambda name, values: helper.make_tensor(p + name, TensorProto.INT64, [len(values)], values)
    model.graph.initializer.extend([const("m1", [-1]), const("s0", [0]), const("s3", [3]), const("s4", [4]), const("s5", [5])])
    nodes = [
        helper.make_node("Shape", [a], [p + "sa"]),
        helper.make_node("Shape", [b], [p + "sb"]),
        helper.make_node("Slice", [p + "sa", p + "s0", p + "s3"], [p + "lead"]),
        helper.make_node("Slice", [p + "sa", p + "s3", p + "s5"], [p + "ta"]),
        helper.make_node("Slice", [p + "sb", p + "s3", p + "s5"], [p + "tb"]),
        helper.make_node("Concat", [p + "m1", p + "ta"], [p + "a3s"], axis=0),
        helper.make_node("Concat", [p + "m1", p + "tb"], [p + "b3s"], axis=0),
        helper.make_node("Reshape", [a, p + "a3s"], [p + "a3"]),
        helper.make_node("Reshape", [b, p + "b3s"], [p + "b3"]),
        helper.make_node("MatMul", [p + "a3", p + "b3"], [p + "c3"], name=node.name + "_rank3"),
        helper.make_node("Slice", [p + "sa", p + "s3", p + "s4"], [p + "m"]),
        helper.make_node("Slice", [p + "sb", p + "s4", p + "s5"], [p + "n"]),
        helper.make_node("Concat", [p + "lead", p + "m", p + "n"], [p + "cs"], axis=0),
        helper.make_node("Reshape", [p + "c3", p + "cs"], [out]),
    ]
    return nodes


for suffix in sys.argv[1:]:
    suffix = "" if suffix == "fp32" else "_" + suffix
    for graph in ("model", "vision_encoder", "audio_encoder"):
        base = f"{graph}{suffix}.onnx"
        model = onnx.load(os.path.join(src, "onnx", base), load_external_data=False)
        allowzero = 0
        for node in model.graph.node:
            if node.op_type == "Reshape":
                for attribute in node.attribute:
                    if attribute.name == "allowzero" and attribute.i == 1:
                        attribute.i = 0
                        allowzero += 1
        targets = rank5_matmuls(model)
        rebuilt = []
        for node in model.graph.node:
            if node.name in targets:
                rebuilt.extend(rewrite_matmul(model, node, len(rebuilt)))
            else:
                rebuilt.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rebuilt)
        onnx.save(model, os.path.join(dst, "onnx", base))
        data = os.path.join(dst, "onnx", base + "_data")
        if not os.path.exists(data):
            os.link(os.path.join(src, "onnx", base + "_data"), data)
        print(f"{base}: Reshape allowzero 1->0: {allowzero}, rank-5 MatMul -> rank 3: {len(targets)}")
