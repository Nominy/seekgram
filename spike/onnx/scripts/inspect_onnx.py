"""Prints inputs, outputs, opsets and operator histogram of ONNX graphs (no weights loaded).
Usage: uv run --no-project --with onnx python scripts/inspect_onnx.py models/embeddinggemma-2-ONNX/onnx/*_fp16.onnx"""
import collections
import sys

import onnx


def shape(value):
    tensor = value.type.tensor_type
    dims = [d.dim_param or d.dim_value for d in tensor.shape.dim]
    return f"{onnx.TensorProto.DataType.Name(tensor.elem_type)}{dims}"


for path in sys.argv[1:]:
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    print(f"== {path}\n  ir={model.ir_version} producer={model.producer_name} {model.producer_version} opsets={[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    initializers = {i.name for i in graph.initializer}
    for value in graph.input:
        if value.name not in initializers:
            print(f"  in  {value.name}: {shape(value)}")
    for value in graph.output:
        print(f"  out {value.name}: {shape(value)}")

    def walk(g, counter):
        for node in g.node:
            counter[(node.domain or "") + ("::" if node.domain else "") + node.op_type] += 1
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    walk(attr.g, counter)
                for sub in attr.graphs:
                    walk(sub, counter)
        return counter

    ops = walk(graph, collections.Counter())
    print(f"  nodes={sum(ops.values())} initializers={len(graph.initializer)}")
    print("  ops: " + ", ".join(f"{k}x{v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])))
