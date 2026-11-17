"""Compressed and unpacked size of the DLLs each runtime route needs, read from the archives'
zip central directories (wheels, nupkgs and the ORT zip are all zip files). An app can fetch
just these members with HTTP range requests, or download the whole archive.
Usage: <python> scripts/archive_members.py"""
import os
import sys
import zipfile

sys.dont_write_bytecode = True
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtimes", "archives")
ROUTES = {
    "cuda12": {
        "ort-cuda12-1.32.0.zip": ["onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll"],
        "nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl": ["cudart64_12.dll"],
        "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl": ["cublas64_12.dll", "cublasLt64_12.dll"],
        "nvidia_cudnn_cu12-9.28.1.4-py3-none-win_amd64.whl": ["cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll", "cudnn_engines_precompiled64_9.dll", "cudnn_heuristic64_9.dll", "cudnn_engines_runtime_compiled64_9.dll", "cudnn_engines_tensor_ir64_9.dll"],
    },
    "cuda13": {
        "onnxruntime_gpu-1.32.0-cp312-cp312-win_amd64.whl": ["onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll"],
        "nvidia_cuda_runtime-13.5.37-py3-none-win_amd64.whl": ["cudart64_13.dll"],
        "nvidia_cublas-13.10.0.20-py3-none-win_amd64.whl": ["cublas64_13.dll", "cublasLt64_13.dll"],
        "nvidia_cudnn_cu13-9.28.1.4-py3-none-win_amd64.whl": ["cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll", "cudnn_engines_precompiled64_9.dll", "cudnn_heuristic64_9.dll", "cudnn_engines_runtime_compiled64_9.dll", "cudnn_engines_tensor_ir64_9.dll"],
    },
    "directml": {
        "ort-directml-1.24.4.nupkg": ["runtimes/win-x64/native/onnxruntime.dll"],
        "directml-1.15.4.nupkg": ["bin/x64-win/DirectML.dll"],
    },
}
for route, archives in ROUTES.items():
    whole = packed = unpacked = 0
    print(f"== {route}")
    for archive, members in archives.items():
        path = os.path.join(A, archive)
        whole += os.path.getsize(path)
        with zipfile.ZipFile(path) as z:
            for member in members:
                info = next(i for i in z.infolist() if i.filename.endswith("/" + member) or i.filename == member)
                packed += info.compress_size
                unpacked += info.file_size
                print(f"  {archive} :: {info.filename}  packed {info.compress_size/1e6:.1f} MB  unpacked {info.file_size/1e6:.1f} MB")
    print(f"  TOTAL whole archives {whole/1e6:.1f} MB | needed members packed {packed/1e6:.1f} MB | on disk {unpacked/1e6:.1f} MB")
