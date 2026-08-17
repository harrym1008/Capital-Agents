import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import multiprocessing as mp


def getGpuMemoryMb() -> float:
    import torch
    freeBytes, totalBytes = torch.cuda.mem_get_info()
    usedMb = (totalBytes - freeBytes) / (1024 * 1024)
    return float(usedMb)


def measurePytorchFp32Worker(resultQueue, modelId: str, batchSizes: list[int], dummyText: str):
    import torch
    from finbert.finbert_engines import PytorchCudaInferenceEngine

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseMemMb = getGpuMemoryMb()

    print("[FP32 PyTorch] Loading engine...")
    engine = PytorchCudaInferenceEngine(modelId=modelId, maxSeqLen=768)
    torch.cuda.synchronize()

    loadedMemMb = getGpuMemoryMb()
    modelWeightsMb = loadedMemMb - baseMemMb

    batchResults = {}
    for bs in batchSizes:
        texts = [dummyText] * bs

        # Warmup
        _ = engine.infer(texts)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        for _ in range(20):
            _ = engine.infer(texts)
        torch.cuda.synchronize()

        peakMemMb = getGpuMemoryMb()
        torchPeakAllocatedMb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        batchResults[bs] = {
            "inferMemMb": peakMemMb,
            "netInferDeltaMb": peakMemMb - baseMemMb,
            "torchPeakAllocatedMb": torchPeakAllocatedMb
        }

    resultQueue.put({
        "name": "FP32 PyTorch (SDPA)",
        "baseMemMb": baseMemMb,
        "loadedMemMb": loadedMemMb,
        "modelWeightsMb": modelWeightsMb,
        "batchResults": batchResults
    })


def measureOnnxFp32Worker(resultQueue, onnxPath: str, modelId: str, batchSizes: list[int], dummyText: str):
    import torch
    from finbert.finbert_engines import OnnxCudaInferenceEngine

    torch.cuda.empty_cache()
    baseMemMb = getGpuMemoryMb()

    print("[FP32 ONNX Runtime] Loading engine...")
    engine = OnnxCudaInferenceEngine(onnxPath, maxSeqLen=768)
    torch.cuda.synchronize()

    loadedMemMb = getGpuMemoryMb()
    modelWeightsMb = loadedMemMb - baseMemMb

    batchResults = {}
    for bs in batchSizes:
        texts = [dummyText] * bs

        # Warmup
        _ = engine.infer(texts)
        torch.cuda.synchronize()

        for _ in range(20):
            _ = engine.infer(texts)
        torch.cuda.synchronize()

        peakMemMb = getGpuMemoryMb()
        batchResults[bs] = {
            "inferMemMb": peakMemMb,
            "netInferDeltaMb": peakMemMb - baseMemMb,
            "torchPeakAllocatedMb": peakMemMb - loadedMemMb
        }

    resultQueue.put({
        "name": f"FP32 ONNX Runtime ({engine.provider})",
        "baseMemMb": baseMemMb,
        "loadedMemMb": loadedMemMb,
        "modelWeightsMb": modelWeightsMb,
        "batchResults": batchResults
    })


def measureTrtFp8Worker(resultQueue, enginePath: str, modelId: str, batchSizes: list[int], dummyText: str):
    import torch
    from finbert.finbert_engines import TrtCudaInferenceEngine

    torch.cuda.empty_cache()
    baseMemMb = getGpuMemoryMb()

    print("[FP8 TensorRT] Loading engine...")
    engine = TrtCudaInferenceEngine(enginePath, maxSeqLen=768)
    torch.cuda.synchronize()

    loadedMemMb = getGpuMemoryMb()
    modelWeightsMb = loadedMemMb - baseMemMb

    batchResults = {}
    for bs in batchSizes:
        texts = [dummyText] * bs

        try:
            # Warmup
            _ = engine.infer(texts)
            torch.cuda.synchronize()

            for _ in range(20):
                _ = engine.infer(texts)
            torch.cuda.synchronize()

            peakMemMb = getGpuMemoryMb()
            batchResults[bs] = {
                "inferMemMb": peakMemMb,
                "netInferDeltaMb": peakMemMb - baseMemMb,
                "torchPeakAllocatedMb": peakMemMb - loadedMemMb,
                "error": None
            }
        except Exception as e:
            batchResults[bs] = {
                "inferMemMb": 0.0,
                "netInferDeltaMb": 0.0,
                "torchPeakAllocatedMb": 0.0,
                "error": str(e)
            }

    resultQueue.put({
        "name": "FP8 TensorRT",
        "baseMemMb": baseMemMb,
        "loadedMemMb": loadedMemMb,
        "modelWeightsMb": modelWeightsMb,
        "batchResults": batchResults
    })


def runWorkerInIsolatedProcess(targetWorker, *args) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=targetWorker, args=(queue, *args))
    process.start()
    process.join()

    if not queue.empty():
        return queue.get()
    else:
        return {"name": "Failed", "error": "Subprocess exited without returning data."}


def printVramReport(resultsList: list[dict], batchSizes: list[int], seqLen: int):
    print("\n" + "=" * 110)
    print(f"  FINBERT VRAM USAGE BENCHMARK (Sequence Length = {seqLen} tokens)")
    print("  (Each engine was measured in a clean, isolated process)")
    print("=" * 110)

    # 1. Model Static Weights & Initialization Table
    print(f"\n1. MODEL INITIALIZATION & WEIGHTS FOOTPRINT:")
    print(f"  {'Engine / Format':<32}{'CUDA Baseline (MB)':<22}{'Loaded VRAM (MB)':<22}{'Model Weights / Init Delta (MB)'}")
    print("  " + "-" * 105)
    for res in resultsList:
        name = res.get("name", "Unknown")
        base = res.get("baseMemMb", 0.0)
        loaded = res.get("loadedMemMb", 0.0)
        delta = res.get("modelWeightsMb", 0.0)
        print(f"  {name:<32}{base:<22.1f}{loaded:<22.1f}{delta:.1f} MB")

    # 2. Peak Active VRAM During Inference Across Batch Sizes
    print(f"\n2. TOTAL ACTIVE VRAM DURING INFERENCE (Model + Activations + CUDA Driver):")
    header = f"  {'Engine / Format':<32}" + "".join([f"BS={bs:<2} Total (MB)    " for bs in batchSizes])
    print(header)
    print("  " + "-" * 105)
    for res in resultsList:
        name = res.get("name", "Unknown")
        row = f"  {name:<32}"
        bResults = res.get("batchResults", {})
        for bs in batchSizes:
            bInfo = bResults.get(bs, {})
            if bInfo.get("error"):
                row += f"{'ERR (Profile)':<18}"
            else:
                totalMb = bInfo.get("inferMemMb", 0.0)
                row += f"{totalMb:<18.1f}"
        print(row)

    # 3. Net VRAM Footprint Dedicated to Model & Inference (Total VRAM - Empty CUDA Baseline)
    print(f"\n3. NET MODEL + INFERENCE FOOTPRINT (Excluding CUDA Base Context):")
    headerNet = f"  {'Engine / Format':<32}" + "".join([f"BS={bs:<2} Net (MB)      " for bs in batchSizes])
    print(headerNet)
    print("  " + "-" * 105)
    for res in resultsList:
        name = res.get("name", "Unknown")
        row = f"  {name:<32}"
        bResults = res.get("batchResults", {})
        for bs in batchSizes:
            bInfo = bResults.get(bs, {})
            if bInfo.get("error"):
                row += f"{'ERR (Profile)':<18}"
            else:
                netMb = bInfo.get("netInferDeltaMb", 0.0)
                row += f"{netMb:<18.1f}"
        print(row)

    print("\n" + "=" * 110 + "\n")


def main():
    modelId = "tabularisai/ModernFinBERT"
    onnxPath = "finbert/models/ModernFinBERT_fp32.onnx"
    enginePath = "finbert/models/ModernFinBERT_fp8.engine"

    batchSizes = [1, 4, 8, 16]
    seqLen = 768

    # A realistic sample headline replicated to fill typical token sequences
    dummyText = "Operating profit for the twelve-month financial period rose by twenty-four percent year-over-year beating market consensus."

    results = []

    # Run PyTorch FP32 in fresh isolated process
    print("\n--- Starting FP32 PyTorch VRAM Measurement ---")
    ptRes = runWorkerInIsolatedProcess(measurePytorchFp32Worker, modelId, batchSizes, dummyText)
    results.append(ptRes)

    # Run ONNX FP32 in fresh isolated process
    print("\n--- Starting FP32 ONNX Runtime VRAM Measurement ---")
    onnxRes = runWorkerInIsolatedProcess(measureOnnxFp32Worker, onnxPath, modelId, batchSizes, dummyText)
    results.append(onnxRes)

    # Run TRT FP8 in fresh isolated process
    print("\n--- Starting FP8 TensorRT VRAM Measurement ---")
    trtRes = runWorkerInIsolatedProcess(measureTrtFp8Worker, enginePath, modelId, batchSizes, dummyText)
    results.append(trtRes)

    printVramReport(results, batchSizes, seqLen)


if __name__ == "__main__":
    main()
