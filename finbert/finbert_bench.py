import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import numpy as np

import tensorrt as trt
import torch
import onnxruntime as ort
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scipy.spatial.distance import cosine
from scipy.special import softmax, kl_div

from finbert.finbert_engines import TrtInferenceEngine, OnnxInferenceEngine
from finbert.finbert_quantise import loadFinancialPhraseBank


def runPytorchModelFp32(model, tokeniser, texts, batchSize, benchBatchSizes, timedRuns):
    # Validation inference over the full dataset
    print(f"Validating FP32 PyTorch model...")
    allLogits = []
    for start in range(0, len(texts), batchSize):
        end = min(start + batchSize, len(texts))
        encoded = tokeniser(texts[start:end], padding="longest", truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**encoded).logits.cpu().numpy()
        allLogits.append(logits)
    logits = np.concatenate(allLogits, axis=0)

    # Latency benchmark across batch sizes (CUDA-event timing)
    print(f"Benchmarking FP32 PyTorch model...")
    benchResults = []
    for bs in benchBatchSizes:
        benchTexts = loadFinancialPhraseBank(256, returnRightSide=True)[0]
        random.shuffle(benchTexts)
        benchTexts = benchTexts[:bs]
        encoded = tokeniser(benchTexts, padding="longest", truncation=True, return_tensors="pt")
        inputIds = encoded["input_ids"].cuda()
        attentionMask = encoded["attention_mask"].cuda()

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)
        latencies = []
        with torch.no_grad():
            for _ in range(timedRuns):
                startEvent.record()
                model(input_ids=inputIds, attention_mask=attentionMask)
                endEvent.record()
                torch.cuda.synchronize()
                latencies.append(startEvent.elapsed_time(endEvent))

        meanMs = float(np.mean(latencies))
        benchResults.append((bs, meanMs, (bs / meanMs) * 1000))

    return logits, benchResults


def runTensorRtModelFp8(engine, tokeniser, texts, batchSize, benchBatchSizes, timedRuns):
    # Validation inference over the full dataset
    print(f"Validating FP8 TensorRT engine...")
    logits = engine.infer(texts, batchSize=batchSize)

    # Latency benchmark across batch sizes
    print(f"Benchmarking FP8 TensorRT engine...")
    benchResults = []
    for bs in benchBatchSizes:
        benchTexts = loadFinancialPhraseBank(256, returnRightSide=True)[0]
        random.shuffle(benchTexts)
        benchTexts = benchTexts[:bs]
        encoded = tokeniser(benchTexts, padding="longest", truncation=True, return_tensors="pt")
        inputIds = encoded["input_ids"].numpy()
        attentionMask = encoded["attention_mask"].numpy()

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)
        latencies = []
        for _ in range(timedRuns):
            startEvent.record()
            engine.infer(inputIds, attentionMask)
            endEvent.record()
            torch.cuda.synchronize()
            latencies.append(startEvent.elapsed_time(endEvent))

        meanMs = float(np.mean(latencies))
        benchResults.append((bs, meanMs, (bs / meanMs) * 1000))

    return logits, benchResults


def runOnnxModelFp32(engine, tokeniser, texts, batchSize, benchBatchSizes, timedRuns):
    # Validation inference over the full dataset
    print(f"Validating FP32 ONNX Runtime model...")
    logits = engine.infer(texts, batchSize=batchSize)

    # Latency benchmark across batch sizes
    print(f"Benchmarking FP32 ONNX Runtime model...")
    benchResults = []
    for bs in benchBatchSizes:
        benchTexts = loadFinancialPhraseBank(256, returnRightSide=True)[0]
        random.shuffle(benchTexts)
        benchTexts = benchTexts[:bs]
        encoded = tokeniser(benchTexts, padding="longest", truncation=True, return_tensors="pt")
        inputIds = encoded["input_ids"].numpy()
        attentionMask = encoded["attention_mask"].numpy()

        latencies = []
        for _ in range(timedRuns):
            start = time.perf_counter()
            engine.infer(inputIds, attentionMask)
            latencies.append((time.perf_counter() - start) * 1000)

        meanMs = float(np.mean(latencies))
        benchResults.append((bs, meanMs, (bs / meanMs) * 1000))

    return logits, benchResults


def computeMetrics(baseModelLogits, testModelLogits):
    diff = np.abs(baseModelLogits - testModelLogits)
    mae = float(np.mean(diff))
    maxAe = float(np.max(diff))

    # Per-sample cosine similarity between logits
    cosineSims = []
    for i in range(baseModelLogits.shape[0]):
        sim = 1.0 - cosine(baseModelLogits[i], testModelLogits[i])
        cosineSims.append(sim)
    meanCosineSim = float(np.mean(cosineSims))

    # Find where the models agree
    fp32Preds = np.argmax(baseModelLogits, axis=1)
    quantPreds = np.argmax(testModelLogits, axis=1)
    agreeRate = float(np.mean(fp32Preds == quantPreds))

    # KL divergence between softmaxed logits
    fp32Probs = softmax(baseModelLogits, axis=1)
    quantProbs = softmax(testModelLogits, axis=1)
    klDivs = []
    for i in range(fp32Probs.shape[0]):
        klDiv = float(np.sum(kl_div(fp32Probs[i], quantProbs[i])))
        klDivs.append(klDiv)
    meanKlDiv = float(np.mean(klDivs))

    return {
        "mae": mae,
        "maxAe": maxAe,
        "meanCosineSim": meanCosineSim,
        "agreeRate": agreeRate,
        "meanKlDiv": meanKlDiv,
        "fp32Preds": fp32Preds,
        "quantPreds": quantPreds
    }



def main():
    modelId = "tabularisai/ModernFinBERT"
    enginePath = "finbert/models/ModernFinBERT_fp8.engine"
    fp32OnnxPath = "finbert/models/ModernFinBERT_fp32.onnx"

    # ModernFinBERT label indices -> class names (0=Bearish, 1=Neutral, 2=Bullish)
    labelNames = ["bearish", "neutral", "bullish"]

    # Validation set: 75Agree sentences NOT in the AllAgree FP8 calibration set
    texts, trueLabels = loadFinancialPhraseBank(256, returnRightSide=True)

    validateBatchSize = 8
    seqLen = 512         # typical sentence length for financial text
    timedRuns = 100
    testBatchSizes = [1, 2, 4, 8, 16, 32]

    print("Loading FP32 PyTorch model...")
    tokeniser = AutoTokenizer.from_pretrained(modelId, cache_dir="data/models", model_max_length=1024)
    model = AutoModelForSequenceClassification.from_pretrained(modelId).cuda().eval()

    print("Loading TensorRT FP8 engine...")
    trtEngine = TrtInferenceEngine(enginePath)

    print("Loading FP32 ONNX Runtime model...")
    onnxEngine = OnnxInferenceEngine(fp32OnnxPath)

    gpuName = torch.cuda.get_device_name(0)
    import onnxruntime as ort
    try:
        import tensorrt as trt
        trtVersionStr = trt.__version__
    except ImportError:
        trtVersionStr = "Not installed"

    print(f"\nGPU: {gpuName} | CUDA: {torch.version.cuda} | PyTorch: {torch.__version__} | "
          f"TensorRT: {trtVersionStr} | ONNX Runtime: {ort.__version__} ({onnxEngine.provider})")


    print(f"\nRunning validation ({len(texts)} sentences) and benchmark...")
    fp32Logits, fp32Bench = runPytorchModelFp32(model, tokeniser, texts, validateBatchSize, testBatchSizes, timedRuns)
    onnxLogits, onnxBench = runOnnxModelFp32(onnxEngine, tokeniser, texts, validateBatchSize, testBatchSizes, timedRuns)
    trtLogits, trtBench = runTensorRtModelFp8(trtEngine, tokeniser, texts, validateBatchSize, testBatchSizes, timedRuns)

    fp8Metrics = computeMetrics(fp32Logits, trtLogits)
    fp32OnnxMetrics = computeMetrics(fp32Logits, onnxLogits)


    print(f"\n{'=' * 100}")
    print("  MODEL VALIDATION")
    print(f"{'=' * 100}")
    print("\nFP32 ONNX vs FP32 PyTorch:")
    print(f"    Mean Absolute Error (MAE): {fp32OnnxMetrics['mae']:.6f}")
    print(f"   Max Absolute Error (MaxAE): {fp32OnnxMetrics['maxAe']:.6f}")
    print(f"       Mean Cosine Similarity: {fp32OnnxMetrics['meanCosineSim']:.6f}")
    print(f"         Prediction Agreement: {fp32OnnxMetrics['agreeRate'] * 100:.2f}%")
    print(f"           Mean KL Divergence: {fp32OnnxMetrics['meanKlDiv']:.6f}")
    print("\nFP8 TensorRT vs FP32 PyTorch:")
    print(f"    Mean Absolute Error (MAE): {fp8Metrics['mae']:.6f}")
    print(f"   Max Absolute Error (MaxAE): {fp8Metrics['maxAe']:.6f}")
    print(f"       Mean Cosine Similarity: {fp8Metrics['meanCosineSim']:.6f}")
    print(f"         Prediction Agreement: {fp8Metrics['agreeRate'] * 100:.2f}%")
    print(f"           Mean KL Divergence: {fp8Metrics['meanKlDiv']:.6f}")

    print(f"\n{'=' * 100}")
    print(f"  MODEL BENCHMARKING (timed runs={timedRuns}) ")
    print(f"{'=' * 100}")

    print(f"  {'Batch':<7}{'FP32 (ms/batch)':<20}{'TRT (ms/batch)':<20}{'ONNX (ms/batch)':<20}"
          f"{'TRT (% uplift)':<20}{'ONNX (% uplift)':<20}{'FP32 (samples/s)':<20}{'TRT(samples/s)':<20}{'ONNX(samples/s)'}")

    for i, bs in enumerate(testBatchSizes):
        fp32Ms, fp32Tp = fp32Bench[i][1], fp32Bench[i][2]
        trtMs, trtTp = trtBench[i][1], trtBench[i][2]
        onnxMs, onnxTp = onnxBench[i][1], onnxBench[i][2]
        print(f"  {bs:<7}{fp32Ms:<20.2f}{trtMs:<20.2f}{onnxMs:<20.2f}"
              f"{fp32Ms / trtMs*100:<20.2f}{fp32Ms / onnxMs*100:<20.2f}"
              f"{fp32Tp:<20.1f}{trtTp:<20.1f}{onnxTp:.1f}")

    print("\n")

if __name__ == "__main__":
    main()
