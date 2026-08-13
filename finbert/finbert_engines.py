import os
import numpy as np


def logitsToPredictions(logits: np.ndarray) -> list[dict]:
    labelNames = ["bearish", "neutral", "bullish"]
    maxLogits = np.max(logits, axis=-1, keepdims=True)
    expLogits = np.exp(logits - maxLogits)
    probs = expLogits / np.sum(expLogits, axis=-1, keepdims=True)

    predictions = []
    for i in range(logits.shape[0]):
        maxIdx = int(np.argmax(probs[i]))
        maxScore = float(probs[i][maxIdx])
        predictions.append({
            "label": labelNames[maxIdx],
            "score": maxScore
        })
    return predictions


# TRT inference helper, lazily loads modules to save VRAM and avoid import errors if env doesn't support TRT
class TrtInferenceEngine:
    def __init__(self, enginePath: str, tokenizer=None):
        import tensorrt as trt
        import torch

        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(enginePath, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {enginePath}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("tabularisai/ModernFinBERT", cache_dir="data/models", model_max_length=1024)
        self.tokenizer = tokenizer

    def _inferRaw(self, inputIds: np.ndarray, attentionMask: np.ndarray) -> np.ndarray:
        import torch
        batchSize, seqLen = inputIds.shape
        self.context.set_input_shape("input_ids", (batchSize, seqLen))
        self.context.set_input_shape("attention_mask", (batchSize, seqLen))

        dInputIds = torch.from_numpy(inputIds.astype(np.int64)).cuda().contiguous()
        dAttMask = torch.from_numpy(attentionMask.astype(np.int64)).cuda().contiguous()

        numClasses = self.engine.get_tensor_shape("logits")[-1]
        if numClasses < 0:
            numClasses = 3
        dOutput = torch.empty((batchSize, numClasses), dtype=torch.float32, device="cuda").contiguous()

        self.context.set_tensor_address("input_ids", dInputIds.data_ptr())
        self.context.set_tensor_address("attention_mask", dAttMask.data_ptr())
        self.context.set_tensor_address("logits", dOutput.data_ptr())

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return dOutput.cpu().numpy()

    def infer(self, inputs, attentionMask: np.ndarray = None, batchSize: int = 8) -> np.ndarray:
        if isinstance(inputs, np.ndarray):
            return self._inferRaw(inputs, attentionMask)

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if not inputs:
            return np.empty((0, 3), dtype=np.float32)

        allLogits = []
        numSamples = len(inputs)
        for startIdx in range(0, numSamples, batchSize):
            endIdx = min(startIdx + batchSize, numSamples)
            batchTexts = inputs[startIdx:endIdx]
            encoded = self.tokenizer(
                batchTexts,
                padding="longest",
                truncation=True,
                max_length=1024,
                return_tensors="np"
            )
            batchInputIds = encoded["input_ids"]
            batchAttMask = encoded["attention_mask"]
            batchLogits = self._inferRaw(batchInputIds, batchAttMask)
            allLogits.append(batchLogits)

        return np.concatenate(allLogits, axis=0)


# ONNX inference engine, lazily loads modules to save VRAM and avoid import errors if env doesn't support ONNX
class OnnxInferenceEngine:
    def __init__(self, onnxPath: str, tokenizer=None):
        import onnxruntime as ort

        availableProviders = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in availableProviders:
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(onnxPath, providers=providers)
        self.provider = self.session.get_providers()[0]

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("tabularisai/ModernFinBERT", cache_dir="data/models", model_max_length=1024)
        self.tokenizer = tokenizer


    def _inferRaw(self, inputIds: np.ndarray, attentionMask: np.ndarray) -> np.ndarray:
        inputs = {
            "input_ids": inputIds.astype(np.int64),
            "attention_mask": attentionMask.astype(np.int64)
        }
        outputs = self.session.run(["logits"], inputs)
        return outputs[0]


    def infer(self, inputs, attentionMask: np.ndarray = None, batchSize: int = 8) -> np.ndarray:
        if isinstance(inputs, np.ndarray):
            return self._inferRaw(inputs, attentionMask)

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if not inputs:
            return np.empty((0, 3), dtype=np.float32)

        allLogits = []
        numSamples = len(inputs)
        for startIdx in range(0, numSamples, batchSize):
            endIdx = min(startIdx + batchSize, numSamples)
            batchTexts = inputs[startIdx:endIdx]
            encoded = self.tokenizer(
                batchTexts,
                padding="longest",
                truncation=True,
                max_length=1024,
                return_tensors="np"
            )
            batchInputIds = encoded["input_ids"]
            batchAttMask = encoded["attention_mask"]
            batchLogits = self._inferRaw(batchInputIds, batchAttMask)
            allLogits.append(batchLogits)

        return np.concatenate(allLogits, axis=0)


def isTensorRtSupported() -> bool:
    try:
        import tensorrt as trt
        import torch
        if not torch.cuda.is_available():
            return False

        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)

        if builder is None:
            return False
        return True
        
    except Exception:
        return False


def getBestInferenceEngine():
    trtPath = "finbert/models/ModernFinBERT_fp8.engine"
    onnxPath = "finbert/models/ModernFinBERT_fp32.onnx"

    # 1. Check if TensorRT is supported and engine file exists
    if isTensorRtSupported():
        if os.path.exists(trtPath):
            try:
                engine = TrtInferenceEngine(trtPath)
                _ = engine.infer(["Financial market sentiment analysis initialisation warmup."])
                print(f"[FinBERT Engine] Successfully loaded TensorRT model from {trtPath}")
                return engine, "TensorRT", trtPath
            except Exception as e:
                print(f"[FinBERT Engine] TensorRT load failed: {e}. Defaulting to ONNX...")

    # 2. Default to ONNX Runtime if TensorRT is not available or failed
    if os.path.exists(onnxPath):
        try:
            engine = OnnxInferenceEngine(onnxPath)
            _ = engine.infer(["Financial market sentiment analysis initialisation warmup."])
            print(f"[FinBERT Engine] Successfully loaded ONNX model from {onnxPath} ({engine.provider})")
            return engine, "ONNX", onnxPath
        except Exception as e:
            print(f"[FinBERT Engine] ONNX load failed: {e}")

    print("[FinBERT Engine] Neither TensorRT nor ONNX model could be loaded.")
    return None, None, None
