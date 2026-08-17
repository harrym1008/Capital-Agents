import os
import threading
from abc import ABC, abstractmethod
from typing import Optional
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


# Base class for all FinBERT inference engines
class BaseInferenceEngine(ABC):
    def __init__(self, modelPathOrId: str, optimalBatchSize: int = 16, maxSeqLen: int = 768):
        self.modelPathOrId = modelPathOrId
        self.optimalBatchSize = optimalBatchSize
        self.maxSeqLen = maxSeqLen
        self.engineType = self.__class__.__name__
        self.lock = threading.RLock()

        self.importModules()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "tabularisai/ModernFinBERT",
            cache_dir="data/models",
            model_max_length=self.maxSeqLen
        )

    @abstractmethod
    def importModules(self) -> None:
        pass

    @abstractmethod
    def _inferRaw(self, inputIds: np.ndarray, attentionMask: np.ndarray) -> np.ndarray:
        pass

    def infer(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 3), dtype=np.float32)

        with self.lock:
            allLogits = []
            numSamples = len(texts)
            for startIdx in range(0, numSamples, self.optimalBatchSize):
                endIdx = min(startIdx + self.optimalBatchSize, numSamples)
                batchTexts = texts[startIdx:endIdx]
                encoded = self.tokenizer(
                    batchTexts,
                    padding="longest",
                    truncation=True,
                    max_length=self.maxSeqLen,
                    return_tensors="np"
                )
                batchInputIds = encoded["input_ids"]
                batchAttMask = encoded["attention_mask"]
                batchLogits = self._inferRaw(batchInputIds, batchAttMask)
                allLogits.append(batchLogits)

            return np.concatenate(allLogits, axis=0)


# TensorRT FP8 Inference Engine (requires CUDA and TensorRT)
class TrtCudaInferenceEngine(BaseInferenceEngine):
    def __init__(self, enginePath: str = "finbert/models/ModernFinBERT_fp8.engine", optimalBatchSize: int = 16, maxSeqLen: int = 768):
        self.enginePath = enginePath
        super().__init__(modelPathOrId=enginePath, optimalBatchSize=optimalBatchSize, maxSeqLen=maxSeqLen)

        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.trt.init_libnvinfer_plugins(self.logger, "")

        with open(self.enginePath, "rb") as f:
            runtime = self.trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.enginePath}")

        self.context = self.engine.create_execution_context()
        self.stream = self.torch.cuda.Stream()

    def importModules(self) -> None:
        import tensorrt as trt
        import torch
        self.trt = trt
        self.torch = torch

    def _inferRaw(self, inputIds: np.ndarray, attentionMask: np.ndarray) -> np.ndarray:
        batchSize, seqLen = inputIds.shape
        self.context.set_input_shape("input_ids", (batchSize, seqLen))
        self.context.set_input_shape("attention_mask", (batchSize, seqLen))

        dInputIds = self.torch.from_numpy(inputIds.astype(np.int64)).cuda().contiguous()
        dAttMask = self.torch.from_numpy(attentionMask.astype(np.int64)).cuda().contiguous()

        numClasses = self.engine.get_tensor_shape("logits")[-1]
        if numClasses < 0:
            numClasses = 3
        dOutput = self.torch.empty((batchSize, numClasses), dtype=self.torch.float32, device="cuda").contiguous()

        self.context.set_tensor_address("input_ids", dInputIds.data_ptr())
        self.context.set_tensor_address("attention_mask", dAttMask.data_ptr())
        self.context.set_tensor_address("logits", dOutput.data_ptr())

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return dOutput.cpu().numpy()


# ONNX Runtime CUDA FP32 Inference Engine (requires CUDA and ONNX Runtime)
class OnnxCudaInferenceEngine(BaseInferenceEngine):
    """ONNX Runtime FP32 Inference Engine running on CUDAExecutionProvider."""

    def __init__(self, onnxPath: str = "finbert/models/ModernFinBERT_fp32.onnx", optimalBatchSize: int = 8, maxSeqLen: int = 768):
        self.onnxPath = onnxPath
        super().__init__(modelPathOrId=onnxPath, optimalBatchSize=optimalBatchSize, maxSeqLen=maxSeqLen)

        cudaOptions = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": "1",
        }
        self.session = self.ort.InferenceSession(self.onnxPath, providers=[("CUDAExecutionProvider", cudaOptions)])
        self.provider = self.session.get_providers()[0]

    def importModules(self) -> None:
        import onnxruntime as ort
        import torch
        self.ort = ort
        self.torch = torch

    def _inferRaw(self, inputIds: np.ndarray, attentionMask: np.ndarray) -> np.ndarray:
        inputs = {
            "input_ids": inputIds.astype(np.int64),
            "attention_mask": attentionMask.astype(np.int64)
        }
        outputs = self.session.run(["logits"], inputs)
        return outputs[0]


# PyTorch CUDA FP32 Inference Engine (requires CUDA and PyTorch)
class PytorchCudaInferenceEngine(BaseInferenceEngine):
    def __init__(self, modelId: str = "tabularisai/ModernFinBERT", optimalBatchSize: int = 16, maxSeqLen: int = 768):
        self.modelId = modelId
        super().__init__(modelPathOrId=modelId, optimalBatchSize=optimalBatchSize, maxSeqLen=maxSeqLen)

        self.model = self.AutoModelForSequenceClassification.from_pretrained(
            self.modelId,
            attn_implementation="sdpa"
        ).cuda().eval()

    def importModules(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification
        self.torch = torch
        self.AutoModelForSequenceClassification = AutoModelForSequenceClassification

    def _inferRaw(self, inputIds: np.ndarray, attentionMask: np.ndarray) -> np.ndarray:
        tInputIds = self.torch.from_numpy(inputIds.astype(np.int64)).cuda()
        tAttMask = self.torch.from_numpy(attentionMask.astype(np.int64)).cuda()

        with self.torch.no_grad():
            outputs = self.model(input_ids=tInputIds, attention_mask=tAttMask)
            logits = outputs.logits.cpu().numpy()
        return logits


# Onnx Runtime CPU FP32 Inference Engine (requires just ONNX runtime)
class OnnxCpuInferenceEngine(BaseInferenceEngine):
    """ONNX Runtime FP32 Inference Engine running on CPUExecutionProvider."""

    def __init__(self, onnxPath: str = "finbert/models/ModernFinBERT_fp32.onnx", optimalBatchSize: int = 4, maxSeqLen: int = 768):
        self.onnxPath = onnxPath
        super().__init__(modelPathOrId=onnxPath, optimalBatchSize=optimalBatchSize, maxSeqLen=maxSeqLen)

        self.session = self.ort.InferenceSession(self.onnxPath, providers=["CPUExecutionProvider"])
        self.provider = "CPUExecutionProvider"

    def importModules(self) -> None:
        import onnxruntime as ort
        self.ort = ort

    def _inferRaw(self, inputIds: np.ndarray, attentionMask: np.ndarray) -> np.ndarray:
        inputs = {
            "input_ids": inputIds.astype(np.int64),
            "attention_mask": attentionMask.astype(np.int64)
        }
        outputs = self.session.run(["logits"], inputs)
        return outputs[0]


# PyTorch CPU FP32 Inference Engine (requires just PyTorch)
class PytorchCpuInferenceEngine(BaseInferenceEngine):
    """Native PyTorch FP32 Inference Engine running on CPU."""

    def __init__(self, modelId: str = "tabularisai/ModernFinBERT", optimalBatchSize: int = 4, maxSeqLen: int = 768):
        self.modelId = modelId
        super().__init__(modelPathOrId=modelId, optimalBatchSize=optimalBatchSize, maxSeqLen=maxSeqLen)

        self.model = self.AutoModelForSequenceClassification.from_pretrained(
            self.modelId,
            attn_implementation="eager"
        ).to("cpu").eval()

    def importModules(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification
        self.torch = torch
        self.AutoModelForSequenceClassification = AutoModelForSequenceClassification

    def _inferRaw(self, inputIds: np.ndarray, attentionMask: np.ndarray) -> np.ndarray:
        tInputIds = self.torch.from_numpy(inputIds.astype(np.int64)).to("cpu")
        tAttMask = self.torch.from_numpy(attentionMask.astype(np.int64)).to("cpu")

        with self.torch.no_grad():
            outputs = self.model(input_ids=tInputIds, attention_mask=tAttMask)
            logits = outputs.logits.numpy()
        return logits



# Gets the best available inference engine for FinBERT, ideally the best performing (trt) then falling back to the next best option if not available
def getBestInferenceEngine() -> BaseInferenceEngine | None:
    trtPath = "finbert/models/ModernFinBERT_fp8.engine"
    onnxPath = "finbert/models/ModernFinBERT_fp32.onnx"
    modelId = "tabularisai/ModernFinBERT"

    import torch
    hasCuda = torch.cuda.is_available()

    inferenceTest = ["Financial market sentiment analysis initialisation warmup."]

    # 1. FP8 TensorRT on CUDA (Batch Size = 16)
    if hasCuda and isTensorRtSupported():
        if os.path.exists(trtPath):
            try:
                engine = TrtCudaInferenceEngine(trtPath)
                _ = engine.infer(inferenceTest)
                print(f"[FinBERT Engine] Successfully loaded TensorRT FP8 model from {trtPath} (Batch Size: {engine.optimalBatchSize})")
                return engine
            except Exception as e:
                print(f"[FinBERT Engine] TensorRT load failed: {e}. Falling back to next engine...")

    # 2. FP32 ONNX on CUDA (Batch Size = 8 to constrain VRAM)
    if hasCuda and os.path.exists(onnxPath):
        try:
            engine = OnnxCudaInferenceEngine(onnxPath)
            _ = engine.infer(inferenceTest)
            print(f"[FinBERT Engine] Successfully loaded ONNX CUDA model from {onnxPath} (Batch Size: {engine.optimalBatchSize})")
            return engine
        except Exception as e:
            print(f"[FinBERT Engine] ONNX CUDA load failed: {e}. Falling back to PyTorch CUDA...")

    # 3. FP32 PyTorch on CUDA (Batch Size = 16)
    if hasCuda:
        try:
            engine = PytorchCudaInferenceEngine(modelId=modelId)
            _ = engine.infer(inferenceTest)
            print(f"[FinBERT Engine] Successfully loaded PyTorch FP32 CUDA model (Batch Size: {engine.optimalBatchSize})")
            return engine
        except Exception as e:
            print(f"[FinBERT Engine] PyTorch CUDA load failed: {e}. Falling back to CPU engines...")

    # 4. FP32 ONNX on CPU (Batch Size = 4)
    if os.path.exists(onnxPath):
        try:
            engine = OnnxCpuInferenceEngine(onnxPath)
            _ = engine.infer(inferenceTest)
            print(f"[FinBERT Engine] Successfully loaded ONNX CPU model from {onnxPath} (Batch Size: {engine.optimalBatchSize})")
            return engine
        except Exception as e:
            print(f"[FinBERT Engine] ONNX CPU load failed: {e}")

    # 5. FP32 PyTorch on CPU (Batch Size = 4)
    try:
        engine = PytorchCpuInferenceEngine(modelId=modelId)
        _ = engine.infer(inferenceTest)
        print(f"[FinBERT Engine] Successfully loaded PyTorch CPU model (Batch Size: {engine.optimalBatchSize})")
        return engine
    except Exception as e:
        print(f"[FinBERT Engine] PyTorch CPU load failed: {e}")

    print("[FinBERT Engine] Neither TensorRT, ONNX, nor PyTorch model could be loaded.")
    return None


# Global singleton engine instance and initialization lock
sentimentEngine: Optional[BaseInferenceEngine] = None
engineLoadLock = threading.RLock()
engineLoadAttempted = False


def getSentimentEngine() -> Optional[BaseInferenceEngine]:
    global sentimentEngine, engineLoadAttempted

    with engineLoadLock:
        if not engineLoadAttempted:
            engineLoadAttempted = True
            sentimentEngine = getBestInferenceEngine()
            if sentimentEngine is not None:
                try:
                    warmupText = ["Financial market sentiment analysis initialisation warmup."]
                    _ = sentimentEngine.infer(warmupText)
                except Exception as warmupError:
                    print(f"[Sentiment Engine] Prewarm encountered an issue: {warmupError}")

        return sentimentEngine


def preloadSentimentModelAsync() -> threading.Thread:
    thread = threading.Thread(target=getSentimentEngine, daemon=True, name="SentimentModelPreloader")
    thread.start()
    return thread
