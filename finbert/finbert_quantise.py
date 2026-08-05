import random
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings from modelopt asking about triton
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="modelopt")

import torch
import tensorrt as trt
import modelopt.torch.quantization as mtq
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FIN_PHRASE_BANK_PATH = "finbert/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"

def loadFinancialPhraseBank(sampleCount, returnRightSide=False):
    texts = []
    trueLabels = []

    with open(FIN_PHRASE_BANK_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("@", 1)
            if len(parts) == 2:
                texts.append(parts[0].strip())
                trueLabels.append(parts[1].strip())

    if len(texts) > sampleCount:
        random.seed(42)
        random.shuffle(texts)
        texts = texts[:sampleCount] if not returnRightSide else texts[sampleCount:]
        trueLabels = trueLabels[:sampleCount] if not returnRightSide else trueLabels[sampleCount:]

    for i in range(len(trueLabels)):
        trueLabels[i] = ["bearish", "neutral", "bullish"][["negative", "neutral", "positive"].index(trueLabels[i].lower())]
        # Map FinancialPhraseBank labels to ModernFinBERT labels

    return texts, trueLabels


def convertToFp32Onnx(modelId, onnxPath):
    print(f"[Step 1] Loading FP32 model and tokeniser...")
    model = AutoModelForSequenceClassification.from_pretrained(modelId).cuda()
    # tokeniser = AutoTokenizer.from_pretrained(modelId)

    dummyInput = (
        torch.ones(1, 1024, dtype=torch.long, device="cuda"),
        torch.ones(1, 1024, dtype=torch.long, device="cuda"),
    )

    print(f"[Step 2] Exporting FP32 model to ONNX...")
    torch.onnx.export(
        model,
        dummyInput,
        onnxPath,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=17,
        dynamo=False
    )
    print(f"  FP32 ONNX model exported to {onnxPath}")
    return onnxPath


def calibrateAndQuantiseIntoOnnx(modelId, onnxSuffix, quantConfig, maxSamples):
    print(f"[Step 1] Loading model, tokeniser and calibration data...")
    model = AutoModelForSequenceClassification.from_pretrained(modelId).cuda()
    tokeniser = AutoTokenizer.from_pretrained(modelId, cache_dir="data/models", model_max_length=1024)

    # Quantisation data requires a representation of the input data, but not that much. 
    calibrationTexts = loadFinancialPhraseBank(maxSamples)[0]       # Dont load the true labels

    encodedInputs = tokeniser(calibrationTexts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")

    def forwardLoop(modelToCalibrate):
        numSamples = encodedInputs["input_ids"].shape[0]
        with torch.no_grad():
            for i in range(min(maxSamples, numSamples)):
                inputBatch = {k: v[i:i+1] for k, v in encodedInputs.items()}
                modelToCalibrate(**inputBatch)

    print(f"[Step 2] Calibrating model...")
    quantisedModel = mtq.quantize(model, quantConfig, forwardLoop)

    print(f"[Step 3] Exporting to ONNX...")
    onnxPath = f"{modelId.split('/')[-1]}_{onnxSuffix}.onnx"
    dummyInput = (encodedInputs["input_ids"][:1], encodedInputs["attention_mask"][:1])

    torch.onnx.export(
        quantisedModel,
        dummyInput,
        onnxPath,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=17,
        dynamo=False
    )
    print(f"  ONNX model exported to {onnxPath}")
    return onnxPath


def compileTensorRtEngine(onnxPath, enginePath):
    print(f"[Step 4] Configuring TensorRT engine...")
    trtLogger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trtLogger)
    
    network = builder.create_network()
    parser = trt.OnnxParser(network, trtLogger)

    with open(onnxPath, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model. Errors:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1024 ** 3))

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", min=(1, 1), opt=(4, 128), max=(8, 1024))
    profile.set_shape("attention_mask", min=(1, 1), opt=(4, 128), max=(8, 1024))
    config.add_optimization_profile(profile)

    print(f"[Step 5] Building TensorRT engine...")
    serialisedEngine = builder.build_serialized_network(network, config)

    if not serialisedEngine:
        raise RuntimeError("Failed to build TensorRT engine.")

    with open(enginePath, "wb") as f:
        f.write(serialisedEngine)

    print(f"  TensorRT engine compiled and saved to {enginePath}")



def main():
    # All models are enforced a maximum of 1024 tokens
    modelsDir = "finbert/models"

    # Download the ModernFinBERT model and export it to FP32 ONNX
    print(f"Converting FP32 model to ONNX...")
    onnxPathFp32 = convertToFp32Onnx(
        "tabularisai/ModernFinBERT",
        onnxPath=f"{modelsDir}/ModernFinBERT_fp32.onnx"
    )

    # Build the FP8 TensorRT engine
    print(f"\n\nCalibrating and quantising model to FP8 ONNX...")
    onnxPathFp8 = calibrateAndQuantiseIntoOnnx(
        "tabularisai/ModernFinBERT",
        onnxSuffix="fp8",
        quantConfig=mtq.FP8_DEFAULT_CFG,
        maxSamples=256
    )
    compileTensorRtEngine(onnxPathFp8, f"{modelsDir}/ModernFinBERT_fp8.engine")
    os.remove(onnxPathFp8)          # Delete intermediate ONNX file

if __name__ == "__main__":
    main()