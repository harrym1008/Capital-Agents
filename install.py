import os
import shutil
import subprocess
import sys

# Locate or create the project's virtual environment
venvPython = None
if sys.platform == "win32":
    venvPython = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")
else:
    venvPython = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")


if not os.path.exists(venvPython):
    print("Virtual environment not found, creating ./venv...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)


def isNvidiaGpuPresent() -> bool:
    # Run nvidia-smi to check for NVIDIA GPU presence or fallback to the dll check on Windows
    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if result.returncode == 0:
                return True
        except Exception:
            pass

    if sys.platform == "win32":
        systemRoot = os.environ.get("SystemRoot", "C:\\Windows")
        cudaDll = os.path.join(systemRoot, "System32", "nvcuda.dll")
        if os.path.isfile(cudaDll):
            return True

    return False


def runPipInstall(packages: list[str], extraArgs: list[str] = None) -> None:
    # Run pip install in the virtual environment with optional extra arguments
    command = [venvPython, "-m", "pip", "install"]
    if extraArgs:
        command.extend(extraArgs)
    command.extend(packages)
    print(f"\n[Installer] Executing: {' '.join(command)}")
    subprocess.run(command, check=True)


def verifyTorchCuda() -> bool:
    # Verify if torch.cuda.is_available() returns True in the virtual environment
    checkScript = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    try:
        proc = subprocess.run([venvPython, "-c", checkScript], check=False)
        return proc.returncode == 0
    except Exception:
        return False


def downloadOnnxModel(targetDir: str = "finbert/models") -> None:
    # Downloads ModernFinBERT_fp32.onnx from Hugging Face if not present
    onnxPath = os.path.join(targetDir, "ModernFinBERT_fp32.onnx")
    if os.path.exists(onnxPath):
        print(f"\n[Installer] ONNX model already present at {onnxPath}.")
        return

    print("\n[Installer] Downloading ModernFinBERT_fp32.onnx from Hugging Face (harrym1008/ModernFinBERT-fp32-onnx)...")
    os.makedirs(targetDir, exist_ok=True)
    downloadScript = (
        "from huggingface_hub import hf_hub_download\n"
        "hf_hub_download(repo_id='harrym1008/ModernFinBERT-fp32-onnx', filename='ModernFinBERT_fp32.onnx', local_dir='finbert/models')\n"
    )
    subprocess.run([venvPython, "-c", downloadScript], check=True)
    print(f"[Installer] Successfully downloaded ONNX model to {onnxPath}.")


def setupFinbertEnvironment(installTensorRt: bool = True) -> None:
    equalsWidth = max(39, len(venvPython) + 17)
    print("=" * equalsWidth)
    print( "  CapitalAgents Environment Installer")
    print(f"  Target Venv: {venvPython}")
    print("=" * equalsWidth)

    hasGpu = isNvidiaGpuPresent()
    cudaActive = False

    if hasGpu:
        print("[Installer] NVIDIA GPU detected. Installing PyTorch cu132...")
        try:
            runPipInstall(["torch"], extraArgs=["--index-url", "https://download.pytorch.org/whl/cu132"])
            cudaActive = verifyTorchCuda()
            if cudaActive:
                print("[Installer] PyTorch cu132 installed and CUDA verified successfully!")
            else:
                print("[Installer] PyTorch cu132 installed, but torch.cuda.is_available() returned False.")
        except Exception as error:
            print(f"[Installer] Failed to install PyTorch cu132: {error}")

    if cudaActive:
        print("CUDA is available. FinBERT will use GPU acceleration.")
    else:
        print("CUDA is not available. FinBERT will use CPU fallback.")


    # Fallback to CPU if no GPU or CUDA installation failed
    if not cudaActive:
        print("\n[Installer] Configuring CPU environment...")
        runPipInstall(["torch"], extraArgs=["--index-url", "https://download.pytorch.org/whl/cpu"])

    # Configure ONNX Runtime (avoid conflicting CPU and GPU wheels)
    if cudaActive:
        print("\n[Installer] Configuring ONNX Runtime GPU (CUDA)...")
        subprocess.run([venvPython, "-m", "pip", "uninstall", "-y", "onnxruntime"], check=False)
        try:
            runPipInstall(["onnxruntime-gpu"])
        except Exception as onnxError:
            print(f"[Installer] onnxruntime-gpu failed ({onnxError}), falling back to CPU ONNX...")
            runPipInstall(["onnxruntime"])

        # Verify ONNX Runtime GPU installation
        if installTensorRt:
            print("\n[Installer] Attempting TensorRT installation...")
            try:
                runPipInstall(["tensorrt"])
                proc = subprocess.run([venvPython, "-c", "import tensorrt"], check=False)
                if proc.returncode == 0:
                    print("[Installer] TensorRT successfully verified!")
                else:
                    print("[Installer] TensorRT import failed; FinBERT will use ONNX CUDA fallback.")
            except Exception as trtError:
                print(f"[Installer] TensorRT skipped: {trtError} (ONNX CUDA fallback will be used).")
    else:
        print("\n[Installer] Configuring ONNX Runtime CPU...")
        subprocess.run([venvPython, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu"], check=False)
        runPipInstall(["onnxruntime"])

    # Base dependencies
    print("\n[Installer] Installing base dependencies from requirements-base.txt...")
    runPipInstall(["-r", "requirements-base.txt"])

    # Download pre-built ONNX model
    downloadOnnxModel()

    # Final Engine Verification
    print("\n" + "=" * 50)
    print("[Installer] Verifying FinBERT Engine Selection...")
    print("=" * 50)
    testCode = ("""
from finbert.finbert_engines import getBestInferenceEngine
engine = getBestInferenceEngine()
result = engine.infer(["Company stock price rises 50 percent after stellar earnings report."])
if engine:
    print(f'--> SUCCESS: Active FinBERT Engine is {engine.engineType}. Warmed up, inference tested.')
else:
    print('--> WARNING: No FinBERT engine could be initialised.')
""")
    subprocess.run([venvPython, "-c", testCode], check=False)
    print("\n[Installer] Setup complete!\n")


if __name__ == "__main__":
    installTensorrt = input("Do you wish to install TensorRT? Type n if you do not know what it is. (y/n): ").lower() == "y"
    setupFinbertEnvironment(installTensorRt=installTensorrt)
