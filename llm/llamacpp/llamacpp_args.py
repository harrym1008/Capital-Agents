import os
import time
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

LLAMACPP_PORT = 9081
LLAMACPP_EXECUTABLE = "llama-server.exe"

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llamacpp_config.json")

LOCKED_ARGS = {
    "--host": "127.0.0.1",
    "--port": "9081",
    "--log-verbosity": "4"
}

DISALLOWED_USER_KEYS = {"--host", "-h", "--port", "-p", "--log-verbosity", "-lv"}

SAMPLING_FLAGS = {
    "--temp",
    "--top-p",
    "--top-k",
    "--min-p",
    "--presence-penalty",
    "--repeat-penalty",
    "--frequency-penalty"
}


def isDisallowedKey(keyStr: str) -> bool:
    if not keyStr:
        return False
    return keyStr.strip().lower() in DISALLOWED_USER_KEYS


def cleanUserArgs(rawArgs: Any) -> List[Dict[str, Any]]:
    cleaned = []
    if isinstance(rawArgs, list):
        for item in rawArgs:
            if isinstance(item, dict):
                k = str(item.get("key", "")).strip()
                if not k or isDisallowedKey(k):
                    continue
                v = str(item.get("value", ""))
                enabled = bool(item.get("enabled", True))
                cleaned.append({"key": k, "value": v, "enabled": enabled})
    elif isinstance(rawArgs, dict):
        for k, v in rawArgs.items():
            kClean = str(k).strip()
            if not kClean or isDisallowedKey(kClean):
                continue
            cleaned.append({"key": kClean, "value": str(v), "enabled": True})
    return cleaned


def getDefaultConfig() -> Dict[str, Any]:
    return {
        "executablePath": "llama-server.exe",
        "lastUsedModelId": "",
        "globalArgs": [],
        "models": []
    }


def loadConfig() -> Dict[str, Any]:
    import json
    if not os.path.exists(CONFIG_FILE_PATH):
        defaultCfg = getDefaultConfig()
        saveConfig(defaultCfg)
        return defaultCfg
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            cfg["lastUsedModelId"] = str(cfg.get("lastUsedModelId", "")).strip()
            cfg["globalArgs"] = cleanUserArgs(cfg.get("globalArgs", []))
            if "models" in cfg and isinstance(cfg["models"], list):
                for m in cfg["models"]:
                    if isinstance(m, dict):
                        m["args"] = cleanUserArgs(m.get("args", []))
            return cfg
    except Exception:
        return getDefaultConfig()


def saveConfig(configData: Dict[str, Any]) -> bool:
    import json
    try:
        cleanedData = {
            "executablePath": configData.get("executablePath", "llama-server.exe").strip() or "llama-server.exe",
            "lastUsedModelId": str(configData.get("lastUsedModelId", "")).strip(),
            "globalArgs": cleanUserArgs(configData.get("globalArgs", [])),
            "models": []
        }
        for m in configData.get("models", []):
            if isinstance(m, dict):
                cleanedData["models"].append({
                    "id": m.get("id", f"model_{int(time.time()*1000)}"),
                    "alias": m.get("alias", "").strip(),
                    "modelPath": m.get("modelPath", "").strip(),
                    "executablePath": m.get("executablePath", "").strip(),
                    "args": cleanUserArgs(m.get("args", []))
                })
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(cleanedData, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving Llama.cpp config: {e}")
        return False


def findModelConfig(modelIdentifier: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Finds a model config from llamacpp_config.json by ID, alias, filename, or fallback.
    Returns (modelConfig, None) if found, or (None, errorMessage) if not found.
    """
    if not modelIdentifier:
        return None, "No model identifier provided."
    
    cfg = loadConfig()
    models = cfg.get("models", [])
    if not models:
        return None, "No models configured in Llama.cpp setup. Please configure a GGUF model first."
    
    cleanIdent = str(modelIdentifier).strip()
    
    # 1. Match by exact ID
    for m in models:
        if m.get("id") == cleanIdent:
            return m, None
            
    # 2. Match by exact alias (case-insensitive)
    for m in models:
        if m.get("alias", "").strip().lower() == cleanIdent.lower():
            return m, None
            
    # 3. Match by filename (with or without .gguf)
    for m in models:
        mPath = m.get("modelPath", "")
        fName = os.path.basename(mPath)
        fNameWithoutExt = fName[:-5] if fName.lower().endswith(".gguf") else fName
        if fName.lower() == cleanIdent.lower() or fNameWithoutExt.lower() == cleanIdent.lower():
            return m, None

    # 4. Fallback: If only 1 model configured, use it
    if len(models) == 1:
        return models[0], None
        
    return None, f"Model '{cleanIdent}' was not found in Llama.cpp configuration."


def buildLlamaCppCommandLine(modelIdentifier: str, allowParallel: bool = True, argOverrides: Optional[dict] = None) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Constructs the exact executable path and argument list to launch llama-server.exe.
    Enforces locked parameters (--host 127.0.0.1, --port 9081, -lv 4), merges global and per-model
    arguments, omits empty sampling parameters, and sets model/parallel options.
    Returns (executablePath, commandArgs, modelConfig).
    """
    modelConfig, error = findModelConfig(modelIdentifier)
    if error:
        raise ValueError(error)
        
    modelPath = modelConfig.get("modelPath", "").strip()
    if not modelPath:
        raise ValueError(f"Model '{modelConfig.get('alias', modelIdentifier)}' has no GGUF file path configured.")
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"GGUF model file not found at '{modelPath}'.")
        
    cfg = loadConfig()
    
    # Resolve executable: model override > global executable > llama-server.exe
    executablePath = modelConfig.get("executablePath", "").strip() or cfg.get("executablePath", "").strip() or LLAMACPP_EXECUTABLE
    
    # Prepare argument lists
    globalArgsList = cleanUserArgs(cfg.get("globalArgs", []))
    modelArgsList = cleanUserArgs(modelConfig.get("args", []))
    
    # Active per-model keys (that are enabled)
    activeModelKeys = {a["key"].strip().lower() for a in modelArgsList if a.get("enabled", True)}
    
    commandArgs = [executablePath]
    
    # 1. Enforce forced/locked flags
    commandArgs.extend(["--host", "127.0.0.1"])
    commandArgs.extend(["--port", str(LLAMACPP_PORT)])
    commandArgs.extend(["-lv", "4"])
    
    # 2. Add -m <modelPath>
    commandArgs.extend(["-m", modelPath])
    
    # 3. Add enabled global arguments (unless overridden by active model arg)
    for gArg in globalArgsList:
        if not gArg.get("enabled", True):
            continue
        k = gArg.get("key", "").strip()
        v = str(gArg.get("value", "")).strip()
        if not k or isDisallowedKey(k):
            continue
        if k.lower() in activeModelKeys:
            # Overridden by model
            continue
        # Check if empty sampling param
        if k.lower() in SAMPLING_FLAGS and v == "":
            continue
        commandArgs.append(k)
        if v != "":
            commandArgs.append(v)
            
    # 4. Add enabled per-model arguments
    for mArg in modelArgsList:
        if not mArg.get("enabled", True):
            continue
        k = mArg.get("key", "").strip()
        v = str(mArg.get("value", "")).strip()
        if not k or isDisallowedKey(k):
            continue
        # Check if empty sampling param
        if k.lower() in SAMPLING_FLAGS and v == "":
            continue
        commandArgs.append(k)
        if v != "":
            commandArgs.append(v)
            
    # 5. Apply any programmatic argOverrides
    if argOverrides and isinstance(argOverrides, dict):
        for k, v in argOverrides.items():
            if isDisallowedKey(str(k)):
                continue
            commandArgs.append(str(k))
            if str(v) != "":
                commandArgs.append(str(v))
                
    # 6. If allowParallel is False, ensure -np 1
    if not allowParallel:
        hasParallelArg = any(arg in ("-np", "--parallel") for arg in commandArgs)
        if not hasParallelArg:
            commandArgs.extend(["-np", "1"])
            
    return executablePath, commandArgs, modelConfig


def validateGgufPath(filePath: str) -> Tuple[bool, Any]:
    if not filePath:
        return False, "File path is required."
    cleanPath = filePath.strip().strip('"').strip("'")
    if not os.path.exists(cleanPath):
        return False, f"File not found: {cleanPath}"
    if not os.path.isfile(cleanPath):
        return False, f"Path is not a regular file: {cleanPath}"
    if not cleanPath.lower().endswith(".gguf"):
        return False, "File does not have a .gguf extension."
    try:
        with open(cleanPath, "rb") as f:
            header = f.read(4)
            if header != b"GGUF":
                return False, f"Invalid GGUF header magic. Expected 'GGUF', got '{header}'."
    except Exception as e:
        return False, f"Error reading file header: {str(e)}"
    
    fileSize = os.path.getsize(cleanPath)
    fileName = os.path.basename(cleanPath)
    return True, {"fileName": fileName, "filePath": cleanPath, "fileSizeBytes": fileSize}


def getLlamaCppModelsList() -> List[Dict[str, Any]]:
    cfg = loadConfig()
    lastUsedId = cfg.get("lastUsedModelId", "").strip()
    modelsList = []
    for modelItem in cfg.get("models", []):
        alias = modelItem.get("alias", "").strip()
        modelPath = modelItem.get("modelPath", "").strip()
        fileName = os.path.basename(modelPath) if modelPath else "unknown.gguf"
        displayName = alias if alias else fileName
        modelsList.append({
            "id": modelItem.get("id", ""),
            "alias": alias,
            "displayName": displayName,
            "fileName": fileName,
            "modelPath": modelPath,
            "args": modelItem.get("args", []),
            "executablePath": modelItem.get("executablePath", "")
        })

    # Sort models alphabetically by display name / alias (case-insensitive)
    modelsList.sort(key=lambda m: (m["displayName"] or m["alias"] or m["fileName"]).lower())

    # Hoist last used model to index 0 if specified
    if lastUsedId:
        matchingIdx = next((i for i, m in enumerate(modelsList) if m["id"] == lastUsedId or m["alias"].lower() == lastUsedId.lower()), None)
        if matchingIdx is not None and matchingIdx > 0:
            lastUsedModel = modelsList.pop(matchingIdx)
            modelsList.insert(0, lastUsedModel)

    return modelsList


def openNativeGgufFileDialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selectedPath = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF Models", "*.gguf")]
        )
        root.destroy()
        return selectedPath or ""
    except Exception as e:
        print(f"Error opening native GGUF file dialog: {e}")
        return ""


def openNativeExecutableFileDialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selectedPath = filedialog.askopenfilename(
            title="Select Llama Server Executable",
            filetypes=[("Executable Files", "*.exe"), ("All Files", "*.*")]
        )
        root.destroy()
        return selectedPath or ""
    except Exception as e:
        print(f"Error opening native executable file dialog: {e}")
        return ""
