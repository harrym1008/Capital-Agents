import os
import time
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from llm.server_config_store import loadServerConfig, saveServerConfig

LLAMACPP_PORT = 9081
LLAMACPP_EXECUTABLE = "llama-server.exe"

DISALLOWED_USER_KEYS = {
    "--host", "-h",
    "--port", "-p",
    "--log-verbosity", "-lv",
    "-m", "--model"
}

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
                key = str(item.get("key", "")).strip()
                if not key or isDisallowedKey(key):
                    continue
                val = str(item.get("value", ""))
                enabled = bool(item.get("enabled", True))
                cleaned.append({"key": key, "value": val, "enabled": enabled})
    elif isinstance(rawArgs, dict):
        for key, val in rawArgs.items():
            keyClean = str(key).strip()
            if not keyClean or isDisallowedKey(keyClean):
                continue
            cleaned.append({"key": keyClean, "value": str(val), "enabled": True})
    return cleaned


def getDefaultConfig() -> Dict[str, Any]:
    return {
        "executablePath": LLAMACPP_EXECUTABLE,
        "lastUsedModelId": "",
        "globalArgs": [],
        "models": []
    }


def loadConfig() -> Dict[str, Any]:
    cfg = loadServerConfig().get("llamacpp", {})
    if not isinstance(cfg, dict):
        cfg = getDefaultConfig()
    cfg["lastUsedModelId"] = str(cfg.get("lastUsedModelId", "")).strip()
    cfg["globalArgs"] = cleanUserArgs(cfg.get("globalArgs", []))
    if "models" in cfg and isinstance(cfg["models"], list):
        for model in cfg["models"]:
            if isinstance(model, dict):
                model["args"] = cleanUserArgs(model.get("args", []))
    return cfg


def saveConfig(configData: Dict[str, Any]) -> bool:
    try:
        cleanedData = {
            "executablePath": configData.get("executablePath", LLAMACPP_EXECUTABLE).strip() or LLAMACPP_EXECUTABLE,
            "lastUsedModelId": str(configData.get("lastUsedModelId", "")).strip(),
            "globalArgs": cleanUserArgs(configData.get("globalArgs", [])),
            "models": []
        }
        for model in configData.get("models", []):
            if isinstance(model, dict):
                cleanedData["models"].append({
                    "id": model.get("id", f"model_{int(time.time() * 1000)}"),
                    "alias": model.get("alias", "").strip(),
                    "modelPath": model.get("modelPath", "").strip(),
                    "executablePath": model.get("executablePath", "").strip(),
                    "args": cleanUserArgs(model.get("args", []))
                })
        fullConfig = loadServerConfig()
        fullConfig["llamacpp"] = cleanedData
        return saveServerConfig(fullConfig)
    except Exception as err:
        print(f"Error saving Llama.cpp config: {err}")
        return False


def findModelConfig(modelIdentifier: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Finds a model config from server_config.json by ID, alias, filename, or fallback.
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
    for model in models:
        if model.get("id") == cleanIdent:
            return model, None
            
    # 2. Match by exact alias (case-insensitive)
    for model in models:
        if model.get("alias", "").strip().lower() == cleanIdent.lower():
            return model, None
            
    # 3. Match by filename (with or without .gguf)
    for model in models:
        modelPath = model.get("modelPath", "")
        fileName = os.path.basename(modelPath)
        fileNameWithoutExt = fileName[:-5] if fileName.lower().endswith(".gguf") else fileName
        if fileName.lower() == cleanIdent.lower() or fileNameWithoutExt.lower() == cleanIdent.lower():
            return model, None

    # 4. Fallback: If only 1 model configured, use it
    if len(models) == 1:
        return models[0], None
        
    return None, f"Model '{cleanIdent}' was not found in Llama.cpp configuration."


def buildLlamaCppCommandLine(
    modelIdentifier: str, 
    allowParallel: bool = True, 
    argOverrides: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[str], Dict[str, Any]]:
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
    activeModelKeys = {arg["key"].strip().lower() for arg in modelArgsList if arg.get("enabled", True)}
    
    commandArgs = [executablePath]
    
    # 1. Enforce locked system flags
    commandArgs.extend(["--host", "127.0.0.1"])
    commandArgs.extend(["--port", str(LLAMACPP_PORT)])
    commandArgs.extend(["-lv", "4"])
    commandArgs.append("--context-shift")
    
    # 2. Add -m <modelPath>
    commandArgs.extend(["-m", modelPath])
    
    # 3. Add enabled global arguments (unless overridden by active model arg)
    for globalArg in globalArgsList:
        if not globalArg.get("enabled", True):
            continue
        key = globalArg.get("key", "").strip()
        val = str(globalArg.get("value", "")).strip()
        if not key or isDisallowedKey(key) or key.lower() in activeModelKeys:
            continue
        # Skip empty sampling flags
        if key.lower() in SAMPLING_FLAGS and val == "":
            continue
        commandArgs.append(key)
        if val != "":
            commandArgs.append(val)
            
    # 4. Add enabled per-model arguments
    for modelArg in modelArgsList:
        if not modelArg.get("enabled", True):
            continue
        key = modelArg.get("key", "").strip()
        val = str(modelArg.get("value", "")).strip()
        if not key or isDisallowedKey(key):
            continue
        # Skip empty sampling flags
        if key.lower() in SAMPLING_FLAGS and val == "":
            continue
        commandArgs.append(key)
        if val != "":
            commandArgs.append(val)
            
    # 5. Apply any programmatic argOverrides
    if argOverrides and isinstance(argOverrides, dict):
        for overrideKey, overrideVal in argOverrides.items():
            keyClean = str(overrideKey).strip()
            if isDisallowedKey(keyClean):
                continue
            commandArgs.append(keyClean)
            valStr = str(overrideVal).strip()
            if valStr != "":
                commandArgs.append(valStr)
                
    # 6. If allowParallel is False, ensure single parallel slot
    if not allowParallel:
        hasParallelArg = any(arg in ("-np", "--parallel") for arg in commandArgs)
        if not hasParallelArg:
            commandArgs.extend(["-np", "1"])
            
    return executablePath, commandArgs, modelConfig


def validateGGUFPath(filePath: str) -> Tuple[bool, Any]:
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
        with open(cleanPath, "rb") as fileHandle:
            header = fileHandle.read(4)
            if header != b"GGUF":
                return False, f"Invalid GGUF header magic. Expected 'GGUF', got '{header}'."
    except Exception as err:
        return False, f"Error reading file header: {str(err)}"
    
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
        matchingIdx = next(
            (i for i, m in enumerate(modelsList) if m["id"] == lastUsedId or m["alias"].lower() == lastUsedId.lower()), 
            None
        )
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
    except Exception as err:
        print(f"Error opening native GGUF file dialog: {err}")
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
            filetypes=[("Llama Server Executable", "llama-server.exe"), ("Other Compatible Executables", "*.exe")]
        )
        root.destroy()
        return selectedPath or ""
    except Exception as err:
        print(f"Error opening native executable file dialog: {err}")
        return ""
