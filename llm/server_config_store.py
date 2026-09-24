import os
import json

SERVER_CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_config.json")

DEFAULT_LLAMACPP_GLOBAL_ARGS = [
    {"key": "--flash-attn", "value": "on", "enabled": True},
    {"key": "--cache-type-k", "value": "q8_0", "enabled": True},
    {"key": "--cache-type-v", "value": "q8_0", "enabled": True},
    {"key": "--jinja", "value": "", "enabled": True},
    {"key": "--load-mode", "value": "dio", "enabled": True},
    {"key": "--metrics", "value": "", "enabled": True},
    {"key": "-np", "value": "2", "enabled": True},
    {"key": "--ctx-size", "value": "131072", "enabled": True},
    {"key": "--cache-ram", "value": "4096", "enabled": True},
    {"key": "-b", "value": "2048", "enabled": True},
    {"key": "-ub", "value": "512", "enabled": True},
    {"key": "--reasoning", "value": "on", "enabled": True},
    {"key": "--split-mode", "value": "none", "enabled": True}
]

DEFAULT_LLAMACPP_MODEL_ARGS = [
    {"key": "-fit", "value": "on", "enabled": True},
    {"key": "--fit-target", "value": "2800", "enabled": True},
]

DEFAULT_CONFIG = {
    "lastSelectedProvider": "llamacpp",
    "llamacpp": {
        "executablePath": "llama-server.exe",
        "lastUsedModelId": "",
        "globalArgs": DEFAULT_LLAMACPP_GLOBAL_ARGS,
        "models": []
    },
    "openrouter": {
        "lastUsedModelId": "",
        "providerRouter": "auto"
    },
    "openaicompatible": {
        "baseUrl": "",
        "apiKey": "",
        "modelName": ""
    }
}


def deepCopy(obj):
    # Perform deep copy of JSON serialisable object
    return json.loads(json.dumps(obj))


def mergeDefaults(cfg):
    # Merge loaded dictionary with default provider schema
    merged = {
        "lastSelectedProvider": "llamacpp",
        "llamacpp": deepCopy(DEFAULT_CONFIG["llamacpp"]),
        "openrouter": deepCopy(DEFAULT_CONFIG["openrouter"]),
        "openaicompatible": deepCopy(DEFAULT_CONFIG["openaicompatible"]),
    }
    if isinstance(cfg, dict):
        providerRaw = cfg.get("lastSelectedProvider")
        if isinstance(providerRaw, str) and providerRaw.strip():
            merged["lastSelectedProvider"] = providerRaw.strip()
        for provider in ("llamacpp", "openrouter", "openaicompatible"):
            section = cfg.get(provider)
            if isinstance(section, dict):
                for key, value in section.items():
                    if value is not None:
                        merged[provider][key] = value
    return merged


def loadServerConfig():
    # Read server configuration file from disk or initialise with defaults
    if not os.path.exists(SERVER_CONFIG_FILE_PATH):
        cfg = deepCopy(DEFAULT_CONFIG)
        saveServerConfig(cfg)
        return cfg
    try:
        with open(SERVER_CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return mergeDefaults(raw)
    except Exception:
        cfg = deepCopy(DEFAULT_CONFIG)
        saveServerConfig(cfg)
        return cfg


def saveServerConfig(configData):
    # Persist updated server configuration dictionary to JSON file
    try:
        cfg = mergeDefaults(configData or {})

        providerVal = cfg.get("lastSelectedProvider")
        if not isinstance(providerVal, str) or not providerVal.strip():
            cfg["lastSelectedProvider"] = "llamacpp"
        else:
            cfg["lastSelectedProvider"] = providerVal.strip()

        for provider in ("llamacpp", "openrouter", "openaicompatible"):
            if not isinstance(cfg[provider], dict):
                cfg[provider] = deepCopy(DEFAULT_CONFIG[provider])

        llamacpp = cfg["llamacpp"]
        if not isinstance(llamacpp.get("globalArgs"), list):
            llamacpp["globalArgs"] = []
        if not isinstance(llamacpp.get("models"), list):
            llamacpp["models"] = []

        with open(SERVER_CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True
    
    except Exception as e:
        print(f"Error saving server config: {e}")
        return False


def getSection(provider):
    # Return specific provider configuration section
    cfg = loadServerConfig()
    section = cfg.get(provider)
    return section if isinstance(section, dict) else deepCopy(DEFAULT_CONFIG.get(provider, {}))


def setSectionValues(provider, values):
    # Update key-value pairs within a specific provider section
    if not isinstance(values, dict):
        return False
    
    cfg = loadServerConfig()

    for key, value in values.items():
        if isinstance(value, str):
            value = value.strip()
        cfg[provider][key] = value
        
    return saveServerConfig(cfg)
