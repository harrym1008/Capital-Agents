import os
import json

SERVER_CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_config.json")

DEFAULT_CONFIG = {
    "lastSelectedProvider": "llamacpp",
    "llamacpp": {
        "executablePath": "llama-server.exe",
        "lastUsedModelId": "",
        "globalArgs": [],
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
    return json.loads(json.dumps(obj))


def mergeDefaults(cfg):
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
    cfg = loadServerConfig()
    section = cfg.get(provider)
    return section if isinstance(section, dict) else deepCopy(DEFAULT_CONFIG.get(provider, {}))


def setSectionValues(provider, values):
    if not isinstance(values, dict):
        return False
    cfg = loadServerConfig()
    for key, value in values.items():
        if isinstance(value, str):
            value = value.strip()
        cfg[provider][key] = value
    return saveServerConfig(cfg)
