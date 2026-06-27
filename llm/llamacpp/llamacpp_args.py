from enum import Enum


LLAMACPP_PORT = 9081
LLAMACPP_EXECUTABLE = "llama-server.exe"

# THINKING_BUDGET = 128
# SUMMARISE_THINK_BUDGET = 512
THINKING_BUDGET_MESSAGE = "... my thinking allowance has been exhausted. I shall now produce my final response.\n"

MODELS_FOLDER = "I:\\LLM\\"


class LlamaCppModel(Enum):
    GEMMA_4_26B_A4B = "Gemma-4-26B-E4B"
    GEMMA_4_12B = "Gemma-4-12B"
    GEMMA_4_E4B = "Gemma-4-E4B"
    GEMMA_4_E2B = "Gemma-4-E2B"
    GEMMA_4_E2B_CPU = "Gemma-4-E2B-CPU"
    
    QWEN_36_27B = "Qwen-3.6-27B"
    QWEN_36_35B_A3B = "Qwen-3.6-35B-A3B"

    QWEN_35_9B = "Qwen-3.5-9B"
    QWEN_35_800M = "Qwen-3.5-0.8B"

    MINICPM5_1B = "MiniCPM5-1B"
    LFM25_8B_A1B = "LFM-2.5-8B-A1B"
    GPT_OSS_20B = "GPT-OSS-20B"


EMPTY_ARG = _ = ""
LLAMACPP_MODEL_TO_ARGS = {
    LlamaCppModel.GEMMA_4_26B_A4B: {
        "-m":               f"{MODELS_FOLDER}Gemma4\\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.5",
        "--top-p":          "0.95",
        "--top-k":          "64",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "--mlock":          _,
        "--fit":            "on",
        "--fit-target":     "1000",
        # "--reasoning-budget":           str(THINKING_BUDGET),     # To be set directly in the HTTP request body
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.GEMMA_4_12B: {
        "-m":               f"{MODELS_FOLDER}Gemma4\\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.5",
        "--top-p":          "0.95",
        "--top-k":          "64",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "-ngl":             "99",
        "--mlock":          _,
        "--spec-type":                  "draft-mtp",
        "--model-draft":                f"{MODELS_FOLDER}Gemma4\\mtp\\gemma-4-12B-it-Q8_0-MTP.gguf",
        "--spec-draft-n-max":           "2",
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.GEMMA_4_E4B: {
        "-m":               f"{MODELS_FOLDER}Gemma4\\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.5",
        "--top-p":          "0.95",
        "--top-k":          "64",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "4096",
        "-ub":              "1024",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "-ngl":             "99",
        "--mlock":          _,
        "--spec-type":                  "draft-mtp",
        "--model-draft":                f"{MODELS_FOLDER}Gemma4\\mtp\\gemma-4-E4B-it-Q8_0-MTP.gguf",
        "--spec-draft-n-max":           "2",
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.GEMMA_4_E2B: {
        "-m":               f"{MODELS_FOLDER}Gemma4\\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.5",
        "--top-p":          "0.95",
        "--top-k":          "64",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "8192",
        "-ub":              "1024",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "-ngl":             "99",
        "--mlock":          _,
        "--spec-type":                  "draft-mtp",
        "--model-draft":                f"{MODELS_FOLDER}Gemma4\\mtp\\gemma-4-E2B-it-Q8_0-MTP.gguf",
        "--spec-draft-n-max":           "2",
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.GEMMA_4_E2B_CPU: {
        "-m":               f"{MODELS_FOLDER}Gemma4\\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.5",
        "--top-p":          "0.95",
        "--top-k":          "64",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "--jinja":          _,
        "-np":              "1",
        "--cache-ram":      "1024",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "-ngl":             "0",
        "--context-shift":  _,
        "--mlock":          _,
        # "--spec-type":                  "draft-mtp",
        # "--model-draft":                f"{MODELS_FOLDER}Gemma4\\mtp\\gemma-4-E2B-it-Q8_0-MTP.gguf",
        # "--spec-draft-n-max":           "2",
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },
    
    LlamaCppModel.MINICPM5_1B: {
        "-m":               f"{MODELS_FOLDER}Others\\MiniCPM5-1B-Q8_0.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.5",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "4096",
        "-ub":              "1024",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "-ngl":             "99",
        "--mlock":          _,
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },
    
    LlamaCppModel.LFM25_8B_A1B: {
        "-m":               f"{MODELS_FOLDER}Others\\LFM2.5-8B-A1B-UD-Q6_K.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.4",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "4096",
        "-ub":              "1024",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "-ngl":             "99",
        "--mlock":          _,
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.QWEN_36_27B: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3.6-27B-Bartowski-IQ3_XS.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.5",
        "--top-p":          "0.95",
        "--top-k":          "20",
        "--min-p":          "0.0",
        "--presence-penalty": "0.0",
        "--repeat-penalty":   "1.0",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "-ngl":             "99",
        "--mlock":          _,
        "--chat-template-kwargs":       '{"preserve_thinking":true}',
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.QWEN_36_35B_A3B: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3.6-35B-A3B-MTP-UD-Q3_K_M.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.5",
        "--top-p":          "0.95",
        "--top-k":          "20",
        "--min-p":          "0.0",
        "--presence-penalty": "0.0",
        "--repeat-penalty":   "1.0",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "--fit":            "on",
        "--fit-target":     "1000",
        "--mlock":          _,
        "--chat-template-kwargs":       '{"preserve_thinking":true}',
        "--spec-type":                  "draft-mtp",
        "--spec-draft-n-max":           "2",
        "--spec-draft-ngl":             "99",
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.GPT_OSS_20B: {
        "-m":               f"{MODELS_FOLDER}Others\\gpt-oss-20b-Q6_K.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.65",
        "--top-p":          "1.0",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "--mlock":          _,
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.QWEN_35_9B: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3.5-9B-MTP-Q8_0.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.6",
        "--top-p":          "0.95",
        "--top-k":          "20",
        "--min-p":          "0.0",
        "--presence-penalty": "1.5",
        "--repeat-penalty":   "1.0",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "--mlock":          _,
        "--chat-template-kwargs":       '{"preserve_thinking":true}',
        "--spec-type":                  "draft-mtp",
        "--spec-draft-n-max":           "2",
        "--spec-draft-ngl":             "99",
        "--reasoning":      "on",
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.QWEN_35_800M: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3.5-0.8B-MTP-Q8_0.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.6",
        "--top-p":          "1.0",
        "--top-k":          "20",
        "--min-p":          "0.0",
        "--presence-penalty": "2.0",
        "--repeat-penalty":   "1.0",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--no-mmap":        _,
        "--metrics":        _,
        "-b":               "8192",
        "-ub":              "2048",
        "--jinja":          _,
        "-np":              "1",
        "--kv-offload":     _,
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # 65k context should be enough for almost every use case
        "--mlock":          _,
        "--chat-template-kwargs":       '{"preserve_thinking":true}',
        "--spec-type":                  "draft-mtp",
        "--spec-draft-n-max":           "2",
        "--spec-draft-ngl":             "99",
        "--reasoning":      "on",
        # "--reasoning-budget":           str(THINKING_BUDGET),
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    }
}