from enum import Enum

import os
from dotenv import load_dotenv
load_dotenv()

LLAMACPP_PORT = 9081
LLAMACPP_SUMMARY_PORT = 9082
LLAMACPP_EXECUTABLE = "llama-server.exe"

# THINKING_BUDGET = 128
# SUMMARISE_THINK_BUDGET = 512
THINKING_BUDGET_MESSAGE = "... my thinking allowance has been exhausted. I shall now produce my final response.\n"

MODELS_FOLDER = os.getenv("BASE_LLM_DIRECTORY") or "I:\\LLM\\"


class LlamaCppModel(Enum):
    GEMMA_4_26B_A4B = "Gemma-4-26B-E4B"
    GEMMA_4_12B = "Gemma-4-12B"
    GEMMA_4_E4B = "Gemma-4-E4B"
    GEMMA_4_E2B = "Gemma-4-E2B"
    GEMMA_4_E2B_CPU = "Gemma-4-E2B-CPU"
    
    QWEN_36_27B = "Qwen-3.6-27B"
    QWEN_36_35B_A3B = "Qwen-3.6-35B-A3B"

    QWEN_35_9B = "Qwen-3.5-9B"
    QWEN_35_2B = "Qwen-3.5-2B"
    QWEN_35_800M = "Qwen-3.5-0.8B"
    
    QWEN_3_1700M = "Qwen-3-1.7B"
    QWEN_3_600M = "Qwen-3-0.6B"
    GPT_OSS_20B = "GPT-OSS-20B"

    TERNARY_BONSAI_27B = "Ternary-Bonsai-27B"

    FORCE_FAILURE_TEST = "FORCE_FAILURE_TEST"

    SUMMARY_MODEL = "Summary-Model"


EMPTY_ARG = _ = ""
EXECUTABLE_ARG_OVERRIDE = "EXECUTABLE_ARG_OVERRIDE"

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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",    # 65k context per slot (np=2) should be enough for almost every use case
        "--fit":            "on",
        "--fit-target":     "1000",
        "--reasoning":      "on",
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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "-ngl":             "99",
        "--spec-type":                  "draft-mtp",
        "--model-draft":                f"{MODELS_FOLDER}Gemma4\\mtp\\gemma-4-12B-it-Q8_0-MTP.gguf",
        "--spec-draft-n-max":           "2",
        "--spec-draft-ngl":             "99",
        "--reasoning":      "on",
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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "4096",
        "-ub":              "1024",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "-ngl":             "99",
        "--spec-type":                  "draft-mtp",
        "--model-draft":                f"{MODELS_FOLDER}Gemma4\\mtp\\gemma-4-E4B-it-Q8_0-MTP.gguf",
        "--spec-draft-n-max":           "2",
        "--spec-draft-ngl":             "99",
        "--reasoning":      "on",
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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "8192",
        "-ub":              "1024",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "-ngl":             "99",
        "--spec-type":                  "draft-mtp",
        "--model-draft":                f"{MODELS_FOLDER}Gemma4\\mtp\\gemma-4-E2B-it-Q8_0-MTP.gguf",
        "--spec-draft-n-max":           "2",
        "--spec-draft-ngl":             "99",
        "--reasoning":      "on",
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.GEMMA_4_E2B_CPU: {
        "-m":               f"{MODELS_FOLDER}Gemma4\\gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.5",
        "--top-p":          "0.95",
        "--top-k":          "64",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--load-mode":      "dio",
        "--metrics":        _,
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "1024",
        "--ctx-size":       "131072",
        "-ngl":             "0",
        "--context-shift":  _,
        # "--spec-type":                  "draft-mtp",
        # "--model-draft":                f"{MODELS_FOLDER}Gemma4\\mtp\\gemma-4-E2B-it-Q8_0-MTP.gguf",
        # "--spec-draft-n-max":           "2",
        # "--spec-draft-ngl":             "0",
        "--reasoning":      "on",
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.QWEN_36_27B: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3.6-27B-MTP-UD-IQ3_XXS.gguf",
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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "-ngl":             "99",
        "--reasoning":      "on",
        "--reasoning-preserve":   _,
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.QWEN_36_35B_A3B: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "--fit":            "on",
        "--fit-target":     "1000",
        "--reasoning":      "on",
        "--reasoning-preserve":   _,
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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "--reasoning":      "on",
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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "--reasoning-preserve":         _,
        "--spec-type":                  "draft-mtp",
        "--spec-draft-n-max":           "2",
        "--spec-draft-ngl":             "99",
        "--reasoning":      "on",
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.QWEN_35_2B: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3.5-2B-MTP-UD-Q8_K_XL.gguf",
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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "4096",
        "-ub":              "1024",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "--reasoning-preserve":         _,
        "--spec-type":                  "draft-mtp",
        "--spec-draft-n-max":           "2",
        "--spec-draft-ngl":             "99",
        "--reasoning":      "on",
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
        "--load-mode":      "dio",
        "--metrics":        _,
        "-b":               "8192",
        "-ub":              "2048",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "--reasoning-preserve":         _,
        "--spec-type":                  "draft-mtp",
        "--spec-draft-n-max":           "2",
        "--spec-draft-ngl":             "99",
        "--reasoning":      "on",
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.QWEN_3_1700M: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3-1.7B-Q4_K_M.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.6",
        "--top-p":          "0.95",
        "--top-k":          "20",
        "--min-p":          "0.0",
        "--presence-penalty": "1.6",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--load-mode":      "dio",
        "--metrics":        _,
        # "-b":               "8192",
        # "-ub":              "2048",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # Qwen 3 has max context of 32k per slot
        "--reasoning":      "on",
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.QWEN_3_600M: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3-0.6B-UD-Q4_K_XL.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.15",
        "--top-p":          "0.8",
        "--top-k":          "20",
        "--repeat-penalty": "1.1",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--load-mode":      "dio",
        "--metrics":        _,
        # "-b":               "8192",
        # "-ub":              "2048",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # Qwen 3 has max context of 32k per slot
        "--reasoning":      "on",
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    LlamaCppModel.TERNARY_BONSAI_27B: {
        EXECUTABLE_ARG_OVERRIDE: f"I:\\llamacpp-other\\bonsai-ternary-27b\\llama-server.exe",
        "-m":               f"{MODELS_FOLDER}Bonsai\\Ternary-Bonsai-27B-Q2_0.gguf",
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
        "--mlock":          _,
        "--metrics":        _,
        "-b":               "2048",
        "-ub":              "512",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "131072",
        "-ngl":             "99",
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },

    #==============================================================

    LlamaCppModel.FORCE_FAILURE_TEST: {
        "-m":               f"{MODELS_FOLDER}Qwen\\MODEL_THAT_DOESNT_EXIST.gguf",
        "--port":           str(LLAMACPP_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.15",
        "--top-p":          "0.8",
        "--top-k":          "20",
        "--repeat-penalty": "1.1",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "--load-mode":      "dio",
        "--metrics":        _,
        # "-b":               "8192",
        # "-ub":              "2048",
        "--jinja":          _,
        "-np":              "2",
        "--cache-ram":      "4096",
        "--ctx-size":       "65536",    # Qwen 3 has max context of 32k per slot
        "--reasoning":      "on",
        "--reasoning-budget-message":   THINKING_BUDGET_MESSAGE
    },


    #==============================================================

    LlamaCppModel.SUMMARY_MODEL: {
        "-m":               f"{MODELS_FOLDER}Qwen\\Qwen3-0.6B-UD-Q4_K_XL.gguf",
        "--port":           str(LLAMACPP_SUMMARY_PORT),
        "--host":           "127.0.0.1",
        "--temp":           "0.15",
        "--top-p":          "0.8",
        "--top-k":          "20",
        "--repeat-penalty": "1.1",
        "--flash-attn":     "on",
        "--cache-type-k":   "q8_0",
        "--cache-type-v":   "q8_0",
        "-b":               "8192",
        "-ub":              "2048",
        "--load-mode":      "mlock",
        "--jinja":          _,
        "-np":              "5",
        "--cache-ram":      "512",
        "--ctx-size":       "30600",    # Each slot gets 6120 tokens to work with... if this is not enough,
        "-ngl":             "99",       # the model will split the large text into smaller chunks and summarise them in multiple iterations 
        "--reasoning":      "on",
        "--reasoning-budget-message":   "OK, I am out of thinking budget. Time to produce the summary."
    }
}