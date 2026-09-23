# CapitalAgents 
A multi-agent, locally run, portfolio management system using multiple roles which LLM agents will complete their analyses under.

# Current Page: Llamacpp Setup Page

In this page, the user can setup the models that they wish to use with llama.cpp. The `Save Configuration` button should be used commonly to save the configuration of the models that the user has selected.

On the left hand side, the user can select the `llama-server.exe` executable file that they wish to use for the llama.cpp engine itself. Additionally, there are default argument flags. Clicking the plus button at the bottom of the screen adds a command line argument. Command line arguments are split into two sides, the left side for the argument name and the right side for the argument value. The right side is optional. The recommended list of arguments is as follows:

- `--flash-attn` : `on` (This enables flash attention, which is a more efficient attention mechanism for transformers.)
- `--cache-type-k` : `q8_0` (This sets the cache type to q8_0, which is a quantisation method for reducing model context window size and improving inference speed.)
- `--cache-type-v` : `q8_0`
- `--jinja` : EMPTY (This enables the use of Jinja templates for generating prompts and responses.)
- `--load-mode` : `dio` (Sets the load mode to dio, direct I/O, improving performance and heavily reducing memory usage for large models.)
- `--metrics` : EMPTY (Allows tracking of the performance of the engine in the top-right corner of the boardroom pages.)
- `-np` : `2`  (Sets the number of inference slots. Set to either 1 or 2, setting to two allows two agents to run simultaneously, but will reduce the inference speed of each agent. Also uses more memory since it needs two whole context windows.)
- `--ctx-size`: `131072`   (Set this to the number of slots, multiplied by 65536. This is the required context size (the system will work below this in memory constrained environments, but it may forget some of its initial context if it reaches capacity (through the use of context shifting)).)
- `--cache-ram` : `4096` (Allows 4GB of system RAM to save context checkpoints, this means the system can cache context and not have to prompt-process every time.)
- `-b`: `2048` (sets the batch size for inference, reduce or increase if there is VRAM to spare or if the system is running out of VRAM)
- `-ub`: `512` (same as above, should be 2 or 4 times smaller than the batch size)
- `--reasoning` : `on`     (Enables reasoning mode, which allows the model to perform more complex reasoning tasks.)


On the right hand side, the user can select and add models from the file system. Clicking `Add Model` opens up a prompt to provide a `GGUF` file, commonly downloaded from https://huggingface.co. Once adding a model, you can provide an alias for the model, which is a user-friendly name. Below are two sections, the Per-Model Arguments and the Sampling Parameters. The Sampling Parameters are used to control the model's output. Inside the huggingface.co model card from the model that the user downloaded, it says which sampling parameters are recommended for that model. The most important are the top 3 (temperature, top p and top k). The Per Model Arguments section is used to provide model-specific command line arguments. These are more complex, only mess with these if you know what you are doing.

You should ask the user what their graphics card is or LLM hardware acceleration. The following are recommended models for use with CapitalAgents. All models are from the Gemma 4 series, and use Quantisation Aware Training (meaning they hold accuracy better than other quantised models). The models are as follows:

For GPUs with very low VRAM or no GPU at all: https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF is recommended. 
For GPUs with 8GB of VRAM: https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF is recommended.
For GPUs with 12-16GB of VRAM: https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF is recommended.
For GPUs with >20GB of VRAM: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF is recommended.
Gemma 4 26B-A4B will be extremely fast on 24GB of VRAM, but advise the user that because it is a mixture of experts model, it can run on lower VRAM GPUs, as low as 8GB. Advise that it will be a little slower and use a lot more VRAM, however. To use this large model, it is recommended that the user activates the `--fit` flag with a `--fit-target` of 2500. This large allowance is necessary, because sentiment analysis uses the GPU for inference, and this model needs to exist in VRAM (the sentiment model uses 1.5-3GB VRAM based on your GPU (different GPUs use different engines)). 2500 is the recommendation but tell the user by monitoring how much VRAM is being used while a boardroom analysis is running, they can change `fit-target` accordingly to use as much of their VRAM as possible.

You should use your best judgement to recommend the best model for the user based on their hardware. 


Tell the user that if they are having problems, consider using an older version of llamacpp - this was tested as of 15 September 2026.