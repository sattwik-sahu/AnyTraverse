# NVLM

##  From NVIDIA

> Today (September 17th, 2024), we introduce NVLM 1.0, a family of frontier-class multimodal large language models (LLMs) that achieve state-of-the-art results on vision-language tasks, rivaling the leading proprietary models (e.g., GPT-4o) and open-access models (e.g., Llama 3-V 405B and InternVL 2). Remarkably, NVLM 1.0 shows improved text-only performance over its LLM backbone after multimodal training.
>
>
> In this repo, we are open-sourcing NVLM-1.0-D-72B (decoder-only architecture), the decoder-only model weights and code for the community.

## Reasons for Abandon

- Man this is **BIG!** Really, really huge (😉)
- Cannot load this. It requires 46 shards of weights to download, all of them `~4 Gb` in size!
- Macaw GPU cannot take this load 🥲 _And I care about the poor fellow_.

> NVLM? Not today, but someday 🪨
