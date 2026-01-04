# `AutoRound` Quantization

`llm-compressor` supports [AutoRound](https://aclanthology.org/2024.findings-emnlp.662.pdf), an advanced quantization technique that delivers **high-accuracy**, **low-bit quantization**. The quantized results are fully compatible with `compressed-tensors` and can be served directly with vLLM.

AutoRound introduces three trainable parameters (V, α, and β) to optimize rounding values and clipping ranges during quantization. The method processes each decoder layer sequentially, using block-wise output reconstruction error as the training objective to fine-tune these parameters. This approach combines the efficiency of post-training quantization with the adaptability of parameter tuning, delivering robust compression for large language models while maintaining strong performance.

## Installation

To get started, install:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
git clone https://github.com/vllm-project/compressed-tensors
cd compressed-tensors
pip install -e .
```

## Quickstart

The example includes an end-to-end script for applying the AutoRound quantization algorithm.

```bash
python3 mistral3_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-NVFP4-AutoRound` is ready to be loaded into vLLM.

### 4) Evaluate Accuracy

With the model created, we can now load and run in vLLM (after installing).

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-NVFP4-AutoRound")
```

We can evaluate accuracy with `lm_eval` (`pip install lm-eval==0.4.9.1`):
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

Run the following to test accuracy on GSM-8K:

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-NVFP4-AutoRound",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 1000 \
  --batch_size 'auto'
```

We can see the resulting scores look good!

```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|                                                         
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|                                                         
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.697|±  |0.0145|                                                         
|     |       |strict-match    |     5|exact_match|↑  |0.640|±  |0.0152|
```
> Note: quantized model accuracy may vary slightly due to nondeterminism.

### Known Issues
Currently, `llm-compressor` supports applying AutoRound only on the `wNa16` quantization schemes. Support for additional schemes is planned. You can follow progress in the [RFC](https://github.com/vllm-project/llm-compressor/issues/1968).

### Questions or Feature Request?

Please open up an issue on [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) or [intel/auto-round](https://github.com/intel/auto-round).
