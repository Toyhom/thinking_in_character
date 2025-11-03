# Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning

## Project Overview
This repository provides the official code and data pipeline for the paper "Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning". It includes scripts for data construction, model training.

## Directory Structure
```
thinking_in_character/
│
├── data/                # Store raw and processed data
├── scripts/             # All Python scripts for data processing
│   ├── data_dpo_convert_sft.py
│   ├── long_reason_example.py
│   ├── long_reason_sample_contrastive_general.py
│   └── long_reason_sample_contrastive_specific.py
├── configs/             # YAML config files for training
│   ├── llama3_lora_reasonsft_aware_rbench.yaml
│   └── llama3_lora_reasonsft_aware_rbench_rso.yaml
├── run_data_process.sh  # One-click data construction script
├── requirements.txt     # Python dependencies
└── README.md
```

## 0.Environment Setup
Install the [llama_factory](https://github.com/hiyouga/LLaMA-Factory).

## 1.Data Construction
All data processing can be done with a single shell script. You can specify the data and model directories as arguments:

Our datas are available on HuggingFace:
- [Datas](https://huggingface.co/datasets/Toyhom/thinking_in_character_datas)

```bash
bash run_data_process.sh data/ models/
```
- `<data_dir>`: Directory for input/output data (default: `data/`)
- `<model_dir>`: Directory for model checkpoints (default: `models/`)

Each Python script in `scripts/` also supports command-line arguments for flexible usage.

## 2.Training
We use [llama_factory](https://github.com/hiyouga/LLaMA-Factory) for model training.

### Dataset Configuration
Copy your dataset info file to the LLaMA-Factory directory:
```bash
cp data/dataset_info.json /path/to/LLaMA-Factory/data/
```

### Stage 1: Role Identity Activation Training
```bash
llamafactory-cli train configs/llama3_lora_reasonsft_aware_rbench.yaml
```

### Model Merge
If you need to merge LoRA weights:
```bash
llamafactory-cli export configs/llama3_lora_sft.yaml
```

### Stage 2: Reasoning Style Optimization Training
```bash
llamafactory-cli train configs/llama3_lora_reasonsft_aware_rbench_rso.yaml
```

## 3.Inference Examples

### vLLM Inference
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "../role_bench_result/llama3_lora_reasonsft_aware_rbench_contrastive"

llm_engine = LLM(
    model=model_path,
    max_model_len=8000,
    dtype="bfloat16",
    tensor_parallel_size=4,  # Set according to your hardware
    enable_prefix_caching=True,
    cpu_offload_gb=0,
    gpu_memory_utilization=0.9,
    max_num_seqs=100,
    enable_lora=True,
    max_lora_rank=128,
    trust_remote_code=True
)
model_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=4096,
    stop=["<|eot_id|>", "<|reserved_special_token"]
)

messages = [
    {"role": "system", "content": "You are John Coffey, your description is: a gentle giant with extraordinary healing abilities who is wrongfully convicted of murder and sentenced to death row. ..."},
    {"role": "user", "content": "John Coffey, what are some examples of the profound changes you bring about in the lives of the prison guards and inmates?"}
]

prompt = model_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = llm_engine.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

### Transformers Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "../role_bench_result/llama3_lora_reasonsft_aware_rbench_contrastive"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

messages = [
    {"role": "system", "content": "You are John Coffey, your description is: a gentle giant with extraordinary healing abilities who is wrongfully convicted of murder and sentenced to death row. ..."},
    {"role": "user", "content": "John Coffey, what are some examples of the profound changes you bring about in the lives of the prison guards and inmates?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```





