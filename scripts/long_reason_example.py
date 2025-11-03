import json
import argparse
import copy
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import torch

# Set device count automatically
num_gpus = torch.cuda.device_count()

# Format input for roleplay scoring
def format_input(example):
    system = example['role']
    role_name = system.split(', your description is: ')[0].split('You are ')[1].strip()
    role_description = system.split(', your description is: ')[1].split('. Now')[0].strip()
    input_text_format = '''<|im_start|>system\n{{'name': '{role_name}', 'description': '{role_description}'}}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>'''
    input_text = input_text_format.format(role_name=role_name, role_description=role_description, query=example['context'], response=example['model_output'])
    return input_text

# Score roleplay quality

def score_roleplay(text, base_model, tokenizer):
    input_text = text
    max_seq_length = 4096
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_seq_length, add_special_tokens=False).to(base_model.pretrained_model.device)
    with torch.no_grad():
        _, _, values = base_model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        attention_mask = inputs["attention_mask"]
        effective_lengths = attention_mask.sum(dim=-1, keepdim=True) - 1
        score = values.gather(dim=-1, index=effective_lengths).squeeze(-1)
        score = score.item()
    return score

def score_with_cache(system, query, response, base_model, tokenizer):
    roleplay_text = format_input({
        'role': system,
        'context': query,
        'model_output': response,
    })
    roleplay_scores = score_roleplay(roleplay_text, base_model, tokenizer)
    return roleplay_scores

def format_prompt(system_prompt, user_prompt, label, model_tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = model_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def generate_responses(rolebench_data, model_path, tokenizer_path):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, max_model_len=10000, dtype="bfloat16", tensor_parallel_size=num_gpus, enable_prefix_caching=True, cpu_offload_gb=0, gpu_memory_utilization=0.9, max_num_seqs=100)
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.98,
        top_k=500,
        max_tokens=4096,
        n=1,
    )
    model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompts = []
    index_list = []
    for index, item in enumerate(rolebench_data):
        system_prompt = item['conversations'][0]['value']
        user_prompt = next(conv['value'] for conv in item['conversations'] if conv['from'] == 'human')
        label_response = item['conversations'][2]['value']
        role_name = system_prompt.split(', your description is: ')[0].split('You are ')[1].strip()
        if role_name != "Peter Parker":
            continue
        prompt = format_prompt(system_prompt, user_prompt, label_response, model_tokenizer)
        prompts.append(prompt)
        index_list.append(index)
    outputs = llm.generate(prompts, sampling_params)
    updated_rolebench_data = []
    rolebench_data_split = rolebench_data
    for i, output in enumerate(outputs):
        for response in output.outputs:
            generated_text = response.text.strip()
            if "<|reserved_special_token" in generated_text:
                generated_text = generated_text.split("<|reserved_special_token")[0].strip()
            if "</think>" in generated_text:
                generated_text_score = generated_text.split("</think>")[-1]
                generated_text_score = generated_text_score.replace("\n","").replace("\"","")
            else:
                continue
            new_item = copy.deepcopy(rolebench_data_split[index_list[i]])
            new_item['conversations'][-1] = {
                "from": "gpt",
                "value": "<think>\n" + generated_text
            }
            updated_rolebench_data.append(new_item)
    return updated_rolebench_data

def process_and_generate(data_path, model_path, tokenizer_path, output_path):
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        rolebench_data = json.load(f)
    final_data = generate_responses(rolebench_data, model_path, tokenizer_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {output_path}")
    return final_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate long reason examples for roleplay data.")
    parser.add_argument('--data_path', type=str, required=True, help='Input data JSON file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer', default=None)
    parser.add_argument('--output_path', type=str, required=True, help='Output JSON file')
    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    process_and_generate(args.data_path, args.model_path, args.tokenizer_path, args.output_path) 
