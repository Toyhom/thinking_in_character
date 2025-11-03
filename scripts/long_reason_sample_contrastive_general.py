import json
import argparse
import copy
from transformers import AutoTokenizer

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

def generate_responses(rolebench_data, model_path):
    from vllm import LLM, SamplingParams
    import torch
    num_gpus = torch.cuda.device_count()
    llm = LLM(model=model_path, max_model_len=8000, dtype="bfloat16", tensor_parallel_size=num_gpus, enable_prefix_caching=True, cpu_offload_gb=0, gpu_memory_utilization=0.8, max_num_seqs=100)
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.98,
        top_k=500,
        max_tokens=4096,
        n=1,
        stop=["<|eot_id|>","<|reserved_special_token"]
    )
    model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    system_specific = ' The thought process generated this time must conform to the following requirements to match the character and the atmosphere of the current Context Type.\nStyle Core: Vivid and imaginative / Emotionally resonant / Intuition-driven and associative\nFocus: The thought process should primarily reflect the character\'s emotional reactions / personal values / past experiences / peculiar associations.\nLanguage Features: The language used in the thoughts should align with the character profile, exhibiting features like rich in detail / assertive tone / specific metaphors.\nContext Matching: The depth and complexity of the reasoning should be appropriate for the current context thoughts can be deeper analysis is needed in a serious situation.'
    system_general = ' The thought process generated this time must conform to the following requirements to match the character and the atmosphere of the current Context Type.\nStyle Core: Vivid and imaginative / Rigorous and logical / Intuition-driven and associative\nFocus: The thought process should primarily reflect the character\'s personal values / pragmatic considerations / peculiar associations.\nLanguage Features: The language used in the thoughts should align with the character profile, exhibiting features like concise and direct / hesitant tone / specific slang.\nContext Matching: The depth and complexity of the reasoning should be appropriate for the current context thoughts can be simple and associative in a lighthearted context.'
    prompts_positive = []
    for item in rolebench_data:
        system_prompt = item['conversations'][0]['value'] + system_general
        user_prompt = next(conv['value'] for conv in item['conversations'] if conv['from'] == 'human')
        label_response = item['conversations'][2]['value']
        prompt = format_prompt(system_prompt, user_prompt, label_response, model_tokenizer)
        prompts_positive.append(prompt)
    prompts_negative = []
    for item in rolebench_data:
        system_prompt = item['conversations'][0]['value'] + system_specific
        user_prompt = next(conv['value'] for conv in item['conversations'] if conv['from'] == 'human')
        label_response = item['conversations'][2]['value']
        prompt = format_prompt(system_prompt, user_prompt, label_response, model_tokenizer)
        prompts_negative.append(prompt)
    outputs_positive = llm.generate(prompts_positive, sampling_params)
    outputs_negative = llm.generate(prompts_negative, sampling_params)
    updated_rolebench_data = []
    rolebench_data_split = rolebench_data
    for i, output in enumerate(outputs_positive):
        new_item = copy.deepcopy(rolebench_data_split[i])
        del new_item['conversations'][-1]
        response_positive = outputs_positive[i].outputs[0]
        generated_text = response_positive.text.strip()
        if "<|reserved_special_token" in generated_text:
            generated_text = generated_text.split("<|reserved_special_token")[0].strip()
        new_item['chosen'] = {
            "from": "gpt",
            "value": generated_text
        }
        response_negative = outputs_negative[i].outputs[0]
        generated_text = response_negative.text.strip()
        if "<|reserved_special_token" in generated_text:
            generated_text = generated_text.split("<|reserved_special_token")[0].strip()
        new_item['rejected'] = {
            "from": "gpt",
            "value": generated_text
        }
        label_response = rolebench_data_split[i]['conversations'][2]['value']
        new_item["label"] = label_response
        updated_rolebench_data.append(new_item)
    return updated_rolebench_data

def process_and_generate(data_path, model_path, output_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        rolebench_data = json.load(f)
    final_data = generate_responses(rolebench_data, model_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {output_path}")
    return final_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate general contrastive samples for roleplay data.")
    parser.add_argument('--data_path', type=str, required=True, help='Input data JSON file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--output_path', type=str, required=True, help='Output JSON file')
    args = parser.parse_args()
    process_and_generate(args.data_path, args.model_path, args.output_path) 