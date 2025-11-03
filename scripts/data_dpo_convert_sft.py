import json
import random
import argparse
import os

def main(args):
    # Load data from two json files
    with open(args.input_file_1, 'r', encoding='utf-8') as f:
        data_1 = json.load(f)
    with open(args.input_file_2, 'r', encoding='utf-8') as f:
        data_2 = json.load(f)

    datas = data_1 + data_2
    new_datas = []

    # For each data, add both 'chosen' and 'rejected' conversations
    for data in datas:
        conversations = data['conversations'].copy()
        conversations.append(data["chosen"])
        new_datas.append({"conversations": conversations})

        conversations = data['conversations'].copy()
        conversations.append(data["rejected"])
        new_datas.append({"conversations": conversations})

    # Sample 1/10 of the data
    new_datas = random.sample(new_datas, len(new_datas) // 10)

    # Save the processed data
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(new_datas, f, ensure_ascii=False)

    print(f"Processed data saved to {args.output_file}, total samples: {len(new_datas)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DPO data to SFT format.")
    parser.add_argument('--input_file_1', type=str, required=True, help='Path to the first input JSON file')
    parser.add_argument('--input_file_2', type=str, required=True, help='Path to the second input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output JSON file')
    args = parser.parse_args()
    main(args) 