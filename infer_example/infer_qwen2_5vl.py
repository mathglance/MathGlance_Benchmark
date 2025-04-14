
import argparse
import os
import json
import math
from tqdm import tqdm
from typing import List, Dict
import torch

import PIL.Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

prompt_choice = {
    "none": "",
    "single": "Answer the question using a single word or phrase.",
    "multimath": "\nPlease reason step by step, and put your final answer within \\boxed{}.\nEach step is placed on a new line, using the following format: \nStep X (Mathematical theorem/basis used): Detailed solution steps. \nAnswer: \\boxed{}"
}

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

system_prompt = "Let me provide some key details about the image, including its foreground and background colors, the number of objects, and their respective spatial locations. The foreground elements consist of distinct geometric shapes, each with specific sizes, orientations, and positions, while the background color provides contrast to ensure clear visibility. You can refer to this information to answer the following question."
def main(args, chunk_size=512):
    
    dtype = torch.bfloat16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=dtype, device_map="auto"
)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # load data
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    answers_file_dir = os.path.dirname(answers_file)
    os.makedirs(answers_file_dir, exist_ok=True)
    answers_file = os.path.join(answers_file_dir, f"{args.chunk_idx}_{os.path.basename(answers_file)}")
    ans_file = open(answers_file, "w")

    for line in tqdm(questions, total=len(questions)):
        query= line['question']
        query += prompt_choice[args.prompt]
        image_file = line["image"]
        image_path = os.path.join(args.image_folder, image_file)

        messages = [
                        {
                            # "role": "system",
                            # "content": [
                            #     {
                            #         "type": "text", "text": system_prompt
                            #     },
                            # ],
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {"type": "text", "text": query},
                            ],
                        }
                    ]
        text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        inputs = inputs.to(model.device, dtype=dtype)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                    ]
            outputs = processor.batch_decode(
                                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )


        response = outputs[-1]
        line['model_output'] = outputs
        line['response'] = response

        ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--prompt", type=str, choices=['none', 'single', 'multimath'])
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="chunk size for the model for prefiiling. "
                             "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                             "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    main(args)
