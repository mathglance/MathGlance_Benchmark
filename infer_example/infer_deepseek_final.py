import argparse
import os
import json
import math
from tqdm import tqdm
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
import PIL.Image
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor


system_prompt = "Let me provide some key details about the image, including its foreground and background colors, the number of objects, and their respective spatial locations. The foreground elements consist of distinct geometric shapes, each with specific sizes, orientations, and positions, while the background color provides contrast to ensure clear visibility. You can refer to this information to answer the following question."

prompt_choice = {
    "none": "",
    "single": "Answer the question using a single word or phrase.",
    "multimath": "\nPlease reason step by step, and put your final answer within \\boxed{}.\nEach step is placed on a new line, using the following format: \nStep X (Mathematical theorem/basis used): Detailed solution steps. \nAnswer: \\boxed{}"
}
# from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    pil_images = []
    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
    return pil_images


def main(args, chunk_size=512):
    
    dtype = torch.bfloat16

    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    vl_gpt = vl_gpt.cuda().eval()

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
        if line['problem_version'] in ["task_grd"]:
            content = f"<image>\n <|ref|>{query}<|/ref|>."
        else:
            content = f"<image>\n {query}"
        conversation = [
                            {
                                "role": "<|User|>",
                                "content": content,
                                "images": [
                                    f"{os.path.join(args.image_folder, image_file)}",
                                ],
                            },
                            {"role": "<|Assistant|>", "content": ""},
                        ]
        
        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)

        prepare_inputs = vl_chat_processor.__call__(
                            conversations=conversation,
                            images=pil_images,
                            force_batchify=True,
                            # system_prompt=system_prompt
                            system_prompt=''
                        ).to(vl_gpt.device, dtype=dtype)

        with torch.no_grad():

            if chunk_size == -1:
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                past_key_values = None
            else:
                # incremental_prefilling when using 40G GPU for vl2-small
                inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=chunk_size
                )

            # run the model to get the response
            outputs = vl_gpt.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,

                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,

                do_sample=False,
                temperature=0,
                top_p=None,
                # repetition_penalty=1.1,
                num_beams=1,

                use_cache=True,
            )

            outputs = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)

        response = outputs
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
