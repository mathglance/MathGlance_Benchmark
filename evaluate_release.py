"""
Description: This is an official release version of evaluation code for MATHGLANCE benchmark, which presents in
``MATHGLANCE: Multimodal Large Language Models Do Not Know Where to Look in Mathematical Diagrams, arXiv 2025``
Module name: evaluate_release.py, version: 1.0.0
Function: main file of this module

Authors: AI4Math Team - Wei Tang, Shan Zhang, and Aotian Chen
Creation Date: Jan 10, 2025
Last Modified: April 12, 2025
Version: release - V1.0

Modification History:
- April 12, 2025 - Wei Tang - release version - V1.0
"""

from collections import defaultdict
import re
from tqdm import tqdm
import time
import json
from utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number
import os
import argparse
import torch
from PIL import Image
from grd_eval_utils import GRDUtils
from extract_answer_utils import extract_answer_qwen, extract_answer_llava, extract_answer_other_models

def evaluate(answer_file, regen_answer=False):
    lines = load_jsonl(answer_file)
    grd_utils = GRDUtils(args.image_folder, args.model_type, args.iou)

    for line in tqdm(lines, desc='gen_correct'):
        if line['problem_version'] in ["task_cls", "task_cnt", "task_rlat"]:
            raw_exampe = id_raw[line['sample_index']]

            gt_answer = str(raw_exampe['answer']).strip()
            gt_answer_list = gt_answer.split('\n')
            if len(gt_answer_list) > 1:
                line['correct'] = False
                continue
            gt_answer_value = gt_answer

            if 'model_answer' not in line or regen_answer:
                if line['response'] == None:
                    model_answer == ''
                else:
                    model_answer = line['response'].strip()
                # 
                for c in 'ABCDE':
                    if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                        model_answer = c
                
                # ################################################
                # match pattern for choice question（cls/cnt/rlat)
                if args.model_type in ["qwen"]:
                    model_answer = extract_answer_qwen(model_answer)
                elif args.model_type in ["ours"] or args.model_type in ["llava"]:
                    model_answer = extract_answer_llava(model_answer)
                elif args.model_type in ["internvl", "internvl2", "internvl_X_2_5", "gpt4o", "gpto1"]:
                    question_text = raw_exampe['question']
                    model_answer = extract_answer_other_models(model_answer, question_text, args.model_type)
                # #################################################

                if is_number(model_answer.split('is ')[-1].rstrip('.')):
                    model_answer = model_answer.split('is ')[-1].rstrip('.')
                if 'oxed{' not in model_answer:
                    for flag in ['Answer:', 'the answer is']:
                        raw_model_answer = model_answer
                        model_answer = model_answer.split(flag)[-1].strip()
                        if flag in raw_model_answer:
                            model_answer = model_answer.split('\n')[0].split('. ')[0]
                        flag = flag.replace('the', 'The')
                        raw_model_answer = model_answer
                        model_answer = model_answer.split(flag)[-1].strip()
                        if flag in raw_model_answer:
                            model_answer = model_answer.split('\n')[0].split('. ')[0]
                elif model_answer.count('oxed{') > 1:
                    model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
                    
                model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').lstrip('option').strip()
                line['model_answer'] = model_answer
            else:
                model_answer = line['model_answer']
            line['correct'] = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
        elif line['problem_version'] in ["task_grd"]:
            grd_utils.eval(line)
        else:
            raise ValueError(f"no task of {line['problem_version']}")
    save_jsonl(answer_file, lines, t_stamp=False)


# #################################################
# # styles for final statistics
def math_level_subject_acc(answer_file, cot_rec_err_threhold=150):
    print(f"Current test on {answer_file}")
    lines = load_jsonl(answer_file)
    
    ###############################################
    # 1. statistics according to task type
    # using defaultdict for creating groups
    grouped_data = defaultdict(list)

    for item in lines:
        source = item['problem_version']
        grouped_data[source].append(item)
    
    all_results_dict = {}
    
    for source, group in grouped_data.items():
        results_dict = {}
        for line in tqdm(group, desc='math_level_subject_acc'):
            correct = line['correct']
            raw_exampe = id_raw[line['sample_index']]
            type = raw_exampe['problem_version']
            
            for key in [
                '-all', 
                f'-{type}'
                ]:
                if key not in results_dict:
                    results_dict[key] = [0,0]
                results_dict[key][0] += 1 if correct else 0
                results_dict[key][1] += 1


        for key in results_dict.keys():
            if results_dict[key][1] == 0:
                results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
            else:
                results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0]/ max(results_dict[key][1], 1)*100, 2)}%'


        results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
        all_results_dict[source] = results_dict
    order = ['task_cls', 'task_cnt', 'task_grd', 'task_rlat']
    all_results_dict = {k: all_results_dict[k] for k in order}
    with open(answer_file.replace('.json', '_result.log'), 'w') as f:
        f.write("***************************General Statistic (task)***********************\n")
        json.dump(all_results_dict, f, indent=4, ensure_ascii=False)
        f.write("\n****************************************************************************\n\n")

    ###############################################
    # 2. statistics according to different sources
    # using defaultdict for creating groups
    grouped_data = defaultdict(list)

    for item in lines:
        source = item['source']
        grouped_data[source].append(item)
    
    all_results_dict = {}
    
    for source, group in grouped_data.items():
        results_dict = {}
        for line in tqdm(group, desc='math_level_subject_acc'):
            correct = line['correct']
            raw_exampe = id_raw[line['sample_index']]
            type = raw_exampe['problem_version']
            
            for key in [
                '-all', 
                f'-{type}'
                ]:
                if key not in results_dict:
                    results_dict[key] = [0,0]
                results_dict[key][0] += 1 if correct else 0
                results_dict[key][1] += 1


        for key in results_dict.keys():
            if results_dict[key][1] == 0:
                results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
            else:
                results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0]/ max(results_dict[key][1], 1)*100, 2)}%'


        results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
        all_results_dict[source] = results_dict
    with open(answer_file.replace('.json', '_result.log'), 'a+') as f:
        f.write("***************************General Statistic (source)***********************\n")
        json.dump(all_results_dict, f, indent=4, ensure_ascii=False)
        f.write("\n****************************************************************************\n\n")

    ###############################################
    # 3. statistics according to level
    # using defaultdict for creating groups
    grouped_data = defaultdict(list)

    for item in lines:
        source = item['metadata']['level']
        grouped_data[source].append(item)
    
    all_results_dict = {}
    
    for source, group in grouped_data.items():
        results_dict = {}
        for line in tqdm(group, desc='math_level_subject_acc'):
            correct = line['correct']
            raw_exampe = id_raw[line['sample_index']]
            type = raw_exampe['problem_version']
            
            for key in [
                '-all', 
                f'-{type}'
                ]:
                if key not in results_dict:
                    results_dict[key] = [0,0]
                results_dict[key][0] += 1 if correct else 0
                results_dict[key][1] += 1


        for key in results_dict.keys():
            if results_dict[key][1] == 0:
                results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
            else:
                results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0]/ max(results_dict[key][1], 1)*100, 2)}%'


        results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
        all_results_dict[source] = results_dict
    with open(answer_file.replace('.json', '_result.log'), 'a+') as f:
        f.write("***************************Detailed Statistic (level)***********************\n")
        json.dump(all_results_dict, f, indent=4, ensure_ascii=False)
        f.write("\n****************************************************************************\n\n")
    
    ###############################################
    # 4. statistics according to rec err / cot err
    # using defaultdict for creating groups
    grouped_data = defaultdict(list)
    grouped_plane_geometry_data = defaultdict(list)

    for item in lines:
        source = item['correct']
        if  item["source"] == "plane_geometry":
            grouped_plane_geometry_data[source].append(item)
        grouped_data[source].append(item)
    
    results_dict = {}

    results_dict['-cot_err_plane_geometry'] = 0
    results_dict['-rec_err_plane_geometry'] = 0

    cot_err_list = []
    rec_err_list = []
    for line in tqdm(grouped_plane_geometry_data[False], desc='math_level_subject_acc'):
        if len(line['response']) > cot_rec_err_threhold:
            results_dict['-cot_err_plane_geometry'] += 1
            cot_err_list.append(line)
        else:
            results_dict['-rec_err_plane_geometry'] += 1
            rec_err_list.append(line)
    total= len(grouped_plane_geometry_data[False]) +  len(grouped_plane_geometry_data[True])
    results_dict['-cot_err_plane_geometry'] = f"{results_dict['-cot_err_plane_geometry']}/{total}={round(results_dict['-cot_err_plane_geometry']/ total*100, 2)}%"
    results_dict['-rec_err_plane_geometry'] = f"{results_dict['-rec_err_plane_geometry']}/{total}={round(results_dict['-rec_err_plane_geometry']/ total*100, 2)}%"
    with open(answer_file.replace('.json', '_result.log'), 'a+') as f:
        f.write("***************************Detailed Statistic (cot&rec)***********************\n")
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
        f.write("\n****************************************************************************\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", type=str, default="benchmark_release/data_final/annotation/FINAL_COMBINE_MIX_V2.0.0.json")
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--answers_file", type=str, default="output.jsonl")
    parser.add_argument("--model_type", type=str, default="ours")
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--cot_rec_err_threhold", type=float, default=150)
    args = parser.parse_args()

    id_raw = {example['sample_index']: example for example in json.load(open(args.gt_file))}

    evaluate(args.answers_file, True)
    math_level_subject_acc(args.answers_file, args.cot_rec_err_threhold)
