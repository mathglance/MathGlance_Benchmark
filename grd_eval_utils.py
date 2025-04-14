"""
Description: This is an official release version of evaluation code for MATHGLANCE benchmark, which presents in
``MATHGLANCE: Multimodal Large Language Models Do Not Know Where to Look in Mathematical Diagrams, arXiv 2025``
Module name: grd_eval_utils.py, version: 1.0.0
Function: differt methods for handle grounding tasks and get the iou and results from model response

Authors: AI4Math Team - Wei Tang, Shan Zhang, and Aotian Chen
Creation Date: Jan 10, 2025
Last Modified: April 12, 2025
Version: release - V1.0

Modification History:
- April 12, 2025 - Wei Tang - release version - V1.0
"""
from PIL import Image
import torch
from torchvision.ops.boxes import box_area
import os
import re
import torch.nn.functional as F
import numpy as np

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def extract_bbox_internvl(response):
    """
    extract bbox from internvl answer
    
    Args:
        response: text response from internvl
    
    Returns:
        bbox: bbox in format [x1, y1, x2, y2] or None
    """
    
    # check for format (x1, y1, x2, y2) or [x1, y1, x2, y2]
    bracket_patterns = [
        r'\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)', # (x1,y1,x2,y2)
        r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'  # [x1,y1,x2,y2]
    ]
    
    for pattern in bracket_patterns:
        matches = re.search(pattern, response)
        if matches:
            return [float(matches.group(1)), float(matches.group(2)), 
                   float(matches.group(3)), float(matches.group(4))]
    
    # check for corner pattern, e.g. "Top-Left Corner: (x1, y1)" or "Bottom-Right Corner: (x2, y2)"
    corner_pattern = re.compile(r'Top-left corner:\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\).*?Bottom-right corner:\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)', re.DOTALL)
    corner_match = corner_pattern.search(response)
    
    if corner_match:
        return [float(corner_match.group(1)), float(corner_match.group(2)),
                float(corner_match.group(3)), float(corner_match.group(4))]
    
    # if not find formatted coordinates, try obtain 4 digits
    numbers = re.findall(r'(\d+(?:\.\d+)?)', response)
    if len(numbers) >= 4:
        return [float(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[3])]
    
    # if all methods failed, return None
    return None

def extract_bbox_internvl2(response):
    """
    Extracts the bounding box coordinates from the response of the InternVL2 model.
    
    Args:
        response: text response from the model
    
    Returns:
        list: list of four numbers representing the bounding box coordinates (x1, y1, x2, y2) or None
    """
    
    # check for format (x1, y1, x2, y2) or [x1, y1, x2, y2]
    brackets_patterns = [
        r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]',
        r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
    ]
    
    for pattern in brackets_patterns:
        matches = re.search(pattern, response)
        if matches:
            return [float(matches.group(1)), float(matches.group(2)), 
                   float(matches.group(3)), float(matches.group(4))]
    
    # check for corner pattern, e.g. "Top-Left Corner: (x1, y1)" or "Bottom-Right Corner: (x2, y2)"
    corner_pattern = re.compile(r'Top-Left Corner.*?\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\).*?Bottom-Right Corner.*?\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)', re.DOTALL)
    corner_match = corner_pattern.search(response)
    
    if corner_match:
        return [float(corner_match.group(1)), float(corner_match.group(2)),
                float(corner_match.group(3)), float(corner_match.group(4))]
    
    # check for latex-like pattern, e.g.  "\( (-1, -1) \)" or "\( (1, 1) \)"
    latex_pattern = re.compile(r'\\{1,2}\(\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*\\{1,2}\).*?\\{1,2}\(\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*\\{1,2}\)', re.DOTALL)
    latex_match = latex_pattern.search(response)
    
    if latex_match:
        return [float(latex_match.group(1)), float(latex_match.group(2)),
                float(latex_match.group(3)), float(latex_match.group(4))]
    
    # if all methods fail, try extracting all coordinates from the text
    # here we assume bbox coner exists as (x,y) format
    coord_pairs = re.findall(r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', response)
    
    if len(coord_pairs) >= 2:
        # assume the first pair is the top-left corner and the last pair is the bottom-right corner
        return [float(coord_pairs[0][0]), float(coord_pairs[0][1]),
                float(coord_pairs[-1][0]), float(coord_pairs[-1][1])]
    
    # if still not found, try to find any 4 digits
    numbers = re.findall(r'(-?\d+(?:\.\d+)?)', response)
    if len(numbers) >= 4:
        # try to find formats like "x1 = ... = 350" 
        x1_match = re.search(r'x1\s*=.*?=\s*(\d+(?:\.\d+)?)', response)
        y1_match = re.search(r'y1\s*=.*?=\s*(\d+(?:\.\d+)?)', response)
        x2_match = re.search(r'x2\s*=.*?=\s*(\d+(?:\.\d+)?)', response)
        y2_match = re.search(r'y2\s*=.*?=\s*(\d+(?:\.\d+)?)', response)
        
        if x1_match and y1_match and x2_match and y2_match:
            return [float(x1_match.group(1)), float(y1_match.group(1)),
                    float(x2_match.group(1)), float(y2_match.group(1))]
        
        # if there are no clear x1,y1,x2,y2, using the former 4 digits as the results
        return [float(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[3])]
    
    # if all methods failed, return None
    return None


def extract_bbox_gpt4o(response):
    """
    extract bbox from gpt4o's answer
    
    Args:
        response: text response from gpt4o
    
    Returns:
        bbox: [x1, y1, x2, y2] or None
    """
    # 1. check for if model say it can not provide coordinates
    if re.search(r"I'm unable to provide|unable to provide|can't provide", response, re.IGNORECASE):
        return None
    
    # 2. check for corner pattern, e.g. "Top-Left Corner: (x1, y1)" or "Bottom-Right Corner: (x2, y2)"
    corner_pattern = re.compile(r'(?:Top-left|Bottom-left) corner:?\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\).*?(?:Top-right|Bottom-right) corner:?\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)', re.DOTALL)
    match = corner_pattern.search(response)
    if match:
        return [float(match.group(1)), float(match.group(2)), 
                float(match.group(3)), float(match.group(4))]
    
    # 3. check for statements like "Minimum x: x1", "Maximum x: x2", "Minimum y: y1", "Maximum y: y2" 
    min_x = re.search(r'Minimum x:?\s*(\d+(?:\.\d+)?)', response)
    max_x = re.search(r'Maximum x:?\s*(\d+(?:\.\d+)?)', response)
    min_y = re.search(r'Minimum y:?\s*(\d+(?:\.\d+)?)', response)
    max_y = re.search(r'Maximum y:?\s*(\d+(?:\.\d+)?)', response)
    
    if min_x and max_x and min_y and max_y:
        return [float(min_x.group(1)), float(min_y.group(1)), 
                float(max_x.group(1)), float(max_y.group(1))]
    
    # 4. try to extract from pair digits, e.g., "(0, 1)" or "(4, 4)"
    coord_pairs = re.findall(r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)', response)
    if len(coord_pairs) >= 2:
        return [float(coord_pairs[0][0]), float(coord_pairs[0][1]), 
                float(coord_pairs[-1][0]), float(coord_pairs[-1][1])]
    
    # 5. if all methods failed, return None
    return None


class GRDUtils:
    def __init__(self, image_folder, model_type, iou):
        self.image_folder = image_folder
        self.model_type = model_type
        self.iou = iou
        if self.model_type in ["qwen"]:
            self.PATTERN = re.compile(r'\((.*?)\),\((.*?)\)')
            self.AUX_PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")
        elif self.model_type in ["deepseek"]:
            self.PATTERN = re.compile(r'\d+')
        elif self.model_type in ["ours"]:
            self.PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")
        elif self.model_type in ["llava"]:
            self.PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")
        elif self.model_type in ["qwen2_5"]:
            self.PATTERN = re.compile(r'\[(\d+), (\d+), (\d+), (\d+)\]')
        elif self.model_type in ["internvl"]:
            self.PATTERN = re.compile(r'\d+')
        elif self.model_type in ["internvl2"]:
            self.PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")
        elif self.model_type in ["internvl_X_2_5"]:
            self.PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")
        elif self.model_type in ["gpt4o"]:
            self.PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")
        elif self.model_type in ["gpto1"]:
            self.PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")
        else:
            raise ValueError("Invalid model type")

    def eval(self, line):
        if self.model_type in ["qwen"]:
            self.eval_qwen(line)
        elif self.model_type in ["deepseek"]:
            self.eval_deepseek(line)
        elif self.model_type in ["ours"]:
            self.eval_ours(line)
        elif self.model_type in ["llava"]:
            self.eval_llava(line)
        elif self.model_type in ["qwen2_5"]:
            self.eval_qwen2_5(line)
        elif self.model_type in ["internvl"]:
            self.eval_internvl(line)
        elif self.model_type in ["internvl2"]:
            self.eval_internvl2(line)
        elif self.model_type in ["internvl_X_2_5"]:
            self.eval_internvl_x_2_5(line)
        elif self.model_type in ["gpt4o"]:
            self.eval_gpt4o(line)
        elif self.model_type in ["gpto1"]:
            self.eval_gpto1(line)
        else:
            raise ValueError("Invalid model type")

        
    
    def eval_qwen(self, line):
        aux_flag = False
        predict_bbox = re.findall(self.PATTERN, line['response'])
        if predict_bbox == []:
            predict_bbox = re.findall(self.AUX_PATTERN, line['response'])
            aux_flag = True
        
        if aux_flag:
            try:
                predict_bbox = [float(num) for num in predict_bbox]
                target_bbox = torch.tensor(line['answer'], dtype=torch.float32).view(-1, 4)
                predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).clone().detach().view(-1, 4)
                img_path = line["image"]
                image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')
                h = image.height
                w = image.width
                predict_bbox[:,0::2] *= w
                predict_bbox[:,1::2] *= h
            except:
                predict_bbox = (0., 0., 0., 0.)
        else:    
            try:
                if ',' not in predict_bbox[0][0] or ',' not in predict_bbox[0][1]:
                    predict_bbox = (0., 0., 0., 0.)
                else:
                    x1, y1 = [
                                float(tmp) for tmp in predict_bbox[0][0].split(',')
                            ]
                    x2, y2 = [
                                float(tmp) for tmp in predict_bbox[0][1].split(',')
                            ]
                    predict_bbox = (x1, y1, x2, y2)
            except:
                    predict_bbox = (0., 0., 0., 0.)
            target_bbox = torch.tensor(line['answer'],
                                        dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox,
                                                    dtype=torch.float32).view(-1, 4) / 999
            img_path = line["image"]
            image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')
            predict_bbox[:, 0::2] *= image.width
            predict_bbox[:, 1::2] *= image.height
        
        try:
            iou, _ = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
        except:
            iou = 0
        if iou >= self.iou:
            line['correct'] = True
        else:
            line['correct'] = False

    def eval_deepseek(self, line):
        predict_bbox = re.findall(self.PATTERN, line['response'])
        try:
            predict_bbox = [int(num) for num in predict_bbox]
            target_bbox = torch.tensor(line['answer'],
                                       dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox,
                                                    dtype=torch.float32).view(-1, 4) / 999
            img_path = line["image"]
            image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')
            predict_bbox[:, 0::2] *= image.width
            predict_bbox[:, 1::2] *= image.height
            iou, _ = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= self.iou:
                line['correct'] = True
            else:
                line['correct'] = False
        except:
            line['correct'] = False

    def eval_ours(self, line):
        predict_bbox = re.findall(self.PATTERN, line['response'])
        try:
            predict_bbox = [float(num) for num in predict_bbox]
            target_bbox = torch.tensor(line['answer'], dtype=torch.float32).view(-1, 4)[0]
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).clone().detach().view(-1, 4)[0]
            img_path = line["image"]
            
            image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')
            h = image.height
            w = image.width

            if h > w:
                predict_bbox[0::2] *= h
                predict_bbox[0::2] -= (h - w) // 2
                predict_bbox[1::2] *= h
                predict_bbox = F.relu(predict_bbox)
            elif h < w:
                predict_bbox[0::2] *= w
                predict_bbox[1::2] *= w
                predict_bbox[1::2] -= (w - h) // 2
                predict_bbox = F.relu(predict_bbox)
            elif h == w:
                predict_bbox[0::2] *= w
                predict_bbox[1::2] *= h
            iou, _ = box_iou(predict_bbox.unsqueeze(0), target_bbox.unsqueeze(0))
            iou = iou.item()
            if iou >= self.iou:
                line['correct'] = True
            else:
                line['correct'] = False
        except:
            line['correct'] = False

    def eval_llava(self, line):
        predict_bbox = re.findall(self.PATTERN, line['response'])
        try:
            predict_bbox = [float(num) for num in predict_bbox]
            target_bbox = torch.tensor(line['answer'], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).clone().detach().view(-1, 4)
            img_path = line["image"]
            image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')
            h = image.height
            w = image.width
            if h > w:
                predict_bbox[:,0::2] *= h
                predict_bbox[:,0::2] -= (h - w) // 2
                predict_bbox[:,1::2] *= h
                predict_bbox = F.relu(predict_bbox)
            elif h < w:
                predict_bbox[:,0::2] *= w
                predict_bbox[:,1::2] *= w
                predict_bbox[:,1::2] -= (w - h) // 2
                predict_bbox = F.relu(predict_bbox)
            elif h == w:
                predict_bbox[:,0::2] *= w
                predict_bbox[:,1::2] *= h
            iou, _ = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= self.iou:
                line['correct'] = True
            else:
                line['correct'] = False
        except:
            line['correct'] = False

    def eval_qwen2_5(self, line):
        try:
            predict_bbox = re.findall(self.PATTERN, line['response'])[-1]
            predict_bbox = [int(num) for num in predict_bbox]
            target_bbox = torch.tensor(line['answer'],
                                       dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, 
                                       dtype=torch.float32).view(-1, 4)
            img_path = line["image"]
            image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')

            iou, _ = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= self.iou:
                line['correct'] = True
            else:
                line['correct'] = False
        except:
            line['correct'] = False

    def eval_internvl(self, line):
        predict_bbox = extract_bbox_internvl(line['response'])
        
        if not predict_bbox:
            line['correct'] = False
            return
        
        try:
            target_bbox = torch.tensor(line['answer'], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            
            img_path = line["image"]
            image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')
            h = image.height
            w = image.width
            
            if torch.max(predict_bbox) <= 1.0:
                predict_bbox[:, 0::2] *= w
                predict_bbox[:, 1::2] *= h
            elif torch.max(predict_bbox) <= 1000.0 and torch.min(predict_bbox) >= 0.0:
                predict_bbox[:, 0::2] *= w / 1000.0
                predict_bbox[:, 1::2] *= h / 1000.0
                
            iou, _ = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            
            if iou >= self.iou:
                line['correct'] = True
            else:
                line['correct'] = False
                
        except Exception as e:
            line['correct'] = False

    def eval_internvl2(self, line):
        predict_bbox = extract_bbox_internvl2(line['response'])
        
        if not predict_bbox:
            line['correct'] = False
            return
        
        try:
            target_bbox = torch.tensor(line['answer'], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            
            img_path = line["image"]
            image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')
            h = image.height
            w = image.width
            
            if torch.max(torch.abs(predict_bbox)) <= 1.0:
                predict_bbox = predict_bbox + 1
                predict_bbox[:, 0::2] *= w / 2.0
                predict_bbox[:, 1::2] *= h / 2.0
            elif torch.max(predict_bbox) <= 1.0 and torch.min(predict_bbox) >= 0.0:
                predict_bbox[:, 0::2] *= w
                predict_bbox[:, 1::2] *= h
            elif torch.max(predict_bbox) <= 1000.0 and torch.min(predict_bbox) >= 0.0:
                predict_bbox[:, 0::2] *= w / 1000.0
                predict_bbox[:, 1::2] *= h / 1000.0
            
            iou, _ = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            
            if iou >= self.iou:
                line['correct'] = True
            else:
                line['correct'] = False
                
        except Exception as e:
            line['correct'] = False

    def eval_internvl_x_2_5(self, line):
        line['correct'] = False
        
        coord_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', line['response'])
        if coord_match:
            try:
                predict_bbox = [float(coord_match.group(1)), float(coord_match.group(2)), 
                               float(coord_match.group(3)), float(coord_match.group(4))]
                
                target_bbox = torch.tensor(line['answer'], dtype=torch.float32).view(-1, 4)
                predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
                
                iou, _ = box_iou(predict_bbox, target_bbox)
                iou = iou.item()
                
                if iou >= self.iou:
                    line['correct'] = True
            except Exception as e:
                line['correct'] = False

    def eval_gpt4o(self, line):
        predict_bbox = extract_bbox_gpt4o(line['response'])

        if not predict_bbox:
            line['correct'] = False
            return

        try:
            target_bbox = torch.tensor(line['answer'], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)

            img_path = line["image"]
            image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')
            h = image.height
            w = image.width

            if torch.max(predict_bbox) <= 10.0:
                scale_x = w / 5.0
                scale_y = h / 5.0

                predict_bbox[:, 0::2] *= scale_x
                predict_bbox[:, 1::2] *= scale_y

            if predict_bbox[0, 0] > predict_bbox[0, 2]:
                predict_bbox[0, 0], predict_bbox[0, 2] = predict_bbox[0, 2], predict_bbox[0, 0]
            if predict_bbox[0, 1] > predict_bbox[0, 3]:
                predict_bbox[0, 1], predict_bbox[0, 3] = predict_bbox[0, 3], predict_bbox[0, 1]

            iou, _ = box_iou(predict_bbox, target_bbox)
            iou = iou.item()

            if iou >= self.iou:
                line['correct'] = True
            else:
                line['correct'] = False

        except Exception as e:
            line['correct'] = False


    def eval_gpto1(self, line):
        if not line['response'] or line['response'].strip() == "":
            line['correct'] = False
            return
        
        coord_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', line['response'])
        if not coord_match:
            coord_match = re.search(r'\((\d+),\s*(\d+)\).*?\((\d+),\s*(\d+)\)', line['response'])
        
        if coord_match:
            try:
                predict_bbox = [float(coord_match.group(1)), float(coord_match.group(2)), 
                               float(coord_match.group(3)), float(coord_match.group(4))]
                
                target_bbox = torch.tensor(line['answer'], dtype=torch.float32).view(-1, 4)
                predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
                
                iou, _ = box_iou(predict_bbox, target_bbox)
                iou = iou.item()
                
                if iou >= self.iou:
                    line['correct'] = True
                else:
                    line['correct'] = False
            except Exception as e:
                line['correct'] = False
        else:
            line['correct'] = False
