"""
Description: This is an official release version of evaluation code for MATHGLANCE benchmark, which presents in
``MATHGLANCE: Multimodal Large Language Models Do Not Know Where to Look in Mathematical Diagrams, arXiv 2025``
Module name: extract_answer_utils.py, version: 1.0.0
Function: differt methods for extract multi-choice question answers from model response

Authors: AI4Math Team - Wei Tang, Shan Zhang, and Aotian Chen
Creation Date: Jan 10, 2025
Last Modified: April 12, 2025
Version: release - V1.0

Modification History:
- April 12, 2025 - Wei Tang - release version - V1.0
"""

import re

# ###############################################
# written by Wei Tang
# 2025.04.12
# extract multi-choice answers for differt models
# ###############################################
def extract_answer_qwen(response):
    matches = re.findall(
        r'(?:Answer:\s*([A-D])|([A-D]):)',  # match Answer:X or X:
        response,
        flags=re.IGNORECASE
    )
    
    options = []
    for m in matches:
        if m[0]:
            options.append(m[0].upper())
        elif m[1]:
            options.append(m[1].upper())
    
    # return the last matched option (usually are the final answer)
    return options[-1] if options else ''


# ###############################################
# written by Shan Zhang
# 2025.04.12
# extract multi-choice answers for differt models
# ###############################################
def  extract_answer_llava(response):
    pattern=r'([A-D]):\s*([\w-]+)'
    match = re.search(pattern, response)
    if match:
        answer_choice = match.group(1) or match.group(2)
        return answer_choice
    pattern0 = r"Answer:([A-D])" 
    match = re.search(pattern0, response)
    if match:
        return match.group(1)
    pattern1 = r'Answer:\s*([A-D])'
    match = re.search(pattern1, response)
    if match:
        return match.group(1)
    return response


# ###############################################
# written by Aotian Chen
# 2025.04.12
# extract multi-choice answers for differt models
# ###############################################
def extract_answer_other_models(response, question_text=None, model_type="gpt4o"):
    """
    support for ["internvl", "internvl2", "internvl_X_2_5", "gpt4o", "gpto1"]
    
    the unified extract answer method for other models, we unified the pre-processing logic for different answer format,
    including Yes/No, direct letter options, and descriptive answers.
    """
    # handle the case where the response is empty or None
    if not response or response.strip() == "":
        return ""
    
    response_stripped = response.strip()

    # add: handle single alpha response: "A", "B", "C", "D"
    if response_stripped in "ABCD" or response_stripped in ["A.", "B.", "C.", "D."]:
        return response_stripped[0].upper()
    
    for c in 'ABCDE':
        if (response_stripped.endswith(f" {c}.") or 
            response_stripped.endswith(f" ({c}).") or 
            response_stripped.startswith(f"{c}\n") or 
            response_stripped.startswith(f"({c})\n") or 
            response_stripped.startswith(f"({c}) {c}\n")):
            return c
    
    # add: handle the middle pattern, e.g. end with "(C)." but no space before it
    if response_stripped.endswith(")."):
        bracket_end_pattern = re.compile(r'\(([A-D])\)\.$', re.IGNORECASE)
        match = bracket_end_pattern.search(response_stripped)
        if match:
            return match.group(1).upper()
    
    # add: handle the answer occur in the end of response, e.g., "Therefore, the correct answer to the question is C"
    end_answer_pattern = re.compile(r'(?:therefore|thus|so|hence|consequently|as a result),?\s+(?:the)?\s*(?:correct|right|appropriate)?\s*(?:answer|solution|choice|option)(?:\s+to\s+(?:the|this)\s+question)?\s+is\s+([A-D])[\.\s]*$', re.IGNORECASE)
    match = end_answer_pattern.search(response_stripped)
    if match:
        return match.group(1).upper()
    
    # add: handle the variable pattern for the answer occur in the end of response
    simple_end_pattern = re.compile(r'(?:answer|choice|option)\s+is\s+([A-D])[\.\s]*$', re.IGNORECASE)
    match = simple_end_pattern.search(response_stripped)
    if match:
        return match.group(1).upper()
    
    # 1. check for format like "... (A)"
    bracket_option_pattern = re.compile(r'\(([A-D])\)', re.IGNORECASE)
    match = bracket_option_pattern.search(response)
    if match:
        return match.group(1).upper()
    
    # 2. check for format that start with the choice alpha like "A." or "A:"
    if len(response_stripped) >= 2 and response_stripped[0] in "ABCD" and response_stripped[1] in [".", ":"]:
        return response_stripped[0].upper()
    
    # 3. check for if there are clear format like "A: " or "C:"
    direct_option_pattern = re.compile(r'(?:^|\s|\.)([A-D]):(?:\s|$)', re.IGNORECASE)
    match = direct_option_pattern.search(response)
    if match:
        return match.group(1).upper()
    
    # 4. check for format "Answer: X" or "X:"
    option_matches = re.findall(r'(?:Answer:\s*([A-D])|([A-D]):)', response, flags=re.IGNORECASE)
    if option_matches:
        options = []
        for m in option_matches:
            if m[0]:
                options.append(m[0].upper())
            elif m[1]:
                options.append(m[1].upper())
        if options:
            return options[-1]
    
    # 5. check for if there are clear citation of choice alpha, e.g."(Choice B)" or "B: ellipse"
    choice_pattern = re.compile(r'(?:Choice |answer is:?|:)\s*([A-D])[:\)]', re.IGNORECASE)
    match = choice_pattern.search(response)
    if match:
        return match.group(1).upper()
    
    # 6. variations of 5.
    letter_pattern = re.compile(r'(?:So,? the )?(?:correct )?(?:answer|choice) is:?\s*([A-D])[\s\.]', re.IGNORECASE)
    match = letter_pattern.search(response)
    if match:
        return match.group(1).upper()
    
    # 7. check if there are format of "correct answer is X" (specially for internvl_X_2_5)
    if model_type == "internvl_x_2_5":
        answer_pattern = re.compile(r'correct answer (?:to the question )?is ([A-D]):', re.IGNORECASE)
        match = answer_pattern.search(response)
        if match:
            return match.group(1).upper()
        
        # check if there are no answer statements in the responese
        if re.search(r'correct answer (?:to the question )?is not listed', response, re.IGNORECASE):
            return ''
    
    # add：handle the format like "The total number of shapes in the picture is 1."
    if question_text:
        number_choices = {}
        choice_pattern = re.compile(r'([A-D]):(.*?)(?=\n[A-D]:|$)', re.DOTALL)
        choice_matches = choice_pattern.findall(question_text)
        
        for letter, description in choice_matches:
            number_match = re.search(r'\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b', description.lower())
            if number_match:
                number_choices[number_match.group(1).lower()] = letter
        
        total_pattern = re.compile(r'(?:total|number|count|there are|there is).*?(?:shape|object|figure|triangle|rectangle|circle|square).*?(?:is|are)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b', re.IGNORECASE)
        match = total_pattern.search(response)
        if match:
            number_found = match.group(1).lower()
            if number_found in number_choices:
                return number_choices[number_found].upper()
    
    # add：handle questions about Yes/No
    if question_text and ('Yes' in question_text or 'No' in question_text):

        yes_no_pattern = re.compile(r'([A-D]):\s*Yes|([A-D]):\s*No', re.IGNORECASE)
        yes_no_matches = yes_no_pattern.findall(question_text)
        
        yes_option = ""
        no_option = ""
        
        for match in yes_no_matches:
            if match[0]:  # A:Yes format
                yes_option = match[0]
            elif match[1]:  # A:No format
                no_option = match[1]
        
        if yes_option and re.search(r'\byes\b', response_stripped, re.IGNORECASE):
            return yes_option.upper()
        elif no_option and re.search(r'\bno\b', response_stripped, re.IGNORECASE):
            return no_option.upper()
    
    # 8. extract options from questions and check it for answer in the response
    if question_text:
        choices = {}
        choice_pattern = re.compile(r'([A-D]):(.*?)(?=\n[A-D]:|$)', re.DOTALL)
        choice_matches = choice_pattern.findall(question_text)
        
        # key word list for shapes
        shape_keywords = [
            'ellipse', 'circle', 'triangle', 'square', 'parallelogram', 'segment', 
            'trapezoid', 'quadrilateral', 'rectangle', 'quadrangle', 'isosceles', 
            'right triangle', 'scalene', 'equilateral'
        ]
        
        # key words list for counting
        number_keywords = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'zero', 'one', 
                          'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        
        for letter, description in choice_matches:
            description_lower = description.strip().lower()
            choices[description_lower] = letter
            
            for keyword in description_lower.split():
                if len(keyword) > 3 and keyword in shape_keywords:
                    choices[keyword] = letter
                if keyword in number_keywords:
                    choices[keyword] = letter
        
        response_lower = response.lower()
        
        for description, letter in choices.items():
            if description in response_lower and len(description) > 5:
                return letter
        
        for shape in shape_keywords:
            if shape in response_lower:
                if shape in choices:
                    # addtional check: if the response truely says the answer, rather than random mention
                    pattern = r'(?:is|are|appears to be|identified as|classified as)(?:\s+(?:an?|the))?\s+' + re.escape(shape)
                    if re.search(pattern, response_lower):
                        return choices[shape]
                        
                for description, letter in choices.items():
                    if shape in description and len(description) > len(shape):
                        shape_mention_pattern = r'(?:is|are|appears to be|identified as|classified as)(?:\s+(?:an?|the))?\s+' + re.escape(shape)
                        if re.search(shape_mention_pattern, response_lower):
                            return letter
        
        for number in number_keywords:
            if number in response_lower:
                if number in choices:
                    number_pattern = r'(?:there (?:is|are)|contains?|has|have|total of|count of|number of)\s+' + re.escape(number)
                    if re.search(number_pattern, response_lower):
                        return choices[number]
    
    # 9. if all methods fail, return empty string
    return ""