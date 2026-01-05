import re
from typing import List
from .answer_parser import extract_answer_from_response

def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    answer = re.sub(r"\s+", " ", answer.strip())
    answer = re.sub(r"\\sqrt\s*\{?\s*(\d+)\s*\}?", r"sqrt\1", answer)
    answer = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"\1/\2", answer)
    return answer.lower()

def answers_match(response_answer: str, correct_answers: List[str]) -> bool:
    if not response_answer or not correct_answers:
        return False
    norm_resp = normalize_answer(response_answer)
    for corr in correct_answers:
        norm_corr = normalize_answer(corr)
        if norm_resp == norm_corr:
            return True
        if norm_resp in norm_corr or norm_corr in norm_resp:
            return True
        resp_nums = re.findall(r"\d+\.?\d*", norm_resp)
        corr_nums = re.findall(r"\d+\.?\d*", norm_corr)
        if len(resp_nums) == len(corr_nums) and len(resp_nums) > 0:
            all_match = True
            for rn, cn in zip(resp_nums, corr_nums):
                try:
                    if abs(float(rn) - float(cn)) > 0.1:
                        all_match = False
                        break
                except ValueError:
                    all_match = False
                    break
            if all_match:
                return True
    return False

def calculate_accuracy(responses: List[str], correct_answers: List[str]) -> float:
    if not responses:
        return 0.0
    correct = 0
    for rsp in responses:
        if answers_match(extract_answer_from_response(rsp), correct_answers):
            correct += 1
    return correct / len(responses)