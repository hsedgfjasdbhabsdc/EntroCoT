import json
import re
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI
import logging

REJECTED_FILE = ""
ALL_ZERO_FILE = ""
RECOVERED_FROM_REJECTED = ""
REWRITTEN_FROM_ALL_ZERO = ""
FAILED_REJECTED_FILE = ""
FAILED_ALL_ZERO_FILE = ""
ORIGINAL_DATASET = ""

API_KEY = ""
API_BASE = ""
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
MAX_WORKERS = 400
MAX_ROLLOUTS = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_id_to_answers(original_jsonl: str) -> Dict[str, List[str]]:
    """Build id -> correct_answers mapping from the original dataset."""
    mapping: Dict[str, List[str]] = {}
    with open(original_jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            if "id" not in d:
                continue
            answers = extract_correct_answers_static(
                next((c["value"] for c in d.get("conversations", []) if c.get("from") == "assistant"), "")
            )
            mapping[d["id"]] = answers
    logger.info(f"Loaded {len(mapping)} id→answers from {original_jsonl}")
    return mapping


def extract_correct_answers_static(assistant_response: str) -> List[str]:
    """Extract answers statically, same logic as in QwenReliabilityFilter."""
    boxed = re.findall(r"\\boxed\s*\{(.*?)\}", assistant_response, re.DOTALL)
    if boxed:
        return [re.sub(r"\s+", " ", ans.strip()) for ans in boxed if ans.strip()]
    for pat in [r"Final Answer:\s*(.*?)(?:\n|$)", r"答案[：:]\s*(.*?)(?:\n|$)", r"Answer:\s*(.*?)(?:\n|$)"]:
        m = re.search(pat, assistant_response, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            return [ans] if ans else []
    return []


class DataRecoveryAgent:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE, timeout=12000.0)

    def normalize_answer(self, answer: str) -> str:
        if not answer:
            return ""
        answer = re.sub(r"\s+", " ", answer.strip())
        answer = re.sub(r"\\sqrt\s*\{?\s*(\d+)\s*\}?", r"sqrt\1", answer)
        answer = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"\1/\2", answer)
        return answer.lower()

    def extract_answer_from_response(self, text: str) -> str:
        boxed = re.findall(r"\\boxed\s*\{(.*?)\}", text, re.DOTALL)
        if boxed:
            return boxed[-1].strip()
        for pat in [r"Final Answer:\s*(.*?)(?:\n|$)", r"答案[：:]\s*(.*?)(?:\n|$)", r"Answer:\s*(.*?)(?:\n|$)"]:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()
        return ""

    def answers_match(self, response_answer: str, correct_answers: List[str]) -> bool:
        if not response_answer or not correct_answers:
            return False
        norm_resp = self.normalize_answer(response_answer)
        for corr in correct_answers:
            norm_corr = self.normalize_answer(corr)
            if norm_resp == norm_corr:
                return True
            if norm_resp in norm_corr or norm_corr in norm_resp:
                return True
            resp_nums = re.findall(r"\d+\.?\d*", norm_resp)
            corr_nums = re.findall(r"\d+\.?\d*", norm_corr)
            if len(resp_nums) == len(corr_nums) and len(resp_nums) > 0:
                try:
                    all_match = True
                    for rn, cn in zip(resp_nums, corr_nums):
                        if abs(float(rn) - float(cn)) > 0.1:
                            all_match = False
                            break
                    if all_match:
                        return True
                except ValueError:
                    pass
        return False

    def call_deepseek(self, messages: List[Dict], temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=11000,
                top_p=0.95,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API Call Failed: {e}")
            return ""

    def fix_rejected_sample(self, data: Dict) -> Tuple[Optional[Dict], Dict]:
        user_question = ""
        truncated_cot = ""
        system_prompt = data.get("system", "You are a helpful assistant.")
        for c in data["conversations"]:
            if c["from"] == "user":
                user_question = c["value"]
            elif c["from"] == "assistant":
                truncated_cot = c["value"]
        if not user_question or not truncated_cot:
            return None, data

        prompt_content = (
            f"Question: {user_question}\n\n"
            f"Below is the beginning of the thinking process for this question:\n"
            f"'''\n{truncated_cot}\n'''\n\n"
            f"Please carefully read the beginning above, follow its logic, "
            f"and continue writing to complete the thinking process and provide the final answer in \\boxed{{}}."
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_content}]

        for i in range(MAX_ROLLOUTS):
            current_temp = min(0.7 + i * 0.05, 1.0)
            response_text = self.call_deepseek(messages, temperature=current_temp)
            extracted_ans = self.extract_answer_from_response(response_text)
            if self.answers_match(extracted_ans, data.get("correct_answers", [])):
                clean_continuation = response_text.replace("<think>", "").replace("</think>", "")
                final_cot = f"{truncated_cot}\n{clean_continuation}"
                new_data = data.copy()
                new_data["conversations"] = [
                    c if c["from"] != "assistant" else {"from": "assistant", "value": final_cot}
                    for c in data["conversations"]
                ]
                new_data["recovery_source"] = f"rejected_fix_attempt_{i+1}"
                return new_data, None
        return None, data

    def rewrite_all_zero_sample(self, data: Dict) -> Tuple[Optional[Dict], Dict]:
        user_question = ""
        system_prompt = data.get("system", "You are a helpful AI assistant, who always ready to help user.")
        for c in data["conversations"]:
            if c["from"] == "user":
                user_question = c["value"]
                break
        if not user_question:
            return None, data

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_question}\n\nPlease solve this step-by-step and put the final answer in \\boxed{{}}."},
        ]

        for i in range(MAX_ROLLOUTS):
            current_temp = min(0.7 + i * 0.05, 1.0)
            response_text = self.call_deepseek(messages, temperature=current_temp)
            extracted_ans = self.extract_answer_from_response(response_text)
            if self.answers_match(extracted_ans, data.get("correct_answers", [])):
                new_data = data.copy()
                new_data["conversations"] = [
                    {"from": "user", "value": user_question},
                    {"from": "assistant", "value": response_text},
                ]
                new_data["recovery_source"] = f"all_zero_rewrite_attempt_{i+1}"
                return new_data, None
        return None, data


def process_file(input_file: str, success_file: str, failed_file: str, mode: str, agent: DataRecoveryAgent, id2answers: Dict[str, List[str]]) -> None:
    logger.info(f"Start processing {input_file} in mode: {mode}")
    data_list = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("id") in id2answers:
                    d["correct_answers"] = id2answers[d["id"]]
                else:
                    logger.warning("Missing id %s in original dataset", d.get("id"))
                    continue
                data_list.append(d)
    except FileNotFoundError:
        logger.error("File not found: %s", input_file)
        return

    logger.info("Loaded %s samples. Max rollouts per sample: %s", len(data_list), MAX_ROLLOUTS)
    success_count, fail_count = 0, 0

    with open(success_file, 'w', encoding='utf-8') as f_success, \
         open(failed_file, 'w', encoding='utf-8') as f_fail:

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_data = {
                executor.submit(agent.fix_rejected_sample if mode == 'fix' else agent.rewrite_all_zero_sample, d): d
                for d in data_list
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_data), total=len(data_list), desc=f"Processing {mode}"):
                try:
                    success_data, failed_data = future.result()
                    if success_data:
                        f_success.write(json.dumps(success_data, ensure_ascii=False) + "\n")
                        success_count += 1
                    else:
                        f_fail.write(json.dumps(failed_data, ensure_ascii=False) + "\n")
                        fail_count += 1
                except Exception as e:
                    logger.error("Error processing item: %s", e)

    logger.info("Finished %s —— Success: %s (saved to %s), Failed: %s (saved to %s)",
                mode, success_count, success_file, fail_count, failed_file)


def main():
    id2answers = load_id_to_answers(ORIGINAL_DATASET)
    agent = DataRecoveryAgent()

    logger.info(">>> Phase 1: Fixing Rejected Samples...")
    process_file(REJECTED_FILE, RECOVERED_FROM_REJECTED, FAILED_REJECTED_FILE, 'fix', agent, id2answers)

    logger.info(">>> Phase 2: Rewriting All-Zero Samples...")
    process_file(ALL_ZERO_FILE, REWRITTEN_FROM_ALL_ZERO, FAILED_ALL_ZERO_FILE, 'rewrite', agent, id2answers)


if __name__ == "__main__":
    main()