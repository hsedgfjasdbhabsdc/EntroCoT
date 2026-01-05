import re
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from .logging_config import get_logger
from .constants import DEFAULT_MAX_SEGMENTS, DEFAULT_MAX_WORKERS
from .models import init_entropy_client
from .entropy import get_token_entropies
from .answer_parser import extract_correct_answers, extract_answer_from_response
from .prompts import build_prompt
from .rollout import RolloutClient
from .metrics import calculate_accuracy
from .data_io import save_jsonl, save_results

logger = get_logger(__name__)

class QwenReliabilityFilter:
    def __init__(
        self,
        api_url: str,
        api_key: str = None,
        entropy_api_base: str = None,
        entropy_api_key: str = None,
        entropy_model: str = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        request_timeout: int = 1200,
        max_segments: int = DEFAULT_MAX_SEGMENTS,
    ):
        logger.info(f"Initializing Filter. Rollout API: {api_url}")
        self.max_workers   = max_workers
        self.request_timeout = request_timeout
        self.max_segments  = max_segments

        self.rollout = RolloutClient(api_url, api_key, timeout=request_timeout)

        if entropy_api_key and entropy_api_base:
            logger.info(f"Initializing Entropy Client: {entropy_api_base} with model {entropy_model or 'default'}")
            init_entropy_client(entropy_api_base, entropy_api_key, entropy_model)
        else:
            logger.warning("Entropy API config missing! Will fallback to sentence splitting.")

        # quick connectivity check
        test = self.rollout.generate([{"role": "user", "content": "Hello"}], max_new_tokens=10)
        if test.startswith("Error"):
            logger.error(f"Rollout API connection failed: {test}")
        else:
            logger.info("Rollout API connection successful!")

    # ---------- thin wrappers ----------
    def extract_think_content(self, text: str) -> str:
        if not text:
            return ""
        m = re.search(r"<think>(.*?)(?:</think>|$)", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def split_think_content(self, think_content: str, user_question: str, num_segments: int = None) -> List[str]:
        if not think_content:
            return [""] * self.max_segments
        if num_segments is None:
            num_segments = self.max_segments

        token_data, high_idx = get_token_entropies(user_question, think_content, top_k=50)
        if token_data and len(token_data) > num_segments and high_idx:
            total_len = len(token_data)
            third = total_len // 3
            front_idx = [i for i in high_idx if i < third]
            mid_idx   = [i for i in high_idx if third <= i < 2 * third]
            back_idx  = [i for i in high_idx if i >= 2 * third]

            a, b, c = len(front_idx), len(mid_idx), len(back_idx)
            total_he = a + b + c
            if total_he == 0:
                raise ValueError("No high-entropy tokens found")
            need_cuts = num_segments - 1
            kf = max(1, round(need_cuts * a / total_he))
            km = max(1, round(need_cuts * b / total_he))
            kb = need_cuts - kf - km
            cut_pos = (
                self._pick_most_spread(front_idx, kf)
                + self._pick_most_spread(mid_idx, km)
                + self._pick_most_spread(back_idx, kb)
            )
            cut_pos = sorted(set(cut_pos))

            segments, start = [], 0
            for pos in cut_pos:
                if pos >= len(token_data):
                    continue
                seg_tokens = token_data[start:pos]
                text = (
                    "".join(t["token"] for t in seg_tokens)
                    .replace("Ġ", " ")
                    .replace("Ċ", "\n")
                )
                segments.append(text)
                start = pos
            text = (
                "".join(t["token"] for t in token_data[start:])
                .replace("Ġ", " ")
                .replace("Ċ", "\n")
            )
            segments.append(text)
            while len(segments) < num_segments:
                segments.append("")
            return segments

        logger.info("Fallback to sentence splitting.")
        sentences = re.split(r"[.!?。！？]\s*", think_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= num_segments:
            return sentences + [""] * (num_segments - len(sentences))
        seg_len = len(sentences) // num_segments
        segs = []
        for i in range(num_segments):
            if i == num_segments - 1:
                segs.append(". ".join(sentences[i * seg_len :]))
            else:
                segs.append(". ".join(sentences[i * seg_len : (i + 1) * seg_len]))
        return segs

    def _pick_most_spread(self, idx_list: List[int], k: int) -> List[int]:
        if k <= 0:
            return []
        if len(idx_list) <= k:
            return idx_list[:]
        idx_list = sorted(idx_list)
        picked = {0, len(idx_list) - 1}
        while len(picked) < k:
            best_gap, best_i = -1, -1
            for i in range(1, len(idx_list)):
                if i in picked or i - 1 in picked:
                    continue
                gap = idx_list[i] - idx_list[i - 1]
                if gap > best_gap:
                    best_gap, best_i = gap, i
            if best_i == -1:
                break
            picked.add(best_i)
        return [idx_list[i] for i in sorted(picked)]

    def rollout_single_context_concurrent_with_raw(
        self, messages: List[Dict], num_rollouts: int = 8
    ) -> Tuple[List[str], List[str]]:
        return self.rollout.rollout(messages, num_rollouts, self.max_workers)

    def process_single_data_concurrent(self, data: Dict, num_rollouts: int = 8) -> Tuple[Dict, bool, Optional[int], bool]:
        result = {
            "id": data.get("id"),
            "original_question": "",
            "correct_answers": [],
            "think_segments": [],
            "accuracy_records": [],
            "all_non_decreasing": False,
            "first_down_idx": None,
            "has_recovery": False,
            "max_accuracy": 0.0,
            "all_zero": False,
            "rollout_raw_answers": {},
        }

        try:
            system_prompt = data.get("system", "You are a helpful assistant.")
            conversations = data.get("conversations", [])
            user_question, assistant_response = "", ""
            for c in conversations:
                if c["from"] == "user" and not user_question:
                    user_question = c["value"]
                elif c["from"] == "assistant":
                    assistant_response = c["value"]
            if not user_question or not assistant_response:
                result["error"] = "Missing question or response"
                return result, False, None, False

            think_content = self.extract_think_content(assistant_response)
            if not think_content:
                think_content = assistant_response

            correct_answers = extract_correct_answers(assistant_response)
            if not think_content:
                result["error"] = "No think content found"
                return result, False, None, False

            segments = self.split_think_content(think_content, user_question, self.max_segments)
            test_range = range(0, min(len(segments), self.max_segments))

            seg_tasks = []
            for num_segs in test_range:
                msgs = build_prompt(user_question, segments, num_segs, system_prompt)
                seg_tasks.append((num_segs, msgs))

            logger.info(f"Data {data.get('id')}: Starting concurrent processing of {len(seg_tasks)} entropy-based scenarios...")

            all_responses = {}
            all_raw_responses = {}

            def _work(args):
                num_segs, msgs = args
                valid, raw = self.rollout_single_context_concurrent_with_raw(msgs, num_rollouts)
                return num_segs, valid, raw

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(seg_tasks), self.max_workers)) as exe:
                for f in concurrent.futures.as_completed([exe.submit(_work, t) for t in seg_tasks]):
                    num_segs, valid, raw = f.result()
                    all_responses[num_segs] = valid
                    all_raw_responses[num_segs] = raw

            accuracy_records = []
            for num_segs in test_range:
                valid = all_responses.get(num_segs, [])
                raw   = all_raw_responses.get(num_segs, [])
                acc   = calculate_accuracy(valid, correct_answers)
                accuracy_records.append(
                    {
                        "num_segments": num_segs,
                        "accuracy": acc,
                        "prompt_length": sum(len(msg["content"]) for msg in seg_tasks[num_segs][1] if "content" in msg),
                        "rollout_answers": raw,
                    }
                )

            accuracies = [rec["accuracy"] for rec in accuracy_records]

            all_non_decreasing = True
            first_down_idx = None
            has_recovery = False
            max_accuracy = max(accuracies) if accuracies else 0.0

            for i in range(1, len(accuracies)):
                if accuracies[i] < accuracies[i-1]:
                    all_non_decreasing = False
                    first_down_idx = i
                    for j in range(i+1, len(accuracies)):
                        if accuracies[j] > accuracies[i-1]:
                            has_recovery = True
                            break
                    break

            all_zero = all(acc == 0.0 for acc in accuracies)

            result.update({
                "original_question": user_question,
                "correct_answers": correct_answers,
                "think_segments": segments,
                "accuracy_records": accuracy_records,
                "all_non_decreasing": all_non_decreasing,
                "first_down_idx": first_down_idx,
                "has_recovery": has_recovery,
                "max_accuracy": max_accuracy,
                "all_zero": all_zero,
                "rollout_raw_answers": all_raw_responses,
            })
            return result, all_non_decreasing, first_down_idx, all_zero

        except Exception as e:
            logger.error(f"Error processing data {data.get('id', 'unknown')}: {str(e)}")
            result["error"] = str(e)
            return result, False, None, False

    def _build_rejected_sample(self, original_data: Dict, think_segments: List[str], first_down_idx: Optional[int]) -> Dict:
        if first_down_idx is None or first_down_idx == 0:
            return original_data
        kept_cot = "".join(think_segments[:first_down_idx])
        new_assistant = f"<think>{kept_cot}"
        rejected_sample = original_data.copy()
        rejected_sample["conversations"] = [
            {**c, "value": new_assistant} if c["from"] == "assistant" else c
            for c in original_data["conversations"]
        ]
        return rejected_sample

    def process_single_data_wrapper(self, args) -> Tuple[Dict, bool, Optional[Dict], Optional[Dict], Optional[Dict]]:
        data, num_rollouts = args
        result, all_non_dec, first_down_idx, all_zero = self.process_single_data_concurrent(data, num_rollouts)

        reliable_data, rejected_data, all_zero_data = None, None, None

        if all_zero:
            all_zero_data = data
        elif all_non_dec or result.get("has_recovery", False):
            reliable_data = data
        else:
            rejected_data = self._build_rejected_sample(data, result["think_segments"], first_down_idx)

        return result, all_non_dec, reliable_data, rejected_data, all_zero_data

    def process_dataset_concurrent(
        self,
        data_list: List[Dict],
        output_file: str,
        reliable_file: str,
        rejected_file: str,
        all_zero_file: str,
        num_rollouts: int,
        batch_size: int,
    ) -> None:
        all_results, reliable_list, rejected_list, all_zero_list = [], [], [], []
        tasks = [(d, num_rollouts) for d in data_list]

        for batch_start in range(0, len(tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            logger.info(f"Processing batch {batch_start//batch_size + 1}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                batch_results = list(
                    tqdm(exe.map(self.process_single_data_wrapper, batch_tasks), total=len(batch_tasks))
                )

            for res, all_non_dec, rel_data, rej_data, az_data in batch_results:
                all_results.append(res)
                if az_data:
                    all_zero_list.append(az_data)
                elif rel_data:
                    reliable_list.append(rel_data)
                if rej_data:
                    rejected_list.append(rej_data)

            save_results(all_results, output_file)
            save_jsonl(reliable_list,   reliable_file)
            save_jsonl(rejected_list,  rejected_file)
            save_jsonl(all_zero_list,  all_zero_file)

        total = len(data_list)
        recovery_count = sum(1 for res in all_results if res.get("has_recovery", False))
        logger.info(f"Filter done —— all-zero: {len(all_zero_list)} | "
                    f"reliable (non-decreasing): {sum(1 for res in all_results if res.get('all_non_decreasing', False))} | "
                    f"reliable (with recovery): {recovery_count} | "
                    f"rejected: {len(rejected_list)} | "
                    f"ratio: {len(all_zero_list)/total:.2%} / {len(reliable_list)/total:.2%} / {len(rejected_list)/total:.2%}")