import re
from typing import List

def extract_correct_answers(assistant_response: str) -> List[str]:
    boxed = re.findall(r"\\boxed\s*\{(.*?)\}", assistant_response, re.DOTALL)
    if boxed:
        return [re.sub(r"\s+", " ", ans.strip()) for ans in boxed if ans.strip()]

    m = re.search(r"The answer is:\s*(.+?)(?:\n|$)", assistant_response, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        ans = re.sub(r"[。！？.!?,，]+$", "", ans)
        return [ans] if ans else []

    for pat in [
        r"Final Answer:\s*(.*?)(?:\n|$)",
        r"答案[：:]\s*(.*?)(?:\n|$)",
        r"Answer:\s*(.*?)(?:\n|$)",
    ]:
        m = re.search(pat, assistant_response, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            return [ans] if ans else []
    return []

def extract_answer_from_response(text: str) -> str:
    boxed = re.findall(r"\\boxed\s*\{(.*?)\}", text, re.DOTALL)
    if boxed:
        return boxed[-1].strip()
    for pat in [
        r"The answer is:\s*(.+?)(?:\n|$)",
        r"Final Answer:\s*(.*?)(?:\n|$)",
        r"答案[：:]\s*(.*?)(?:\n|$)",
        r"Answer:\s*(.*?)(?:\n|$)",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            ans = re.sub(r"[。！？.!?,，]+$", "", ans)
            return ans
    return ""