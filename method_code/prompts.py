from typing import List, Dict

def build_prompt(
    original_question: str,
    think_segments: List[str],
    num_segments: int,
    system_prompt: str = "",
) -> List[Dict]:
    if num_segments == 0:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_question + " Place the numerical part of the final answer in \\boxed."},
        ]
    num_segments = max(0, min(num_segments, len(think_segments)))
    partial_think = "".join(s for s in think_segments[:num_segments] if s)
    user_content = f"""The question is: "{original_question}"

And the following is part of the idea. Follow and continue to finish the answer based on it (Remember to place the numerical part of the final answer in \\boxed.):

<think>{partial_think}"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]