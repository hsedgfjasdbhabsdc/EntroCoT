import numpy as np
import re
from typing import List, Tuple, Dict
from .models import get_entropy_client, get_entropy_model
from .logging_config import get_logger

logger = get_logger(__name__)

def get_token_entropies(
    user_question: str, think_content: str, top_k: int = 50
) -> Tuple[List[Dict], List[int]]:
    """
    Returns (token_data, high_entropy_global_indices)
    token_data := [{"token": str, "entropy": float}, ...]
    """
    client = get_entropy_client()
    if not client or not think_content:
        return [], []

    prompt = f"{user_question}\n\nAssistant: <think>{think_content}</think>"
    try:
        response = client.completions.create(
            model=get_entropy_model(),
            prompt=prompt,
            temperature=0.0,
            logprobs=5,
            max_tokens=1,
            echo=True,
        )
        if not response.choices:
            return [], []
        logprobs = response.choices[0].logprobs
        if not logprobs or not logprobs.tokens:
            return [], []

        tokens          = logprobs.tokens
        top_logprobs    = logprobs.top_logprobs
        data, global_idx = [], []
        temp_buffer, start_index = "", -1

        for i, tok in enumerate(tokens):
            clean = tok.replace("Ġ", " ")
            temp_buffer += clean
            if "<think>" in temp_buffer and start_index == -1:
                start_index = i + 1
                temp_buffer = ""
                continue
            if start_index == -1:
                continue
            if "</think>" in temp_buffer or "</" in clean:
                break

            entropy = 0.0
            if top_logprobs[i] is not None:
                try:
                    probs = [np.exp(lp) for lp in top_logprobs[i].values()]
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                        entropy = -sum(p * np.log2(p) for p in probs if p > 1e-10)
                except Exception:
                    entropy = 0.0
            data.append({"token": tok, "entropy": entropy})
            global_idx.append(i - start_index)

        if len(data) < top_k:
            top_k = len(data)
        top_indices = sorted(range(len(data)), key=lambda x: data[x]["entropy"], reverse=True)[:top_k]
        high_entropy_pos = [global_idx[i] for i in top_indices]
        return data, high_entropy_pos

    except Exception as e:
        logger.error(f"Entropy calculation failed: {e}")
        return [], []