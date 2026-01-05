import json
from typing import List, Dict
from .logging_config import get_logger

logger = get_logger(__name__)

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
    return data

def save_jsonl(data_list: List[Dict], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for d in data_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def save_results(data: List[Dict], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)