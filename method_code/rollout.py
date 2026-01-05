import requests
import concurrent.futures
from typing import List, Dict, Tuple
from .logging_config import get_logger

logger = get_logger(__name__)

class RolloutClient:
    def __init__(self, api_url: str, api_key: str = None, timeout: int = 1200):
        self.api_url = api_url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def generate(self, messages: List[Dict], max_new_tokens: int = 8192, temperature: float = 0.7) -> str:
        payload = {
            "model": "Qwen/Qwen3-4B-Instruct",
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.1,
            "stream": False,
        }
        try:
            rsp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
            if rsp.status_code != 200:
                logger.error(f"Rollout API request failed - Status: {rsp.status_code}")
                return f"Error: API request failed, status code {rsp.status_code}, response: {rsp.text}"
            result = rsp.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            if "error" in result:
                return f"Error: {result['error']}"
            return "Error: Unable to parse API response"
        except requests.exceptions.Timeout:
            return "Error: API request time out"
        except requests.exceptions.ConnectionError:
            return "Error: API connection error"
        except Exception as e:
            return f"Error: {str(e)}"

    def rollout(self, messages: List[Dict], num_rollouts: int = 8, max_workers: int = 10) -> Tuple[List[str], List[str]]:
        def _one(_):
            return self.generate(messages, temperature=0.7)

        valid, raw = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_rollouts, max_workers)) as exe:
            futures = [exe.submit(_one, i) for i in range(num_rollouts)]
            for f in concurrent.futures.as_completed(futures):
                txt = f.result()
                raw.append(txt)
                if not txt.startswith("Error"):
                    valid.append(txt)
        return valid, raw