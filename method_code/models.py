from openai import OpenAI
from .constants import DEFAULT_ENTROPY_MODEL

_entropy_client = None
_entropy_model   = DEFAULT_ENTROPY_MODEL

def init_entropy_client(base_url: str, api_key: str, model: str = None):
    global _entropy_client, _entropy_model
    _entropy_client = OpenAI(base_url=base_url, api_key=api_key, timeout=1200.0)
    if model:
        _entropy_model = model

def get_entropy_client():
    return _entropy_client

def get_entropy_model():
    return _entropy_model