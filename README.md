
# EntroCoT

A **modular, scalable** algorithm for **detecting and filtering unreliable reasoning paths** in large-scale reasoning datasets.  

---

## What it does

1. **Entropy-guided segmentation**  
   Uses token-level entropy to locate the most **ambiguous** places inside a chain-of-thought.

2. **Rollout-based accuracy curve**  
   For each incremental segment it prompts the model **n times** and measures **answer accuracy**.

3. **Automatic triage**  
   Every sample is classified into **exactly one** bucket:
   - ✅ **reliable** – accuracy is non-decreasing or recovers after a drop  
   - ❌ **rejected** – accuracy drops and **never** recovers (partial COT is stored for later **recovery**)  
   - ⚠️ **all-zero** – every segment gives **0 % accuracy** (candidate for **full rewrite**/check whether it is too difficult)

4. **Ready for recovery(optional)**  
   Outputs three clean JSONL files so you can immediately run the companion **DataRecovery** process to fix rejected & all-zero samples.

---

## File Layout

```
method_code/          
├── core.py              # ReliabilityFilter class 
├── entropy.py           # Token-level entropy calculation
├── rollout.py           # Concurrent rollout client
├── metrics.py           # Accuracy / answer-matching logic
├── prompts.py           # Prompt templates
├── answer_parser.py     # answer extraction
├── data_io.py           # for JSONL
├── models.py            # OpenAI client
├── constants.py         # Default hyper-parameters settings
└── logging_config.py    # logging config

data_recovery.py         # fixes rejected & all-zero samples (for reference)
```
---


## Requirements

```bash

pip install numpy requests openai tqdm

```


---


## Quick Start

```python
from method_code import QwenReliabilityFilter
from method_code.data_io import load_jsonl

filter = QwenReliabilityFilter(
    api_url="http://your-qwen-endpoint/v1/chat/completions",
    api_key="your-key",                       # rollout API
    entropy_api_base="http://deepseek-endpoint/v1",
    entropy_api_key="your-key",               # entropy API
    entropy_model="deepseek-ai/DeepSeek-R1",
    max_workers=1000,
    max_segments=5,        
    request_timeout=10000,
)

# Any math JSONL with "conversations" format
data = load_jsonl("numina_train.jsonl")

filter.process_dataset_concurrent(
    data_list=data,
    output_file="entropy_report.json",   
    reliable_file="reliable.jsonl",      # reliable samples
    rejected_file="rejected.jsonl",      # truncate reasoning paths
    all_zero_file="all_zero.jsonl",      # rewrite later
    num_rollouts=8,      # accuracy estimated with 8 rollouts
    batch_size=1000,     
)
```

---

## Performance

| Model             | Dataset    | Size | Method              | GSM8K     | MATH      | GaoKao    | Odyssey   | Olympiad  | AMC23     | Avg.      |
| ----------------- | ---------- | ---- | ------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Meta-Llama3.1-8B  | MetaMathQA | 395k | Direct SFT          | **77.03** | 33.80     | 23.64     | 7.46      | 6.22      | 7.50      | 25.94     |
|                   |            | 332k | EntroCoT-random     | 74.83     | 31.40     | 23.90     | 6.68      | 5.93      | 5.00      | 24.62     |
|                   |            | 358k | EntroCoT-w/o-greedy | 75.01     | 32.60     | 23.22     | 6.78      | **6.84**  | 8.50      | 25.49     |
|                   |            | 344k | EntroCoT-full       | 76.89     | **35.80** | **27.01** | **7.97**  | 6.81      | **15.00** | **28.25** |
| Meta-Llama3.1-8B  | NuminaMath | 859k | Direct SFT          | 72.10     | 37.20     | 32.73     | **20.82** | 13.04     | 19.00     | 32.48     |
|                   |            | 515k | EntroCoT-random     | 71.34     | 39.24     | 36.67     | 19.69     | 12.86     | 19.00     | 33.13     |
|                   |            | 395k | EntroCoT-w/o-greedy | 70.96     | 39.80     | 38.96     | 17.48     | 12.00     | 17.50     | 32.78     |
|                   |            | 480k | EntroCoT-full       | **76.00** | **41.20** | **40.00** | 19.54     | **14.37** | **20.00** | **35.19** |
| Qwen2.5-Math-1.5B | MetaMathQA | 395k | Direct SFT          | 48.60     | 33.84     | 33.72     | 17.12     | 10.28     | 7.50      | 25.18     |
|                   |            | 332k | EntroCoT-random     | 45.40     | 33.92     | 35.58     | 16.45     | **12.12** | 8.50      | 25.33     |
|                   |            | 358k | EntroCoT-w/o-greedy | 47.43     | 34.44     | 35.12     | 16.20     | 10.67     | 8.50      | 25.39     |
|                   |            | 344k | EntroCoT-full       | **50.19** | **34.56** | **37.35** | **17.23** | 11.14     | **15.00** | **27.58** |
| Qwen2.5-Math-1.5B | NuminaMath | 859k | Direct SFT          | 70.90     | 54.64     | 46.07     | 21.44     | 19.73     | 32.50     | 40.88     |
|                   |            | 515k | EntroCoT-random     | 71.01     | 52.12     | 44.21     | 22.67     | 21.48     | 32.50     | 40.67     |
|                   |            | 395k | EntroCoT-w/o-greedy | 73.09     | 48.20     | 40.52     | 20.31     | 18.07     | 35.00     | 39.20     |
|                   |            | 480k | EntroCoT-full       | **74.65** | **59.60** | **48.80** | **23.40** | **24.35** | **45.50** | **46.05** |


---

## Optional: Data Recovery

After filtering, recovery process can be run:

```
python data_recovery.py

```

It will:
- **continue** truncated COTs (`rejected`)  
- **rewrite** completely wrong solutions (`all_zero`)  


---



