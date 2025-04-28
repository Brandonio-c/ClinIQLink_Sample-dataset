# Sample Submission Guide

This directory shows a minimal, ready-to-run submission for the **ClinIQLink** evaluations.  
You can run it “as-is” and easily swap in your own model.

---

## 1. Folder Structure

```
sample_submission/
├── submit.py                  # Main driver 
├── README.md                   # This document
├── model_submission/           # Put your weights or scripts here
│   └── snapshots/
│       └── gpt2/               # Example: Hugging Face "gpt2" snapshot (or your own model)
│           ├── config.json
│           ├── pytorch_model.bin
│           ├── tokenizer.json
│           └── …               # etc.
└── submission_template/        # Prompt templates (keep unchanged unless necessary)
    ├── MC_template.prompt
    ├── list_template.prompt
    ├── multi_hop_template.prompt
    ├── multi_hop_inverse_template.prompt
    ├── short_template.prompt
    ├── short_inverse_template.prompt
    ├── tf_template.prompt
    └── README.md
```

To download a Hugging Face model into `model_submission/snapshots/`, you can use:

```bash
transformers-cli download gpt2 --cache-dir model_submission/snapshots
```

or any other model snapshots you would like to use 
---

## 2. Quick Start (Local CPU)

```bash
# Optional: create a clean environment
python -m venv cliniq_env
source cliniq_env/bin/activate
pip install --upgrade pip transformers torch

# Run the submission
python submit.py --mode local --chunk_size 4 --max_length 200 --num_tf 1 --num_mc 1 --num_list 1 --num_short 1 --num_short_inv 1 --num_multi 1 --num_multi_inv 1
```

- No edits required.
- The script detects the model placed under `model_submission/snapshots/` and runs it automatically.

To use a different Hugging Face model:

1. Place the model snapshot under `model_submission/snapshots/Model-Name/`.
2. Simply run:

```bash
python submit.py --mode local --chunk_size 4 --max_length 200 --num_tf 1 --num_mc 1 --num_list 1 --num_short 1 --num_short_inv 1 --num_multi 1 --num_multi_inv 1
```

---

## 4. Modifying `submit.py` (Optional)

You typically do not need to modify `submit.py`.  
Only change it if:

- You need to load a non-standard checkpoint format (e.g., DeepSpeed),
- You want to customize batch handling,
- You are integrating additional modules like retrieval-augmented generation (RAG).

Key functions you can modify:

| Function | Purpose |
|----------|---------|
| `load_participant_model(self)` | How your model is loaded |
| `load_participant_pipeline(self)` | How inference is handled |

---

## 5. Prompt Templates

The `submission_template/` folder contains `.prompt` files for each QA type.  
These templates define how questions are framed when given to the model.  
Only adjust them if necessary.

---

## 6. Submission Requirements

A valid submission must:

- Follow the folder structure described above.
- Place all model weights and scripts under `model_submission/snapshots/`.
- Ensure `submit.py` runs locally without errors.
- Avoid using external LLM APIs during inference.

More information and detailed rules can be found on the ClinIQLink Challenge website:  
[https://cliniqlink.org](https://cliniqlink.org)

