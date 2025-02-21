# ClinIQLink Challenge page

See our challenge page here: https://brandonio-c.github.io/ClinIQLink-2025/


# ClinIQLink Sample Dataset

This repository provides a **sample dataset** for participants in the **ClinIQLink 2025 - LLM Lie Detector Task**. The dataset is intended **only for format verification** and does not contain training data. Participants should use this dataset to ensure their submissions conform to the required structure before submitting to [CodaBench](https://www.codabench.org/competitions/5117/).

## Dataset Structure
```
ClinIQLink_Sample-dataset/
├── sample_QA_pairs/                        # Sample question-answer pairs for format testing
│   ├── list.json
│   ├── MC.json
│   ├── multi_hop.json
│   ├── short.json
│   ├── TF.json
│   ├── QA_Dataset.csv
│ 
├── sample_submission/                  # Example of a correctly formatted submission
│   ├── submit.py                       # Main submission script (participants modify this)
│   ├── submit_GPT-2_example.py         # Example implementation using GPT-2
│   ├── submit.sh                       # SLURM submission script for HPC environments
│   ├── README.md                       # This document
│   ├── submission_template/            # Blank template for participants
│       ├── MC_template.prompt
│       ├── list_template.prompt
│       ├── multi_hop_template.prompt
│       ├── multi_hop_inverse_template.prompt
│       ├── short_template.prompt
│       ├── short_inverse_template.prompt
│       ├── tf_template.prompt
│       ├── README.md                   # Instructions for using the templates
│ 
├── README.md                               # Main repository documentation
└── LICENSE                                 # License restricting dataset use
```

## How to Use
1. Review the **sample_QA_pairs** directory for example question-answer pairs.
2. Generate your model's responses using the provided submit.py script.
3. See score generated from submisson script.
4. Ensure the submission matches the expected format before uploading to [CodaBench](https://www.codabench.org/competitions/5117/).

## Submission Format
Each QA pair follow a strict JSON format. Example:
```json
[
     {
        "answer": "answer",
        "question": "question?",
        "source": {
            "isbn": "ISBN",
            "page": 001,
            "paragraph_id": "para_id"
        },
        "type": "type"
    },
]
```
Refer to **sample_QA_Pairs/** for a complete examples for each qa pair type.

These json files are used in conjunction with the submission prompt templates to prompt your LLM. Example:

```
You are a highly knowledgeable medical expert. Answer the following short answer question accurately and concisely. Your final answer must be no more than 100 words.

Question:
{question}

Expected response output format:
Final Answer: <your concise answer here (max 100 words)>
```

The QA pair data and prompt templates are combined by the submit.py file and are used to determine scores for QA pair answers. 

Source text information will be obscured from the sample dataset to ensure participants do not attempt to recreate the dataset/ just finetune over the source texts to "game" the benchmark. 

## Important Links
- Official Challenge Website: [ClinIQLink 2025](https://brandonio-c.github.io/ClinIQLink-2025/)
- CodaBench Benchmark: [View Challenge](https://www.codabench.org/competitions/5117/)

## License
This dataset is **strictly for format validation** and **must not** be used for model training, fine-tuning, or any other purpose beyond submission verification.

For inquiries, contact **Brandon Colelough** at **brandon.colelough@nih.gov**.

