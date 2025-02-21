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
│   ├── MC_template.prompt
│   ├── list_template.prompt
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
1. Review the **sample_questions/** directory for example question-answer pairs.
2. Generate your model's responses using the provided question formats.
3. Structure your answers based on **submission_template.json**.
4. Compare your submission with the **sample_submission/** example.
5. Ensure the submission matches the expected format before uploading to [CodaBench](https://www.codabench.org/competitions/5117/).

## Submission Format
Each submission must follow a strict JSON format. Example:
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
Refer to **sample_submission/** for a complete example.
Source text information will be obscured from the sample dataset to ensure participants do not attempt to recreate the dataset/ just finetune over the source texts to "game" the benchmark. 

## Important Links
- Official Challenge Website: [ClinIQLink 2025](https://brandonio-c.github.io/ClinIQLink-2025/)
- CodaBench Benchmark: [View Challenge](https://www.codabench.org/competitions/5117/)

## License
This dataset is **strictly for format validation** and **must not** be used for model training, fine-tuning, or any other purpose beyond submission verification.

For inquiries, contact **Brandon Colelough** at **brandon.colelough@nih.gov**.

