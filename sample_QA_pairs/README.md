# Sample QA Pairs

This folder contains sample question-answer pairs for **ClinIQLink 2025 - LLM Lie Detector Task**. These examples are provided to help participants understand the expected input format for their models. Each question type aligns with the official benchmark structure.

## Structure  
```
sample_QA_pairs/
├── list.json          # List-based questions
├── MC.json            # Multiple-choice questions
├── multi_hop.json     # Multi-hop reasoning questions
├── short.json         # Short-answer questions
├── TF.json            # True/False questions
├── QA_dataset.csv     # CSV containing filenames of QA pair files in this directory
└── source_text.json   # Source text references
```

## Question Types  

### **1. List-Based Questions (`list.json`)**  
Questions requiring a **list of correct answers**.  


### **2. Multiple-Choice Questions (`MC.json`)**  
Questions where one option is the correct answer.  


### **3. Multi-Hop Questions (`multi_hop.json`)**  
Questions requiring **step-by-step reasoning** to arrive at the correct answer.  


### **4. Short-Answer Questions (`short.json`)**  
Open-ended questions that require a concise factual response.  


### **5. True/False Questions (`TF.json`)**  
Binary questions that require a **true** or **false** response.  


### **6. Source Text (`source_text.json`)**  
Contains **excerpts from reference materials** used to verify the accuracy of answers.  


## Usage  
1. Use these **example QA pairs** to test the **input-output pipeline** for your model.  
2. Ensure that your submission follows the required JSON format in **`submission_template/`**.  
3. The **source_text.json** provides references to ensure models align with verifiable medical knowledge.  

For details on generating a submission, see the **`sample_submission/`** directory.