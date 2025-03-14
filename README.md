# Prompt Injection Detection Project

This project presents a prompt injection detection model based on various Natural Language Processing (NLP) techniques. The model is trained on the SPML dataset.

## Datasheet

**[SPML Dataset](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection?row=0)**

### Description
The SPML (System Prompt Meta Language) dataset is designed to evaluate and enhance the security of chatbots based on large language models (LLMs). It includes system prompts and user inputs, both safe and malicious, to test the robustness of chatbots against prompt injection attacks.

### Source
University of Washington, Paul G. Allen School of Computer Science & Engineering

### Data Format
- **File Type:** CSV
- **Data Structure:** The dataset is described in a CSV file containing examples of system prompts and user interactions.

### Dataset Size
- **Total Records:** 800 different system prompts and 2.9k different user inputs, totaling 16k rows.
- **Size:** Not specified

### Variables/Columns
- **System Prompts**
- **User Inputs (safe and malicious)**
- **Security Labels (safe/malicious)**

### Data Types
- **Text (system prompts and user inputs)**
- **Categories (security labels)**

### Preprocessing
- Data cleaning to remove outliers

### Potential Uses
- Evaluating the security of LLM-based chatbots
- Developing methods to detect and prevent prompt injection attacks
- Writing prompt injections to benchmark NLP models
