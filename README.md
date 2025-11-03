# T5 Text Summarization using PyTorch

This project demonstrates **fine-tuning the T5 Transformer model** for
text summarization using **PyTorch** and the HuggingFace Transformers
library.

## Features

-   Fine-tune T5 for summarization
-   Preprocess dataset
-   Tokenize text
-   Train & evaluate model
-   Save & load model

## Installation

### 1. Create environment

    conda create -n t5env python=3.10 -y
    conda activate t5env

### 2. Install dependencies

    pip install -r requirements.txt

## Run Notebook

    jupyter notebook

Open file: `notebooks/t5_summarization.ipynb`

## Inference Example

    from transformers import T5Tokenizer, T5ForConditionalGeneration

    model = T5ForConditionalGeneration.from_pretrained("./model")
    tokenizer = T5Tokenizer.from_pretrained("./tokenizer")

    text = "Your long article text here..."

    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt")
    summary_ids = model.generate(input_ids, max_length=120, min_length=30)

    print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
