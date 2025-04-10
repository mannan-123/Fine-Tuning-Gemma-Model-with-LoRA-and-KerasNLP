# Fine-Tuning Gemma Model with KerasNLP and LoRA

This project demonstrates how to fine-tune the [Gemma Causal Language Model](https://keras.io/api/keras_nlp/models/gemma/gemma_causal_lm/) using [KerasNLP](https://keras.io/keras_nlp/) on code-based instruction datasets. The process includes data transformation, model inference, and Low Rank Adaptation (LoRA) fine-tuning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
  - [Install Dependencies](#install-dependencies)
  - [Configure Environment](#configure-environment)
- [Dataset Preparation](#dataset-preparation)
- [Model Loading and Inference](#model-loading-and-inference)
- [Fine-Tuning with LoRA](#fine-tuning-with-lora)
- [Post Fine-Tuning Inference](#post-fine-tuning-inference)
- [Improvements & Next Steps](#improvements--next-steps)
- [References](#references)

---

## 📌 Overview

This tutorial showcases:

- Using Keras 3 with the JAX backend.
- Downloading and transforming an instruction-based dataset from Hugging Face.
- Inference using the pre-trained `code_gemma_2b_en` model.
- Applying LoRA for lightweight fine-tuning.
- Re-running inference to observe improvements in generated responses.

---

## ⚙️ Setup Instructions

### Install Dependencies

```bash
!pip install -q -U keras-nlp
!pip install -q -U keras>=3
!pip install huggingface_hub
```

### Configure Environment

If you're using **Google Colab**, ensure Kaggle API credentials are loaded:

```python
from google.colab import userdata
import os

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

Set the backend to `jax` and optimize memory:

```python
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
```

---

## 📊 Dataset Preparation

We use the `"iamtarun/python_code_instructions_18k_alpaca"` dataset from Hugging Face.

```python
from huggingface_hub import hf_hub_download
import pandas as pd
import json

df = pd.read_parquet("/content/train-00000-of-00001-8b6e212f3e1ece96.parquet")

data = []
template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

for _, row in df.iterrows():
    data.append(json.dumps({"text": template.format(instruction=row['instruction'], response=row['output'])}))

with open("transformed_dataset.jsonl", "w") as file:
    for item in data[:1000]:  # Use first 1000 samples
        file.write(item + "\n")
```

---

## 🧠 Model Loading and Inference

Load the **Gemma** model and perform inference before fine-tuning.

```python
import keras
import keras_nlp

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("code_gemma_2b_en")
gemma_lm.summary()

# Run initial inference
prompt = template.format(instruction="write me  code to print fibonacci series", response="")
sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)

gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))
```

---

## 🔧 Fine-Tuning with LoRA

We enable **LoRA** to fine-tune efficiently with fewer parameters.

```python
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.preprocessor.sequence_length = 512

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

gemma_lm.fit(data[:1000], epochs=1, batch_size=1)
```

---

## 🔮 Post Fine-Tuning Inference

Generate text again after fine-tuning to see improved outputs.

```python
prompt = template.format(instruction="write me  code to fibonacii series using recursive functions", response="")
print(gemma_lm.generate(prompt, max_length=256))
```

---

## 💡 Improvements & Next Steps

To further enhance performance:

- Increase dataset size (full 18k examples)
- Train for more epochs
- Experiment with higher LoRA ranks (8, 16, etc.)
- Fine-tune hyperparameters (learning rate, weight decay)
- Use premium GPU or distributed training for larger models (e.g., Gemma 7B)
