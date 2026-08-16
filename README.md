# Train Your Own Small Language Model (SLM)

[![Open gdm_lab_1_5 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neel-1718/Google-DeepMind-01-Build-Your-Own-Small-Language-Model/blob/main/gdm_lab_1_5_train_your_own_small_language_model.ipynb)
[![Open gdm_lab_1_4 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neel-1718/Google-DeepMind-01-Build-Your-Own-Small-Language-Model/blob/main/gdm_lab_1_4_prepare_the_dataset_for_training_a_slm.ipynb)
[![Open gdm_lab_1_3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neel-1718/Google-DeepMind-01-Build-Your-Own-Small-Language-Model/blob/main/gdm_lab_1_3_compare_n_gram_models_and_transformer_language_models.ipynb)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Neel-1718/Google-DeepMind-01-Build-Your-Own-Small-Language-Model?style=social)](https://github.com/Neel-1718/Google-DeepMind-01-Build-Your-Own-Small-Language-Model/stargazers)


Welcome — this repository contains a short, hands-on course-style set of Colab labs that show how to build, train, and evaluate a small transformer language model (SLM) from scratch using the Africa Galore dataset.

Why this repo
- Practical labs: step-by-step Colab notebooks designed so you can run everything in Colab (GPU recommended).
- Reproducible: requirements.txt and instructions to export/import trained weights.
- Learn by doing: tokenization, preprocessing, model definition, training loop, text generation, and evaluation.

Quick links
- gdm_lab_1_5_train_your_own_small_language_model.ipynb — Train your own SLM (end-to-end). Recommended: run on Colab with GPU.
- gdm_lab_1_4_prepare_the_dataset_for_training_a_slm.ipynb — Prepare dataset: tokenization, vocabulary, encode/decode.
- gdm_lab_1_3_compare_n_gram_models_and_transformer_language_models.ipynb — Compare n-gram vs transformer models; visualizations and analysis.
- inference.py — Minimal CLI for running generation from exported weights.

Quick start (one-minute)
1. Open the main Colab lab: click the 'Open in Colab' badge for gdm_lab_1_5.
2. Run cells in order. For best performance choose Runtime → Change runtime type → GPU.
3. After training, export weights (example in the notebook):

```python
# in Colab after training
model.save_weights('/content/slm_weights.h5')
# or
model.save('/content/slm_savedmodel')
```

4. Download the saved file and run local inference:

```bash
git clone https://github.com/Neel-1718/Google-DeepMind-01-Build-Your-Own-Small-Language-Model.git
cd Google-DeepMind-01-Build-Your-Own-Small-Language-Model
pip install -r requirements.txt
# install ai_foundations (DeepMind package) explicitly:
pip install "git+https://github.com/google-deepmind/ai-foundations.git@main"
# Run inference using exported weights
python inference.py --prompt "Abeni," --num_tokens 20 --weights /path/to/slm_weights.h5
```

Requirements (recommended)
- Python 3.8+
- jax (the notebooks use jax[cuda12]==0.7.2 on supported CUDA setups)
- ai_foundations (install from the DeepMind repo as shown above)
- tensorflow, keras, pandas, requests

See requirements.txt for suggestion pins.

Practical labs (what you'll learn)
- Tokenization & vocab: whitespace tokenizer, special tokens (<PAD>, <UNK>), encode/decode.
- Padding & batching: pad/truncate to fixed max_length, create input/target pairs.
- Model: small transformer via ai_foundations.training.create_model.
- Training: batch training loop, callbacks for sample generation, monitor loss.
- Evaluation: next-token probability plots, sampling modes (greedy vs random), prompt sensitivity.

Contributing
If you find issues or want to add improvements (example notebooks, datasets, weights), please read CONTRIBUTING.md and submit a PR. Stars and forks are appreciated — they help others find the project.

License
This project is licensed under the MIT License — see LICENSE for details.

Contact
Owner: Neel Wagh — https://github.com/Neel-1718

