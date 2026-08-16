# Train Your Own Small Language Model (SLM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14Hvmn6u72Ch46MxSrq-Ld4vAKo2bPZa4?usp=sharing)

Built and trained a custom Small Language Model (SLM) from scratch to understand the core architecture powering today's most advanced AI systems. This repository contains the Colab lab, a minimal inference script, and notes on training and evaluation.

Repository: Neel-1718/Google-DeepMind-01-Build-Your-Own-Small-Language-Model

Summary
- Developer: Neel Wagh
- Dataset: Africa Galore (public dataset used in the Colab lab)
- Model: Transformer-based SLM (~3.5M parameters)
- Training: ~200 epochs in the Colab lab (final reported loss ~0.5995)

Quick start
1. Open the interactive Colab lab (recommended for training and reproducing results): click the "Open In Colab" badge above.
2. To run a local inference test (after exporting weights from Colab), clone the repo and run:

```bash
git clone https://github.com/Neel-1718/Google-DeepMind-01-Build-Your-Own-Small-Language-Model.git
cd Google-DeepMind-01-Build-Your-Own-Small-Language-Model
python inference.py --prompt "Abeni," --num_tokens 20 --weights /path/to/weights
```

Requirements (suggested)
- Python 3.8+
- jax (the notebook used jax[cuda12]==0.7.2) — adjust for your hardware
- ai_foundations (installation from DeepMind's repo as used in the Colab lab)
- tensorflow and keras (the lab used Keras with the JAX backend configuration)
- pandas, requests

Example pip installs (use Colab or a suitable environment):

```bash
pip install "git+https://github.com/google-deepmind/ai-foundations.git@main"
pip install jax[cuda12]==0.7.2  # only on supported CUDA setup
pip install tensorflow pandas requests
```

Model weights
This repository does not include trained model weights by default. In the Colab lab you can save weights using Keras APIs (model.save or model.save_weights). After exporting the weights from Colab, place them in the repo or provide a path/URL and use the --weights flag when running inference.py.

How to export weights from Colab
- To save TensorFlow / Keras checkpoint weights:

```python
model.save_weights('/content/slm_weights.h5')
# or for a SavedModel directory
model.save('/content/slm_savedmodel')
```

- Download the saved file(s) from Colab (Files pane) and add them to a release or an external storage bucket (GCS/S3) if you want others to download them.

Technical / Methods
- Tokenization: a simple whitespace-based tokenizer (SimpleWordTokenizer). Special tokens: <PAD> and <UNK>.
- Preprocessing: tokenization of paragraphs, pad/truncate to a fixed max_length, and create input/target pairs where targets are inputs shifted left by one token.
- Model: A small Transformer implemented via ai_foundations.training.create_model. The Colab lab targeted ~3.5M parameters for efficient experimentation.
- Training: Dataset shuffled and batched (batch_size 32 in the lab), trained for ~200 epochs with learning_rate=1e-4. Training prints regular sample generations to track qualitative progress.
- Evaluation: Next-token probability visualization, greedy and stochastic sampling to inspect generation quality and sensitivity to prompt changes.

Files in this commit
- README.md (this file)
- CHANGELOG.md (release notes / summary)
- inference.py (minimal CLI inference script — needs exported weights)

What I did
I added a README with a direct link to your Colab, a short technical summary and usage instructions, a CHANGELOG summarizing the Colab run, and a minimal inference script that rebuilds the tokenizer and model and can load weights exported from the Colab notebook.

Next steps (optional)
- If you want, I can add a small example notebook for inference, or a release with uploaded weights (if you provide the weights or a download URL).
- I can also add CONTRIBUTING.md / LICENSE files if you'd like a specific license.
