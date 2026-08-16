# Changelog

All notable changes to this project are documented in this file.

## 2026-08-16 — Initial release / Colab lab
- Added interactive Colab lab: "Train Your Own Small Language Model" (link in README).
- Dataset: Africa Galore (public dataset used in the Colab lab).
- Model: Transformer-based Small Language Model (SLM), ~3.5M parameters.
- Training details:
  - epochs: 200
  - batch_size: 32
  - learning_rate: 1e-4
  - final reported training loss (end of run): ~0.5995
- Key artifacts:
  - Colab notebook for full train/evaluate workflow (Open In Colab badge in README)
  - README with quick-start and Technical/Methods summary
  - (Planned) minimal inference script to rebuild tokenizer/model and load weights for generation
- Notes:
  - Trained weights are not included in the repository by default. Use the Colab notebook to export weights (model.save or model.save_weights) and then add them to a release or external storage for others to download.
