#!/usr/bin/env python3
"""
Minimal inference script for the Small Language Model (SLM) used in the Colab lab.

Usage example:
    python inference.py --prompt "Abeni," --num_tokens 20 --weights /path/to/slm_weights.h5

Notes:
- The script will attempt to rebuild the tokenizer from the Africa Galore dataset
  if a vocabulary file is not provided.
- The script expects the same model factory used in the Colab lab:
  ai_foundations.training.create_model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import pandas as pd

# The Colab lab used ai_foundations and Keras with a JAX backend config.
# Ensure these packages are available in your environment before running.
try:
    from ai_foundations import training, generation
except Exception as e:  # pragma: no cover - runtime import error
    raise RuntimeError("ai_foundations package is required. Install via: \n"
                       "pip install \"git+https://github.com/google-deepmind/ai-foundations.git@main\"") from e


class SimpleWordTokenizer:
    UNKNOWN_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    def __init__(self, corpus: Optional[list[str]] = None, vocabulary: Optional[list[str]] = None):
        import re

        self._re = re
        if vocabulary is None:
            if corpus is None:
                raise ValueError("Either corpus or vocabulary must be provided.")
            if isinstance(corpus, str):
                corpus = [corpus]
            tokens = []
            for text in corpus:
                for t in self.space_tokenize(text):
                    tokens.append(t)
            vocabulary = sorted(list(set(tokens)))
            self.vocabulary = [self.PAD_TOKEN] + vocabulary + [self.UNKNOWN_TOKEN]
        else:
            self.vocabulary = vocabulary

        self.vocabulary_size = len(self.vocabulary)
        self.token_to_index = {t: i for i, t in enumerate(self.vocabulary)}
        self.index_to_token = {i: t for i, t in enumerate(self.vocabulary)}
        self.pad_token_id = self.token_to_index[self.PAD_TOKEN]
        self.unknown_token_id = self.token_to_index[self.UNKNOWN_TOKEN]

    def space_tokenize(self, text: str) -> list[str]:
        return [t for t in self._re.split(r" +", text.strip()) if t != ""]

    def encode(self, text: str) -> list[int]:
        unk = self.unknown_token_id
        return [self.token_to_index.get(t, unk) for t in self.space_tokenize(text)]

    def decode(self, indices) -> str:
        if isinstance(indices, int):
            indices = [indices]
        return " ".join([self.index_to_token.get(i, self.UNKNOWN_TOKEN) for i in indices])


AFRICA_GALORE_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json"
)


def load_africa_galore_descriptions() -> list[str]:
    df = pd.read_json(AFRICA_GALORE_URL)
    return df["description"].values.tolist()


def build_tokenizer_or_load(vocab_path: Optional[str], dataset_for_building: bool = True) -> SimpleWordTokenizer:
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        tokenizer = SimpleWordTokenizer(vocabulary=vocab)
        return tokenizer
    else:
        if not dataset_for_building:
            raise FileNotFoundError("No vocabulary file and dataset build disabled.")
        print("Downloading Africa Galore dataset to build tokenizer (this may take a few seconds)...")
        corpus = load_africa_galore_descriptions()
        tokenizer = SimpleWordTokenizer(corpus)
        return tokenizer


def get_model(max_length: int, vocabulary_size: int):
    # Recreate the same model architecture used in the Colab lab.
    model = training.create_model(max_length=max_length, vocabulary_size=vocabulary_size, learning_rate=1e-4)
    return model


def safe_load_weights(model, weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights path not found: {weights_path}")

    # Keras can load from HDF5 or SavedModel dir; try both.
    if os.path.isdir(weights_path):
        # Attempt to load SavedModel (directory)
        try:
            model.load_weights(weights_path)
            return
        except Exception:
            # Some SavedModel layouts require tf.keras.models.load_model; attempt fallback
            import tensorflow as tf

            try:
                loaded = tf.keras.models.load_model(weights_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load SavedModel from {weights_path}: {e}")
            # Copy weights
            model.set_weights(loaded.get_weights())
            return
    else:
        # File (likely HDF5 checkpoint)
        model.load_weights(weights_path)
        return


def run_generation(prompt: str, num_tokens: int, model, tokenizer: SimpleWordTokenizer, sampling: str = "greedy") -> str:
    generated_text, _ = generation.generate_text(
        prompt,
        num_tokens,
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        sampling_mode=sampling,
    )
    return generated_text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run inference with the small language model (SLM).")
    parser.add_argument("--prompt", type=str, default="Abeni,", help="Prompt text.")
    parser.add_argument("--num_tokens", type=int, default=20, help="Number of tokens to generate.")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (e.g. .h5 or SavedModel dir).")
    parser.add_argument("--vocab", type=str, default=None, help="Optional vocabulary JSON file (list of tokens).")
    parser.add_argument("--max_length", type=int, default=299, help="Model max input length used when trained (default: 299).")
    parser.add_argument("--sampling", choices=["greedy", "random"], default="greedy", help="Sampling mode.")

    args = parser.parse_args(argv)

    tokenizer = build_tokenizer_or_load(args.vocab, dataset_for_building=True)
    print(f"Vocabulary size: {tokenizer.vocabulary_size}")

    model = get_model(max_length=args.max_length, vocabulary_size=tokenizer.vocabulary_size)

    print(f"Loading weights from: {args.weights}")
    safe_load_weights(model, args.weights)

    print("Generating text...")
    out = run_generation(args.prompt, args.num_tokens, model, tokenizer, sampling=args.sampling)

    print("\n=== Generated text ===\n")
    print(out)
    print("\n======================\n")


if __name__ == "__main__":
    main()
