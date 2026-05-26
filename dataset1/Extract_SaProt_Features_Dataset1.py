# -*- coding: utf-8 -*-
"""
Extract_SaProt_Features_Dataset1.py

Extract SaProt representations for Dataset1 41-mer candidate sequences.

Input:
    Data/dataset1.npy
        A dict-like npy file containing:
        - sequences: 41-mer peptide sequences centered on K
        - labels   : 0/1 labels

Output:
    features/dataset1_saprot_<pooling>.npy

Default SaProt model:
    westlake-repl/SaProt_650M_AF2

Note:
    This script follows the simplified representation used in this project:
    each amino acid is converted to SaProt token "AA#". The symbol "#" is used
    as a placeholder structural token when real Foldseek 3Di tokens are not used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_LEN = 41
CENTER_INDEX = 20


def clean_seq(seq: str) -> str:
    seq = str(seq).strip().replace(" ", "").replace("\t", "").upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    return seq


def load_dataset(dataset_path: str | Path):
    data = np.load(dataset_path, allow_pickle=True).item()
    if "sequences" not in data or "labels" not in data:
        raise KeyError("dataset npy must contain 'sequences' and 'labels'.")
    sequences = np.asarray([clean_seq(s) for s in data["sequences"]])
    labels = np.asarray(data["labels"], dtype=np.int64)
    return sequences, labels


def check_sequences(sequences: np.ndarray) -> None:
    bad_len = [i for i, s in enumerate(sequences) if len(s) != EXPECTED_LEN]
    if bad_len:
        raise ValueError(f"Found sequences not equal to {EXPECTED_LEN}: {bad_len[:5]}")
    bad_center = [i for i, s in enumerate(sequences) if s[CENTER_INDEX] != "K"]
    if bad_center:
        raise ValueError(f"Found sequences whose center residue is not K: {bad_center[:5]}")


def build_saprot_tokens(seq: str, vocab: set[str], unk_token: str = "<unk>") -> str:
    tokens = []
    for aa in clean_seq(seq):
        if aa in AA20:
            token = aa + "#"
            tokens.append(token if token in vocab else unk_token)
        else:
            token = "X#"
            tokens.append(token if token in vocab else unk_token)
    return " ".join(tokens)


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> np.ndarray:
    """Pool hidden states to sample-level features."""
    # remove special tokens: [CLS] token is usually at position 0.
    # for 41 tokens, amino acid tokens are expected at 1:42.
    aa_hidden = hidden[:, 1:EXPECTED_LEN + 1, :]

    if mode == "center_only":
        return aa_hidden[:, CENTER_INDEX, :].detach().cpu().numpy()

    if mode == "mean":
        return aa_hidden.mean(dim=1).detach().cpu().numpy()

    if mode == "center_mean":
        center = aa_hidden[:, CENTER_INDEX, :]
        mean = aa_hidden.mean(dim=1)
        # keep the same dimensionality by averaging center and window mean
        return ((center + mean) / 2.0).detach().cpu().numpy()

    raise ValueError(f"Unknown pooling mode: {mode}")


def extract_saprot_features(
    sequences: np.ndarray,
    model_path: str,
    pooling: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path, output_hidden_states=True)
    model.to(device)
    model.eval()

    vocab = set(tokenizer.get_vocab().keys())
    all_features = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc=f"SaProt-{pooling}"):
            batch = sequences[start:start + batch_size]
            batch_tokens = [build_saprot_tokens(seq, vocab) for seq in batch]

            inputs = tokenizer(
                batch_tokens,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=EXPECTED_LEN + 2,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.hidden_states[-1]
            features = pool_hidden(hidden, inputs["attention_mask"], pooling)
            all_features.append(features.astype(np.float32))

    return np.concatenate(all_features, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Data/dataset1.npy")
    parser.add_argument("--model", default="westlake-repl/SaProt_650M_AF2")
    parser.add_argument("--out_dir", default="features")
    parser.add_argument("--pooling", default="center_mean", choices=["center_only", "mean", "center_mean"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences, labels = load_dataset(args.dataset)
    check_sequences(sequences)

    X = extract_saprot_features(
        sequences=sequences,
        model_path=args.model,
        pooling=args.pooling,
        batch_size=args.batch_size,
        device=args.device,
    )

    out_file = out_dir / f"dataset1_saprot_{args.pooling}.npy"
    np.save(out_file, X)

    print(f"Saved feature matrix: {out_file}")
    print(f"Shape: {X.shape}; labels: {labels.shape}")


if __name__ == "__main__":
    main()
