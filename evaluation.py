# !/usr/bin/env python3

"""
Evaluation code for Quora paraphrase detection.

model_eval_paraphrase is suitable for the dev (and train) dataloaders where the label information is available.
model_test_paraphrase is suitable for the test dataloader where label information is not available.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
  SonnetsDataset,
)

TQDM_DISABLE = False


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
  label_mapping = {8505: 1, 3919: 0}  # Convert labels to 0/1 for evaluation
  inv_label_mapping = {0: 3919, 1: 8505}  # Convert predictions back to BPE tokens


  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch['attention_mask'], batch['sent_ids'], batch[
      'labels'].flatten()

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()
    
    # print(f"Raw logits: {logits[:10]}")
    # print(f"Raw preds: {preds[:10]}")
    # print(f"Raw labels: {labels[:10]}")  # Debug: Should contain only 8505 or 3919

    
    # map labels
    mapped_labels = np.array([label_mapping.get(label.item(), -1) for label in labels])

    # print(f"Mapped labels: {mapped_labels[:10]}", flush=True)

    
    valid_indices = mapped_labels >= 0
    mapped_labels = mapped_labels[valid_indices]
    preds = preds[valid_indices]

    y_true.extend(mapped_labels.tolist())
    y_pred.extend(preds.tolist())
    sent_ids.extend(b_sent_ids)
    
    # print(f"y_true: {y_true[:10]}", flush=True)
    # print(f"y preds: {y_pred[:10]}", flush=True)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)
  
  mapped_preds = [inv_label_mapping.get(pred, pred) for pred in y_pred]
  print(f"y_true (ground truth labels): {y_true[:10]}")  # Should be 0s and 1s
  print(f"y_pred (model predictions): {mapped_preds[:10]}")  # Should be 0s and 1s

  
  return acc, f1, mapped_preds, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
  inv_label_mapping = {0: 3919, 1: 8505}  # convert predictions to BPE tokens
  
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred, sent_ids = [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()
    mapped_preds = [inv_label_mapping.get(pred, pred) for pred in preds]
    
    y_pred.extend(mapped_preds)
    sent_ids.extend(b_sent_ids)

  return y_pred, sent_ids


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)