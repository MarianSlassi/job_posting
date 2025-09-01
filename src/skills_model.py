import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, NamedTuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score as f1_span, precision_score as p_span, recall_score as r_span, classification_report
from seqeval.scheme import IOB2


class _Span(NamedTuple):
    """Word-level entity span produced from BIO tags."""
    label: str
    start: int  # inclusive word index
    end: int    # inclusive word index


@dataclass
class SkillExtractor:
    """Wrapper around a HF token-classification model for skill extraction with simple text->skills API."""
    model_name: str = "bert-base-cased"
    max_length: int = 256
    lr: float = 5e-5
    batch_train: int = 16
    batch_eval: int = 32
    num_epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    use_fp16: bool = field(default_factory=lambda: torch.cuda.is_available())
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer: Optional[AutoTokenizer] = field(init=False, default=None)
    model: Optional[AutoModelForTokenClassification] = field(init=False, default=None)
    label_list: Optional[List[str]] = field(init=False, default=None)
    label2id: Optional[Dict[str, int]] = field(init=False, default=None)
    id2label: Optional[Dict[int, str]] = field(init=False, default=None)
    trainer: Optional[Trainer] = field(init=False, default=None)
    ds_encoded: Optional[DatasetDict] = field(init=False, default=None)

    # ------------------------------
    # Data prep and training
    # ------------------------------
    @staticmethod
    def read_conll(path: Path) -> Tuple[List[List[str]], List[List[str]]]:
        """Read token<TAB>tag format, sentences separated by blank lines."""
        s_tokens, s_tags = [], []
        tokens, tags = [], []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    if tokens:
                        s_tokens.append(tokens); s_tags.append(tags)
                        tokens, tags = [], []
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                tok, lab = parts
                tokens.append(tok); tags.append(lab)
        if tokens:
            s_tokens.append(tokens); s_tags.append(tags)
        return s_tokens, s_tags

    @staticmethod
    def _ensure_bio(labels: List[str]) -> None:
        """Validate that labels are BIO-like or 'O'."""
        if any(l not in {"O"} and not re.match(r"^[BI]-", l) for l in labels):
            raise ValueError(f"Non-BIO labels found: {labels}")

    @staticmethod
    def _to_hf(tokens: List[List[str]], labels: List[List[str]], label2id: Dict[str, int]) -> Dataset:
        """Convert lists-of-lists to HuggingFace Dataset with integer labels."""
        ids = [[label2id[t] for t in seq] for seq in labels]
        return Dataset.from_dict({"tokens": tokens, "labels": ids})

    def _build_label_maps(self, y_train: List[List[str]], y_val: List[List[str]], y_test: List[List[str]]) -> None:
        """Create label mapping; keep 'O' at index 0 for stability."""
        uniq = sorted({lab for seq in (y_train + y_val + y_test) for lab in seq})
        self._ensure_bio(uniq)
        self.label_list = ["O"] + [l for l in uniq if l != "O"]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}

    def _tokenize_and_align(self, batch: Dict[str, Any], label_all_tokens: bool = False) -> Dict[str, Any]:
        """Tokenize word tokens and align word-level labels to subwords (-100 for non-first subwords)."""
        assert self.tokenizer is not None
        enc = self.tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        all_labels = []
        for i in range(len(batch["tokens"])):
            word_ids = enc.word_ids(batch_index=i)
            labels_i = batch["labels"][i]
            aligned = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)
                elif wid != prev_wid:
                    aligned.append(labels_i[wid])
                else:
                    aligned.append(labels_i[wid] if label_all_tokens else -100)
                prev_wid = wid
            all_labels.append(aligned)
        enc["labels"] = all_labels
        return enc

    def prepare_from_conll(self, train_path: Path, val_path: Path, test_path: Path) -> None:
        """Read CoNLL files, build label maps, encode datasets, and initialize the model."""
        X_train, y_train = self.read_conll(train_path)
        X_val,   y_val   = self.read_conll(val_path)
        X_test,  y_test  = self.read_conll(test_path)

        self._build_label_maps(y_train, y_val, y_test)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        d_train = self._to_hf(X_train, y_train, self.label2id)
        d_val   = self._to_hf(X_val,   y_val,   self.label2id)
        d_test  = self._to_hf(X_test,  y_test,  self.label2id)
        ds = DatasetDict({"train": d_train, "validation": d_val, "test": d_test})

        encoded = DatasetDict({
            "train": ds["train"].map(self._tokenize_and_align, batched=True, remove_columns=ds["train"].column_names),
            "validation": ds["validation"].map(self._tokenize_and_align, batched=True, remove_columns=ds["validation"].column_names),
            "test": ds["test"].map(self._tokenize_and_align, batched=True, remove_columns=ds["test"].column_names),
        })
        self.ds_encoded = encoded

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
        ).to(self.device)

    def fit(self, output_dir: Path, early_stopping_patience: int = 3) -> None:
        """Train with early stopping and keep the best checkpoint by span-level F1."""
        assert self.model and self.tokenizer and self.ds_encoded, "Call prepare_from_conll first"
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            y_true, y_pred = self._align_for_metrics(logits, labels, self.id2label)
            return {
                "precision": p_span(y_true, y_pred, scheme=IOB2),
                "recall":    r_span(y_true, y_pred, scheme=IOB2),
                "f1":        f1_span(y_true, y_pred, scheme=IOB2),
            }

        args = TrainingArguments(
            output_dir=str(output_dir),
            eval_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            per_device_train_batch_size=self.batch_train,
            per_device_eval_batch_size=self.batch_eval,
            num_train_epochs=self.num_epochs,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_steps=100,
            report_to="none",
            fp16=self.use_fp16,
            seed=self.seed,
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.ds_encoded["train"],
            eval_dataset=self.ds_encoded["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )
        self.trainer.train()

    def evaluate(self, split: str = "validation") -> Dict[str, float]:
        """Print a seqeval report and return precision/recall/F1 for the split."""
        assert self.trainer and self.ds_encoded, "Trainer or dataset missing"
        preds = self.trainer.predict(self.ds_encoded[split])
        y_true, y_pred = self._align_for_metrics(preds.predictions, preds.label_ids, self.id2label)
        print(f"\n=== {split.upper()} (span-level, IOB2) ===")
        print(classification_report(y_true, y_pred, scheme=IOB2, digits=4))
        return {
            "precision": float(p_span(y_true, y_pred, scheme=IOB2)),
            "recall": float(r_span(y_true, y_pred, scheme=IOB2)),
            "f1": float(f1_span(y_true, y_pred, scheme=IOB2)),
        }

    def save(self, save_dir: Path) -> None:
        """Save model, tokenizer, and label mapping to a directory."""
        assert self.model and self.tokenizer and self.label_list and self.label2id and self.id2label
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.trainer:
            self.trainer.save_model(str(save_dir))
        else:
            self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        with (save_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
            json.dump({"label_list": self.label_list, "label2id": self.label2id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, load_dir: Path) -> "SkillExtractor":
        """Load model, tokenizer, and label mapping from a directory."""
        with (load_dir / "label_mapping.json").open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        inst = cls()
        inst.label_list = mapping["label_list"]
        inst.label2id = {k: int(v) for k, v in mapping["label2id"].items()}
        inst.id2label = {int(v): k for k, v in inst.label2id.items()}
        inst.tokenizer = AutoTokenizer.from_pretrained(str(load_dir), use_fast=True)
        inst.model = AutoModelForTokenClassification.from_pretrained(str(load_dir)).to(inst.device)
        return inst

    # ------------------------------
    # Inference helpers
    # ------------------------------
    @staticmethod
    def _simple_word_tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Lightweight word tokenizer returning tokens and their [start, end) char spans."""
        tokens, spans = [], []
        for m in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
            tokens.append(m.group(0))
            spans.append((m.start(), m.end()))
        return tokens, spans

    def _predict_bio_for_tokens(self, tokens: List[str]) -> List[str]:
        """Run model on a word-tokenized sequence and return one BIO tag per word."""
        assert self.model and self.tokenizer and self.id2label
        enc = self.tokenizer(
            [tokens],
            is_split_into_words=True,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt"
        )
        word_ids = enc.word_ids(batch_index=0)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits[0].cpu().numpy()
        preds = np.argmax(logits, axis=1)
        tags, used = [], set()
        for idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid not in used:
                tags.append(self.id2label[int(preds[idx])])
                used.add(wid)
        if len(tags) < len(tokens):
            tags += ["O"] * (len(tokens) - len(tags))
        else:
            tags = tags[:len(tokens)]
        return tags

    @staticmethod
    def _bio_to_word_spans(tags: List[str]) -> List[_Span]:
        """Convert BIO tags to word-level spans."""
        spans, start, lab = [], None, None
        for i, t in enumerate(tags):
            if t == "O":
                if lab is not None:
                    spans.append(_Span(lab, start, i - 1))
                    lab, start = None, None
                continue
            bi, typ = t.split("-", 1)
            if bi == "B":
                if lab is not None:
                    spans.append(_Span(lab, start, i - 1))
                lab, start = typ, i
            elif bi == "I":
                pass
        if lab is not None:
            spans.append(_Span(lab, start, len(tags) - 1))
        return spans

    # ------------------------------
    # Public predict API
    # ------------------------------
    def predict(self, text: str, *, unique: bool = True) -> List[str]:
        """Return a list of extracted skills from raw text; optionally deduplicate while preserving first casing."""
        assert self.model and self.tokenizer and self.id2label, "Model is not initialized or loaded"
        tokens, char_spans = self._simple_word_tokenize(text)
        if not tokens:
            return []
        tags = self._predict_bio_for_tokens(tokens)
        word_spans = self._bio_to_word_spans(tags)

        skills: List[str] = []
        seen_norm: set[str] = set()
        for span in word_spans:
            # Keep only spans whose label denotes a skill; typical dataset uses 'SKILL'
            if span.label.upper() != "SKILL" and not span.label.upper().endswith("SKILL"):
                continue
            cs = char_spans[span.start][0]
            ce = char_spans[span.end][1]
            frag = text[cs:ce].strip()
            frag = frag.strip(".,;:!?)(")  # light cleanup
            if not frag:
                continue
            key = frag.lower()
            if unique:
                if key in seen_norm:
                    continue
                seen_norm.add(key)
            skills.append(frag)
        return skills

    # ------------------------------
    # Metrics helpers
    # ------------------------------
    @staticmethod
    def _align_for_metrics(predictions: np.ndarray, labels: np.ndarray, id2label: Dict[int, str]) -> Tuple[List[List[str]], List[List[str]]]:
        """Convert model outputs and label ids into string BIO sequences, skipping -100."""
        preds = np.argmax(predictions, axis=2)
        y_true, y_pred = [], []
        for p_seq, l_seq in zip(preds, labels):
            true_tags, pred_tags = [], []
            for p, l in zip(p_seq, l_seq):
                if int(l) == -100:
                    continue
                true_tags.append(id2label[int(l)])
                pred_tags.append(id2label[int(p)])
            y_true.append(true_tags); y_pred.append(pred_tags)
        return y_true, y_pred
