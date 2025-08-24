from typing import Any, Dict, List, Optional, Union
import os
import numpy as np
import torch
from datasets import Dataset
import evaluate
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

class ClassifierModel:
    def __init__(
        self,
        model_path: str,
        *,
        finetuned: Optional[bool] = None,
        num_labels: Optional[int] = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        problem_type: Optional[str] = None,  # "single_label_classification" | "multi_label_classification" | "regression"
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self.max_length = int(max_length)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.metric_acc = evaluate.load("accuracy")
        self.metric_f1  = evaluate.load("f1")
        self._init_model_and_tokenizer(
            model_path=model_path,
            finetuned=finetuned,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type=problem_type,
        )
        self.collator = DataCollatorWithPadding(self.tokenizer)

    def _init_model_and_tokenizer(
        self,
        *,
        model_path: str,
        finetuned: Optional[bool],
        num_labels: Optional[int],
        id2label: Optional[Dict[int, str]],
        label2id: Optional[Dict[str, int]],
        problem_type: Optional[str],
    ):

        is_new_head = any(x is not None for x in (num_labels, id2label, label2id, problem_type))
        use_finetuned = finetuned if finetuned is not None else not is_new_head

        if use_finetuned:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            cfg = AutoConfig.from_pretrained(
                model_path,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                problem_type=problem_type,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=cfg)

        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label or {i: f"LABEL_{i}" for i in range(self.model.config.num_labels)}
        self.label2id = self.model.config.label2id or {v: k for k, v in self.id2label.items()}
        self.problem_type = (
            problem_type
            or getattr(self.model.config, "problem_type", None)
            or ("regression" if self.model.config.num_labels == 1 else "single_label_classification")
        )

    def reload(
        self,
        model_path: str,
        *,
        finetuned: Optional[bool] = None,
        num_labels: Optional[int] = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        problem_type: Optional[str] = None,
    ):
        self._init_model_and_tokenizer(
            model_path=model_path,
            finetuned=finetuned,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type=problem_type,
        )

    def _tokenize(self, ds: Dataset) -> Dataset:
        def _enc(batch):
            return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=self.max_length)
        remove_cols = [c for c in ds.column_names if c not in ("text", "labels")]
        return ds.map(_enc, batched=True, remove_columns=remove_cols)

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        logits, y_true = eval_pred
        if self.problem_type == "regression":
            y_pred = logits.squeeze(-1)
            y_true = y_true.astype(np.float64); y_pred = y_pred.astype(np.float64)
            mse = float(np.mean((y_pred - y_true) ** 2))
            mae = float(np.mean(np.abs(y_pred - y_true)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) or 1.0
            r2 = 1.0 - ss_res / ss_tot
            return {"mse": mse, "mae": mae, "r2": r2}
        if self.problem_type == "multi_label_classification":
            probs = 1.0 / (1.0 + np.exp(-logits))
            subset_acc = float((probs >= 0.5).astype(int).__eq__(y_true).all(axis=1).mean())
            return {"subset_accuracy": subset_acc}
        y_pred = np.argmax(logits, axis=1)
        acc = self.metric_acc.compute(predictions=y_pred, references=y_true)["accuracy"]
        f1m = self.metric_f1.compute(predictions=y_pred, references=y_true, average="macro")["f1"]
        return {"accuracy": float(acc), "macro_f1": float(f1m)}

    def fit(
        self,
        train_data: Dataset,
        eval_data: Optional[Dataset] = None,
        train_config: Optional[TrainingArguments] = None,
        save_dir: Optional[str] = None,
        early_stopping_patience: Optional[int] = 3,
    ) -> Dict[str, Any]:
        train_ds = self._tokenize(train_data)
        eval_ds = self._tokenize(eval_data) if eval_data is not None else None

        if train_config is None:
            train_config = TrainingArguments(
                output_dir="./checkpoints",
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                learning_rate=2e-5,
                weight_decay=0.01,
                warmup_ratio=0.06,
                logging_steps=100,
                evaluation_strategy="epoch" if eval_ds is not None else "no",
                save_strategy="epoch",
                load_best_model_at_end=eval_ds is not None,
                report_to="none",
                fp16=torch.cuda.is_available(),
            )

        callbacks = []
        if eval_ds is not None and early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

        trainer = Trainer(
            model=self.model,
            args=train_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=self.collator,
            compute_metrics=self._compute_metrics if eval_ds is not None else None,
            callbacks=callbacks,
        )

        trainer.train()
        metrics = {}
        if eval_ds is not None:
            metrics.update({k: float(v) for k, v in trainer.evaluate().items() if isinstance(v, (int, float))})

        out_dir = save_dir or train_config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        trainer.save_model(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(self.device).eval()
        self.id2label = self.model.config.id2label or self.id2label
        self.label2id = self.model.config.label2id or self.label2id
        self.problem_type = getattr(self.model.config, "problem_type", self.problem_type)
        return metrics

    @torch.inference_mode()
    def predict(self, data: Union[str, List[str]], return_probas: bool = True, batch_size: int = 32):
        single = isinstance(data, str)
        texts = [data] if single else list(data)
        results: List[Dict[str, Any]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            use_amp = self.device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self.model(**enc).logits

            if self.problem_type == "regression":
                vals = logits.squeeze(-1).detach().cpu().tolist()
                results.extend({"value": float(v)} for v in vals)
            elif self.problem_type == "multi_label_classification":
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                for row in probs:
                    labels = [self.id2label[i] for i, p in enumerate(row) if p >= 0.5]
                    scores = [float(p) for p in row if p >= 0.5]
                    item = {"labels": labels, "scores": scores}
                    if return_probas:
                        item["probas"] = {self.id2label[i]: float(p) for i, p in enumerate(row)}
                    results.append(item)
            else:
                probs = torch.softmax(logits, dim=-1)
                top_scores, top_ids = probs.max(dim=-1)
                for j in range(probs.size(0)):
                    pred_id = int(top_ids[j].item())
                    pred_label = self.id2label.get(pred_id, str(pred_id))
                    item = {"label": pred_label, "score": float(top_scores[j].item())}
                    if return_probas:
                        item["probas"] = {self.id2label[k]: float(probs[j, k].item()) for k in range(probs.size(1))}
                    results.append(item)

        return results[0] if single else results

    def evaluate(self, eval_data: Dataset, eval_batch_size: int = 32) -> Dict[str, float]:
        eval_ds = self._tokenize(eval_data)
        args = TrainingArguments(output_dir="./eval_tmp", per_device_eval_batch_size=eval_batch_size, report_to="none")
        trainer = Trainer(
            model=self.model,
            args=args,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=self.collator,
            compute_metrics=self._compute_metrics,
        )
        metrics = trainer.evaluate()
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}



