"""LLM report reader: Qwen3-8B reads each report into 12 labels (notebook cell 1).

Run it with scripts/label_reports_llm.py (notebook cell 2). Needs `transformers`
and a GPU. Its output (llm_labels.csv) is not used by the training pipeline,
which trains on the rule-based labels of rules.py.
"""
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from .. import config as C

def competition_labels(data_path=None):
    """The 12 finding columns of train.csv, as the competition names them.

    In the notebook this was cell 1's global LABELS, read when the cell ran (and
    later replaced by cell 3's labeler names); the reader keeps its own copy."""
    data_path = C.DATA_PATH if data_path is None else data_path
    labels = pd.read_csv(os.path.join(data_path, 'train.csv')).columns[2:].to_list()
    assert 'Report' not in labels and len(labels) == 12, labels
    return labels
 
RULES = """Rules:
- Judge each finding separately. A report can have several findings, or none.
- present: the report states the finding.
- uncertain: the report hedges ("possible", "suspected", "cannot be excluded").
- absent: the report negates it or calls it normal or intact ("no effusion", "ACL intact").
- not_mentioned: the report says nothing about it.
- Reports may be in any language.
- "evidence" must be a short phrase copied exactly from the report, or "" if not_mentioned."""
 
# status -> score, so the output can be compared with train.csv (only the order matters for AUC)
SCORE = {"present": 1.0, "uncertain": 0.6, "not_mentioned": 0.2, "absent": 0.0}
 
 
class LLMReportReader :
    def __init__(self, report, temperature, model_path="Qwen/Qwen3-8B", batch_size=8) :
        self.report = report            # csv name without .csv, e.g. "train"
        self.temperature = temperature
        self.batch_size = batch_size    # reports per generate call; lower it if the GPU runs out of memory
        self.labels = competition_labels()
        from transformers import AutoModelForCausalLM, AutoTokenizer   # optional dependency, only for this reader
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        # float16 because T4 GPUs don't support bfloat16; device_map="auto" splits the model over both GPUs
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
        self.model.eval()
 
    def _read_report(self) :
        report = pd.read_csv(os.path.join(C.DATA_PATH, str(self.report) + '.csv'))
        id_col = report.columns[0]
        return report[[id_col, 'Report']].dropna(subset=['Report'])
 
    def build_system_prompt(self, labels=None) -> str:
        labels = self.labels if labels is None else labels
        lines = [
            "You read knee MRI radiology reports and say, for each finding below, what the report says about it.",
            "",
            "Findings:",
        ]
        lines += [f"- {label}" for label in labels]
        lines += [
            "",
            RULES,
            "",
            'Respond with JSON only, one key per finding: '
            '{"<finding>": {"evidence": "<exact short quote>", "status": "present|uncertain|absent|not_mentioned"}, ...}',
        ]
        return "\n".join(lines)
 
    def build_user_prompt(self, report) -> str :
        return f"Report:\n{report}"
 
    def _call_llm(self, reports) -> list :
        """Takes a list of reports, returns one dict of scores per report."""
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.build_system_prompt()},
                 {"role": "user", "content": self.build_user_prompt(report)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,      # otherwise Qwen3 writes a long reasoning block before the JSON
            )
            for report in reports
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        if self.temperature > 0:
            sampling = dict(do_sample=True, temperature=self.temperature)
        else:
            sampling = dict(do_sample=False)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=768,
                                         pad_token_id=self.tokenizer.pad_token_id, **sampling)
        # keep only the newly generated tokens
        texts = self.tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.last_outputs = texts           # raw model answers, for checking why a score is NaN
        del inputs, output
        torch.cuda.empty_cache()
 
        results = []
        for text in texts:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            try:
                answer = json.loads(match.group(0))
            except (AttributeError, json.JSONDecodeError):     # no JSON or broken JSON
                answer = {}
            if not isinstance(answer, dict):
                answer = {}
            # match finding names and statuses loosely: "Present", "not mentioned", "baker's" all count
            answer = {str(k).strip().lower(): v for k, v in answer.items()}
            scores = {}
            for label in self.labels:
                item = answer.get(label.lower())
                status = item.get("status") if isinstance(item, dict) else None
                status = str(status).strip().lower().replace(" ", "_")
                scores[label] = SCORE.get(status, np.nan)
            results.append(scores)
        return results
 
    def predict(self, n=None) -> pd.DataFrame :
        reports = self._read_report()
        if n is not None:
            reports = reports.head(n)       # run on a few reports first
        id_col = reports.columns[0]
        ids, texts = reports[id_col].tolist(), reports['Report'].tolist()
        rows = []
        start = 0
        bs = self.batch_size
        ok = 0                              # batches in a row that fitted in memory
        while start < len(texts):
            batch_ids = ids[start:start + bs]
            batch = texts[start:start + bs]
            try:
                batch_scores = self._call_llm(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                ok = 0
                if bs > 1:                  # halve the batch and retry the same reports
                    bs //= 2
                    print(f"out of GPU memory, trying {bs} reports at a time")
                    continue
                print(f"{batch_ids[0]}: report too long for GPU memory, skipped")
                batch_scores = [{}]
            else:
                ok += 1
                if ok >= 10 and bs < self.batch_size:   # long reports have passed, try bigger batches again
                    bs, ok = bs * 2, 0
            for study_id, scores in zip(batch_ids, batch_scores):
                rows.append({id_col: study_id, **scores})
            start += len(batch)
            print(f"{start}/{len(texts)} reports done")
        return pd.DataFrame(rows, columns=[id_col] + self.labels)
 
