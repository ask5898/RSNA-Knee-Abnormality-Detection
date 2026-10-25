"""Read every training report with Qwen3-8B into 12 labels (notebook cell 2).

    python scripts/label_reports_llm.py --check     # a test sentence and 16 reports only
    python scripts/label_reports_llm.py             # every report -> llm_labels.csv

Needs `transformers`, a GPU (two T4s on Kaggle) and Internet to fetch the model
the first time. Hours for the full training set. The training pipeline uses the
rule-based labels; this output is for comparison or as an alternative label set.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rsna_knee.cli import add_common_args, apply_common_args  # noqa: E402
from rsna_knee.labeling.llm import LLMReportReader  # noqa: E402


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="Hugging Face model id or local path")
    p.add_argument("--batch-size", type=int, default=16, help="reports per generate call")
    p.add_argument("--out", default="llm_labels.csv")
    p.add_argument("--check", action="store_true", help="only the test sentence and the first 16 reports")
    args = p.parse_args(argv)
    apply_common_args(args)

    reader = LLMReportReader("train", temperature=0, model_path=args.model, batch_size=args.batch_size)
    print(reader._call_llm(["Complete ACL tear. Moderate joint effusion. No fracture."]))
    print(reader.predict(n=16))     # check a few reports before running everything
    if args.check:
        return
    llm_labels = reader.predict()
    llm_labels.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(llm_labels)} reports")


if __name__ == "__main__":
    main()
