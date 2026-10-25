"""Train and save the model to MODEL_DIR (notebook cells 12 and 13).

    python scripts/train.py                          # LoRA fine-tuning (config.FINETUNE = True)
    python scripts/train.py --method frozen          # frozen ConvNeXt features + 5 fold heads
    python scripts/train.py --set FT_EPOCHS=5 --set FT_BATCH=2

Needs train_set.csv (scripts/build_train_set.py) and the competition data. The
first run downloads the ConvNeXt weights; MODEL_DIR then holds everything the
offline submission needs (backbone.pt, head_fold*.pt, config.json).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rsna_knee import config as C  # noqa: E402
from rsna_knee.cli import add_common_args, apply_common_args  # noqa: E402


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--method", choices=["lora", "frozen"],
                   help="lora: fine-tune ConvNeXt with LoRA; frozen: frozen features + fold heads "
                        f"(default: {'lora' if C.FINETUNE else 'frozen'}, from config.FINETUNE)")
    args = p.parse_args(argv)
    apply_common_args(args)
    if args.method:
        C.update(FINETUNE=args.method == "lora")
    if not C.TRAIN:
        print("TRAIN is False (the hidden-test rerun of a Kaggle submission never trains): nothing to do")
        return

    if C.FINETUNE:
        from rsna_knee.training.lora import train_lora
        train_lora()
    else:
        from rsna_knee.training.frozen import train_frozen
        train_frozen()


if __name__ == "__main__":
    main()
