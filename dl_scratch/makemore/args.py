"""
Refactored from https://github.com/karpathy/makemore/blob/master/makemore.py
"""

import argparse
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: Optional[int] = None  # length of the input sequences of integers
    vocab_size: Optional[int] = None  # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4


def parse_args() -> argparse.Namespace:
    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        default="names.txt",
        help="input file with things one per line",
    )
    parser.add_argument(
        "--work-dir", "-o", type=str, default="out", help="output working directory"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="when this flag is used, we will resume optimization from existing model in the workdir",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="just sample from the model and quit, don't train",
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=4,
        help="number of data workers for both train/test",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="max number of optimization steps to run for, or -1 for infinite.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to use for compute, examples: cpu|cuda|cuda:2|mps",
    )
    parser.add_argument("--seed", type=int, default=3407, help="seed")
    # sampling
    parser.add_argument(
        "--top-k", type=int, default=-1, help="top-k for sampling, -1 means no top-k"
    )
    # model
    parser.add_argument(
        "--type",
        type=str,
        default="transformer",
        help="model class type to use, bigram|mlp|rnn|gru|bow|transformer",
    )
    parser.add_argument("--n-layer", type=int, default=4, help="number of layers")
    parser.add_argument("--n-head", type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument(
        "--n-embd", type=int, default=64, help="number of feature channels in the model"
    )
    parser.add_argument(
        "--n-embd2", type=int, default=64, help="number of feature channels elsewhere in the model"
    )
    # optimization
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="batch size during optimization"
    )
    parser.add_argument("--learning-rate", "-l", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight-decay", "-w", type=float, default=0.01, help="weight decay")
    return parser.parse_args()
