# -*- coding: utf-8 -*-
from os.path import join
import argparse
import time

import utils.checkpoint as checkpoint


def get_args():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint")

    # feed them to the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", type=str2bool, default=True)

    # task.
    parser.add_argument("--ptl", type=str, default="bert")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--model_scheme", type=str, default="vector_bag_sentence")
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--max_seq_len", type=int, default=128)

    # training and learning scheme
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr_for_mask", type=float, default=None)

    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)

    parser.add_argument("--momentum_factor", default=0.9, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)

    parser.add_argument(
        "--weight_decay", default=0, type=float, help="weight decay (default: 1e-4)"
    )
    parser.add_argument("--drop_rate", default=0.0, type=float)

    # masking.
    parser.add_argument("--masking_scheduler_conf", type=str, default="")
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--init_scale", type=float, default=2e-2)
    parser.add_argument(
        "--controlled_init",
        type=str,
        default=None,
        choices=["magnitude", "uniform", "magnitude_and_uniform", "double_uniform"],
    )
    parser.add_argument("--mask_biases", default=False, type=str2bool)

    parser.add_argument("--do_BL", default=True, type=str2bool)  # do baseline
    parser.add_argument("--do_MS", default=False, type=str2bool)  # do masking
    parser.add_argument("--do_tuning_on_MS", type=str2bool, default=False)
    parser.add_argument("--do_tuning_on_MS_scheme", type=str, default=None)

    parser.add_argument("--ptl_req_grad", default=True, type=str2bool)
    parser.add_argument("--classifier_req_grad", default=True, type=str2bool)
    parser.add_argument("--mask_classifier", default=False, type=str2bool)
    parser.add_argument("--mask_ptl", default=True, type=str2bool)

    parser.add_argument("--random_init_ptl", default=None, type=str)

    parser.add_argument("--structured_masking", default=None, type=str)
    parser.add_argument("--structured_masking_types", default=None, type=str)
    parser.add_argument(
        "--force_masking", type=str, choices=["all", "bert", "classifier"],
    )

    parser.add_argument("--name_of_masker", default="MaskedLinear1", type=str)

    parser.add_argument(
        "--layers_to_mask", default="0,1,2,3,4,5,6,7,8,9,10,11", type=str
    )

    # do cosine lr
    parser.add_argument("--do_cosinelr", default=False, type=str2bool)
    parser.add_argument("--num_snapshots", default=-1, type=int)

    # miscs
    parser.add_argument("--data_path", default=RAW_DATA_DIRECTORY, type=str)
    parser.add_argument("--checkpoint", default=TRAINING_DIRECTORY, type=str)
    parser.add_argument("--manual_seed", type=int, default=7, help="manual seed")
    parser.add_argument("--eval_every_batch", default=60, type=int)
    parser.add_argument("--summary_freq", default=100, type=int)
    parser.add_argument("--time_stamp", default=None, type=str)
    parser.add_argument("--train_fast", default=True, type=str2bool)
    parser.add_argument("--track_time", default=True, type=str2bool)
    parser.add_argument("--early_stop", default=None, type=float)

    """meta info."""
    parser.add_argument("--experiment", type=str, default="debug")
    parser.add_argument("--job_id", type=str, default="/tmp/jobrun_logs")
    parser.add_argument("--script_path", default="exp/", type=str)
    parser.add_argument("--script_class_name", default=None, type=str)
    parser.add_argument("--num_jobs_per_node", default=1, type=int)

    # device
    parser.add_argument(
        "--python_path", type=str, default="$HOME/conda/envs/pytorch-py3.6/bin/python"
    )
    parser.add_argument(
        "-j",
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument("--world", default="0", type=str)

    # parse conf.
    conf = parser.parse_args()
    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
