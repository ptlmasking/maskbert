from collections import defaultdict, Counter
import copy
import torch
import numpy as np
import types
import utils.eval_meters as eval_meters
from .finetuner import _task2batched_fn
from .base import BaseTrainer


class Ensembler(BaseTrainer):
    def __init__(self, conf, logger, data_iter):
        super(Ensembler, self).__init__(conf, logger, None)
        self.trn_dl, self.val_dl, self.tst_dl = None, None, None
        self.task_metrics = data_iter.metrics
        self.batch_to_device = types.MethodType(_task2batched_fn[data_iter.task], self)
        self._wrap_datasplits(data_iter)
        self.model_ptl = conf.ptl

        self.uid2gold = {"val_dl": {}, "tst_dl": {}}

        self.uid2logits = {
            "val_dl": defaultdict(lambda: np.zeros((data_iter.num_labels))),
            "tst_dl": defaultdict(lambda: np.zeros((data_iter.num_labels))),
        }

        self.uid2probs = copy.deepcopy(self.uid2logits)

        self.uid2preds = {
            "val_dl": defaultdict(list),
            "tst_dl": defaultdict(list),
        }
        if not conf.ensemble_snapshots:
            self.exp_checker = defaultdict(dict)
        else:
            self.exp_checker = None

    def infer(self, model, state_dict_, state_dict):
        if not self.conf.ensemble_snapshots:
            self.exp_checker[state_dict_]["prev_eval_res"] = state_dict["eval_res"]
            model.load_state_dict(state_dict["state_dict"])
        else:
            model.load_state_dict(state_dict)
        model = self._parallel_to_device(model)
        model.eval()

        eval_res = {}
        with torch.no_grad():
            for eval_name in ("val_dl", "tst_dl"):
                eval_dl = getattr(self, eval_name)
                if not eval_dl:
                    eval_res[eval_name] = None
                    continue
                all_golds, all_preds, uid_golds = [], [], []
                for batched in eval_dl:
                    uids, golds, batched, _golds = self.batch_to_device(batched)
                    if self.conf.model_scheme == "postagging":
                        logits, bert_out, *_ = self._model_forward(model, **batched)
                    else:
                        logits, *_ = self._model_forward(model, **batched)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    preds = (
                        torch.argmax(logits, dim=-1, keepdim=False)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    golds = golds.detach().cpu().numpy()
                    logits = logits.detach().cpu().numpy()
                    probs = probs.detach().cpu().numpy()

                    for u_idx, uid in enumerate(uids):
                        self.uid2logits[eval_name][uid] += logits[u_idx]
                        self.uid2probs[eval_name][uid] += probs[u_idx]
                        self.uid2preds[eval_name][uid].append(preds[u_idx])
                        self.uid2gold[eval_name][uid] = golds[u_idx]

                    all_preds.extend(preds)
                    all_golds.extend(golds)

                eval_res[eval_name] = {}
                for task_metric in self.task_metrics:
                    eval_fn = getattr(eval_meters, task_metric)
                    eval_res[eval_name][task_metric] = eval_fn(all_preds, all_golds)

                if not self.conf.ensemble_snapshots:
                    self.exp_checker[state_dict_]["redo_eval_res"] = eval_res

        if not self.conf.ensemble_snapshots:
            for metric, val in self.exp_checker[state_dict_]["redo_eval_res"].items():
                if self.exp_checker[state_dict_]["prev_eval_res"][metric] is None:
                    continue
                assert val == self.exp_checker[state_dict_]["prev_eval_res"][metric]

    def ensemble(self):
        for eval_name in ("val_dl", "tst_dl"):
            if not getattr(self, eval_name):
                continue

            golds = sorted(
                self.uid2gold[eval_name].items(), key=lambda x: int(x[0].split("-")[-1])
            )
            if len(golds) == 0:
                continue  # no data on eval_name, e.g., glue does not have test.
            ensembled_logits = list(
                map(
                    lambda x: (x[0], np.argmax(x[1])),
                    sorted(
                        self.uid2logits[eval_name].items(),
                        key=lambda x: int(x[0].split("-")[-1]),
                    ),
                )
            )

            ensembled_probs = list(
                map(
                    lambda x: (x[0], np.argmax(x[1])),
                    sorted(
                        self.uid2probs[eval_name].items(),
                        key=lambda x: int(x[0].split("-")[-1]),
                    ),
                )
            )

            ensembled_preds = self._ensemble_preds(eval_name)

            assert (
                len(golds)
                == len(ensembled_logits)
                == len(ensembled_preds)
                == len(ensembled_probs)
            )

            _, golds = zip(*golds)
            _, ensembled_logits = zip(*ensembled_logits)
            _, ensembled_probs = zip(*ensembled_probs)
            _, ensembled_preds = zip(*ensembled_preds)

            self.log_fn("\n")
            self.log_fn(f"showing ensemble results on {eval_name}")
            avg_accs = []

            if not self.conf.ensemble_snapshots:
                for experiment_, results in self.exp_checker.items():
                    self.log_fn(f"{experiment_}")
                    prev_res = results["prev_eval_res"][eval_name]
                    redo_res = results["redo_eval_res"][eval_name]
                    self.log_fn(f"previous results on {eval_name}: {prev_res}")
                    self.log_fn(f"recomputed results on {eval_name} {redo_res}")
                    avg_accs.append(redo_res["accuracy"])

            self.log_fn(f"averaged scores of ensemble {np.mean(avg_accs)}")
            self.log_fn(f"std of ensemble {np.std(avg_accs)}")

            self.log_fn(f"ensembled results on {eval_name}\n")
            for ensembled_labels in (
                "ensembled_logits",
                "ensembled_probs",
                "ensembled_preds",
            ):
                self.log_fn(f"Ensemble method: {ensembled_labels}")
                ensembled_labels = eval(ensembled_labels)
                for task_metric in self.task_metrics:
                    eval_fn = getattr(eval_meters, task_metric)
                    eval_res = eval_fn(ensembled_labels, golds)
                    self.log_fn(f"{task_metric}: {eval_res}")
                self.log_fn("\n")

    def _ensemble_preds(self, eval_name):
        def _arg_dist_max(pair):
            uid, preds_from_models = pair
            # there might be ties, but less probably if we have severl models
            preds_dist = sorted(
                list(Counter(preds_from_models).items()),
                key=lambda x: x[1],
                reverse=True,
            )
            return (uid, preds_dist[0][0])

        return sorted(
            map(_arg_dist_max, self.uid2preds[eval_name].items()),
            key=lambda x: int(x[0].split("-")[-1]),
        )

    def _model_forward(self, model, **kwargs):
        if (
            self.model_ptl == "roberta" or self.model_ptl == "distilbert"
        ) and "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        return model(**kwargs)
