from collections import defaultdict, Counter
import copy
import torch
import numpy as np
import types
import scipy
import utils.eval_meters as eval_meters
from .finetuner import _task2batched_fn
from .base import BaseTrainer


class NEREnsembler(BaseTrainer):
    def __init__(self, conf, logger, data_iter):
        super(NEREnsembler, self).__init__(conf, logger, None)
        self.trn_dl, self.val_dl, self.tst_dl = None, None, None
        self.task_metrics = data_iter.metrics
        self.batch_to_device = types.MethodType(_task2batched_fn[data_iter.task], self)
        self._wrap_datasplits(data_iter)
        self.model_ptl = conf.ptl
        self.uid2gold = {"val_dl": {}, "tst_dl": {}}
        self.uid2logits = {
            "val_dl": defaultdict(list),
            "tst_dl": defaultdict(list),
        }
        self.uid2probs = copy.deepcopy(self.uid2logits)
        self.uid2preds = {
            "val_dl": defaultdict(list),
            "tst_dl": defaultdict(list),
        }
        self.data_iter_t2i = data_iter.pdata.i2t

    def infer(self, model, state_dict_, state_dict):
        model.load_state_dict(state_dict["state_dict"])
        model = self._parallel_to_device(model)
        model.eval()

        eval_res = {}
        with torch.no_grad():
            for eval_name in ("val_dl", "tst_dl"):
                eval_dl = getattr(self, eval_name)
                if not eval_dl:
                    eval_res[eval_name] = None
                    continue
                all_golds_ner, all_preds_ner = [], []
                for batched in eval_dl:
                    uids, golds, batched, _golds = self.batch_to_device(batched)
                    logits, bert_out, bert_out_logits, *_ = self._model_forward(
                        model, **batched
                    )
                    assert bert_out.shape == _golds.shape
                    if_tgts = batched["if_tgts"]
                    for sent_idx in range(_golds.shape[0]):
                        uid = uids[sent_idx]
                        self.uid2logits[eval_name][uid].append(
                            bert_out_logits[sent_idx][if_tgts[sent_idx]].cpu().numpy()
                        )
                        self.uid2gold[eval_name][uid] = (
                            _golds[sent_idx][if_tgts[sent_idx]].cpu().numpy()
                        )
                        self.uid2preds[eval_name][uid].append(
                            bert_out[sent_idx][if_tgts[sent_idx]].cpu().numpy()
                        )
                        self.uid2probs[eval_name][uid].append(
                            scipy.special.softmax(
                                bert_out_logits[sent_idx][if_tgts[sent_idx]]
                                .cpu()
                                .numpy(),
                                axis=-1,
                            )
                        )

    def ensemble(self):
        for eval_name in ("val_dl", "tst_dl"):
            if not getattr(self, eval_name):
                continue

            golds = list(
                sorted(
                    self.uid2gold[eval_name].items(),
                    key=lambda x: int(x[0].split("-")[-1]),
                )
            )

            if len(golds) == 0:
                continue  # no data on eval_name, e.g., glue does not have test.

            ensembled_logits = []

            for uid, sent_logits in sorted(
                self.uid2logits[eval_name].items(),
                key=lambda x: int(x[0].split("-")[-1]),
            ):
                sent_logits = sum(sent_logits)
                sent_preds = np.argmax(sent_logits, axis=-1)
                ensembled_logits.append((uid, sent_preds))

            ensembled_probs = []

            for uid, sent_probs in sorted(
                self.uid2probs[eval_name].items(),
                key=lambda x: int(x[0].split("-")[-1]),
            ):
                sent_probs = sum(sent_probs)
                sent_preds = np.argmax(sent_probs, axis=-1)
                ensembled_probs.append((uid, sent_preds))

            ensembled_preds, for_std = self._ensemble_preds(eval_name)

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
            self.log_fn(f"ensembled results on {eval_name}\n")

            _golds = []
            for sent in golds:
                _golds.append([self.data_iter_t2i[w] for w in sent])

            for ensembled_labels in (
                "ensembled_logits",
                "ensembled_probs",
                "ensembled_preds",
            ):
                self.log_fn(f"Ensemble method: {ensembled_labels}")
                ensembled_labels = eval(ensembled_labels)
                for task_metric in self.task_metrics:
                    eval_fn = getattr(eval_meters, task_metric)
                    _ensembled_labels = []
                    for sent in ensembled_labels:
                        _ensembled_labels.append([self.data_iter_t2i[w] for w in sent])

                    eval_res = eval_fn(_ensembled_labels, _golds)
                    self.log_fn(f"{task_metric}: {eval_res}")
                self.log_fn("\n")

            avg_accs = []
            for exp_idx in [0, 1, 2, 3]:
                inidividual_preds = []
                for uid, list_of_preds in for_std:
                    idx_pred = list_of_preds[exp_idx]
                    inidividual_preds.append([self.data_iter_t2i[w] for w in idx_pred])
                avg_accs.append(eval_fn(inidividual_preds, _golds))
            self.log_fn(f"averaged scores of ensemble {np.mean(avg_accs)}")
            self.log_fn(f"std of ensemble {np.std(avg_accs)}")

    def _ensemble_preds(self, eval_name):
        for_std = []
        most_commons = []
        for uid, list_of_preds in sorted(
            self.uid2preds[eval_name].items(), key=lambda x: int(x[0].split("-")[-1]),
        ):
            list_of_preds = [list(l) for l in list_of_preds]
            sent = []
            for idx in range(len(list_of_preds[0])):
                preds = [p[idx] for p in list_of_preds]
                pred = Counter(preds).most_common(1)[0][0]
                sent.append(pred)
            most_commons.append((uid, sent))
            for_std.append((uid, list_of_preds))
        return most_commons, for_std

    def _model_forward(self, model, **kwargs):
        if (
            self.model_ptl == "roberta" or self.model_ptl == "distilbert"
        ) and "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        return model(**kwargs)
