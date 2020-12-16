from .finetuner import _task2batched_fn
from .base import BaseTrainer
from tqdm import tqdm
from os.path import join
from collections import Counter

import torch
import types
import utils.eval_meters as eval_meters
import json


class Inferencer(BaseTrainer):
    def __init__(self, conf, logger, data_iter):
        super(Inferencer, self).__init__(conf, logger, None)
        self.trn_dl, self.val_dl, self.tst_dl = None, None, None
        self.task_metrics = data_iter.metrics
        self.batch_to_device = types.MethodType(_task2batched_fn[data_iter.task], self)
        self._wrap_datasplits(data_iter)

    def infer(self, model, state_dict):
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
                all_golds, all_preds, all_uids = [], [], []
                all_cls_vectors = []
                for batch_idx, batched in enumerate(tqdm(eval_dl)):
                    uids, golds, batched, _ = self.batch_to_device(batched)
                    logits, hiddens, attns = self._model_forward(model, **batched)
                    preds = (
                        torch.argmax(logits, dim=-1, keepdim=False)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    golds = golds.detach().cpu().numpy()
                    logits = logits.detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_golds.extend(golds)

                    all_uids.extend(uids)
                    all_cls_vectors.append(hiddens[-1][:, 0, :].detach().cpu())

                self._dump_hidden_state_CLS(
                    eval_name, torch.cat(all_cls_vectors).numpy(), all_uids, all_golds,
                )
                print(f"{eval_name} distribution: {Counter(all_golds)}")
                eval_res[eval_name] = {}
                for task_metric in self.task_metrics:
                    eval_fn = getattr(eval_meters, task_metric)
                    eval_res[eval_name][task_metric] = eval_fn(all_preds, all_golds)
        self.log_fn(json.dumps(eval_res))

    def _dump_hidden_state_CLS(self, eval_name, vectors, uids, golds):
        """write out representation of [CLS] from layer 11"""
        with open(join(self.conf.log_root, f"CLS,11,{eval_name}.tsv"), "w") as f:
            assert (
                vectors.shape[0] == len(uids) == len(golds)
            ), f"{vectors.shape[0]} {len(uids)} {len(golds)}"
            for i in range(vectors.shape[0]):
                myvec = []
                for j in range(vectors.shape[1]):
                    myvec.append(str(vectors[i, j]))
                myvec = " ".join(myvec)
                f.write(f"{uids[i]}\t{golds[i]}\t{myvec}\n")

    def _model_forward(self, model, **kwargs):
        if self.conf.ptl == "roberta" and "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        return model(**kwargs)
