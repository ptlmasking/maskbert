# -*- coding: utf-8 -*-
import time
import numpy as np
import torch

from .base import BaseTrainer
from .hooks.base_hook import HookContainer
from utils.stat_tracker import RuntimeTracker
from utils.timer import Timer
import utils.eval_meters as eval_meters
import optim as optim
import types


import understanding.mode_connectivity as mode_connectivity


def seqcls_batch_to_device(self, batched):
    uids = batched[0]
    input_ids, golds, attention_mask, token_type_ids = map(
        lambda x: x.cuda(), batched[1:]
    )
    return (
        uids,
        golds,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )


def tagging_batch_to_device(self, batched):
    uids = batched[0]
    input_ids, attention_mask, _golds, if_tgts = map(lambda x: x.cuda(), batched[1:])

    golds = []
    for b_step in range(_golds.shape[0]):
        gold = _golds[b_step][if_tgts[b_step]]
        golds.append(gold)

    return (
        uids,
        torch.cat(golds, dim=0),
        {"input_ids": input_ids, "attention_mask": attention_mask, "if_tgts": if_tgts,},
    )


_task2batched_fn = {
    "mrpc": seqcls_batch_to_device,
    "sst2": seqcls_batch_to_device,
    "mnli": seqcls_batch_to_device,
    "qqp": seqcls_batch_to_device,
    "cola": seqcls_batch_to_device,
    "qnli": seqcls_batch_to_device,
    "rte": seqcls_batch_to_device,
    "posptb": tagging_batch_to_device,
    "swag": seqcls_batch_to_device,
    "agnews": seqcls_batch_to_device,
    "trec": seqcls_batch_to_device,
    "dbpedia": seqcls_batch_to_device,
}


class BertFinetuner(BaseTrainer):
    def __init__(self, conf, logger, data_iter):
        super(BertFinetuner, self).__init__(conf, logger)
        self.trn_dl, self.val_dl, self.tst_dl = None, None, None
        self.task_metrics = data_iter.metrics
        self.batch_to_device = types.MethodType(_task2batched_fn[data_iter.task], self)
        self._wrap_datasplits(data_iter)

        # init for mode connectivity.
        self.num_bends = conf.model_connectivity_num_bends
        self.curve_type = conf.model_connectivity_curve_type
        self.rng_state = np.random.RandomState(conf.seed)

        # logging tools.
        self.tracker = RuntimeTracker(metrics_to_track=[])
        self.timer = Timer(
            verbosity_level=1 if True else 0, log_fn=logger.log_metric, on_cuda=True,
        )
        self.model_ptl = conf.ptl

    def train(self, model, hooks=None):
        # init the model for the training.
        opt, model = self._init_training(model)
        self.model = model  # set the model for hooks
        self.model.train()

        # init the hook for the training.
        hook_container = HookContainer(world_env={"trainer": self}, hooks=hooks)

        num_batch = len(self.trn_dl)
        self.log_fn(f"[INFO]: start training for task: {self.conf.task}")
        for epoch in range(1, self.conf.num_epochs + 1):
            for batched in self.trn_dl:
                self._update_model_with_coeffs()

                # update the step counter.
                self._batch_step += 1
                self._epoch_step = self._batch_step / num_batch

                with self.timer("load_data", epoch=self._epoch_step):
                    uids, golds, batched = self.batch_to_device(batched)

                # forward for "pretrained model+classifier".
                with self.timer("forward_pass", epoch=self._epoch_step):
                    logits, *_ = self._model_forward(**batched)
                    # the cross entropy by default uses reduction='mean'
                    loss = self.criterion(logits, golds)

                # backward for "pretrained model+classifier".
                with self.timer("backward_pass", epoch=self._epoch_step):
                    self.tracker.update_metrics(
                        metric_stat=[loss.item()], n_samples=len(logits)
                    )
                    loss.backward()

                with self.timer("perform_update", epoch=self._epoch_step):
                    opt.step()
                    opt.zero_grad()

                # logging.
                self.log_fn_json(
                    name="training",
                    values={
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "step": self._batch_step,
                        "epoch": self._epoch_step,
                        "loss": self.tracker.stat["loss"].val,
                        "avg_loss": self.tracker.stat["loss"].avg,
                    },
                    tags={"split": "train"},
                    display=True,
                )
                hook_container.on_batch_end()

                # display the timer info.
                if self._batch_step % self.conf.summary_freq == 0:
                    print(self.timer.summary())

            self._epoch_step += 1
            self.tracker.reset()
            self.logger.save_json()

        # scan the curve.
        self.log_fn(f"[INFO] Scan the curve.")
        self.scan_curves()
        self.logger.save_json()
        hook_container.on_train_end()

    def _update_model_with_coeffs(self, t=None):
        # init for mode connectivity.
        if t is None:
            t = self.rng_state.uniform(0, 1)
        if "bezier" in self.curve_type:
            coeffs = mode_connectivity.bezier_curve(t, num_bends=self.num_bends)
        elif self.curve_type == "poly_chain":
            coeffs = mode_connectivity.poly_chain(t, num_bends=self.num_bends)
        elif self.curve_type == "linear_curve":
            coeffs = mode_connectivity.linear_curve(t, num_bends=self.num_bends)
        else:
            raise NotImplementedError("the curve type is not supported yet.")

        for module in self.model.modules():
            if hasattr(module, "coeffs"):
                module.coeffs = coeffs
        return coeffs

    def scan_curves(self):
        list_of_t = np.linspace(
            start=0, stop=1, num=self.conf.model_connectivity_num_curve_points
        )

        self.model.eval()
        eval_res = {}
        message = ""
        for eval_name in ("val_dl", "tst_dl"):
            eval_dl = getattr(self, eval_name)
            if not eval_dl:
                message += f" skip evaluation on {eval_name}."
                eval_res[eval_name] = None
                continue
            message += f" finished evaluation on {eval_name}."

            # init.
            eval_res[eval_name] = {}
            for idx, t in enumerate(list_of_t):
                if idx % 10 == 0:
                    self.log_fn(f"scaned the curve: ({idx}/{len(list_of_t)}).")

                all_losses, all_golds, all_preds = [], [], []
                eval_res[eval_name][idx] = {}

                # update the model and evaluate the model.
                self._update_model_with_coeffs(t)
                for batched in eval_dl:
                    uids, golds, batched = self.batch_to_device(batched)
                    with torch.no_grad():
                        logits, *_ = self._model_forward(**batched)
                        loss = self.criterion(logits, golds).mean().item()
                        preds = torch.argmax(logits, dim=-1, keepdim=False)
                        all_losses.append(loss)
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_golds.extend(golds.detach().cpu().numpy())

                for task_metric in self.task_metrics:
                    eval_fn = getattr(eval_meters, task_metric)
                    eval_res[eval_name][idx][task_metric] = eval_fn(
                        all_preds, all_golds
                    )

            # logging.
            self.log_fn(f"[INFO] Finished evaluation: {message}")
            self.log_fn_json(
                name="evaluation",
                values={
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "step": self._batch_step,
                    "epoch": self._epoch_step,
                    "loss": loss,
                    **eval_res[eval_name],
                },
                tags={"split": eval_name},
                display=True,
            )
            self.log_fn(
                f"[INFO] Eval results on {eval_name} @ batch_step {self._batch_step}, "
                f"avg loss: {np.mean(all_losses)}."
            )
            for m_name, m_score in eval_res[eval_name].items():
                self.log_fn(
                    f"[INFO] Eval results on {eval_name} @ batch_step {self._batch_step},"
                    f"{m_name}: {m_score}."
                )
        return eval_res

    def _model_forward(self, **kwargs):  # accepting *args is removed for safety...
        if self.model_ptl == "roberta" and "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        return self.model(**kwargs)

    def _init_training(self, model):
        model = self._parallel_to_device(model)

        # define the param to optimize.
        params = [
            {
                "params": [value],
                "name": key,
                "weight_decay": self.conf.weight_decay,
                "param_size": value.size(),
                "nelement": value.nelement(),
                "lr": self.conf.lr,
            }
            for key, value in model.named_parameters()
            if value.requires_grad
        ]

        # create the optimizer.
        if self.conf.optimizer == "adam":
            opt = optim.Adam(
                params,
                lr=self.conf.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.conf.weight_decay,
            )
        else:
            raise NotImplementedError("this optimizer is not supported yet.")
        opt.zero_grad()
        model.zero_grad()
        self.log_fn(f"Initialize the optimizer: {self.conf.optimizer}")
        return opt, model
