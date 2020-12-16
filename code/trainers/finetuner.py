# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn as nn

from .base import BaseTrainer
from .hooks.base_hook import HookContainer
from .hooks.cosine_lr_scheduler import LRScheduler
from utils.stat_tracker import RuntimeTracker
from utils.timer import Timer
import utils.eval_meters as eval_meters
import optim as optim
import types
from collections import Counter


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
        None,
    )


def tagging_batch_to_device(self, batched):
    uids = batched[0]
    input_ids, attention_mask, _golds, if_tgts = map(lambda x: x.cuda(), batched[1:])

    golds = []
    for b_step in range(_golds.shape[0]):
        gold = _golds[b_step][if_tgts[b_step]]
        golds.append(gold)

    if self.conf.task != "conll2003":
        return (
            uids,
            torch.cat(golds, dim=0),
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "if_tgts": if_tgts,
            },
            None,
        )
    return (
        uids,
        torch.cat(golds, dim=0),
        {"input_ids": input_ids, "attention_mask": attention_mask, "if_tgts": if_tgts,},
        _golds,
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
    "yelp2": seqcls_batch_to_device,
    "semeval16": seqcls_batch_to_device,
    "conll2003": tagging_batch_to_device,
    "germeval": seqcls_batch_to_device,
    "imdb": seqcls_batch_to_device,
}


class BertFinetuner(BaseTrainer):
    def __init__(self, conf, logger, data_iter):
        super(BertFinetuner, self).__init__(conf, logger)
        self.trn_dl, self.val_dl, self.tst_dl = None, None, None
        self.task_metrics = data_iter.metrics
        self.batch_to_device = types.MethodType(_task2batched_fn[data_iter.task], self)
        self._wrap_datasplits(data_iter)

        # logging tools.
        self.tracker = RuntimeTracker(metrics_to_track=[])
        self.timer = Timer(
            verbosity_level=1 if conf.track_time else 0,
            log_fn=logger.log_metric,
            on_cuda=True,
        )
        self.model_ptl = conf.ptl
        if self.conf.task == "conll2003":
            self.data_iter_t2i = data_iter.pdata.i2t

    def train(self, model, masker, hooks=None):
        # init the model for the training.
        opt, model = self._init_training(model)
        self.model = model  # set the model for hooks
        self.model.train()

        # init the masker for the training.
        self.masker = masker  # set the masker for hooks
        self.masker_scheduler = masker.masker_scheduler if masker is not None else None
        self._init_sparsity_control()

        # init lr scheduler hook, I put it here since opt is defined after init_training
        if self.conf.do_cosinelr:
            scheduler = LRScheduler(
                len(self.trn_dl),
                self.conf.num_epochs,
                self.conf.num_snapshots,
                opt,
                self.conf.checkpoint_root,
            )
            hooks.append(scheduler)

        # init the hook for the training.
        hook_container = HookContainer(world_env={"trainer": self}, hooks=hooks)

        num_batch = len(self.trn_dl)
        self.log_fn(f"[INFO]: start training for task: {self.conf.task}")
        hook_container.on_train_begin()
        for epoch in range(1, self.conf.num_epochs + 1):
            for batched in self.trn_dl:
                self._batch_step += 1
                self._epoch_step = self._batch_step / num_batch

                with self.timer("validation", epoch=self._epoch_step):
                    if self._batch_step % self.conf.eval_every_batch == 0:
                        eval_res = self.evaluate()
                        hook_container.on_validation_end(eval_res=eval_res)

                with self.timer("load_data", epoch=self._epoch_step):
                    uids, golds, batched, _ = self.batch_to_device(batched)

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

                # try to control the mask sparsity via the lagrangian loss.
                with self.timer("control_sparsity_backward", epoch=self._epoch_step):
                    current_sparsity = (
                        self.masker_scheduler.get_sparsity_over_whole_model(
                            self.model, self.masker
                        )
                        if self.masker_scheduler is not None
                        else torch.tensor(0)
                    )
                    if (
                        self.masker_scheduler is not None
                        and self.masker_scheduler.is_skip is False
                    ):
                        _, target_sparsity, _ = self.masker_scheduler.step(
                            cur_epoch=self._epoch_step
                        )
                        lagrangian_loss = self.lambda_ * (
                            (target_sparsity - current_sparsity) ** 2
                        )
                        lagrangian_loss.backward()

                with self.timer("perform_update", epoch=self._epoch_step):
                    opt.step()
                    if (
                        self.masker_scheduler is not None
                        and self.masker_scheduler.is_skip is False
                    ):
                        self.sparsity_optimizer.step()
                        self.sparsity_optimizer.zero_grad()
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
                        "current_sparsity": current_sparsity.item(),
                        "target_sparsity": target_sparsity
                        if self.masker_scheduler is not None
                        and self.masker_scheduler.is_skip is False
                        else -1,
                    },
                    tags={"split": "train"},
                    display=True,
                )
                hook_container.on_batch_end()

                # early stopping.
                self._best_epoch_step = hook_container.hooks[0].best_step / num_batch
                if (
                    self.conf.early_stop is not None
                    and self._epoch_step - self._best_epoch_step > self.conf.early_stop
                ):
                    self.log_fn(
                        f"Early-stopping: current epoch={self._epoch_step},"
                        f"best_epoch={self._best_epoch_step}."
                    )
                    self.logger.save_json()
                    hook_container.on_train_end()
                    return

                # display the timer info.
                if (
                    self.conf.track_time
                    and self._batch_step % self.conf.summary_freq == 0
                ):
                    print(self.timer.summary())

            self._epoch_step += 1
            self.tracker.reset()
            self.logger.save_json()
        hook_container.on_train_end()

    def evaluate(self):
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
            all_losses, all_golds, all_preds = [], [], []
            all_golds_ner, all_preds_ner = [], []
            for batched in eval_dl:
                # golds is used for compute loss, _golds used for i2t convertion
                uids, golds, batched, _golds = self.batch_to_device(batched)
                with torch.no_grad():
                    if self.conf.model_scheme == "postagging":
                        logits, bert_out, *_ = self._model_forward(**batched)
                    else:
                        logits, *_ = self._model_forward(**batched)
                    loss = self.criterion(logits, golds).mean().item()
                    preds = torch.argmax(logits, dim=-1, keepdim=False)
                    all_losses.append(loss)
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_golds.extend(golds.detach().cpu().numpy())
                    if self.conf.task == "conll2003":
                        assert bert_out.shape == _golds.shape
                        if_tgts = batched["if_tgts"]
                        for sent_idx in range(_golds.shape[0]):
                            sent_gold = _golds[sent_idx][if_tgts[sent_idx]]
                            sent_pred = bert_out[sent_idx][if_tgts[sent_idx]]
                            all_preds_ner.append(
                                [
                                    self.data_iter_t2i[label_id.item()]
                                    for label_id in sent_pred
                                ]
                            )
                            all_golds_ner.append(
                                [
                                    self.data_iter_t2i[label_id.item()]
                                    for label_id in sent_gold
                                ]
                            )
            eval_res[eval_name] = {}
            for task_metric in self.task_metrics:
                eval_fn = getattr(eval_meters, task_metric)
                if len(all_golds_ner) == 0:
                    eval_res[eval_name][task_metric] = eval_fn(all_preds, all_golds)
                    self.log_fn(
                        f"[INFO]: gold distribution on {eval_name}: {Counter(all_golds)}"
                    )
                else:
                    eval_res[eval_name][task_metric] = eval_fn(
                        all_preds_ner, all_golds_ner
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
                    f"{m_name}: {m_score:.4f}."
                )
        self.model.train()
        return eval_res

    def _model_forward(self, **kwargs):  # accepting *args is removed for safety...
        if (
            self.model_ptl == "roberta" or self.model_ptl == "distilbert"
        ) and "token_type_ids" in kwargs:
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
                "lr": self.conf.lr_for_mask
                if self.conf.lr_for_mask is not None and "mask" in key
                else self.conf.lr,
            }
            for key, value in model.named_parameters()
            if value.requires_grad
        ]

        # create the optimizer.
        if self.conf.optimizer == "adam":
            opt = optim.Adam(
                params,
                lr=self.conf.lr,
                betas=(self.conf.adam_beta_1, self.conf.adam_beta_2),
                eps=self.conf.adam_eps,
                weight_decay=self.conf.weight_decay,
            )
        elif self.conf.optimizer == "sgd":
            opt = torch.optim.SGD(
                params,
                lr=self.conf.lr,
                momentum=self.conf.momentum_factor,
                weight_decay=self.conf.weight_decay,
                nesterov=self.conf.use_nesterov,
            )
        elif self.conf.optimizer == "signsgd":
            opt = optim.SignSGD(
                params,
                lr=self.conf.lr,
                momentum=self.conf.momentum_factor,
                weight_decay=self.conf.weight_decay,
                nesterov=self.conf.use_nesterov,
            )
        else:
            raise NotImplementedError("this optimizer is not supported yet.")
        opt.zero_grad()
        model.zero_grad()
        self.log_fn(f"Initialize the optimizer: {self.conf.optimizer}")
        return opt, model

    def _init_sparsity_control(self):
        if self.masker_scheduler is not None and self.masker_scheduler.is_skip is False:
            self.log_fn(f"Initialize to control the sparsity.")
            # init the parameters.
            self.lambda_ = nn.Parameter(torch.tensor(0.0).cuda())

            # init the optimizer for the lambda2.
            # if we also include lambda1 then the optimization becomes unstable.
            self.sparsity_optimizer = torch.optim.Adam([self.lambda_], weight_decay=0)
            self.sparsity_optimizer.param_groups[0]["lr"] = (
                -1.0
                if "lambdas_lr" not in self.conf.masking_scheduler_conf_
                else -self.conf.masking_scheduler_conf_["lambdas_lr"]
            )
