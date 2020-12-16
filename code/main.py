# -*- coding: utf-8 -*-
import os
import copy
import torch

from parameters import get_args
from trainers.hooks import EvaluationRecorder, SparsityRecorder
from trainers.finetuner import BertFinetuner

import predictors.linear_predictors as linear_predictors
import predictors.random_reinit as random_reinit
import configs.task_configs as task_configs
import masking.maskers as maskers
import masking.sparsity_control as sp_control
import utils.checkpoint as checkpoint
import utils.logging as logging
import utils.param_parser as param_parser


config = dict(
    ptl="bert",
    model="bert-base-uncased",
    task="mrpc",
    model_scheme="vector_cls_sentence",
    experiment="debug",
    max_seq_len=128,
    lr=1e-3,
    world="0",
    batch_size=32,
    eval_every_batch=60,
    num_epochs=10,
    do_BL=False,
    do_MS=True,
    ptl_req_grad=False,
    classifier_req_grad=False,
    mask_classifier=True,
    mask_ptl=True,
    layers_to_mask="2,3,4,5,6,7,8,9,10,11",
    train_fast=True,
    num_snapshots=10,
    masking_scheduler_conf="lambdas_lr=0,sparsity_warmup=automated_gradual_sparsity,final_sparsity=0.05,sparsity_warmup_interval_epoch=0.1,init_epoch=0,final_epoch=1",
)


def init_task(conf):
    classes = linear_predictors.ptl2classes[conf.ptl]
    tokenizer = classes.tokenizer.from_pretrained(conf.model)
    data_iter = task_configs.task2dataiter[conf.task](
        conf.task, conf.model, tokenizer, conf.max_seq_len
    )
    conf.logger.log(f"Creating and loading pretrained {conf.ptl.upper()} model.")

    if conf.model_scheme == "vector_cls_sentence":
        model = classes.seqcls.from_pretrained(
            conf.model,
            num_labels=data_iter.num_labels,
            cache_dir=conf.pretrained_weight_path,
        )
    elif conf.model_scheme == "postagging":
        model = classes.postag.from_pretrained(
            conf.model,
            out_dim=data_iter.num_labels,
            cache_dir=conf.pretrained_weight_path,
        )
    elif conf.model_scheme == "multiplechoice":
        model = classes.multiplechoice.from_pretrained(
            conf.model, cache_dir=conf.pretrained_weight_path,
        )

    # we can have the choice to randomly (and partially) initialize the ptl.
    random_reinit.random_init_ptl(conf, model)
    return model, data_iter


def confirm_experiment(conf, model):
    # we will first put the model on the device (to reduce the initialization time).
    model = model.cuda()

    # this fn does not override conf so the path etc should be fine
    for name, param in model.named_parameters():
        param.requires_grad = False

    if conf.do_BL:
        assert (not conf.do_MS) and (not conf.mask_classifier)
        for param_name, param in model.named_parameters():
            if conf.ptl in param_name and conf.ptl_req_grad:
                param.requires_grad = True
            # note, roberta has one extra linear layer in the ``classifier'' namespace
            if "classifier" in param_name and conf.classifier_req_grad:
                param.requires_grad = True
            conf.logger.log("{} -> {}".format(param_name, param.requires_grad))
        return (model, None)

    elif conf.do_MS:
        assert (not conf.do_BL) and (not conf.ptl_req_grad)
        # for do_masking, the top classification layer is either
        # (i) masked without explicit training (ii) explict training
        if conf.classifier_req_grad:
            for _, param in model.classifier.named_parameters():
                param.requires_grad = True
            assert not conf.mask_classifier
        # else:
        # assert conf.mask_classifier
        masker = init_masker(conf, model)
        return (model, masker)
    raise ValueError("one of do_BL, do_MS must be true!")


def init_masker(conf, model):
    # init the masker scheduler.
    masker_scheduler = sp_control.MaskerScheduler(conf)

    # init the masker.
    masker = maskers.Masker(
        masker_scheduler=masker_scheduler,
        log_fn=conf.logger.log,
        mask_biases=conf.mask_biases,
        structured_masking_info={
            "structured_masking": conf.structured_masking,
            "structured_masking_types": conf.structured_masking_types_,
            "force_masking": conf.force_masking,
        },
        threshold=conf.threshold,
        init_scale=conf.init_scale,
        which_ptl=conf.ptl,
        controlled_init=conf.controlled_init,
    )

    # assuming mask all stuff in one transformer block, absorb bert.pooler directly
    weight_types = ["K", "Q", "V", "AO", "I", "O", "P"]

    # parse the get the names of layers to be masked.
    names_tobe_masked = set()
    if conf.mask_ptl:
        names_tobe_masked = maskers.chain_module_names(
            conf.ptl, conf.layers_to_mask_, weight_types
        )
    if conf.mask_classifier:
        if conf.ptl == "bert" or conf.ptl == "distilbert":
            names_tobe_masked.add("classifier")
        elif conf.ptl == "roberta":
            if (
                conf.model_scheme == "postagging"
                or conf.model_scheme == "multiplechoice"
            ):
                names_tobe_masked.add("classifier")
            elif conf.model_scheme == "vector_cls_sentence":
                names_tobe_masked.add("classifier.dense")
                names_tobe_masked.add("classifier.out_proj")

    # patch modules.
    masker.patch_modules(
        model=model,
        names_tobe_masked=names_tobe_masked,
        name_of_masker=conf.name_of_masker,
    )
    return masker


def init_recorders(conf, masker):
    # the log_fn of recorder_hook uses the self._trainer.log_fn (pls check the base_trainer.)
    if conf.do_cosinelr:
        assert conf.num_snapshots > 1
        print(
            f" initializing the cosine LR scheduler:"
            f" number of snapshots: {conf.num_snapshots}"
        )

    state_recorder = EvaluationRecorder(
        init_state_where_=os.path.join(conf.checkpoint_root, "init_state"),
        where_=os.path.join(conf.checkpoint_root, "best_state"),
        which_metric=task_configs.task2main_metric[conf.task],
    )

    if masker is not None:
        sparsity_recorder = SparsityRecorder(
            where_=f"{os.path.join(conf.checkpoint_root, f'{conf.task},sparsities')}",
            init_masks=masker.init_masks,
        )
        return [state_recorder, sparsity_recorder]
    return [state_recorder]


def finetune_on_fixed_masks(conf, model, masker, data_iter):
    _conf = copy.deepcopy(conf)
    _model = copy.deepcopy(model)
    assert _conf.do_MS

    # turn off the gradients for masks.
    for name, param in _model.named_parameters():
        if "mask" in name:
            param.requires_grad = False

    # remove the masks for classifier.
    if "classifier_wo_mask" in _conf.do_tuning_on_MS_scheme_:
        _model.classifier = maskers.MaskedLinear0(
            weight=_model.classifier.weight, bias=_model.classifier.bias,
        )
    # turn on the gradients for some layers of the berts for fine-tuning.
    if "ptl_req_grad" in _conf.do_tuning_on_MS_scheme_:
        for name, param in getattr(_model, conf.ptl).named_parameters():
            if "mask" not in name:
                param.requires_grad = True
    if "classifier_req_grad" in _conf.do_tuning_on_MS_scheme_:
        for name, param in _model.classifier.named_parameters():
            if "mask" not in name:
                param.requires_grad = True

    # specify the optimizer
    if "adam" in _conf.do_tuning_on_MS_scheme_:
        _conf.optimizer = "adam"
    else:
        raise NotImplementedError("please specify the optimizer")

    # init the recorders.
    recorder_hooks = init_recorders(_conf, masker=None)

    # init the trainer (i.e. finetuner.)
    _conf.logger.log(
        "Finetuning on the fixed masks: Initialized tasks, masks, recorders, and initing the trainer."
    )
    trainer = BertFinetuner(_conf, logger=_conf.logger, data_iter=data_iter)

    # training/tuning.
    trainer.train(model=_model, masker=None, hooks=recorder_hooks)


def main(conf):
    # general init.
    if conf.override:
        for name, value in config.items():
            assert type(getattr(conf, name)) == type(value), f"{name} {value}"
            setattr(conf, name, value)
    init_config(conf)

    # init the task.
    model, data_iter = init_task(conf)

    # init the mask.
    model, masker = confirm_experiment(conf, model)

    # init the recorders.
    recorder_hooks = init_recorders(conf, masker)

    # init the trainer (i.e. finetuner.)
    conf.logger.log("Initialized tasks, masks, recorders, and initing the trainer.")
    trainer = BertFinetuner(conf, logger=conf.logger, data_iter=data_iter)

    # training/tuning.
    conf.logger.log("Starting training/validation.")
    trainer.train(model, masker, hooks=recorder_hooks)

    # continue the fine-tuning on the trained masks.
    if conf.do_tuning_on_MS:
        finetune_on_fixed_masks(conf, model, masker, data_iter)

    # update the status.
    conf.logger.log("Finishing training/validation.")
    conf.is_finished = True
    logging.save_arguments(conf)
    os.system(f"echo {conf.checkpoint_root} >> {conf.job_id}")


def init_config(conf):
    conf.is_finished = False
    assert conf.ptl in conf.model

    # configure the training device.
    assert conf.world is not None, "Please specify the gpu ids."
    conf.world = (
        [int(x) for x in conf.world.split(",")]
        if "," in conf.world
        else [int(conf.world)]
    )
    conf.n_sub_process = len(conf.world)

    # init the masking scheduler.
    conf.masking_scheduler_conf_ = (
        param_parser.dict_parser(conf.masking_scheduler_conf)
        if conf.masking_scheduler_conf is not None
        else None
    )
    if conf.masking_scheduler_conf is not None:
        for k, v in conf.masking_scheduler_conf_.items():
            setattr(conf, f"masking_scheduler_{k}", v)

    # init the layers to mask.
    assert conf.layers_to_mask is not None, "Please specify which BERT layers to mask."
    conf.layers_to_mask_ = (
        [int(x) for x in conf.layers_to_mask.split(",")]
        if "," in conf.layers_to_mask
        else [int(conf.layers_to_mask)]
    )

    # init the params for structure pruning.
    if (
        conf.structured_masking is not None
        and conf.structured_masking_types is not None
    ):
        conf.structured_masking_types_ = conf.structured_masking_types.split(",")
    else:
        conf.structured_masking_types_ = None

    # init the params for do_tuning_on_MS_scheme
    if conf.do_tuning_on_MS:
        assert conf.do_tuning_on_MS_scheme is not None
        conf.do_tuning_on_MS_scheme_ = conf.do_tuning_on_MS_scheme.split(",")

    # re-configure batch_size if sub_process > 1.
    if conf.n_sub_process > 1:
        conf.batch_size = conf.batch_size * conf.n_sub_process

    # configure cuda related.
    assert torch.cuda.is_available()
    torch.manual_seed(conf.manual_seed)
    torch.cuda.manual_seed(conf.manual_seed)
    torch.cuda.set_device(conf.world[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True if conf.train_fast else False

    # define checkpoint for logging.
    checkpoint.init_checkpoint(conf)

    # display the arguments' info.
    logging.display_args(conf)

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_root)


if __name__ == "__main__":
    conf = get_args()

    main(conf)
