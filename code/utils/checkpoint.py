# -*- coding: utf-8 -*-
import os
import time
import torch


def init_checkpoint(conf):
    # init checkpoint dir.
    invalid = True
    while invalid:
        time_id = str(int(time.time()))
        conf.time_stamp_ = f"{time_id}_model-{conf.model}_scheme-{conf.model_scheme}_task-{conf.task}_{'do_BL' if conf.do_BL else 'do_MS'}{f'_sp-{conf.masking_scheduler_final_sparsity}' if conf.do_MS else ''}{'_ptl_wgrad' if conf.ptl_req_grad else ''}{'_cl_wgrad' if conf.classifier_req_grad else ''}{'_mask_cl' if conf.mask_classifier else ''}{f'_mask_scheme-{conf.name_of_masker}' if conf.do_MS else ''}_optim-{conf.optimizer}_lr-{conf.lr:.1E}_bs-{conf.batch_size}_s-{conf.manual_seed}"
        conf.checkpoint_root = os.path.join(
            conf.checkpoint,
            conf.task,
            conf.experiment if conf.experiment is not None else "",
            conf.time_stamp_,
        )

        # if the directory does not exists, create them and break the loop.
        if not os.path.exists(conf.checkpoint_root) and build_dirs(
            conf.checkpoint_root
        ):
            invalid = False

    conf.pretrained_weight_path = os.path.join(conf.data_path, "pretrained_weights")
    print(conf.checkpoint_root)
    assert len(os.path.abspath(conf.checkpoint_root)) < 255
    return conf.checkpoint_root


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = os.path.join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def build_dirs(path):
    try:
        os.makedirs(path)
        return True
    except Exception as e:
        print(" encounter error: {}".format(e))
        return False
