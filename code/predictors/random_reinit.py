# -*- coding: utf-8 -*-
from torch import nn

import transformers

import utils.param_parser as param_parser


def random_init_ptl(conf, model):
    def _get_ptl():
        if conf.ptl == "bert":
            return getattr(transformers, "modeling_bert")
        elif conf.ptl == "distilbert":
            return getattr(transformers, "modeling_distilbert")
        elif conf.ptl == "modeling_roberta":
            return getattr(transformers, "modeling_roberta")
        else:
            raise NotImplementedError("invalid ptl.")

    def _init_weights(
        name,
        module,
        reinit_embedding=False,
        reinit_layernorm=False,
        reinit_attention_self=False,
        reinit_attention_output=False,
        reinit_intermediate=False,
        reinit_output=False,
        reinit_pooler=False,
        reinit_classifier=False,
    ):
        """ Initialize the weights """
        # if isinstance(module, (nn.Linear, nn.Embedding)) and reinit_embedding:
        #     module.weight.data.normal_(mean=0.0, std=0.02)

        if isinstance(module, nn.Embedding) and reinit_embedding:
            print("reinit embedding layer.")
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, _get_ptl().BertLayerNorm) and reinit_layernorm:
            print("reinit layer-norm.")
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear):
            flag = False
            if "attention.self" in name and reinit_attention_self:
                print("reinit linear for attention.self.")
                flag = True
            if "attention.output" in name and reinit_attention_output:
                print("reinit linear for attention.output.")
                flag = True
            if "intermediate" in name and reinit_intermediate:
                print("reinit linear for intermediate.")
                flag = True
            if "output" in name and "attention" not in name and reinit_output:
                print("reinit linear for layer output.")
                flag = True
            if "pooler" in name and reinit_pooler:
                print("reinit linear for pooler.")
                flag = True
            if "classifier" in name and reinit_classifier:
                print("reinit linear for classifier.")
                flag = True

            if flag:
                module.weight.data.normal_(mean=0.0, std=0.02)

                if module.bias is not None:
                    module.bias.data.zero_()

    if conf.random_init_ptl is None:
        return model
    else:
        random_init_ptl = (
            param_parser.dict_parser(conf.random_init_ptl)
            if conf.random_init_ptl is not None
            else None
        )

        for _name, _module in model.named_modules():
            _init_weights(
                _name,
                _module,
                reinit_embedding=random_init_ptl["reinit_embedding"]
                if "reinit_embedding" in random_init_ptl
                else False,
                reinit_layernorm=random_init_ptl["reinit_layernorm"]
                if "reinit_layernorm" in random_init_ptl
                else False,
                reinit_attention_self=random_init_ptl["reinit_attention_self"]
                if "reinit_attention_self" in random_init_ptl
                else False,
                reinit_attention_output=random_init_ptl["reinit_attention_output"]
                if "reinit_attention_output" in random_init_ptl
                else False,
                reinit_intermediate=random_init_ptl["reinit_intermediate"]
                if "reinit_intermediate" in random_init_ptl
                else False,
                reinit_output=random_init_ptl["reinit_output"]
                if "reinit_output" in random_init_ptl
                else False,
                reinit_pooler=random_init_ptl["reinit_pooler"]
                if "reinit_pooler" in random_init_ptl
                else False,
                reinit_classifier=random_init_ptl["reinit_classifier"]
                if "reinit_classifier" in random_init_ptl
                else False,
            )
        return model
