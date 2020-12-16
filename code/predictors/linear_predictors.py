from collections import namedtuple
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import (
    BertForSequenceClassification,
    BertForMultipleChoice,
)
from transformers.modeling_roberta import (
    RobertaForSequenceClassification,
    RobertaForMultipleChoice,
    RobertaModel,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
)
from transformers.modeling_distilbert import (
    DistilBertForSequenceClassification,
    DistilBertPreTrainedModel,
    DistilBertModel,
)
from transformers.configuration_roberta import RobertaConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer
import torch
import torch.nn as nn


__all__ = ["ptl2classes"]


Classes = namedtuple("Classes", "seqcls postag multiplechoice tokenizer")


class LinearPredictor(BertPreTrainedModel):
    def __init__(self, bert_config, out_dim, dropout):
        super(LinearPredictor, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        self.classifier = nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self):
        raise NotImplementedError


class BertPOSTagger(LinearPredictor):
    def __init__(self, bert_config, out_dim, dropout=0.1):
        super(BertPOSTagger, self).__init__(bert_config, out_dim, dropout)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        if_tgts=None,
        **kwargs,
    ):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        bert_out = bert_out[0]
        bert_out = self.dropout(bert_out)
        bert_out = self.classifier(bert_out)

        logits = []
        for b_step in range(input_ids.shape[0]):
            preds = bert_out[b_step][if_tgts[b_step]]
            logits.append(preds)
        return (
            torch.cat(logits, dim=0),
            torch.argmax(bert_out, dim=-1, keepdim=False),
            bert_out,
        )


class RobertaPOSTagger(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, roberta_config, out_dim, dropout=0.1):
        super().__init__(roberta_config)
        self.roberta = RobertaModel(roberta_config)
        self.classifier = nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        if_tgts=None,
        **kwargs,
    ):
        roberta_out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        roberta_out = roberta_out[0]
        roberta_out = self.dropout(roberta_out)
        roberta_out = self.classifier(roberta_out)

        logits = []
        for b_step in range(input_ids.shape[0]):
            preds = roberta_out[b_step][if_tgts[b_step]]
            logits.append(preds)
        return (
            torch.cat(logits, dim=0),
            torch.argmax(roberta_out, dim=-1, keepdim=False),
            roberta_out,
        )


class DistilBERTPOSTagger(DistilBertPreTrainedModel):
    def __init__(self, distilbert_config, out_dim, dropout=0.1):
        super().__init__(distilbert_config)
        self.distilbert = DistilBertModel(distilbert_config)
        self.classifier = nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        if_tgts=None,
        **kwargs,
    ):
        distilbert_out = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        distilbert_out = distilbert_out[0]
        distilbert_out = self.dropout(distilbert_out)
        distilbert_out = self.classifier(distilbert_out)

        logits = []
        for b_step in range(input_ids.shape[0]):
            preds = distilbert_out[b_step][if_tgts[b_step]]
            logits.append(preds)
        return (
            torch.cat(logits, dim=0),
            torch.argmax(distilbert_out, dim=-1, keepdim=False),
            distilbert_out,
        )


class DistilBERTForMultipleChoice(DistilBertPreTrainedModel):
    def __init__(self, distilbert_config, dropout=0.1):
        super(DistilBERTForMultipleChoice, self).__init__(distilbert_config)
        self.distilbert = DistilBertModel(distilbert_config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))  # bs * 4 x msl

        attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1))
            if attention_mask is not None
            else None
        )

        position_ids = (
            position_ids.view(-1, position_ids.size(-1))
            if position_ids is not None
            else None
        )

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[0][:, 0, :]  # get cls
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        return (reshaped_logits,)


ptl2classes = {
    "bert": Classes(
        BertForSequenceClassification,
        BertPOSTagger,
        BertForMultipleChoice,
        BertTokenizer,
    ),
    "roberta": Classes(
        RobertaForSequenceClassification,
        RobertaPOSTagger,
        RobertaForMultipleChoice,
        RobertaTokenizer,
    ),
    "distilbert": Classes(
        DistilBertForSequenceClassification,
        DistilBERTPOSTagger,
        DistilBERTForMultipleChoice,
        DistilBertTokenizer,
    ),
    "mbert": Classes(BertForSequenceClassification, None, None, BertTokenizer),
}
