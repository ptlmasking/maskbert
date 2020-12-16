from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from .bert_formatting import (
    glue_example_to_feature,
    tagging_example_to_feature,
    multiplechoice_example_to_feature,
)
from .glue.datasets import *
from .reading_compr.datasets import *
from .tagging.datasets import *
from .textcls.text_classification import *
from .semeval.datasets import *
from .ner.datasets import *
import torch
import os
import pickle
import json
import uuid


task2datadir = {
    "mrpc": "data/glue_data/data/MRPC",
    "sst2": "data/glue_data/data/SST-2",
    "mnli": "data/glue_data/data/MNLI",
    "qqp": "data/glue_data/data/QQP",
    "cola": "data/glue_data/data/CoLA",
    "qnli": "data/glue_data/data/QNLI",
    "rte": "data/glue_data/data/RTE",
    "posptb": "data/tagging/ptb/",
    "swag": "data/reading_compr/swag",
    "squad1": "data/qa/squad/v1.1",
    "agnews": "data/textcls/agnews",
    "trec": "data/textcls/trec",
    "dbpedia": "data/textcls/dbpedia",
    "yelp2": "data/textcls/yelp2",
    "semeval16": "data/semeval/",
    "conll2003": "data/ner/conll2003",
    "germeval": "data/multilingual/germeval19-task2",
    "imdb": "data/textcls/imdbparsed",
}


task2dataset = {
    "mrpc": MRPCDataset,
    "sst2": SST2Dataset,
    "mnli": MNLIDataset,
    "qqp": QQPDataset,
    "cola": COLADataset,
    "qnli": QNLIDataset,
    "rte": RTEDataset,
    "posptb": PTBTDataset,
    "swag": SWAGDataset,
    "agnews": AGNEWSDataset,
    "trec": TRECDataset,
    "dbpedia": DBPEDIADataset,
    "yelp2": YELPDataset,
    "semeval16": SEMEVAL16Dataset,
    "conll2003": CONLL2003Dataset,
    "germeval": GermEvalDataset,
    "imdb": IMDBDataset,
}


task2metrics = {
    "mrpc": ["accuracy"],
    "sst2": ["accuracy"],
    "mnli": ["accuracy"],
    "qqp": ["f1", "accuracy"],
    "cola": ["mcc"],
    "qnli": ["accuracy"],
    "rte": ["accuracy"],
    "posptb": ["accuracy"],
    "swag": ["accuracy"],
    "agnews": ["accuracy"],
    "trec": ["accuracy"],
    "dbpedia": ["accuracy"],
    "yelp2": ["accuracy"],
    "semeval16": ["accuracy"],
    "conll2003": ["f1_score_ner"],
    "germeval": ["accuracy"],
    "imdb": ["accuracy"],
}


class SeqClsDataIter(object):
    def __init__(self, task, model, tokenizer, max_seq_len):
        self.task = task
        self.metrics = task2metrics[task]
        self.pdata = task2dataset[task](task2datadir[task])
        self.num_labels = len(self.pdata.get_labels())
        self.trn_dl = self.wrap_iter(
            task, model, "trn", self.pdata.trn_egs, tokenizer, max_seq_len
        )
        self.val_dl = self.wrap_iter(
            task, model, "val", self.pdata.val_egs, tokenizer, max_seq_len
        )
        if hasattr(self.pdata, "tst_egs"):
            self.tst_dl = self.wrap_iter(
                task, model, "tst", self.pdata.tst_egs, tokenizer, max_seq_len
            )

    def wrap_iter(self, task, model, split, egs, tokenizer, max_seq_len):
        cached_ = os.path.join(
            "data", "cached", f"{task},{max_seq_len},{model},{split},cached.pkl"
        )
        meta_ = cached_.replace(".pkl", ".meta")
        if os.path.exists(cached_):
            print("[INFO] loading cached dataset.")
            with open(meta_, "r") as f:
                meta = json.load(f)
            assert meta["complete"]
            with open(cached_, "rb") as f:
                fts = pickle.load(f)
            if fts["uid"] == meta["uid"]:
                fts = fts["fts"]
            else:
                # will not do self recompute for safety
                raise ValueError("uids of data and meta do not match ...")
        else:
            print("[INFO] computing fresh dataset.")
            fts = glue_example_to_feature(
                self.task, egs, tokenizer, max_seq_len, self.label_list
            )
            uid, complete = str(uuid.uuid4()), True
            try:
                with open(cached_, "wb") as f:
                    pickle.dump({"fts": fts, "uid": uid}, f)
            except:
                complete = False
            with open(meta_, "w") as f:
                json.dump({"complete": complete, "uid": uid}, f)
        return _SeqClsIter(fts)

    @property
    def name(self):
        return self.pdata.name

    @property
    def label_list(self):
        return self.pdata.get_labels()


class _SeqClsIter(torch.utils.data.Dataset):
    def __init__(self, fts):
        self.uides = [ft.uid for ft in fts]
        self.input_idses = torch.as_tensor(
            [ft.input_ids for ft in fts], dtype=torch.long
        )
        self.golds = torch.as_tensor([ft.gold for ft in fts], dtype=torch.long)
        self.attention_maskes = torch.as_tensor(
            [ft.attention_mask for ft in fts], dtype=torch.long
        )
        self.token_type_idses = torch.as_tensor(
            [ft.token_type_ids for ft in fts], dtype=torch.long
        )

    def __len__(self):
        return self.golds.shape[0]

    def __getitem__(self, idx):
        return (
            self.uides[idx],
            self.input_idses[idx],
            self.golds[idx],
            self.attention_maskes[idx],
            self.token_type_idses[idx],
        )


class TaggingDataIter(object):
    def __init__(self, task, model, tokenizer, max_seq_len):
        self.task = task
        self.metrics = task2metrics[task]
        self.pdata = task2dataset[task](task2datadir[task])
        self.num_labels = len(self.pdata.get_labels())
        self.trn_dl = self.wrap_iter(
            task, model, "trn", self.pdata.trn_tagged_sents, tokenizer, max_seq_len
        )
        self.val_dl = self.wrap_iter(
            task, model, "val", self.pdata.val_tagged_sents, tokenizer, max_seq_len
        )
        if hasattr(self.pdata, "tst_tagged_sents"):
            self.tst_dl = self.wrap_iter(
                task, model, "tst", self.pdata.tst_tagged_sents, tokenizer, max_seq_len
            )

    def wrap_iter(self, task, model, which_split, tagged_sents, tokenizer, max_seq_len):
        cached_ = os.path.join(
            "data", "cached", f"{task},{max_seq_len},{model},{which_split},cached.pkl"
        )
        meta_ = cached_.replace(".pkl", ".meta")
        if os.path.exists(cached_):
            print("[INFO] loading cached dataset.")
            with open(meta_, "r") as f:
                meta = json.load(f)
            assert meta["complete"]
            with open(cached_, "rb") as f:
                fts = pickle.load(f)
            if fts["uid"] == meta["uid"]:
                fts = fts["fts"]
            else:
                # will not do self recompute for safety
                raise ValueError("uids of data and meta do not match ...")
        else:
            print("[INFO] computing fresh dataset.")
            fts = tagging_example_to_feature(
                which_split, tagged_sents, tokenizer, self.pdata.t2i, max_seq_len,
            )
            uid, complete = str(uuid.uuid4()), True
            try:
                with open(cached_, "wb") as f:
                    pickle.dump({"fts": fts, "uid": uid}, f)
            except:
                complete = False
            with open(meta_, "w") as f:
                json.dump({"complete": complete, "uid": uid}, f)
        return _TaggingIter(fts)

    @property
    def name(self):
        return self.pdata.name

    @property
    def label_list(self):
        return self.pdata.get_labels()


class _TaggingIter(torch.utils.data.Dataset):
    """ predict the tag of words identified by List ``if_tgt``"""

    def __init__(self, fts):
        self.uides = [ft.uid for ft in fts]
        self.input_idses = torch.as_tensor(
            [ft.input_ids for ft in fts], dtype=torch.long
        )
        self.if_tgtes = torch.as_tensor(
            [ft.sent_if_tgt for ft in fts], dtype=torch.bool
        )
        self.attention_maskes = torch.as_tensor(
            [ft.attention_mask for ft in fts], dtype=torch.long
        )
        self.tags_ides = torch.as_tensor([ft.tags_ids for ft in fts], dtype=torch.long)

    def __len__(self):
        """ NOTE: size of the dataloader refers to number of sentences,
        rather than tags.
        """
        return self.input_idses.shape[0]

    def __getitem__(self, idx):
        return (
            self.uides[idx],
            self.input_idses[idx],
            self.attention_maskes[idx],
            self.tags_ides[idx],
            self.if_tgtes[idx],
        )


class MultipleChoiceDataIter(object):
    def __init__(self, task, model, tokenizer, max_seq_len):
        self.task = task
        self.metrics = task2metrics[task]
        self.pdata = task2dataset[task](task2datadir[task])
        self.trn_dl = self.wrap_iter(
            task, model, "trn", self.pdata.trn_egs, tokenizer, max_seq_len
        )
        self.val_dl = self.wrap_iter(
            task, model, "val", self.pdata.val_egs, tokenizer, max_seq_len
        )
        if hasattr(self.pdata, "tst_egs"):
            self.tst_dl = self.wrap_iter(
                task, model, "tst", self.pdata.tst_egs, tokenizer, max_seq_len
            )

    def wrap_iter(self, task, model, split, egs, tokenizer, max_seq_len):
        cached_ = os.path.join(
            "data", "cached", f"{task},{max_seq_len},{model},{split},cached.pkl"
        )
        meta_ = cached_.replace(".pkl", ".meta")
        if os.path.exists(cached_):
            print("[INFO] loading cached dataset.")
            with open(meta_, "r") as f:
                meta = json.load(f)
            assert meta["complete"]
            with open(cached_, "rb") as f:
                fts = pickle.load(f)
            if fts["uid"] == meta["uid"]:
                fts = fts["fts"]
            else:
                # will not do self recompute for safety
                raise ValueError("uids of data and meta do not match ...")
        else:
            print("[INFO] computing fresh dataset.")
            fts = multiplechoice_example_to_feature(egs, tokenizer, max_seq_len)
            uid, complete = str(uuid.uuid4()), True
            try:
                with open(cached_, "wb") as f:
                    pickle.dump({"fts": fts, "uid": uid}, f)
            except:
                complete = False
            with open(meta_, "w") as f:
                json.dump({"complete": complete, "uid": uid}, f)
        return _MultipleChoiceIter(fts)

    @property
    def name(self):
        return self.pdata.name


class _MultipleChoiceIter(torch.utils.data.Dataset):
    def __init__(self, fts):
        self.uides = [ft.uid for ft in fts]
        self.golds = torch.as_tensor([ft.label for ft in fts], dtype=torch.long)

        self.input_idses = torch.as_tensor(
            self._select_field("input_ids", fts), dtype=torch.long
        )
        self.attention_maskes = torch.as_tensor(
            self._select_field("attention_mask", fts), dtype=torch.long
        )
        self.token_type_idses = torch.as_tensor(
            self._select_field("token_type_ids", fts), dtype=torch.long
        )

    def _select_field(self, field, fts):
        return [[_composed[field] for _composed in ft.composed] for ft in fts]

    def __len__(self):
        return self.golds.shape[0]

    def __getitem__(self, idx):
        return (
            self.uides[idx],
            self.input_idses[idx],
            self.golds[idx],
            self.attention_maskes[idx],
            self.token_type_idses[idx],
        )
