"""
Utilities for loading different BERT models from canonical locations in blob storage
"""

import logging
from typing import List, Optional, Union

from transformers import AutoModel, AutoTokenizer, BertModel

from ..file_utils import LazyEnvVar, LazyPath
from .tiny_bert import TINY_BERT_DIR, TINY_BERT_NAME, ensure_tiny_bert_model

BERT_MODELS_DIR = LazyPath(LazyEnvVar("BERT_MODELS_DIR"))

_SCI_BERT_UNCASED_MODEL_DIR = LazyPath(BERT_MODELS_DIR, "scibert_scivocab_uncased")

_SCI_BERT_CASED_MODEL_DIR = LazyPath(BERT_MODELS_DIR, "scibert_scivocab_cased")


_WWM_PUBMED_BERT_PUBMED_VOCAB_DIR = LazyPath(
    BERT_MODELS_DIR, "wwm_pubmed_bert_pubmed_vocab"
)


# make sure tiny_bert is always available
ensure_tiny_bert_model()
_TINY_BERT_MODEL_DIR = LazyPath(str(TINY_BERT_DIR))

logger = logging.getLogger(__name__)


class PretrainedTransformerModels:

    # cache of loaded models
    _loaded_models = {}

    # cache of loaded tokenizers
    _loaded_tokenizers = {}

    _custom_types_map = {
        "sci_bert": _SCI_BERT_UNCASED_MODEL_DIR,
        "scibert-uncased": _SCI_BERT_UNCASED_MODEL_DIR,
        "scibert-cased": _SCI_BERT_CASED_MODEL_DIR,
        "wwm_pubmed_bert_pubmed_vocab-uncased": _WWM_PUBMED_BERT_PUBMED_VOCAB_DIR,
        TINY_BERT_NAME: _TINY_BERT_MODEL_DIR,
    }

    _custom_types_lowercase = {"sci_bert": True, "bio_bert": False}

    @classmethod
    def get_model_name(cls, model_name: str) -> str:
        """
        Map custom transformer model names (e.g. scibert-uncased) to model locations,
        or return the original name if it is a standard pretrained type (e.g.
        bert-base-uncased). The result of this mapping should be loadable by the
        huggingface transformers library via AutoModel.from_pretrained() /
        AutoTokenizer.from_pretrained()

        :param model_name:
        :return:
        """
        path = cls._custom_types_map.get(model_name)
        if path:
            model_name = path.get_valid_path()
        return model_name

    @classmethod
    def get_tokenizer_name(cls, model_name: str) -> str:
        """
        Map custom transformer model names (e.g. scibert-uncased) to model locations.
        This is mostly the same as get_model_name, but we need some extra logic to
        map biobert to the bert-base-uncased tokenizer.

        :param model_name:
        :return:
        """
        if model_name == "bio_bert":
            # BioBERT uses the bert-base-cased vocabulary
            model_name = "bert-base-cased"
        return cls.get_model_name(model_name)

    @classmethod
    def get_pretrained_model(cls, model_name: str):
        """
        Gets a BertModel object (ie., the huggingface implementation of BERT).
        bert_type may be one of {"sci_bert" or "bert_base"} Note that this returns a
        singleton variable. That is, there is only 1 instance of each model (
        subsequent calls return the same object)
        """
        model_name = cls.get_model_name(model_name)
        model = cls._loaded_models.get(model_name)
        logger.info("loading model %s", model_name)
        if model is None:
            model = AutoModel.from_pretrained(model_name)
        return model

    @classmethod
    def should_do_lower_case(cls, model_name: str):
        # try to determine whether tokens should be lowercased
        if model_name.endswith("-uncased"):
            do_lower_case = True
        elif model_name.endswith("-cased"):
            do_lower_case = False
        else:
            do_lower_case = cls._custom_types_lowercase.get(model_name)

        if do_lower_case is None:
            raise ValueError(
                "Unable to determine if BERT tokenizer should perform "
                "lowercasing, please set do_lower_case explicitly"
            )

        return do_lower_case

    @classmethod
    def get_pretrained_tokenizer(
        cls, model_name: str, do_lower_case: Optional[bool] = None
    ):
        """
        Returns a BertTokenizer object. bert_type may be one of {"sci_bert" or
        "bert_base"} Note that this returns a singleton variable. That is, there is
        only 1 instance of these per notebook.
        """
        if do_lower_case is None:
            do_lower_case = cls.should_do_lower_case(model_name)

        model_name = cls.get_tokenizer_name(model_name)
        tokenizer = cls._loaded_tokenizers.get(model_name)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, do_lower_case=do_lower_case
            )
            cls._loaded_tokenizers[model_name] = tokenizer

        return tokenizer

    @staticmethod
    def _any_match(list_of_matches, str_to_match):
        for l in list_of_matches:
            if l in str_to_match:
                return True
        return False

    @classmethod
    def set_bert_layers_to_train(
        cls,
        bert_model: BertModel,
        layers_to_train: Union[int, List[int]],
        train_pooler: bool = False,
        train_embedding: bool = False,
    ) -> None:
        """
        Set the layers to train, typically for partial fine-tuning.

        :param bert_model: huggingface BERT model
        :param layers_to_train: either an integer specifying how many of the top
        layers to train
        :param train_pooler: if True, train the BERT pooling weights
        :param train_embedding: if True, train the BERT base-level embedding weights
        """
        to_train = {}
        to_freeze = {}
        param_names_to_train = []
        if train_pooler:
            param_names_to_train.append("pooler")
        if train_embedding:
            param_names_to_train.append("embedding")

        # total number of layers in BERT model
        n_layers = bert_model.config.num_hidden_layers

        if isinstance(layers_to_train, int):
            layers_to_train = list(range(n_layers - layers_to_train, n_layers))
        for layer_idx in layers_to_train:
            param_names_to_train.append(f"layer.{layer_idx}.")

        for name, param in bert_model.named_parameters():
            if not cls._any_match(param_names_to_train, name):
                to_freeze[name] = param
            else:
                to_train[name] = param

        logger.info(f"Will train {len(to_train)} BERT parameters")
        logger.info(f"Will freeze {len(to_freeze)} BERT parameters")

        for param in to_train.values():
            param.requires_grad = True
        for param in to_freeze.values():
            param.requires_grad = False
