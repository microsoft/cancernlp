"""
Utilities for testing with tiny pretrained transformer models
"""
import string
from pathlib import Path

from transformers import BertConfig, BertModel

CUR_DIR = Path(__file__).resolve().parent
TINY_BERT_NAME = "tiny-bert-uncased"
TINY_BERT_DIR = CUR_DIR / TINY_BERT_NAME
TINY_BERT_EMBEDDING_DIM = 16


def ensure_tiny_bert_model(force=False):
    """
    Create a tiny BERT model suitable for unit testing, save it locally
    """

    if (
        not force
        and TINY_BERT_DIR.is_dir()
        and (TINY_BERT_DIR / "pytorch_model.bin").exists()
    ):
        # we've already created the model
        return

    TINY_BERT_DIR.mkdir(exist_ok=True)

    printable_chars = list(set(string.printable.lower()))
    printable_chars.remove("\t")
    printable_chars.remove("\n")
    printable_chars.remove(" ")

    # this is the same as bert-base-uncased
    vocab = ["[PAD]"]
    vocab += [f"unused{idx}" for idx in range(100)]
    vocab += ["[UNK]", "[CLS]", "[SEP]"]
    # then just add some stuff
    vocab += "the quick brown fox jumped over the lazy dogs".split()
    # and all the printable characters
    vocab += printable_chars
    vocab += [f"##{_char}" for _char in printable_chars]

    with open(TINY_BERT_DIR / "vocab.txt", "w") as fp:
        fp.write("\n".join(vocab))

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=TINY_BERT_EMBEDDING_DIM,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    )
    model = BertModel(config)
    # we're not going to train this, we just need it to sanity-check models
    model.save_pretrained(TINY_BERT_DIR)
