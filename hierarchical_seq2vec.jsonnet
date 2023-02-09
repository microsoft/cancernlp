// replace values with comment for actual training
local embedding_dim = 16;               // 768
local num_epochs = 1;                   // 500
local cuda_device = -1;                 // 0 for gpu
local bert_model = "tiny-bert-uncased"; // "scibert-uncased"
local train_data_purpose = "SAMPLE";    // "TRAIN"
local test_data_purpose = "SAMPLE";     // "DEV"
local distributed = false;              // true

{
    "dataset_reader": {
        "type": "doc_class_transformer_dataset_reader",
        "is_multilabel": false,
        "label_key_function": "TumorSite",
        "sentence_ann_type": "Annotation.Sentence.FixedLengthSentence",
        "token_ann_type": "Annotation.Token.TransformerInputToken",
        "token_indexer": {
            "tokens": {
                "type": "my-pretrained-indexer",
                "model_name": bert_model
            }
        },
        "preprocessing": {
            "source": {
                "type": "custom-file-source",
                "train_data_path": "data.json",
                "train_val_split": 0.8,
                "test_data_path": "data.json",
            },
            "pipeline":
            [
                {
                    "type": "simple-tokenizer",
                },
                {
                    "type": "wordpiece-tokenizer",
                    "model_name": bert_model,
                    "token_type": "Annotation.Token.SimpleToken",
                },
                {
                    "type": "max-tokens-annotator",
                    "token_type": "Annotation.Token.TransformerInputToken",
                    "max_tokens": 6400,
                },
                {
                    "type": "fixed-length-sentencizer",
                    "token_type": "Annotation.Token.TransformerInputToken",
                    "sentence_length": 128,
                },
            ]
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 8,
        }
    },
    "model": {
        "type": "multi_class_document_classifier",
        "dropout": 0.3,
        "embedding_dropout": 0.3,
        
        "seq2vec_encoder": {
            "type": "hierarchical",
            "token_seq2seq_encoder": {
                "type": "gru",
                "bidirectional": true,
                "dropout": 0.3,
                "hidden_size": embedding_dim,
                "input_size": embedding_dim
            },
            "token_seq2vec_encoder": {
                "type": "attention",
                "attention": {
                    "type": "dot_product",
                    "embedding_dim": 2 * embedding_dim,
                    "key_transform": {
                        "input_dim": embedding_dim * 2,
                        "num_layers": 1,
                        "hidden_dims": embedding_dim * 2,
                        "activations": "linear",
                        "dropout": 0.3,
                    }
                },
                "normalization": {"type": "sparsemax"},
            },
            "sentence_seq2vec_encoder": {
                "type": "attention",
                "attention": {
                    "type": "dot_product",
                    "embedding_dim": 2 * embedding_dim,
                    "key_transform": {
                        "input_dim": embedding_dim * 2,
                        "num_layers": 1,
                        "hidden_dims": embedding_dim * 2,
                        "activations": "linear",
                        "dropout": 0.3,
                    }
                },
                "normalization": {"type": "sparsemax"},
            },
        },
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "batched_embedding",
                    "bert_embedder": {
                        "type": "bert_wrapper",
                        "wrapped_embedder": {
                            "type": "my-pretrained-embedder",
                            "model_name": bert_model,
                            "train_n_layers": 0
                        }
                    }
                }
            }
        }
    },
    "train_data_path": "train",
    "validation_data_path": "validate",
    "test_data_path": "test",
    [if distributed then "distributed"]: {
        "cuda_devices": [0, 1, 2, 3],
    },
    "trainer": {
        [if !distributed then "cuda_device"]: cuda_device,
        "grad_clipping": 5,
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "adam",
            "lr": 0.001,
            "weight_decay": 0.0001
        },
        "patience": 20,
        "validation_metric": "+accuracy"
    },
    "type": "my-experiment",
}
