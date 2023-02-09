import logging
import os
import warnings
from typing import Any, Dict, List, Optional

import torch
from allennlp.commands.train import TrainModel
from allennlp.common import Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data import DataLoader, DatasetReader, Vocabulary
from allennlp.models import Model
from allennlp.training import Trainer

logger = logging.getLogger(__name__)


@TrainModel.register("my-experiment", constructor="from_partial_objects")
class MyTrainModel(TrainModel):
    """
    Subclass from AllenNLP's TrainModel, just add some extra parameters and logic we
    may want.
    """

    def finish(self, metrics: Dict[str, Any]):
        super().finish(metrics=metrics)

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        local_rank: int,
        dataset_reader: DatasetReader,
        train_data_path: str,
        model: Lazy[Model],
        data_loader: Lazy[DataLoader],
        trainer: Lazy[Trainer],
        vocabulary: Lazy[Vocabulary] = Lazy(Vocabulary),
        datasets_for_vocab_creation: List[str] = None,
        validation_dataset_reader: DatasetReader = None,
        validation_data_path: Optional[str] = None,
        validation_data_loader: Lazy[DataLoader] = None,
        test_data_path: Optional[str] = None,
        evaluate_on_test: bool = False,
        batch_weight_key: str = "",
        model_initialization_weights: str = None,
    ) -> "TrainModel":
        """
        Almost a direct copy of this method from the superclass, but adding
        'model_initialization_weights' parameter
        """

        # Train data loader.
        data_loaders: Dict[str, DataLoader] = {
            "train": data_loader.construct(
                reader=dataset_reader, data_path=train_data_path
            )
        }

        # Validation data loader.
        if validation_data_path is not None:
            validation_dataset_reader = validation_dataset_reader or dataset_reader
            if validation_data_loader is not None:
                data_loaders["validation"] = validation_data_loader.construct(
                    reader=validation_dataset_reader, data_path=validation_data_path
                )
            else:
                data_loaders["validation"] = data_loader.construct(
                    reader=validation_dataset_reader, data_path=validation_data_path
                )
                if (
                    getattr(data_loaders["validation"], "batches_per_epoch", None)
                    is not None
                ):
                    warnings.warn(
                        "Using 'data_loader' params to construct validation data "
                        "loader since 'validation_data_loader' params not specified, "
                        "but you have 'data_loader.batches_per_epoch' set which may "
                        "result in different validation datasets for each epoch.",
                        UserWarning,
                    )

        # Test data loader.
        if test_data_path is not None:
            test_dataset_reader = validation_dataset_reader or dataset_reader
            if validation_data_loader is not None:
                data_loaders["test"] = validation_data_loader.construct(
                    reader=test_dataset_reader, data_path=test_data_path
                )
            else:
                data_loaders["test"] = data_loader.construct(
                    reader=test_dataset_reader, data_path=test_data_path
                )

        if datasets_for_vocab_creation:
            for key in datasets_for_vocab_creation:
                if key not in data_loaders:
                    raise ConfigurationError(
                        f"invalid 'dataset_for_vocab_creation' {key}"
                    )

            logger.info(
                "From dataset instances, %s will be considered for vocabulary "
                "creation.",
                ", ".join(datasets_for_vocab_creation),
            )

        instance_generator = (
            instance
            for key, data_loader in data_loaders.items()
            if datasets_for_vocab_creation is None or key in datasets_for_vocab_creation
            for instance in data_loader.iter_instances()
        )

        vocabulary_ = vocabulary.construct(instances=instance_generator)

        model_ = model.construct(vocab=vocabulary_, serialization_dir=serialization_dir)

        if model_initialization_weights is not None:
            logger.info(f"Loading weights from {model_initialization_weights}")
            state_dict = torch.load(model_initialization_weights)
            model_.load_state_dict(state_dict)

        # Initializing the model can have side effect of expanding the vocabulary.
        # Save the vocab only in the primary. In the degenerate non-distributed
        # case, we're trivially the primary. In the distributed case this is safe
        # to do without worrying about race conditions since saving and loading
        # the vocab involves acquiring a file lock.
        if local_rank == 0:
            vocabulary_path = os.path.join(serialization_dir, "vocabulary")
            vocabulary_.save_to_files(vocabulary_path)

        for data_loader_ in data_loaders.values():
            data_loader_.index_with(model_.vocab)

        # We don't need to pass serialization_dir and local_rank here, because they will
        # have been passed through the trainer by from_params already, because they were
        # keyword arguments to construct this class in the first place.
        trainer_ = trainer.construct(
            model=model_,
            data_loader=data_loaders["train"],
            validation_data_loader=data_loaders.get("validation"),
        )
        assert trainer_ is not None

        return cls(
            serialization_dir=serialization_dir,
            model=model_,
            trainer=trainer_,
            evaluation_data_loader=data_loaders.get("test"),
            evaluate_on_test=evaluate_on_test,
            batch_weight_key=batch_weight_key,
        )
