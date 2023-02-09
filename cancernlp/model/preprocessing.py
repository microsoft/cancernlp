"""
Framework for loading and preprocessing a dataset
"""
import copy
import itertools
import logging
import os
from typing import Callable, Iterable, Iterator, List, Tuple, Type

import more_itertools
from allennlp.common import Params, Registrable
from overrides import overrides
from tqdm import tqdm

from ..annotation import AnnotatedText, AnnotatedTextInterface, Corpus

logger = logging.getLogger(__name__)


class CacheableBase:
    """Interface for cacheable components"""

    def get_config(self) -> dict:
        raise NotImplementedError()


class CacheableComponent(Registrable, CacheableBase):
    """Base class of components that retain configuration information in order to
    provide a hash for caching"""

    # Dictionary of parameters from Registrable object.  This should be overridden
    # by particular instances.
    _config_params = None

    @classmethod
    def from_params(
        cls: Type["CacheableComponent"],
        params: Params,
        constructor_to_call: Callable[..., "CacheableComponent"] = None,
        constructor_to_inspect: Callable[..., "CacheableComponent"] = None,
        **extras,
    ) -> "CacheableComponent":
        # We just need to save the params so we can hash them and use the hash to
        # cache this preprocessing.  We're only going to use the config of "pipeline"
        # for the hash.
        #
        # We make a copy because params are 'popped' during object construction,
        # and as_ordered_dict() doesn't make deep copies of lists -- so if we have
        # Registrable objects inside lists, their parameters they may disappear
        # instead of ending up in the config.

        params_dict = copy.deepcopy(params.as_ordered_dict())
        # call superclass version
        constructed_ob = super().from_params(
            params=params,
            constructor_to_call=constructor_to_call,
            constructor_to_inspect=constructor_to_inspect,
            **extras,
        )
        constructed_ob._set_config_params(params_dict)
        return constructed_ob

    def _set_config_params(self, params_dict: dict):
        self._config_params = params_dict

    def get_config(self) -> dict:
        if self._config_params is None:
            raise ValueError("config params not initialized!")

        return self._config_params


class PipelineSource(Registrable):
    """First component in a pipeline, responsible for creating initial dataset by
    name"""

    def __call__(self, dataset_name: str) -> Corpus:
        raise NotImplementedError(
            "Subclasses of PipelineSource should implement __call__"
        )

    def get_dataset_config(self, dataset_name: str) -> dict:
        """The default implementation simply logs the dataset name, but this may be
        overridden by subclasses."""
        return {"dataset_name": dataset_name}


class PipelineComponent(CacheableComponent):
    """
    General pipeline component that takes individual AnnotateTextInterface objects
    and returns a list of zero or more AnnotatedTextInterface objects.  The
    variable-length output can be used for data filtering / data augmentation.

    The reset function resets the state of a PipelineComponent. This is used for
    stateful components.
    """

    @classmethod
    def name(cls) -> str:
        """Human-interpretable name for component, defaults to class name"""
        return cls.__name__

    def process(self, ann_text: AnnotatedTextInterface) -> List[AnnotatedTextInterface]:
        raise NotImplementedError(
            "Subclasses of PipelineComponent should implement process()"
        )

    def reset(self) -> None:
        # default is to do nothing
        pass

    def process_corpus(
        self, corpus: Corpus, show_progress: bool = False, desc=None
    ) -> Corpus:
        if show_progress:
            corpus = tqdm(corpus, desc=desc)
        return Corpus(itertools.chain(*[self.process(doc) for doc in corpus]))


class PipelineProcessorComponent(PipelineComponent):
    """Processing component in a pipeline, responsible for adding annotations.

    The __call__ function of a PipelineComponent should add annotations as necessary
    and return the updated object.

    If there may not be a one-to-one mapping between input objects and output objects
    (e.g. for example filtering or data augmentation), inherit from
    PipelineComponentBase instead.
    """

    @overrides
    def process(self, ann_text: AnnotatedTextInterface) -> List[AnnotatedTextInterface]:
        return [self(ann_text)]

    def __call__(self, ann_text: AnnotatedTextInterface) -> AnnotatedTextInterface:
        raise NotImplementedError(
            "Subclasses of PipelineComponent should implement __call__"
        )


class PipelineError(RuntimeError):
    """Preprocessing pipeline exception"""


class PreprocessingPipeline(Registrable):
    default_implementation = "default"

    def __init__(
        self,
        source: PipelineSource,
        pipeline: List[PipelineComponent] = [],
        cache_dir: str = None,
        do_not_cache: bool = False,
        cache_final_only: bool = False,
        disable_cache: bool = False,
    ):
        self.source = source
        self.pipeline = pipeline
        self.data_cache = None
        self.do_not_cache = do_not_cache
        self.cache_final_only = cache_final_only

    def full_config(self, dataset_name) -> List[dict]:
        """
        The full config consists of the dataset_name and config (__init__ params by
        default) for the PipelineSource and ordered list of PipelineComponents.
        """
        return [self.source.get_dataset_config(dataset_name)] + [
            annotator.get_config() for annotator in self.pipeline
        ]

    def cache_exists(self, dataset_name) -> bool:
        config = self.full_config(dataset_name)
        return bool(self.data_cache and self.data_cache.exists(config))

    def _load_preprocessed_records(
        self,
        dataset_name,
        force_preprocessing: bool = False,
        sc=None,
        npartitions=None,
    ) -> Tuple[
        Iterable[AnnotatedTextInterface], List[dict], List[PipelineProcessorComponent]
    ]:

        # config consists of the dataset_name and config (__init__ params by default)
        # for the PipelineSource and ordered list of PipelineComponents.
        config = self.full_config(dataset_name)
        # look through the config steps in reverse order, trying to load each.  If we
        # can load one, it is the longest set of preprocessing steps available.  For
        # each annotator that is not available, pop it from the list and save it in
        # remaining_annotators so we can perform those processing steps later.
        # Similarly, remove preprocessing steps from config and return a config of
        # just the already-completed steps.
        remaining_annotators = []
        for annotator_idx, annotator in enumerate(reversed(self.pipeline)):
            logger.info(
                f"attempting to load results of annotator {annotator.name()} "
                f"with config {config}"
            )
            records = (
                self.data_cache.load_records_from_cache(
                    config, sc=sc, npartitions=npartitions
                )
                if self.data_cache and not force_preprocessing
                else None
            )
            if records is None:
                logger.info(
                    f"did not find cached result for annotator {annotator.name()}, "
                    "will annotate."
                )
                config.pop()
                remaining_annotators.append(annotator)
                continue
            break
        else:
            # if we didn't find any cached preprocessing steps, go ahead and load the
            # dataset from the PipelineSource.
            logger.info(f"Loading {dataset_name} dataset from source.")
            records = self.source(dataset_name)
            if sc:
                records = sc.parallelize(records, numSlices=npartitions)
        remaining_annotators = list(reversed(remaining_annotators))
        return records, config, remaining_annotators

    def preprocess(
        self,
        dataset_name: str,
        stream: bool = True,
        force_preprocessing: bool = False,
        do_not_cache: bool = None,
        max_records: int = None,
    ) -> Iterable[AnnotatedTextInterface]:
        """
        Run the configured preprocessing on the given dataset_name.  This
        dataset_name is just passed to the PipelineSource (first element of the
        pipeline), so it determines how to convert it to a raw dataset for further
        processing.

        This function attempts to load precomputed results from the data cache.  It
        loads the longest initial sequence of preprocessing steps available in the
        cache unless force_preprocessing was specified in the constructor.

        If `stream` is True and the fully-preprocessed dataset can be loaded from
        disk, a lazy iterator over AnnotatedText objects is returned.

        do_not_cache forces processing not to cache results.  If it is unspecified
        (None), use the value of do_not_cache from constructor.

        `max_records` only loads the specified number of records if given, defaults to
        all records.
        """
        # Use the instance-level value for do_not_cache if not specified in function
        # arguments.  Also force do_not_cache if a limited number of samples are taken.
        do_not_cache = self.do_not_cache if do_not_cache is None else do_not_cache
        # ugly formatting for multiple outputs:
        (
            records,
            completed_config,
            remaining_annotators,
        ) = self._load_preprocessed_records(dataset_name, force_preprocessing)
        if max_records is not None:
            logger.info(
                "Will not cache because max_records is specified "
                f"(max_records={max_records})"
            )
            do_not_cache = True
            records = itertools.islice(records, max_records)
        # perform the preprocessing steps that weren't loaded from the cache, adding
        # each to the cache as it completes.
        for _, is_last, annotator in more_itertools.mark_ends(remaining_annotators):
            logger.info(f"Annotating dataset with {annotator.name()} annotator")
            annotator.reset()
            # We could stream the data here, allowing out-of-memory computation,
            # and write it to disk.  However, this would mean we'd need to re-read
            # the data from disk (possibly slow), and we'd need extra logic to handle
            # the case where the cache isn't available.  So for now we just read the
            # whole dataset into memory if we need to annotate it.  If you run out of
            # memory here, think about using spark for distributed annotation.
            records = annotator.process_corpus(
                records, show_progress=True, desc=f"Annotating {annotator.name()}"
            )
            if len(records) == 0:
                raise PipelineError(
                    f"All records were filtered by component {annotator.name()}, "
                    "perhaps the __call__ method did not return a value? "
                )
            completed_config.append(annotator.get_config())
            # to cache, must have a valid cache, and do_not_cache should be False
            should_cache = self.data_cache and not do_not_cache
            # also, if check if we're only supposed to save the final result
            if self.cache_final_only:
                should_cache = should_cache and is_last
            if should_cache:
                logger.info(
                    f"Saving preprocessed dataset with config {completed_config}"
                )
                try:
                    self.data_cache.save_records_to_cache(
                        config=completed_config, records=records
                    )
                except OSError:
                    logger.exception("Error saving to cache")

        if not stream and not isinstance(records, Corpus):
            # this will load all of the records into memory in a Corpus object
            records = Corpus(records)

        return records

    def spark_preprocess(
        self,
        dataset_name: str,
        sc,
        npartitions: int = None,
        force_preprocessing: bool = False,
        do_not_cache: bool = None,
    ):
        """
        Run the configured preprocessing on the given dataset_name using Spark
        resources.  This dataset_name is just passed to the PipelineSource (first
        element of the pipeline), so it determines how to convert it to a raw dataset
        for further processing.

        This function attempts to load precomputed results from the data cache.  It
        loads the longest initial sequence of preprocessing steps available in the
        cache unless force_preprocessing was specified in the constructor.

        It returns a spark RDD containing the preprocessed records.
        """
        do_not_cache = self.do_not_cache if do_not_cache is None else do_not_cache
        (
            records_rdd,
            completed_config,
            remaining_annotators,
        ) = self._load_preprocessed_records(
            dataset_name,
            force_preprocessing=force_preprocessing,
            sc=sc,
            npartitions=npartitions,
        )

        # perform the preprocessing steps that weren't loaded from the cache, adding
        # each to the cache as it completes.
        for _, is_last, annotator in more_itertools.mark_ends(remaining_annotators):
            annotator.reset()
            logger.info("Annotating dataset using spark")
            records_rdd = records_rdd.flatMap(annotator.process)
            completed_config.append(annotator.get_config())
            # to cache, must have a valid cache, and do_not_cache should be False
            should_cache = self.data_cache and not do_not_cache
            # also, if check if we're only supposed to save the final result
            if self.cache_final_only:
                should_cache = should_cache and is_last
            if should_cache:
                logger.info(
                    f"Saving preprocessed dataset with config {completed_config}"
                )
                try:
                    self.data_cache.save_records_to_cache(
                        config=completed_config, records=records_rdd
                    )
                except OSError:
                    logger.exception("Error saving to cache")
        return records_rdd

    def preprocess_doc(self, doc: AnnotatedText) -> List[AnnotatedText]:
        """Run preprocessing pipeline on the given AnnotatedText.  May return None if
        the doc is filtered by one of the PipelineComponents"""
        docs = [doc]
        for component in self.pipeline:
            docs = list(itertools.chain(*[component.process(doc) for doc in docs]))
        return docs

    def preprocess_corpus(
        self, corpus: Corpus, quiet: bool = False
    ) -> Iterator[AnnotatedText]:
        """Helper function to preprocess the given corpus, returning an iterator of
        processed AnnotatedText objects."""
        for doc in tqdm(corpus, desc="preprocessing", disable=quiet):
            processed_docs = self.preprocess_doc(doc)
            for processed_doc in processed_docs:
                yield processed_doc

    def get_cache_data_file(self, dataset_name):
        """
        Utility / debugging function returning the name of the cache file for the given
        dataset name
        """
        return self.data_cache.get_data_file(self.full_config(dataset_name))

    def get_cache_config_file(self, dataset_name):
        """
        Utility / debugging function returning the name of the cache file for the given
        dataset name
        """
        return self.data_cache.get_config_file(self.full_config(dataset_name))


PreprocessingPipeline.register("default")(PreprocessingPipeline)


class AnnotatorPipelineComponentBase(PipelineProcessorComponent):
    def __init__(self, annotator):
        self.annotator = annotator

    def __call__(self, ann_text: AnnotatedText) -> AnnotatedText:
        ann_text.annotate(self.annotator)
        return ann_text
