import datetime as dt
import json
from typing import Iterable, List, Optional, Union
from uuid import uuid4

from ..annotation import AnnotatedText, Corpus, MultiAnnotatedText
from .preprocessing import PipelineSource


class ClinicalDocument(AnnotatedText):
    def __init__(
        self,
        text: str,
        id: Union[str, None],
        date: Union[str, dt.datetime],
        type: str,
        auto_optimize: bool = False,
    ):

        super().__init__(text_id=id, text=text, auto_optimize=auto_optimize)
        self.type = type
        self.date = date


class PatientDocuments(MultiAnnotatedText):
    """
    Representation of a collection of clinical documents of one patient.
    """

    def __init__(
        self,
        patient_id: str,
        notes: Iterable[ClinicalDocument],
    ):
        """
        A list of ClinicalDocuments can be passed in as 'notes'.  The type-specific
        parameters (path_reports, image_reports, op_notes, progress_notes) are just kept
        for backwards compatibility.
        """
        # sort based on time
        reports = sorted(notes, key=lambda x: x.date)
        doc_id = patient_id + "_" + str(uuid4())
        super().__init__(reports, doc_id=doc_id)
        self.patient_id = patient_id


DATE_FMT = "%Y-%m-%d"


class CancerRegistryEntry(PatientDocuments):
    """
    Representation of annotated cancer registry data
    """

    def __init__(
        self,
        patient_id: str,
        diagnosis_date: str,
        TumorSite: List[str],
        pathologic_t: List[str],
        pathologic_n: List[str],
        pathologic_m: List[str],
        clinical_t: List[str],
        clinical_n: List[str],
        clinical_m: List[str],
        notes: Optional[Iterable[ClinicalDocument]] = None,
    ):
        self.patient_id = patient_id
        self.diagnosis_date = diagnosis_date
        self.TumorSite = TumorSite
        self.pathologic_t = pathologic_t
        self.pathologic_n = pathologic_n
        self.pathologic_m = pathologic_m
        self.clinical_t = clinical_t
        self.clinical_n = clinical_n
        self.clinical_m = clinical_m

        if self.diagnosis_date:
            self.date = dt.datetime.strptime(self.diagnosis_date, DATE_FMT)

        super().__init__(
            patient_id=patient_id,
            notes=notes,
        )

    @classmethod
    def from_data_row(cls, row_item) -> "CancerRegistryEntry":
        """Create PathologyReport objects from a data row"""

        docs = Corpus(
            [
                ClinicalDocument(
                    text=note["text"],
                    id=note["id"],
                    date=dt.datetime.strptime(note["date"], DATE_FMT),
                    type=note["type"],
                )
                for note in row_item["notes"]
            ]
        )

        return cls(
            notes=docs,
            diagnosis_date=row_item["diagnosis_date"],
            patient_id=row_item["patient_id"],
            TumorSite=row_item["TumorSite"],
            pathologic_t=row_item["pathologic_t"],
            pathologic_n=row_item["pathologic_n"],
            pathologic_m=row_item["pathologic_m"],
            clinical_t=row_item["clinical_t"],
            clinical_n=row_item["clinical_n"],
            clinical_m=row_item["clinical_m"],
        )


def _load_data(path):
    with open(path, "r") as read_content:
        dataset = json.load(read_content)

    return Corpus([CancerRegistryEntry.from_data_row(row_item) for row_item in dataset])


@PipelineSource.register("custom-file-source", exist_ok=True)
class CustomFileSource(PipelineSource):
    def __init__(
        self,
        train_data_path: str,
        train_val_split: float = 0.8,
        test_data_path: str = "",
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.train_val_split = train_val_split

    def __call__(self, dataset_name: str):

        if dataset_name in ["train", "validate"]:
            corpus = _load_data(self.train_data_path)
            train, val = corpus.split(self.train_val_split)
            if dataset_name == "train":
                corpus = train
            elif dataset_name == "validate":
                corpus = val
        elif dataset_name == "test":
            corpus = _load_data(self.test_data_path)
        else:
            raise ValueError(f"Unknown dataset_name {dataset_name}!")

        return corpus

    def get_dataset_config(self, dataset_name: str) -> dict:
        if dataset_name in ["train", "validate"]:
            config = {
                "train_data_path": self.train_data_path,
                "train_val_split": self.train_val_split,
                "dataset_name": dataset_name,
            }
            return config
        elif dataset_name == "test":
            config = {
                "test_data_path": self.test_data_path,
                "dataset_name": dataset_name,
            }
            return config
        raise ValueError(f"Unknown dataset_name {dataset_name}!")
