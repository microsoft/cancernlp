import tempfile

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common import util as common_util


config_path = "hierarchical_seq2vec.jsonnet"

params = Params.from_file(config_path)
tmp_dir = tempfile.TemporaryDirectory()

serialization_dir = tmp_dir.name

include_packages = [
    "cancernlp.model",
]

for package_name in include_packages:
    common_util.import_module_and_submodules(package_name)

train_model(
    params=params,
    serialization_dir=serialization_dir,
    include_package=include_packages,
)

print("done")
tmp_dir.cleanup()
