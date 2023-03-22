from lambeq import BobcatParser, TreeReader, spiders_reader, cups_reader, stairs_reader
from lambeq import TensorAnsatz, SpiderAnsatz, MPSAnsatz, AtomicType
from lambeq import PytorchModel, PytorchTrainer, Dataset
from discopy import Dim
from classic_pipeline import *

parser = BobcatParser(verbose = "text")
ansatz = TensorAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})

pip = ClassicPipeline(parser, ansatz)
#pip.add_rewriter_rules(Pipeline.SUPPORTED_RULES[0], Pipeline.SUPPORTED_RULES[1], Pipeline.SUPPORTED_RULES[4])
train_labels, train_circuits = pip.create_circuits_and_labels("code/pipeline/mc_train_data.txt", "n")
test_labels, test_circuits = pip.create_circuits_and_labels("code/pipeline/mc_test_data.txt", "n")

train_set, test_set = pip.create_dataset(train_labels, train_circuits), pip.create_dataset(test_labels, test_circuits)
pip.create_trainer(train_circuits, test_circuits)
pip.train_model(train_set, test_set)
pip.plot()