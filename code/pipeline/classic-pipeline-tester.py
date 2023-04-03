from lambeq import BobcatParser, TreeReader, TreeReaderMode, spiders_reader, cups_reader, stairs_reader
from lambeq import TensorAnsatz, SpiderAnsatz, MPSAnsatz, AtomicType
from discopy import Dim
from classic_pipeline import *

bobcat_parser = BobcatParser(verbose = "text")
spider_parser = spiders_reader
cups_parser = cups_reader
stairs_parser = stairs_reader
tree_parser = TreeReader(mode=TreeReaderMode.RULE_ONLY)


tensor_ansatz = TensorAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})
spider_ansatz = SpiderAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})
mps_ansatz = MPSAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})


pip = ClassicPipeline(bobcat_parser, tensor_ansatz)
pip.add_rewriter_rules(ClassicPipeline.SUPPORTED_RULES[0], ClassicPipeline.SUPPORTED_RULES[1], ClassicPipeline.SUPPORTED_RULES[4])
train_labels, train_circuits = pip.create_circuits_and_labels("code/pipeline/mc_train_data.txt", "n")
test_labels, test_circuits = pip.create_circuits_and_labels("code/pipeline/mc_test_data.txt", "n")

train_set, test_set = pip.create_dataset(train_labels, train_circuits), pip.create_dataset(test_labels, test_circuits)
pip.create_trainer(train_circuits, test_circuits)
pip.train_model(train_set, test_set)
pip.plot()