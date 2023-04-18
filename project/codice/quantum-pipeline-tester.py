from lambeq import BobcatParser
from lambeq import IQPAnsatz, AtomicType
from discopy import Dim
from quantum_pipeline import *

parser = BobcatParser(verbose = "text", root_cats = ("NP", "N"))
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 0}, n_layers = 1, n_single_qubit_params = 3)

pip = QuantumPipeline(parser, ansatz)
train_labels, train_circuits = pip.create_circuits_and_labels("code/pipeline/rp_train_data.txt")
test_labels, test_circuits = pip.create_circuits_and_labels("code/pipeline/rp_test_data.txt")

train_set, test_set = pip.create_dataset(train_labels, train_circuits), pip.create_dataset(test_labels, test_circuits)
pip.create_trainer(train_circuits, test_circuits)
pip.train_model(train_set, test_set)
pip.plot()