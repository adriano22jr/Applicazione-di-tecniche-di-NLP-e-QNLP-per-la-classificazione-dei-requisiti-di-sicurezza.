from lambeq import BobcatParser, Rewriter, TensorAnsatz, AtomicType
from discopy import Dim
from utilities import *
import torch

#objects declaration for syntax-tensor pipeline
bobcat_parser = BobcatParser(verbose = "text")
rewriter = Rewriter(["prepositional_phrase", "determiner", "auxiliary", "connector", "coordination"])
tensor_ansatz = TensorAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})

#retrieving train and test data
train_labels, train_sentences = extract_data("code/pipelines/mc_train_data.txt")
test_labels, test_sentences = extract_data("code/pipelines/mc_test_data.txt")

#creating diagram for each sentence
train_diagrams = bobcat_parser.sentences2diagrams(train_sentences)
test_diagrams = bobcat_parser.sentences2diagrams(test_sentences)

#rewriting and normalizing diagrams
train_normalized_diagrams = [rewriter(diagram).normal_form() for diagram in train_diagrams]
test_normalized_diagrams = [rewriter(diagram).normal_form() for diagram in test_diagrams]

#parameterisation of the diagrams
train_circuits = [tensor_ansatz(diagram) for diagram in train_normalized_diagrams]
test_circuits = [tensor_ansatz(diagram) for diagram in test_normalized_diagrams]

#creating model and trainer
model = PytorchModel.from_diagrams(train_circuits + test_circuits)