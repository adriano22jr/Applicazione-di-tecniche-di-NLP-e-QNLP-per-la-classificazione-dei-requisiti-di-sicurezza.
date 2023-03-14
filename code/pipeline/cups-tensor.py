from lambeq import cups_reader, Rewriter, TensorAnsatz, AtomicType, PytorchModel, PytorchTrainer, Dataset
from discopy import Dim
from utilities import *

#objects declaration for syntax-tensor pipeline
rewriter = Rewriter(["prepositional_phrase", "determiner", "auxiliary", "connector", "coordination"])
tensor_ansatz = TensorAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})

#retrieving train and test data
train_labels, train_sentences = extract_data("code/pipelines/mc_train_data.txt")
test_labels, test_sentences = extract_data("code/pipelines/mc_test_data.txt")

#creating diagram for each sentence
train_diagrams = cups_reader.sentences2diagrams(train_sentences)
test_diagrams = cups_reader.sentences2diagrams(test_sentences)

#rewriting and normalizing diagrams
train_normalized_diagrams = [rewriter(diagram).normal_form() for diagram in train_diagrams]
test_normalized_diagrams = [rewriter(diagram).normal_form() for diagram in test_diagrams]

#parameterisation of the diagrams
train_circuits = [tensor_ansatz(diagram) for diagram in train_normalized_diagrams]
test_circuits = [tensor_ansatz(diagram) for diagram in test_normalized_diagrams]
"""train_circuits = [tensor_ansatz(diagram) for diagram in train_diagrams]
test_circuits = [tensor_ansatz(diagram) for diagram in test_diagrams]"""

#creating model and trainer
model = PytorchModel.from_diagrams(train_circuits + test_circuits)
trainer = PytorchTrainer(
    model = model,
    loss_function = torch.nn.BCEWithLogitsLoss(),
    optimizer = torch.optim.AdamW,
    epochs = 50,
    evaluate_functions = evaluation_metric(),
    evaluate_on_train = True,
    learning_rate = LEARNING_RATE,
    verbose = "text",
    seed = SEED
)

#creating datasets
train_dataset = Dataset(train_circuits, train_labels)
test_dataset = Dataset(test_circuits, test_labels)

#starting training
trainer.fit(train_dataset, test_dataset, evaluation_step = 1, logging_step = 5)