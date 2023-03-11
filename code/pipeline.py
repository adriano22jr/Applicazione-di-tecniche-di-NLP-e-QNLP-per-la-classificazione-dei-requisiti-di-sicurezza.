from lambeq import BobcatParser, AtomicType, TensorAnsatz, PytorchModel, PytorchTrainer, Dataset
from discopy import Dim
import torch

BATCH_SIZE = 30
EPOCHS = 30
LEARNING_RATE = 3e-2
SEED = 0

sig = torch.sigmoid
def accuracy(y_hat, y):
    return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2
eval_metrics = {"acc": accuracy}

def extract_data(dataset):
    label, sentence = [], []
    file = open(dataset, "r")
    for line in file:
        label.append( [float(line[0]), 1 - float(line[0])])
        sentence.append(line[1:].strip())
    return label, sentence

#prendo i dati dal dataset di training
label_data, train_data = extract_data("./code/mc_train_data.txt")

#trasformo le frasi dei dati di training in string-diagrams da parametrizzare
bobcat_parser = BobcatParser(verbose="text")
train_diagrams = bobcat_parser.sentences2diagrams(train_data)

#parametrizzo gli string-diagrams ottenuti, uso un ansats
tensor_ansatz = TensorAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})
train_circuits = [tensor_ansatz(diagram) for diagram in train_diagrams]

#creo il modello e il trainer usando pyTorch
model = PytorchModel.from_diagrams(train_circuits)
trainer = PytorchTrainer(
    model = model,
    loss_function = torch.nn.BCEWithLogitsLoss(),
    optimizer = torch.optim.AdamW,
    learning_rate = LEARNING_RATE,
    epochs = EPOCHS,
    evaluate_functions = eval_metrics,
    evaluate_on_train = True,
    verbose = "text",
    seed = SEED 
)

train_dataset = Dataset(train_circuits, label_data, batch_size = BATCH_SIZE)
trainer.fit(train_dataset, evaluation_step = 1, logging_step = 10)
