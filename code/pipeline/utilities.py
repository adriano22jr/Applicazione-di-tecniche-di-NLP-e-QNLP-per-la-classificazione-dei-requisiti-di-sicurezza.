import torch

def extract_data(dataset):
    label, sentence = [], []
    file = open(dataset, "r")
    for line in file:
        label.append( [float(line[0]), 1 - float(line[0])])
        sentence.append(line[1:].strip())
    return label, sentence

def evaluation_metric():
    sig = torch.sigmoid
    def accuracy(y_hat, y):
        return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2
    eval_metrics = {"acc": accuracy}
    return eval_metrics

SEED = 0
LEARNING_RATE = 3e-2