import torch
import pickle
import numpy as np

def extract_data(dataset):
    label, sentence = [], []
    file = open(dataset, "r")
    for line in file:
        label.append( [float(line[0]), 1 - float(line[0])])
        sentence.append(line[1:].strip())
    return label, sentence

def extract_data_manual(dataset):
    label, sentence = [], []
    file = open(dataset, "r")
    for line in file:
        actual_value = int(line[0])
        label.append(np.array([actual_value, not(actual_value)], dtype=np.float32))
        sentence.append(line[1:].strip())
    return label, sentence

def save_data(filename: str, *data):
    file = open(filename, "wb")
    for item in data:
        pickle.dump(item, file)
    file.close()
    
def load_data(filename: str):
    file = open(filename, "rb")
    labels = pickle.load(file)
    circuits = pickle.load(file)
    file.close()
    return labels, circuits

CLASSIC_SEED = 0
QUANTUM_SEED = 2
LEARNING_RATE = 3e-2