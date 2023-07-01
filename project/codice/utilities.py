import pickle
import torch
import random
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

def k_split(sentences: list, labels: list, k):
    data = []
    for sentence, label in zip(sentences, labels):
        data.append( (sentence, label) )
    
    random.shuffle(data)    
    return [data[i::k] for i in range(k)]

def unpack_data(data: list):
    sentences = []
    labels = []

    for item in data:
        sentences.append(item[0])
        labels.append(item[1])
        
    return sentences, labels

CLASSIC_SEED = 0
QUANTUM_SEED = 2
LEARNING_RATE = 3e-2