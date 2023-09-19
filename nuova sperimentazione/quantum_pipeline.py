from lambeq import *
from utilities import *
import sklearn.metrics as mt
import os, warnings, numpy as np, matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "False"


def accuracy(y_hat, y):
    return mt.accuracy_score(y, np.round(y_hat))

def precision(y_hat, y):
    return mt.precision_score(y, np.round(y_hat), average = "micro")

def recall(y_hat, y):
    return mt.recall_score(y, np.round(y_hat), average = "micro")

def f1score(y_hat, y):
    return mt.f1_score(y, np.round(y_hat), average = "micro")

def precision(y_hat, y):
    return mt.precision_score(y, np.round(y_hat), average = "micro")

eval_metrics = {"acc": accuracy, "prec": precision, "rec": recall, "f1": f1score}


class QuantumPipeline():
    SUPPORTED_RULES = ["auxiliary", "connector", "coordination", "curry", "determiner", "postadverb", "preadverb", "prepositional_phrase", "object_rel_pronoun", "subject_rel_pronoun"]
    
    def __init__(self, parser, ansatz) -> None:
        self.__parser = parser
        self.__ansatz = ansatz
        self.__tokeniser = SpacyTokeniser()
        self.__model = None
        self.__trainer = None
        
    def create_circuits_and_labels(self, dataset: str):
        labels, sentences = extract_data_manual(dataset)
        lower_sentences = [s.lower() for s in sentences]
        tokens = self.__tokeniser.tokenise_sentences(lower_sentences)
        diagrams = self.__parser.sentences2diagrams(tokens, suppress_exceptions = True, tokenised = True)

        normalized_diagrams = [diagram.normal_form() for diagram in diagrams if diagram is not None]    
        edited_labels = [label for (diagram, label) in zip(diagrams, labels) if diagram is not None]   
                  
        circuits = [self.__ansatz(diagram) for diagram in normalized_diagrams]      
        
        return edited_labels, circuits
    
    def create_dataset(self, circuits, labels, shuffle = False):
        return Dataset(circuits, labels, shuffle = shuffle)
    
    def create_model(self, *circuits):
        circuits_model = []
        for circuit in circuits:
            circuits_model = circuits_model + circuit

        self.__model = NumpyModel.from_diagrams(circuits_model, use_jit = False)
        return self.__model
    
    def create_trainer(self, model = None, optimizer = SPSAOptimizer, n_epochs = 0, a_hyp = 0.05, seed = CLASSIC_SEED, evaluate = False):
        self.__trainer = QuantumTrainer(
            model = model,
            loss_function = BinaryCrossEntropyLoss(use_jax = True),
            epochs = n_epochs,
            optimizer = optimizer,
            optim_hyperparams = {'a': a_hyp, 'c': 0.06, 'A':0.01 * n_epochs},
            evaluate_functions = eval_metrics,
            evaluate_on_train = evaluate,
            verbose = "text",
            seed = seed
        )
        
        return self.__trainer
        
    def load_from_model(self, path):
        checkpoint = self.__trainer.load_training_checkpoint(path)
        return checkpoint
        
    def train_and_evaluate(self, train_set, test_set, eval_step, log_step, stop):        
        self.__trainer.fit(train_set, test_set, eval_interval = eval_step, log_interval = log_step, early_stopping_interval = stop)
        
    def train_model(self, train_set, eval_step, log_step, stop = None):
        self.__trainer.fit(train_set, eval_interval = eval_step, log_interval = log_step, early_stopping_interval = stop)
        