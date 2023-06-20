from lambeq import BobcatParser, Rewriter, SpacyTokeniser
from lambeq import IQPAnsatz, AtomicType, remove_cups
from pytket.extensions.qiskit import AerBackend
from lambeq import NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset
from lambeq import BinaryCrossEntropyLoss
from utilities import *
import torchmetrics
import os, warnings, numpy as np, matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "False"


def accuracy(y_hat, y):
    return torchmetrics.functional.accuracy(y_hat, y, "binary")

def precision(y_hat, y):
    return torchmetrics.functional.precision(y_hat, y, "binary")

def recall(y_hat, y):
    return torchmetrics.functional.recall(y_hat, y, "binary")

def f1score(y_hat, y):
    return torchmetrics.functional.f1_score(y_hat, y, "binary")

def precision(y_hat, y):
    return torchmetrics.functional.precision(y_hat, y, "binary")

eval_metrics = {"acc": accuracy, "prec": precision, "rec": recall, "f1": f1score}


class QuantumPipeline():
    SUPPORTED_RULES = ["auxiliary", "connector", "coordination", "curry", "determiner", "postadverb", "preadverb", "prepositional_phrase", "object_rel_pronoun", "subject_rel_pronoun"]
    
    def __init__(self, parser, ansatz) -> None:
        self.__tokeniser = SpacyTokeniser()
        self.__parser = parser
        self.__ansatz = ansatz
        self.__rewriter = Rewriter()
        
    def add_rewriter_rules(self, *rules) -> None:
        self.__rewriter.add_rules(*rules)
        
    def create_circuits_and_labels(self, dataset: str):
        labels, sentences = extract_data_manual(dataset)
        lower_sentences = [s.lower() for s in sentences]
        diagrams = self.__parser.sentences2diagrams(lower_sentences, tokenised = False)
        
        normalized_diagrams = [diagram.normal_form() for diagram in diagrams if diagram is not None]    
        edited_labels = [label for (diagram, label) in zip(diagrams, labels) if diagram is not None]   
                  
        circuits = [self.__ansatz(remove_cups(diagram)) for diagram in normalized_diagrams]      
        
        return edited_labels, circuits
    
    def create_dataset(self, labels, circuits):
        return Dataset(circuits, labels, shuffle = False)
    
    def create_model(self, *circuits):
        circuits_model = []
        for circuit in circuits:
            circuits_model = circuits_model + circuit

        self.__model = NumpyModel.from_diagrams(circuits_model, use_jit = False)
        return self.__model
    
    def create_trainer(self, model = None, n_epochs = 0, a_hyp = 0.05, seed = CLASSIC_SEED):
        self.__trainer = QuantumTrainer(
            model = model,
            loss_function = BinaryCrossEntropyLoss(use_jax = True),
            epochs = n_epochs,
            optimizer = SPSAOptimizer,
            optim_hyperparams = {'a': a_hyp, 'c': 0.06, 'A':0.01 * n_epochs},
            evaluate_functions = eval_metrics,
            evaluate_on_train = True,
            verbose = "text",
            seed = seed
        )
        
        return self.__trainer
        
    def train_model(self, train_set, test_set, eval_step, log_step):        
        self.__trainer.fit(train_set, test_set, evaluation_step = eval_step, logging_step = log_step)
        
    def plot(self):
        fig1, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex = True, sharey='row', figsize=(10, 6))
        
        ax_tl.set_title("Training set")
        ax_tl.set_ylim([0, 3])
        ax_bl.set_ylabel('Accuracy')
        ax_bl.set_ylim([0, 1])
        
        ax_tr.set_title("Test set")
        ax_tr.set_ylim([0, 3])   
        ax_tl.set_ylabel('Loss')
        ax_br.set_ylim([0, 1])
        
        colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        ax_tl.plot(self.__trainer.train_epoch_costs[::10], color = next(colours))
        ax_bl.plot(self.__trainer.train_results['acc'][::10], color = next(colours))
        ax_tr.plot(self.__trainer.val_costs[::10], color = next(colours))
        ax_br.plot(self.__trainer.val_results['acc'][::10], color = next(colours))
        
        plt.show()
        