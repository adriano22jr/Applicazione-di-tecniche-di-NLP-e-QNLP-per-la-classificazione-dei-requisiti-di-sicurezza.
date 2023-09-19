from lambeq import BobcatParser, TreeReader, TreeReaderMode, spiders_reader, cups_reader, stairs_reader, Rewriter
from lambeq import SpacyTokeniser
from lambeq import TensorAnsatz, SpiderAnsatz, MPSAnsatz, AtomicType
from lambeq import PytorchModel, PytorchTrainer, Dataset
from discopy import Dim
from copy import deepcopy
import torchmetrics
from utilities import *
import matplotlib.pyplot as plt


def accuracy(y_hat, y):
    return torchmetrics.functional.accuracy(y_hat, y, "binary")

def precision(y_hat, y):
    return torchmetrics.functional.precision(y_hat, y, "binary")

def recall(y_hat, y):
    return torchmetrics.functional.recall(y_hat, y, "binary")

def f1score(y_hat, y):
    return torchmetrics.functional.f1_score(y_hat, y, "binary")

eval_metrics = {"acc": accuracy, "prec": precision, "rec": recall, "f1": f1score}

class ClassicPipeline():
    SUPPORTED_RULES = ["auxiliary", "connector", "coordination", "curry", "determiner", "postadverb", "preadverb", "prepositional_phrase", "object_rel_pronoun", "subject_rel_pronoun"]
    
    def __init__(self, parser, ansatz) -> None:
        self.__tokeniser = SpacyTokeniser()
        self.__parser = parser
        self.__ansatz = ansatz
        self.__rewriter = Rewriter()
    
    def add_rewriter_rules(self, *rules) -> None:
        self.__rewriter.add_rules(*rules)
        
    def create_circuits_and_labels(self, dataset: str, control = None):
        labels, sentences = extract_data_manual(dataset)
        lower_sentences = [s.lower() for s in sentences]
        tokens = self.__tokeniser.tokenise_sentences(lower_sentences)
        diagrams = self.__parser.sentences2diagrams(tokens, tokenised = True)

        if control.lower() == "y":
            normalized_diagrams = [self.__rewriter(diagram).normal_form() for diagram in diagrams]            
            circuits = [self.__ansatz(diagram) for diagram in normalized_diagrams]            
            return labels, circuits
        
        circuits = [self.__ansatz(diagram) for diagram in diagrams]
        return labels, circuits

    def create_dataset(self, circuits, labels, shuffle = False):
        return Dataset(circuits, labels, shuffle = shuffle)
    
    def create_model(self, *circuits):
        circuits_model = []
        for circuit in circuits:
            circuits_model = circuits_model + circuit

        self.__model = PytorchModel.from_diagrams(circuits_model)
        return self.__model
    
    def create_trainer(self, model = None, loss = None, optimizer = torch.optim.AdamW, n_epochs = 0, lr = LEARNING_RATE, seed = CLASSIC_SEED, evaluate = False):
        self.__trainer = PytorchTrainer(
                model = model,
                loss_function = loss,
                optimizer = optimizer,
                epochs = n_epochs,
                evaluate_functions = eval_metrics,
                evaluate_on_train = evaluate,
                learning_rate = lr,
                verbose = "text",
                seed = seed
        )
        
        return self.__trainer
        
    def train_model(self, train_set, eval_step, log_step):        
        self.__trainer.fit(train_set, evaluation_step = eval_step, logging_step = log_step)
        
    def train_and_evaluate(self, train_set, test_set, eval_step, log_step):
        self.__trainer.fit(train_set, test_set, evaluation_step = eval_step, logging_step = log_step)
            
    def plot(self):
        fig1, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharey='row', figsize=(10, 6))
        
        ax_tl.set_title("Training set")
        ax_tl.set_ylim([0, 3])
        ax_bl.set_ylabel('Accuracy')
        ax_bl.set_ylim([0, 1])
        
        ax_tr.set_title("Test set")
        ax_tr.set_ylim([0, 3])   
        ax_tl.set_ylabel('Loss')
        ax_br.set_ylim([0, 1])
        
        colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        ax_tl.plot(self.__trainer.train_epoch_costs, color = next(colours))
        ax_bl.plot(self.__trainer.train_results['acc'], color = next(colours))
        ax_tr.plot(self.__trainer.val_costs, color = next(colours))
        ax_br.plot(self.__trainer.val_results['acc'], color = next(colours))
        
        plt.show()
