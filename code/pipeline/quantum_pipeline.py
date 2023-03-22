from lambeq import BobcatParser, Rewriter
from lambeq import IQPAnsatz, AtomicType, remove_cups
from lambeq import TketModel, QuantumTrainer, SPSAOptimizer, Dataset
from pytket.extensions.qiskit import AerBackend
from utilities import *
import os, warnings, numpy as np, matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "True"

backend = AerBackend()
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}

loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2
eval_metrics = {"acc": acc}


class QuantumPipeline():
    SUPPORTED_RULES = ["auxiliary", "connector", "coordination", "curry", "determiner", "postadverb", "preadverb", "prepositional_phrase", "object_rel_pronoun", "subject_rel_pronoun"]
    
    def __init__(self, parser, ansatz) -> None:
        self.__parser = parser
        self.__ansatz = ansatz
        self.__rewriter = Rewriter()
        
    def add_rewriter_rules(self, *rules) -> None:
        self.__rewriter.add_rules(*rules)
        
    def create_circuits_and_labels(self, dataset: str, control = None):
        labels, sentences = extract_data(dataset)
        diagrams = self.__parser.sentences2diagrams(sentences, suppress_exceptions = True)

        normalized_diagrams = [self.__rewriter(diagram).normal_form() for diagram in diagrams if diagram is not None]      
        edited_labels = [label for (diagram, label) in zip(diagrams, labels) if diagram is not None]             
        circuits = [self.__ansatz(remove_cups(diagram)) for diagram in normalized_diagrams]            
        
        return edited_labels, circuits
    
    def create_dataset(self, labels, circuits):
        return Dataset(circuits, labels)
    
    def create_trainer(self, *circuits):
        circuits_model = []
        for circuit in circuits:
            circuits_model = circuits_model + circuit

        self.__model = TketModel.from_diagrams(circuits_model, backend_config = backend_config)
        self.__trainer = QuantumTrainer(
            model = self.__model,
            loss_function = loss,
            epochs = 1000,
            optimizer = SPSAOptimizer,
            optim_hyperparams = {'a': 0.05, 'c': 0.06, 'A':0.01 * 1000},
            evaluate_functions = eval_metrics,
            evaluate_on_train = True,
            verbose = "text",
            seed = CLASSIC_SEED
        )
        
    def train_model(self, train_set, test_set):        
        self.__trainer.fit(train_set, test_set, evaluation_step = 1, logging_step = 100)
        
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
        