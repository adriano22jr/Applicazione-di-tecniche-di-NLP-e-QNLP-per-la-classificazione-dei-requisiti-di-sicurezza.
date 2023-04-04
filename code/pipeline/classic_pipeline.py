from lambeq import BobcatParser, TreeReader, TreeReaderMode, spiders_reader, cups_reader, stairs_reader, Rewriter
from lambeq import TensorAnsatz, SpiderAnsatz, MPSAnsatz, AtomicType
from lambeq import PytorchModel, PytorchTrainer, Dataset
from discopy import Dim
from utilities import *
import matplotlib.pyplot as plt


sig = torch.sigmoid
def accuracy(y_hat, y):
    return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2
eval_metrics = {"acc": accuracy}


class ClassicPipeline():
    SUPPORTED_RULES = ["auxiliary", "connector", "coordination", "curry", "determiner", "postadverb", "preadverb", "prepositional_phrase", "object_rel_pronoun", "subject_rel_pronoun"]
    
    def __init__(self, parser, ansatz) -> None:
        self.__parser = parser
        self.__ansatz = ansatz
        self.__rewriter = Rewriter()
    
    def add_rewriter_rules(self, *rules) -> None:
        self.__rewriter.add_rules(*rules)
        
    def create_circuits_and_labels(self, dataset: str, control = None):
        labels, sentences = extract_data(dataset)
        diagrams = self.__parser.sentences2diagrams(sentences)

        if control.lower() == "y":
            normalized_diagrams = [self.__rewriter(diagram).normal_form() for diagram in diagrams]            
            circuits = [self.__ansatz(diagram) for diagram in normalized_diagrams]            
            return labels, circuits
        
        circuits = [self.__ansatz(diagram) for diagram in diagrams]
        return labels, circuits

    def create_dataset(self, labels, circuits):
        return Dataset(circuits, labels)
    
    def create_trainer(self, *circuits):
        circuits_model = []
        for circuit in circuits:
            circuits_model = circuits_model + circuit

        self.__model = PytorchModel.from_diagrams(circuits_model)
        self.__trainer = PytorchTrainer(
                model = self.__model,
                loss_function = torch.nn.BCEWithLogitsLoss(),
                optimizer = torch.optim.AdamW,
                epochs = 70,
                evaluate_functions = eval_metrics,
                evaluate_on_train = True,
                learning_rate = LEARNING_RATE,
                verbose = "text",
                seed = CLASSIC_SEED
        )
        
    def train_model(self, train_set, test_set):        
        self.__trainer.fit(train_set, test_set, evaluation_step = 1, logging_step = 5)
            
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
