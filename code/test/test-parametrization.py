from lambeq import BobcatParser, TreeReader, TreeReaderMode, Rewriter, AtomicType, IQPAnsatz, spiders_reader, cups_reader, stairs_reader
from discopy import grammar

sentence = 'John walks in the park'
parser = BobcatParser(verbose='text')
diagram = parser.sentence2diagram(sentence)

N = AtomicType.NOUN
S = AtomicType.SENTENCE

# Convert string diagram to quantum circuit
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=2)
discopy_circuit = ansatz(diagram)
discopy_circuit.draw(figsize=(15,10))