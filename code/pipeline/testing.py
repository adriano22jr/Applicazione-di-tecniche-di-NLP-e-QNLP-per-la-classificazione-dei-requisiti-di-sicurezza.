from lambeq import BobcatParser, TreeReader, TreeReaderMode, spiders_reader, cups_reader, stairs_reader
from lambeq import TensorAnsatz, SpiderAnsatz, MPSAnsatz, AtomicType
from lambeq import SpacyTokeniser
from discopy import Dim, grammar
from utilities import *

tokeniser = SpacyTokeniser()
parser = BobcatParser(verbose = "progress")
ansatz = TensorAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})
labels, sentences = extract_data("code/datasets/GPS_edited.csv")
tokens = tokeniser.tokenise_sentences(sentences)


"""i = 0
while i < len(tokens):
    try:
        print(f"parsing string {i} of {len(tokens)}")
        diagram = parser.sentence2diagram(tokens[i], tokenised = True)
        i += 1
    except: 
        Exception
        print(tokens[i])
        i += 1
        continue"""

diagrams = parser.sentences2diagrams(tokens, tokenised = True)

i = 0
while i < len(diagrams):
    try:
        print(f"converting diagram {i} of {len(diagrams)}")
        circuit = ansatz(diagrams[i])
        i += 1
    except: 
        Exception
        print(diagrams[i])
        i += 1
        continue
    
"""circuit = ansatz(diagrams[0])
print(circuit)
#circuits = [ansatz(diagram) for diagram in diagrams]
#print(f"Done! Number of diagrams: {len(diagrams)}")"""