from lambeq import BobcatParser, TreeReader, TreeReaderMode, spiders_reader, cups_reader, stairs_reader
from lambeq import TensorAnsatz, SpiderAnsatz, MPSAnsatz, AtomicType
from lambeq import SpacyTokeniser
from discopy import Dim, grammar
from utilities import *

tokeniser = SpacyTokeniser()
parser = BobcatParser(verbose = "progress")
ansatz = TensorAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})

token = tokeniser.tokenise_sentence("The CNG should detect replayed user credentials and/or device credentials ")
print(token)
diagram = parser.sentence2diagram(token, tokenised=True)
print(diagram)



"""
labels, sentences = extract_data("code/datasets/ePurse_edited.csv")
#tokens = tokeniser.tokenise_sentences(sentences)



faults = []
i = 0
count = 0
while i < len(sentences):
    try:
        print(f"parsing string {i} of {len(sentences)}")
        diagram = parser.sentence2diagram(sentences[i], tokenised = True)
        i += 1
    except Exception: 
        faults.append((labels[i], sentences[i]))
        count += 1
        i += 1
        continue
    """
#print(len(faults), count)


