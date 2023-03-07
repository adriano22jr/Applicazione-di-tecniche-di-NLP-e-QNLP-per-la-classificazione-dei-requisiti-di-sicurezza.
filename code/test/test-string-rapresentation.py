from lambeq import BobcatParser, TreeReader, TreeReaderMode, spiders_reader, cups_reader, stairs_reader
from discopy import grammar

sentence = 'John walks in the park'
sentence2 = "John gave Mary a flower"

# Parse the sentence and convert it into a string diagram
parser = BobcatParser(verbose="progress")
diagram = parser.sentence2diagram(sentence)
grammar.draw(diagram, figsize=(14,3), fontsize=12)


spiders_diagram = spiders_reader.sentence2diagram(sentence)
spiders_diagram.draw(figsize=(13,6), fontsize=12)


cups_diagram = cups_reader.sentence2diagram(sentence)
grammar.draw(cups_diagram, figsize=(12,3), fontsize=12)


stairs_diagram = stairs_reader.sentence2diagram(sentence)
stairs_diagram.draw(figsize=(12,5), fontsize=12)


reader = TreeReader()
tree_diagram = reader.sentence2diagram(sentence2)
tree_diagram.draw(figsize=(12,5), fontsize=12)


reader = TreeReader(mode=TreeReaderMode.RULE_ONLY)
tree_diagram = reader.sentence2diagram(sentence2)
tree_diagram.draw(figsize=(12,5), fontsize=12)