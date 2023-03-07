"""from lambeq import BobcatParser
from discopy import grammar

sentence = 'John walks in the park'

# Parse the sentence and convert it into a string diagram
parser = BobcatParser(verbose='suppress')
diagram = parser.sentence2diagram(sentence)

grammar.draw(diagram, figsize=(14,3), fontsize=12)"""


from lambeq import spiders_reader

sentence = 'John walks in the park'
# Create string diagrams based on spiders reader
spiders_diagram = spiders_reader.sentence2diagram(sentence)

# Not a pregroup diagram, we can't use grammar.draw()
spiders_diagram.draw(figsize=(13,6), fontsize=12)