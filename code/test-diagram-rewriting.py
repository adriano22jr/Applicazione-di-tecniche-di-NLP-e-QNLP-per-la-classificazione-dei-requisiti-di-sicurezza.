from lambeq import BobcatParser, TreeReader, TreeReaderMode, Rewriter, spiders_reader, cups_reader, stairs_reader
from discopy import grammar

sentence = 'John that seems to be sus, walks in the park being sussy and walking his fucking dog'
sentence2 = "John gave Mary a flower"

parser = BobcatParser(verbose="suppress")
rewriter = Rewriter(["prepositional_phrase", "determiner", "auxiliary", "connector", "coordination"])

print(Rewriter.available_rules())

"""diagram = parser.sentence2diagram(sentence)
rewrited_diagram = rewriter(diagram).normal_form()

rewrited_diagram.draw(figsize=(14, 3), fontsize=12)"""