def extract_data(dataset):
    label, sentence = [], []
    file = open(dataset, "r")
    for line in file:
        label.append( [float(line[0]), 1 - float(line[0])])
        sentence.append(line[1:].strip())
    return label, sentence