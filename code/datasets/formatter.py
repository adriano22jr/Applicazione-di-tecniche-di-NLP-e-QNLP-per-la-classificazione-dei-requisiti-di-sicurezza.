import collections

file = open("code/datasets/GPS.csv", "r")
new_file = open("code/datasets/GPS_edited.csv", "w")
new_list = []

for line in file:
    words = line.split()
    if words[len(words) - 1].endswith(";sec"):
        words.append("1 ")
        queue = collections.deque(words)
        queue.rotate(1)
        
        new_list = list(queue)
        new_list.pop()
        new_list.append(".")
    
    if words[len(words) - 1].endswith(";nonsec"):
        words.append("0 ")
        queue = collections.deque(words)
        queue.rotate(1)
        
        new_list = list(queue)
        new_list.pop()
        new_list.append(".")

    for item in new_list:
        new_file.write(str(item) + " ")
    new_file.write("\n")