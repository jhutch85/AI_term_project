import json

with open("./questions_corrected.json", 'r') as file:
    data = json.load(file)


for i in range(5):
    print(data[i])
