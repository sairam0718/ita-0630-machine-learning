import csv

def read_data(file_path):
    data = ["D:\ml\data.csv"]
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data

def find_s(data):
    hypothesis = None
    for instance in data:
        if instance['Target Concept'] == 'Malignant (+':
            if hypothesis is None:
                hypothesis = instance.copy()
            else:
                for key, value in instance.items():
                    if key != 'Target Concept' and hypothesis[key] != value:
                        hypothesis[key] = '?'
    return hypothesis

def candidate_elimination(data):
    specific_hypothesis = None
    general_hypothesis = {}
    for instance in data:
        if instance['Target Concept'] == 'Malignant (+':
            if specific_hypothesis is None:
                specific_hypothesis = instance.copy()
                general_hypothesis = {k: ['?'] for k in instance.keys() if k != 'Target Concept'}
            else:
                for key, value in instance.items():
                    if key != 'Target Concept':
                        if specific_hypothesis[key] != value:
                            specific_hypothesis[key] = '?'
                        if value not in general_hypothesis[key]:
                            general_hypothesis[key].append(value)
        else:
            for key, value in instance.items():
                if key != 'Target Concept':
                    if value in general_hypothesis[key]:
                        general_hypothesis[key].remove(value)
                        if len(general_hypothesis[key]) == 0:
                            general_hypothesis[key] = ['?']

    return specific_hypothesis, general_hypothesis

# Read the training data
data = read_data('data.csv')

# Apply FIND-S algorithm
find_s_hypothesis = find_s(data)
print("FIND-S Hypothesis:")
print(find_s_hypothesis)

# Apply Candidate Elimination algorithm
specific, general = candidate_elimination(data)
print("\nCandidate Elimination Hypothesis:")
print("Specific Hypothesis:")
print(specific)
print("General Hypothesis:")
print(general)
