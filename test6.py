# List of sequences
sequences = [
    ['A', 'B', 'C', 'A', 'B'],
    ['B', 'C', 'A', 'D'],
    ['A', 'B', 'E', 'F', 'A'],
    # More sequences...
]

# Sequence of interest
sequence_of_interest = ['C', 'A', 'D']

# Best Two occurrences
count = sum(1 for sequence in sequences for i in range(len(sequence)-1) if sequence[i:i+2] == sequence_of_interest)

#Best Three Occurances
count = sum(1 for sequence in sequences for i in range(len(sequence)-2) if sequence[i:i+3] == sequence_of_interest)


print(f"The sequence {sequence_of_interest} occurs {count} times.")
