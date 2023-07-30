import matplotlib.pyplot as plt

# Given sequences
sequences = [
    [4, 2, 1, 7, 3, 2, 7, 1],
    [1, 2, 7, 6, 1, 4, 5, 3],
    [5, 4, 3, 2, 6, 5, 7, 4],
    [7, 6, 3, 2, 1, 7, 4, 5],
    [2, 1, 4, 3, 6, 7, 5, 1]
]

# Create a new figure
plt.figure(figsize=(10,6))

# Plot each sequence
for i, sequence in enumerate(sequences, start=1):
    plt.plot(sequence, label=f'Sequence {i}')

# Set title and labels
plt.title('Sequences Visualization')
plt.xlabel('Position in Sequence')
plt.ylabel('Value')

# Create a legend
plt.legend()

# Show the plot
plt.show()
