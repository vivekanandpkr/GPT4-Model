import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load log data from a file (assuming log_data.txt contains the log entries)
log_data = []
with open('data.txt', 'r') as file:
    for line in file:
        timestamp, log_text = line.strip().split('\t')
        log_data.append(log_text)

# Tokenize log texts and create a mapping
tokenizer = Tokenizer()
tokenizer.fit_on_texts(log_data)
log_text_to_id = tokenizer.word_index

# Convert log texts to numeric tokens
sequences = tokenizer.texts_to_sequences(log_data)

# Create input-output pairs
input_sequences = []
output_sequences = []
sequence_length = 3

for i in range(len(sequences) - sequence_length):
    input_sequences.append(sequences[i:i + sequence_length])
    output_sequences.append(sequences[i + sequence_length])

input_sequences = np.array(input_sequences)
output_sequences = np.array(output_sequences)

# Pad sequences
max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
output_sequences = pad_sequences(output_sequences, maxlen=1, padding='pre')

# Convert output sequences to one-hot vectors
num_unique_logs = len(log_text_to_id) + 1  # Add 1 for padding token
output_sequences = to_categorical(output_sequences, num_unique_logs)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(input_sequences, output_sequences, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=num_unique_logs, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(num_unique_logs, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
epochs = 100
batch_size = 16
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Use the model for log sequence extraction and root cause analysis
new_log_sequences = [...]  # Replace [...] with the new log sequences to analyze
new_input_sequences = tokenizer.texts_to_sequences(new_log_sequences)
new_input_sequences = pad_sequences(new_input_sequences, maxlen=max_sequence_length, padding='pre')
predicted_sequences = model.predict_classes(new_input_sequences)

# Convert predicted sequences back to log texts
predicted_log_texts = [tokenizer.index_word[idx] for idx in predicted_sequences]

# Extract the root cause from the predicted log texts
root_cause = ""
for log_text in predicted_log_texts:
    root_cause += log_text + " "

print("Root Cause:", root_cause)
