import pandas as pd

# Sample log data in a DataFrame
data = {
    'Log time': ['2023/07/20 08.20', '2023/07/20 08.21', '2023/07/20 08.22', '2023/07/20 08.23', '2023/07/20 08.24'],
    'Log Text': ['Start', 'Laptop temperature', 'Excel Application Start', 'Excel Crashed', 'End']
}
df = pd.DataFrame(data)

# Tokenize log texts and create a mapping
unique_log_texts = df['Log Text'].unique()
log_text_to_id = {text: idx for idx, text in enumerate(unique_log_texts)}

# Convert log texts to numeric tokens
df['Log Text ID'] = df['Log Text'].map(log_text_to_id)

# Create sequences with a sequence length of 3
sequence_length = 3
sequences = [df['Log Text ID'][i:i+sequence_length].tolist() for i in range(len(df)-sequence_length+1)]

# Create input-output pairs
input_sequences = [seq[:-1] for seq in sequences]
output_sequences = [seq[-1] for seq in sequences]

print("Input Sequences:", input_sequences)
print("Output Sequences:", output_sequences)
