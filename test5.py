import pandas as pd
from prefixspan import PrefixSpan

#Laptop Windows Logs
# Ask GPT 4 to create the sample logs for you
# Use Prefixspan to calculate the best occurrences

# Log data
log_data = {
    "log_timestamp": [
        "2023-07-30T09:02:34",
        "2023-07-30T09:02:36",
        "2023-07-30T09:04:02",
        "2023-07-30T09:05:11",
        "2023-07-30T09:10:55",
        # more timestamps...
    ],
    "log_text": [
        "System booted successfully.",
        "Battery level low: 15% remaining.",
        "Failed to connect to WiFi network 'HomeWiFi'.",
        "USB device inserted, Vendor: Sandisk, Product: Cruzer Blade.",
        "CPU temperature: 90C, threshold: 95C.",
        # more log texts...
    ]
}

# Convert to DataFrame
df = pd.DataFrame(log_data)

# Get unique log_text values
unique_log_texts = df['log_text'].unique()

# Create a dictionary mapping log_text to a unique ID
log_text_to_id = {text: f"{i+1:03}" for i, text in enumerate(unique_log_texts)}

# Add the IDs to the DataFrame
df['log_id'] = df['log_text'].map(log_text_to_id)

print(df)
print(log_text_to_id)
print(df['log_id'])

ps = PrefixSpan(df['log_id'])
patterns = ps.frequent(2)  # Find frequent patterns with minimum support of 2

# Print the frequent sequential patterns
for pattern in patterns:
    print(pattern)
