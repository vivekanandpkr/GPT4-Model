#Access GPT 4 via API

import openai
print('hello')

api_key = "sk-sjTp9PDtyyyyyyyyyy" #INsert the api key which you can find it in openai.com
openai.api_key = api_key

def chat_with_gpt3(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can use "text-davinci-002" for GPT-3.5 or other available engines.
        prompt=prompt,
        temperature=0.7,  # Adjust the temperature to control the randomness of the model's responses.
        max_tokens=150,   # Limit the response length (in tokens).
        stop=["\n"]       # Stop generating the response when a newline character is encountered.
    )
    return response.choices[0].text.strip()

# Example usage:
user_input = "Hey GPT, What are all the efficient ways that we can use AI for the benifit of humanity and how does quantum computer works ?.!"
response = chat_with_gpt3(user_input)
print(response)
