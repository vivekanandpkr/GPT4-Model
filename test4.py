import torch
import torch.nn as nn
import math

# Define model
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ninp = ninp
        self.pos_encoder = nn.Embedding(ninp, ninp)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.transformer = nn.Transformer(ninp, nhead, nhid, nlayers)
        self.fc_out = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)


    def forward(self, x):
        x = self.encoder(x) * math.sqrt(self.ninp)
        x = self.pos_encoder(torch.arange(0, x.size(0)).to(x.device).unsqueeze(1)) + x
        output = self.transformer(x, x)
        output = self.fc_out(output)
        return output

# Given sequences
sequences = [
    [2, 3, 6, 1, 7, 9, 4, 5, 2, 8, 2, 7, 7, 8, 4, 2],
    [2, 3, 8, 6, 4, 3, 1, 6, 9, 2, 3, 3, 2, 4, 5, 3],
    [3, 1, 5, 2, 2, 8, 3, 6, 3, 7, 8, 7, 4, 3, 8, 7],
    [6, 7, 9, 4, 9, 7, 7, 6, 6, 2, 4, 4, 9, 2, 8, 7],
    [8, 7, 7, 4, 7, 1, 3, 7, 2, 2, 7, 6, 9, 9, 3, 5],
    [3, 3, 1, 7, 5, 6, 9, 8, 4, 6, 9, 7, 3, 4, 8, 2],
    [6, 3, 8, 3, 7, 3, 4, 7, 2, 7, 1, 4, 4, 4, 6, 8],
    [2, 8, 2, 8, 5, 5, 6, 7, 9, 2, 1, 8, 8, 2, 9, 7],
    [9, 6, 3, 5, 5, 2, 3, 5, 9, 4, 2, 7, 3, 6, 2, 4],
    [5, 5, 2, 7, 6, 8, 6, 8, 9, 7, 6, 7, 1, 5, 5, 2]
]

# Preprocess the data
data = torch.tensor(sequences, dtype=torch.long)

# Parameters
ntokens = 10  # Numbers from 1 to 7 plus potential padding, unique elements
emsize = 10  # Embedding size
nhid = 10  # Hidden size
nlayers = 2  # Number of transformer layers
nhead = 2  # Number of heads in multiheadattention models
dropout = 0.2  # Dropout rate, regularization technique for reducing overfit in nn by preventing complex co-adaptations on training data.
lr = 5.0  # Learning rate, hyper parameter that determines the step size at each iteration while moving towards a min of loss function
epochs = 10  # Number of epochs
batch_size = 10  # Batch size, hyperparameter of gradient descent that controls the number of training samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# Training loop
for epoch in range(1, epochs + 1):
    total_loss = 0.
    for batch, i in enumerate(range(0, data.size(0) - 1, batch_size)):
        data_batch = data[i:i+batch_size].transpose(0, 1).contiguous().to(device)
        optimizer.zero_grad()
        output = model(data_batch[:-1, :])
        loss = criterion(output.view(-1, ntokens), data_batch[1:, :].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % 100 == 0 and batch > 0:
            print('| epoch {:3d} | loss {:5.2f}'.format(epoch, total_loss))

    scheduler.step()

def predict(model, sequence, num_predictions):
    model.eval()
    with torch.no_grad():
        sequence = torch.tensor(sequence).unsqueeze(1).to(device)
        output = model(sequence)
        _, predicted = torch.topk(output, num_predictions)

    return predicted.squeeze().tolist()

# Let's make a prediction using the first sequence
sequence = sequences[3]
num_predictions = 1  # For simplicity, let's predict the most likely next number

predicted = predict(model, sequence, num_predictions)
print("Predicted sequence: ", predicted)
