import glob
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
EOS = n_letters - 1

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicode_to_ascii("O'Néàl"))

def read_lines(filename):
    lines = open(filename, encoding="utf8").read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

category_lines = {}
all_categories = []
i = 0
for filename in glob.glob('data\\names\\*.txt'):
    category = filename.split('\\')[-1].split('.')[0]
    all_categories.append(category)
    i = i + 1
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

print('# categories:', n_categories, all_categories)
      
# RNN model
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
    

## prepairing for training
import random

# Get a random category and random line from that category
def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    return category, line

# One-hot vector for category
def make_category_input(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return Variable(tensor)

# One-hot matrix of first to last letters (not including EOS) for input
def make_chars_input(chars):
    tensor = torch.zeros(len(chars), n_letters)
    for ci in range(len(chars)):
        char = chars[ci]
        tensor[ci][all_letters.find(char)] = 1
    tensor = tensor.view(-1, 1, n_letters)
    return Variable(tensor)

# LongTensor of second letter to end (EOS) for target
def make_target(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    tensor = torch.zeros([len(letter_indexes), 1], dtype=torch.long)
    for i in range(len(line)):
        tensor[i][0] = letter_indexes[i]
    #tensor = torch.LongTensor(letter_indexes)
    return Variable(tensor)


# Make category, input, and target tensors from a random category, line pair
def random_training_set():
    category, line = random_training_pair()
    category_input = make_category_input(category)
    line_input = make_chars_input(line)
    line_target = make_target(line)
    return category_input, line_input, line_target


# Train the network
def train(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()
    loss = 0
    
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()
    optimizer.step()
    
    return output, loss.data[0] / input_line_tensor.size()[0]

import time
import math

def time_since(t):
    now = time.time()
    s = now - t
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


n_epochs = 100000
print_every = 5000
plot_every = 500
all_losses = []
loss_avg = 0 # Zero every plot_every epochs to keep a running average
learning_rate = 0.0005

rnn = RNN(n_letters, 128, n_letters)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

start = time.time()

for epoch in range(1, n_epochs + 1):
    category_input, line_input, line_target = random_training_set()
    output, loss = train(category_input, line_input, line_target)
    loss_avg += loss
    
    if epoch % print_every == 0:
        print('%s (%d %d%%) %.4f' % (time_since(start), epoch, epoch / n_epochs * 100, loss))

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0
        
## ploting the losses
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


## generation
max_length = 20

# Generate given a category and starting letter
def generate_one(category, start_char='A', temperature=0.5):
    category_input = make_category_input(category)
    chars_input = make_chars_input(start_char)
    hidden = rnn.init_hidden()

    output_str = start_char
    
    for i in range(max_length):
        output, hidden = rnn(category_input, chars_input[0], hidden)
        
        # Sample as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Stop at EOS, or add to output_str
        if top_i == EOS:
            break
        else:    
            char = all_letters[top_i]
            output_str += char
            chars_input = make_chars_input(char)

    return output_str

# Get multiple samples from one category and multiple starting letters
def generate(category, start_chars='ABC'):
    for start_char in start_chars:
        print(generate_one(category, start_char))
        
        
generate('English', 'RUS')