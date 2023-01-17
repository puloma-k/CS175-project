import pandas as pd
import torch
import string
import torch.nn as nn
import time
import math

########################################################### 
# collect training dataset from file intp list of reviews, ratings, targets
###########################################################
all_chars = string.printable #features as all ASCII characters
n_chars = len(all_chars)
n_targets = 2 #2 targets (fake as -1 and real as 1)
training_dataset=pd.read_csv('partitions/Xtrain.csv')
training_target=pd.read_csv('partitions/Ytrain.csv')
reviews=training_dataset['Review'].values.tolist()
ratings=training_dataset['Rating'].values.tolist()
targets=training_target['Generated'].values.tolist()


###########################################################
# Turn reviews into a <review_length x 1 x n_chars> tensor
###########################################################
def reviewToTensor(review):
    tensor = torch.zeros(len(review), 1, n_chars)
    for li, letter in enumerate(review):
        tensor[li][0][all_chars.find(letter)] = 1
    return tensor


########################################################### 
# Neural Network Model
###########################################################
class RNN(nn.Module):
    def __init__(self,layer_cnt, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size #...
        # Put the declaration of RNN network here
        self.hidden_layer =  nn.Linear(input_size + hidden_size, hidden_size) #... input and the output of the hidden layer are combined
        self.output_layer =  nn.Linear(input_size + hidden_size, output_size) #... input and the output of the hidden layer are combined
        self.softmax = nn.LogSoftmax(dim=1) #...
        self.layer_cnt=layer_cnt

    def forward(self, input, hidden):
        # Put the computation for forward pass here
        combined = torch.cat((input, hidden),1) #... input and the output of the hidden layer are combined
        for i in range(self.layer_cnt):
            hidden = self.hidden_layer(combined) #... feed the combined input to hidden layers
        output = self.output_layer(combined) #... feed the combined input to output layers
        output = self.softmax(output) #... softmax on output to generate percentages for each class

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = None
def initRNN(n_hidden,n_layer): # n_hidden = number of nodes per layer
    global rnn 
    rnn = RNN(n_layer, n_chars, n_hidden, n_targets)

########################################################### 
# Training function
###########################################################
def train_iteration_CharRNN(learning_rate, target_tensor, review_tensor):
    criterion = nn.NLLLoss()
    hidden = rnn.initHidden()
    rnn.zero_grad()

    # The forward process 
    for i in range(review_tensor.size()[0]):
        output, hidden = rnn(review_tensor[i], hidden) #... iteratively update output and hidden

    # The backward process
    loss = criterion(output, target_tensor) #... calculate loss
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate) #...

    return output, loss.item()

def train_charRNN(learning_rate):
    n_iters=1000#len(reviews) #uncomment to train on entire training dataset
    print_every = 1000

    current_loss = 0

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for iter in range(1, n_iters + 1):
        target = targets[iter-1]
        review = reviews[iter-1]
        target_tensor = torch.tensor([{-1:0,1:1}[target]], dtype=torch.long)
        review_tensor = reviewToTensor(review)
        output, loss = train_iteration_CharRNN(learning_rate, target_tensor, review_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            print('%d %d%% (%s) loss = %.4f' % (iter, iter / n_iters * 100, timeSince(start), loss))
            print('Average loss: %.4f' % (current_loss / print_every))
            current_loss = 0

    torch.save(rnn, 'char-rnn-classification.pt')


########################################################### 
# Predicting function
###########################################################
def predict(input_review):
    print("Prediction for %s:" % input_review)
    hidden = rnn.initHidden()

    # Generate the input for RNN
    review_tensor = reviewToTensor(input_review) #... convert input review to tensor
    for i in range(review_tensor.size()[0]): #...
        output, hidden = rnn(review_tensor[i],hidden) #... iteratively update output and hidden

    # Get the value and index of top K predictions from the output
    # Then apply Softmax function on the scores of all target predictions so we can
    # output the probabilities that this name belongs to different languages.

    softmax_layer = nn.Softmax(dim=1)
    top_prob = softmax_layer(output) #... convert output to percentages, probabilities and pick top
    return top_prob,output

########################################################### 
# Sample test
###########################################################
initRNN(128,2)
train_charRNN(0.01)
top_prob,output=predict('Great pizza. Great service')
print('(target = %s) Probability: (%.2f), Score: (%.2f)' % (-1, top_prob[0][0]*100, output[0][0]))
print('(target = %s) Probability: (%.2f), Score: (%.2f)' % (1, top_prob[0][1]*100, output[0][1]))