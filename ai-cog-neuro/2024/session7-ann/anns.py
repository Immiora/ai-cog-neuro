## create a toy regression dataset
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=300, n_features=10, n_informative=5, n_targets=2, noise=1, random_state=87)

## split into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=87)

## fit a linear regression model on the train set
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train, y_train)

## report performance on train and test set
print(f'Train score: {reg.score(x_train, y_train)}')
print(f'Test score: {reg.score(x_test, y_test)}')

## create a linear regression model using an ANN
import torch

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

## specify the model, objective and learning rule
epochs = 5
model = LinearRegression(inputSize= x_train.shape[-1], outputSize=y_train.shape[-1] )
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

## train an ANN
for epoch in range(epochs):
    inputs = torch.autograd.Variable(torch.from_numpy(x_train).float())
    labels = torch.autograd.Variable(torch.from_numpy(y_train).float())

    # clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)

    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters using backpropagation
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


## get prediction for train and test data using the trained ANN
with torch.no_grad(): # we don't need gradients in the testing phase
    predicted_train = model(torch.autograd.Variable(torch.from_numpy(x_train).float())).data.numpy()
    predicted_test = model(torch.autograd.Variable(torch.from_numpy(x_test).float())).data.numpy()

## calculate the performance score for the ANN model
from sklearn.metrics import r2_score
print(f'Train nn: {r2_score(y_train, predicted_train)}')
print(f'Train nn: {r2_score(y_test, predicted_test)}')

## plot predictions
from matplotlib import pyplot as plt
plt.figure()
plt.subplot(211)
plt.plot(y_train[:,0], 'ko--', label='True train', alpha=0.7)
plt.plot(predicted_train[:,0], 'go--', label='Predict train', alpha=0.4)
plt.legend(loc='best')
plt.subplot(212)
plt.plot(y_test[:,0], 'ko--', label='True test', alpha=0.7)
plt.plot(predicted_test[:,0], 'go--', label='Predict test', alpha=0.4)
plt.legend(loc='best')
plt.show()

##
# train for more epochs, report results, is there any improvement?
# how many epochs do you need to obtain a good r2_score?
# how does the lr parameter affect training?

##
# create a toy dataset for classification using sklearn's make_classification
# create an ANN equivalent to a logistic regression
# train the ANN implementation of a logistic regression and compare its performance to a sklearn logistic regression classifier

##
# create random input data (x) and pass it through an arbitrary function to generate y
# train a deeper neural network with non-linear activation functions to see whether you can approximate the arbitrary function



