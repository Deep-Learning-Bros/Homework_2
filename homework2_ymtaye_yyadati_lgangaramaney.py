## Yared Taye
## Lokesh Gangaramaney
## Yash Yadati
import numpy as np

X_tr = np.reshape(np.load("../age_regression_Xtr.npy"), (-1, 48 * 48))
ytr = np.load("../age_regression_ytr.npy")
X_te = np.reshape(np.load("../age_regression_Xte.npy"), (-1, 48 * 48))
yte = np.load("../age_regression_yte.npy")

## Split Dataset
X_tr, X_val = X_tr[:4000, :], X_tr[4000:, :]
ytr, yval = ytr[:4000], ytr[4000:]

## WEIGHT Generate
def generate_weights():
    sigma = 0.01 ** 0.1
    return sigma * np.random.randn(48*48+1) + 0.5

## Prediction on test set based on weight/bias
def predict(X, W_tilda):
    w = W_tilda[:-1]
    b = W_tilda[-1]
    return X.dot(w) + b


def fmse(y_predit, y_actual):
    err = np.square(y_predit-y_actual)
    err = np.mean(err) * 1/2
    return err


def gradient(X, y, weight, bias, alpha):
    diff = X.dot(weight) + bias - y
    unregularized = X.T.dot(diff.T) / X.shape[0]
    return unregularized + (alpha / X.shape[0])*weight, np.average(diff)


def minibatches(X, y, batchsize):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
        ids = indices[start_idx:start_idx + batchsize]
        yield X[ids], y[ids]



def SGD( X_tr, ytr, StepSize=0.001, alpha=0.01, batchsize=50, epochs=20):
    W_tilda = generate_weights()
    weight = W_tilda[:-1]
    bias = W_tilda[-1]
    for e in range(0, epochs):
        for batch in minibatches(X_tr, ytr, batchsize):
            X_tr_batch, ytr_batch = batch
            g, delta = gradient(X_tr_batch, ytr_batch, weight, bias, alpha)
            weight = weight - (StepSize * g)
            bias = bias - (StepSize * delta)
            W_tilda = np.hstack((weight, bias))

    return W_tilda


def predictions(Train_X, Test_X, Train_y, Test_y, aug_w, StepSize, batches, ep, Wd):
    train_pred = predict(Train_X, aug_w)
    test_pred = predict(Test_X, aug_w)
    train_cost = fmse(train_pred, Train_y)
    test_cost = fmse(test_pred, Test_y)
    print("*"*50)
    print('Current Learning Rate: ', StepSize)
    print('Batches: ', batches)
    print('Epochs', ep)
    print('L2-Reg', Wd)
    print('fMSE for Training Set: ', train_cost)
    print("*"*50)
    print('Current Learning Rate: ', StepSize)
    print('Batches: ', batches)
    print('Epochs', ep)
    print('L2-Reg', Wd)
    print('fMSE for Testing Set: ', test_cost)
    print("*" * 50)



def grid_search():
    ### Grid Search
    LearningRates = [0.0001, 0.001, 0.005, 0.01]
    BatchSizes = [10, 50, 100, 200]
    WeightDecays = [0.02, 0.05, 0.07, 0.1]
    num_epochs = [5, 10, 50, 100]
    for ep in num_epochs:
        for Wd in WeightDecays:
            for batches in BatchSizes:
                for lr in LearningRates:
                    weight = (SGD(X_tr, ytr, alpha=Wd, StepSize=lr, batchsize=batches, epochs=ep))
                    predictions(X_val, X_te, yval, yte, weight, lr, batches, ep, Wd)

def summary():
    ## USED HYPERPARAMETERS

    alpha = 0.02
    epislon = 0.001
    num_batches  = 10
    num_epochs = 100
    weight = (SGD(X_tr, ytr, alpha=alpha, StepSize=epislon, batchsize=num_batches, epochs=num_epochs))
    predictions(X_tr, X_te, ytr, yte, weight, ep=num_epochs, batches=num_batches, StepSize=epislon, Wd=alpha)


summary()