'''
Parameter Update Functions (Optimisers)
'''

# Imports
import numpy as np

# Main Functions
# Update Functions
# @ N Kausik CS21M037
def UpdateFunc_SGD(X, Y, parameters, grads_fn, data, **params):
    '''
    Update parameters using Stoachastic Gradient Descent
    '''
    lr = params["lr"]

    Ws, bs = parameters["Ws"], parameters["bs"]

    # Init grads
    W_grads = []
    b_grads = []
    for i in range(len(Ws)):
        W_grads.append(np.zeros(Ws[i].shape))
        b_grads.append(np.zeros(bs[i].shape))

    # Calculate grads
    n_samples = X.shape[0]
    for i in range(X.shape[0]):
        x, y = X[i].reshape((1, -1)), Y[i].reshape((1, -1))
        
        grads_data = grads_fn["func"](x, y, parameters, **grads_fn["params"])
        for j in range(len(Ws)):
            W_grads[j] += grads_data["dW"][j] / n_samples
            b_grads[j] += grads_data["db"][j] / n_samples

    # Update parameters
    for i in range(len(Ws)):
        W, b = Ws[i], bs[i]
        W_grad, b_grad = W_grads[i], b_grads[i]

        W = W - lr * W_grad
        b = b - lr * b_grad

        Ws[i], bs[i] = W, b

    parameters.update({
        "Ws": Ws,
        "bs": bs
    })
    
    return parameters, data

def UpdateFunc_Momentum(X, Y, parameters, grads_fn, data, **params):
    '''
    Update parameters using Momentum based Gradient Descent
    '''
    lr = params["lr"]
    gamma = params["gamma"]

    Ws, bs = parameters["Ws"], parameters["bs"]

    # Init data if required
    if "W_velocity" not in data:
        data["W_velocity"] = []
        data["b_velocity"] = []
        for j in range(len(Ws)):
            data["W_velocity"].append(np.zeros(Ws[j].shape))
            data["b_velocity"].append(np.zeros(bs[j].shape))

    # Init grads
    W_grads = []
    b_grads = []
    for i in range(len(Ws)):
        W_grads.append(np.zeros(Ws[i].shape))
        b_grads.append(np.zeros(bs[i].shape))

    # Calculate grads
    n_samples = X.shape[0]
    for i in range(X.shape[0]):
        x, y = X[i].reshape((1, -1)), Y[i].reshape((1, -1))
        
        grads_data = grads_fn["func"](x, y, parameters, **grads_fn["params"])
        for j in range(len(Ws)):
            W_grads[j] += grads_data["dW"][j] / n_samples
            b_grads[j] += grads_data["db"][j] / n_samples

    # Update parameters
    for i in range(len(Ws)):
        W, b = Ws[i], bs[i]
        W_grad, b_grad = W_grads[i], b_grads[i]
        W_velocity, b_velocity = data["W_velocity"][i], data["b_velocity"][i]

        W_velocity = gamma * W_velocity + lr * W_grad
        b_velocity = gamma * b_velocity + lr * b_grad
        W = W - W_velocity
        b = b - b_velocity

        data["W_velocity"][i], data["b_velocity"][i] = W_velocity, b_velocity
        Ws[i], bs[i] = W, b

    parameters.update({
        "Ws": Ws,
        "bs": bs
    })
    
    return parameters, data

def UpdateFunc_Nesterov(X, Y, parameters, grads_fn, data, **params):
    '''
    Update parameters using Nesterov Accelerated Gradient Descent
    '''
    lr = params["lr"]
    gamma = params["gamma"]

    Ws, bs = parameters["Ws"], parameters["bs"]

    # Init data if required
    if "W_velocity" not in data:
        data["W_velocity"] = []
        data["b_velocity"] = []
        for j in range(len(Ws)):
            data["W_velocity"].append(np.zeros(Ws[j].shape))
            data["b_velocity"].append(np.zeros(bs[j].shape))

    # Init lookahead grads
    W_grads = []
    b_grads = []
    parameters_lookahead = dict(parameters)
    for i in range(len(Ws)):
        W_grads.append(np.zeros(Ws[i].shape))
        b_grads.append(np.zeros(bs[i].shape))

        W_lookahead = Ws[i] - gamma * data["W_velocity"][i]
        b_lookahead = bs[i] - gamma * data["b_velocity"][i]
        parameters_lookahead["Ws"][i], parameters_lookahead["bs"][i] = W_lookahead, b_lookahead

    # Calculate lookahead grads
    n_samples = X.shape[0]
    for i in range(X.shape[0]):
        x, y = X[i].reshape((1, -1)), Y[i].reshape((1, -1))
        
        grads_data = grads_fn["func"](x, y, parameters_lookahead, **grads_fn["params"])
        for j in range(len(Ws)):
            W_grads[j] += grads_data["dW"][j] / n_samples
            b_grads[j] += grads_data["db"][j] / n_samples

    # Update parameters
    for i in range(len(Ws)):
        W, b = Ws[i], bs[i]
        W_grad, b_grad = W_grads[i], b_grads[i]
        W_velocity, b_velocity = data["W_velocity"][i], data["b_velocity"][i]

        W_velocity = gamma * W_velocity + lr * W_grad
        b_velocity = gamma * b_velocity + lr * b_grad
        W = W - W_velocity
        b = b - b_velocity

        data["W_velocity"][i], data["b_velocity"][i] = W_velocity, b_velocity
        Ws[i], bs[i] = W, b

    parameters.update({
        "Ws": Ws,
        "bs": bs
    })
    
    return parameters, data

# @ Karthikeyan S CS21M028
# TODO: AdaGrad, RMSProp, Adam, NAdam