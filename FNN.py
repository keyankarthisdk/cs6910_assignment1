'''
Feed Forward Neural Network
'''

# Imports
import numpy as np

from FunctionsLibrary.InitFunctions import *
from FunctionsLibrary.ActivationFunctions import *
from FunctionsLibrary.LossFunctions import *
from FunctionsLibrary.UpdateFunctions import *
from FunctionsLibrary.EvalFunctions import *

import wandb

# Main Functions
# @ Karthikeyan S CS21M028
# Init Functions
def Params_Init(layer_sizes, funcs):
    Ws = []
    bs = []
    Ws, bs = funcs["init_fn"]["func"](layer_sizes, **funcs["init_fn"]["params"])

    parameters = {
        "n_layers": len(layer_sizes),
        "layer_sizes": layer_sizes,
        "Ws": Ws,
        "bs" : bs,
        "act_fns": funcs["act_fns"],
        "loss_fn": funcs["loss_fn"]
    }
    for i in range(len(Ws)):
        print(i, "W:", (Ws[i].min(), Ws[i].max()), "b:", (bs[i].min(), bs[i].max()))
    return parameters


# @ N Kausik CS21M037
def ForwardPropogation(X, parameters):
    Ws = parameters["Ws"]
    bs = parameters["bs"]
    act_fns = parameters["act_fns"]

    # Initial a = x
    Os = []
    As = []
    a = np.copy(X).T
    for layer in range(len(Ws)):
        # o = W a(previous layer) + b
        Os.append(np.matmul(Ws[layer], a) + bs[layer])
        # a = activation(o)
        a = act_fns[layer]["func"](Os[layer])
        # Save all activations
        As.append(np.copy(a))

    return a, As, Os


# @ N Kausik CS21M037
def BackwardPropogation(X, y, parameters):
    n_samples = X.shape[0]
    Ws = parameters["Ws"]
    bs = parameters["bs"]
    act_fns = parameters["act_fns"]
    n_layers = len(Ws)+1

    grads = {
        "dA": [0]*(n_layers-1),
        "dW": [0]*(len(Ws)),
        "db": [0]*(len(bs)),
    }
    
    # Find final activations
    a, As, Os = ForwardPropogation(X, parameters)

    for layer in reversed(range(n_layers-1)):
        # Output Layer
        if layer == n_layers-2:
            grads["dA"][layer] = parameters["loss_fn"]["deriv"](a, y.T, **parameters["loss_fn"]["params"])
            # Update Weights
            grads["dW"][layer] = (1/n_samples) * np.matmul(grads["dA"][layer], As[layer-1].T)
            # Update Biases
            grads["db"][layer] = (1/n_samples) * np.sum(grads["dA"][layer], axis=1, keepdims=True)
        # Hidden Layers
        else:
            dA = act_fns[layer]["deriv"](Os[layer])
            grads["dA"][layer] = np.matmul(Ws[layer+1].T, grads["dA"][layer+1]) * dA
            # Update Weights
            if layer == 0: # Input Layer
                grads["dW"][layer] = (1/n_samples) * np.matmul(grads["dA"][layer], X)
            else:
                grads["dW"][layer] = (1/n_samples) * np.matmul(grads["dA"][layer], As[layer-1].T)
            # Update Biases
            grads["db"][layer] = (1/n_samples) * np.sum(grads["dA"][layer], axis=1, keepdims=True)

    return grads


# @ Karthikeyan S CS21M028
# TODO: Model Training (With Epochs and Batch)


# @ Karthikeyan S CS21M028
# TODO: Model Predict