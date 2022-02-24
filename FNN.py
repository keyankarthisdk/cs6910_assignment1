'''
Feed Forward Neural Network
'''

# Imports
import pickle
import numpy as np
from tqdm import tqdm

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
# Propagation Functions
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
def Model_Train(inputs, layer_sizes, funcs, n_epochs=1, batch_size=1, wandb_data={"use_wandb": False}):
    X, Y, X_val, Y_val = inputs["X"], inputs["Y"], inputs["X_val"], inputs["Y_val"]
    Y_indices = np.argmax(Y, axis=1)
    Y_val_indices = np.argmax(Y_val, axis=1)

    history = {
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_datapoints": X.shape[0],
        "layer_sizes": layer_sizes,
        "Ws": [],
        "bs": [],
        "loss": {
            "train": [],
            "val": []
        },
        "eval": {
            "train": [],
            "val": []
        }
    }

    # Intialize parameters
    parameters = Params_Init(layer_sizes, funcs)

    # Record Starting Weights and Biases
    Ws = [np.copy(w) for w in parameters["Ws"]]
    bs = [np.copy(b) for b in parameters["bs"]]
    history["Ws"].append(Ws)
    history["bs"].append(bs)

    # Train model
    for i in tqdm(range(n_epochs)):
        for j in tqdm(range(0, X.shape[0], batch_size), disable=False):
            cur_batch_range = [j, min(j+batch_size, X.shape[0])]
            X_batch = X[cur_batch_range[0]:cur_batch_range[1]]
            Y_batch = Y[cur_batch_range[0]:cur_batch_range[1]]

            # Backpropogate and update parameters
            grads_fn = {
                "func": BackwardPropogation,
                "params": {}
            }
            parameters, funcs["update_fn"]["data"] = funcs["update_fn"]["func"](
                X_batch, Y_batch, 
                parameters, grads_fn, 
                funcs["update_fn"]["data"], **funcs["update_fn"]["params"]
            )

        # Calculate loss and evaluation metric
        W_flat = np.concatenate([w.flatten() for w in parameters["Ws"]])
        b_flat = np.concatenate([b.flatten() for b in parameters["bs"]])
        params_flat = np.concatenate([W_flat, b_flat])
        loss_reg = parameters["loss_fn"]["params"]["l2_lambda"] * np.sqrt(np.sum(params_flat**2)) + \
                    parameters["loss_fn"]["params"]["l1_lambda"] * np.sum(np.abs(params_flat))
        # Training
        Y_train_out, As, Os = ForwardPropogation(X, parameters)
        Y_train_out = Y_train_out.T
        loss_train = loss_reg + parameters["loss_fn"]["func"](Y_train_out, Y, **parameters["loss_fn"]["params"])
        Y_train_out_indices = np.argmax(Y_train_out, axis=-1)
        eval_train = funcs["eval_fn"]["func"](Y_train_out_indices, Y_indices, **funcs["eval_fn"]["params"])
        # Validation
        Y_val_out, As_val, Os_val = ForwardPropogation(X_val, parameters)
        Y_val_out = Y_val_out.T
        loss_val = loss_reg + parameters["loss_fn"]["func"](Y_val_out, Y_val, **parameters["loss_fn"]["params"])
        Y_val_out_indices = np.argmax(Y_val_out, axis=-1)
        eval_val = funcs["eval_fn"]["func"](Y_val_out_indices, Y_val_indices, **funcs["eval_fn"]["params"])
            
        # Record Epoch History
        history["loss"]["train"].append(loss_train)
        history["loss"]["val"].append(loss_val)
        history["eval"]["train"].append(eval_train)
        history["eval"]["val"].append(eval_val)

        if(i%1 == 0):
            print(f"EPOCH {i}:")
            print(f"\tLOSS: TRAIN: {loss_train}, VAL: {loss_val}")
            print(f"\tEVAL: TRAIN: {eval_train} / {Y_indices.shape[0]}, VAL: {eval_val} / {Y_val_indices.shape[0]}")

        # Wandb Logging
        if wandb_data["use_wandb"]:
            wandb.log({
                "epoch": i+1,
                "loss_train": loss_train,
                "loss_val": loss_val,
                "eval_train": eval_train,
                "eval_val": eval_val
            })

    return parameters, history

# Predict Functions
def Model_Predict(X, parameters):
    y_pred, _, _ = ForwardPropogation(X, parameters)
    return y_pred

# Load Save Functions
def Model_Load(path):
    parameters = pickle.load(open(path, "rb"))
    return parameters

def Model_Save(parameters, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(parameters, f)

