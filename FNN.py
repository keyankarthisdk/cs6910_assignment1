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
# TODO: Forward Propogation


# @ N Kausik CS21M037
# TODO: Backward Propogation


# @ Karthikeyan S CS21M028
# TODO: Model Training (With Epochs and Batch)


# @ Karthikeyan S CS21M028
# TODO: Model Predict