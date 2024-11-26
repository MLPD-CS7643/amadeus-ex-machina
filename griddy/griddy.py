import itertools
import json
import os
import torch
import uuid
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from models.griddy_model import GriddyModel
from griddy_solver import GriddySolver

NJOBS = 8
DEVICE = "cuda"
BASE_DIR = Path("./griddy")
JSON_DIR = BASE_DIR / "out"
IMG_DIR = BASE_DIR / "plot"
CSV_FILENAME = "griddy_table.csv"


FIXED_PARAMS = {
    "device": DEVICE,
    "model_class": GriddyModel,
    "solver_class": GriddySolver,
    "optim_class": torch.optim.SGD,
    "loss_class": torch.nn.CrossEntropyLoss,
    "dtype": "float32",
    "warmup": 5,
    "epochs": 100,
    "early_stop": True,
    "early_stop_epochs": 10,
    "steps": [32, 49, 60, 100],
    "print": False,
}

MODEL_PARAMS = {
    "n_blocks": [3],
    "block_depth": [3],
    "pad": [1],
    "stride": [1],
    "k_conv": [3],
    "maxpool": [2],
    "dropout": [0.2, 0.3],
    "out_channels": [64]
}

SGD_PARAMS = {
    "batch_size": [16],
    "lr": [0.03, 0.02],
    "momentum": [0.98, 0.99],
    "weight_decay": [0.00001],
}

PARAM_RANGES = MODEL_PARAMS | SGD_PARAMS


def hit_griddy(param_ranges, fixed_params, n_jobs):
    """
    I am addicted to hitting the griddy.

    Args:
        param_ranges (dict): params to grid search
        fixed_params (dict): params to not grid search (lame)
        n_jobs (int): number of workers

    Returns:
        None
    """
    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR)

    print("\"Hitting the griddy...\" -Ellie")

    keys, values = zip(*param_ranges.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for params in permutations:
        params.update(fixed_params)

    if n_jobs == 1:
        for params in permutations:
            __griddy_iter(params, fixed_params)
    else:
        Parallel(n_jobs=n_jobs)(delayed(__griddy_iter)(params, fixed_params) for params in permutations)

    print("DONE")

def __griddy_iter(params, fixed_params):

    name = str(uuid.uuid1()).replace("-","")

    print(f"START: {name}\nPARAMS: {params}")

    solver_class = fixed_params.pop("solver_class")
    solver = solver_class(**params)
    score, model, stop_epoch, train_history, valid_history, class_score = solver.train()
    print(f"END: {name} - SCORE: {__str_digits(score)}")
    #model.half()
    torch.save(
        model.state_dict(),
        f"./{BASE_DIR}/{__str_digits(score)}_{name}.pth",
    )

    for key in fixed_params.keys():
        params.pop(key, None)

    out = {
        "name": name,
        "size": __model_size(model),
        "score": round(score,4),
        "class_score": class_score,
        "stop_epoch": stop_epoch,
        "train_history": train_history,
        "valid_history": valid_history,
        "params": params
    }

    with open(f"{JSON_DIR}/{__str_digits(score)}_{name}.json", "w") as f:
        json.dump(out, f)

def __model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def __str_digits(num):
    return "{:.4f}".format(num).replace(".", "")


### NASTY PLOT ###

def plot_from_json(json_folder=JSON_DIR, img_folder=IMG_DIR):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder, filename)
            
            # Load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            train_history = data['train_history']
            valid_history = data['valid_history']
            title = "{:.4f}".format(data['score']).replace('.', '') + data['name']
            filename = title
            #title = f"LR:{data['params']['learning_rate']}"
            #title = f"Reg: {data['params']['reg']}"

            __plot_curves(train_history, valid_history, img_folder, filename, title)

def json_to_csv(json_folder=JSON_DIR, csv_filename=CSV_FILENAME):
    results = []
    
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder, filename)
            
            with open(file_path, 'r') as file:
                data = json.load(file)

            entry = {
                'params': str(data['params']),
                'stop_epoch': data['stop_epoch'],
                'size': data['size'],
                'score': data['score'],
                'name': data['name']
            }
            results.append(entry)

    results_df = pd.DataFrame(results)

    results_df.to_csv(os.path.join(json_folder, csv_filename), index=False)

def __plot_curves(train_history, valid_history, img_folder, filename, title):
    plt.figure()
    plt.plot(train_history, label='Training Score')
    plt.plot(valid_history, label='Validation Score')
    plt.title(f"{title}")
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{img_folder}/{filename}.png')
    plt.close()