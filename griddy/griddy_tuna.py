import os
import copy
import inspect
import torch
import optuna
import torch.utils
from collections.abc import Iterable
from models.griddy_model import GriddyModel
from pathlib import Path
from solver import Solver
from enum import Enum, auto


class SearchMethod(Enum):
    """
    CATEGORICAL - search within a defined set of values, ex: [8, 32, 128, 512]
    UNIFORM - find value between two floats, ex: [1.0, 4.0]
    LOG_UNIFORM - same but log-scaled, ex: [0.001, 0.1]
    """
    CATEGORICAL = auto()
    UNIFORM = auto()
    LOG_UNIFORM = auto()


### EXAMPLE PARAM SETUP ###

SOLVER_PARAMS = {
    Solver : {
        "device": "cuda",
        "epochs": 10,
        "early_stop_epochs": 0, # early stop after n epochs without improvement, 0 to disable
        "warmup_epochs": 0, # 0 to disable
        "dtype": "float16",
        "train_dataloader": None, # assume a DataLoader object
        "valid_dataloader": None, # assume a DataLoader object
        "direction": "minimize" # must specify this, even if not used by solver
    }
}

MODEL_PARAMS = {
    GriddyModel: {
        "n_blocks": [3],
        "block_depth": [3],
        "pad": [1],
        "stride": [1],
        "k_conv": [3],
        "maxpool": [2],
        "dropout": [0.2, 0.3],
        "out_channels": [16, 32, 64, 128, SearchMethod.CATEGORICAL],
        # "out_channels": [16, 32, 64, 128] is also valid, CATEGORICAL is assumed in this instance
    }
}

OPTIM_PARAMS = {
    torch.optim.SGD : {
        "lr": [0.001, 0.1, SearchMethod.LOG_UNIFORM],
        "momentum": [0.9, 0.99, SearchMethod.UNIFORM],
        "weight_decay": [0.00001],
    },
    torch.optim.Adam : {
        "lr": [0.03, 0.02, 0.01, 0.1],
        "momentum": [0.98, 0.99],
        "weight_decay": [0.00001],
    }
}

SCHED_PARAMS = {
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts : {
        "T_max": [10],
    },
    torch.optim.lr_scheduler.StepLR : {
        "step_size": [10, 20],
        "gamma" : [0.1, 0.05],
    }
}

CRITERION_PARAMS = {
    torch.nn.CrossEntropyLoss : {}
}

PARAM_SET = {
    "solver": SOLVER_PARAMS,
    "model" : MODEL_PARAMS,
    "optim" : OPTIM_PARAMS,
    "sched" : SCHED_PARAMS,
    "criterion" : CRITERION_PARAMS,
}

def hit_griddy(study_name, param_set, out_dir, n_trials, n_jobs, prune, resume):
    """
    I am addicted to hitting the griddy.

    Args:
        study_name (str)
        param_set (dict): full set of params following example pattern
        out_dir: (Path or str): folder to save output
        n_trials (int): number of trials
        n_jobs (int): number of workers
        prune (bool): enable optuna pruning
        resume (bool): resume existing study if one exists

    Returns:
        None
    """
    print("\"Hitting the griddy...\" -Ellie")

    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, f"{study_name}.db")
    storage_path = f'sqlite:///{full_path}'

    if not resume:
        try:
            optuna.delete_study(study_name=study_name, storage=storage_path)
        except:
            pass

    study = optuna.create_study(study_name=study_name, direction=__get_direction(param_set), storage=storage_path, load_if_exists=resume)
    objective = __create_objective(param_set, out_dir, prune)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print("DONE")

def __create_objective(param_set, save_dir, prune):
    # optuna objective function
    def objective(trial):
        model = __instantiate_class_with_trial_params(trial, 'model', copy.deepcopy(param_set), enforce_single_class=True)
        optimizer = __instantiate_class_with_trial_params(trial, 'optim', copy.deepcopy(param_set), pass_through_kwargs={'params': model.parameters()})
        scheduler = __instantiate_class_with_trial_params(trial, 'sched', copy.deepcopy(param_set), pass_through_kwargs={'optimizer': optimizer})
        criterion = __instantiate_class_with_trial_params(trial, 'criterion', copy.deepcopy(param_set))
        solver:Solver = __instantiate_class_with_trial_params(trial, 'solver', copy.deepcopy(param_set), enforce_single_class=True, pass_through_kwargs={'model': model, 'optimizer': optimizer, 'scheduler': scheduler, 'criterion': criterion, 'optuna_prune': prune})
        best_metric = solver.train_and_evaluate(trial)
        #if len(trial.study.trials_dataframe().dropna(subset=['value'])) > 0 and trial.study.best_trial.number == trial.number:
        #    torch.save(solver.best_model, Path(save_dir) / f'{solver.best_model.__class__.__name__}_best_model.pth')
        return best_metric
    return objective

def __instantiate_class_with_trial_params(trial, class_group, param_set, pass_through_kwargs:dict=None, enforce_single_class=False):
    """
    Dynamically creates an instance of a class based on Optuna trial suggestions.
    
    Args:
    - trial: The Optuna trial object.
    - class_group: The group in the PARAM_SET that contains the classes, e.g., 'optim'.
    - full_param_set: Full dictionary of all parameters for all classes.
    - enforce_single_class: Set True if this group of classes should only contain one class (like solver).
    
    Returns:
    - An instance of the selected class with the dynamically chosen parameters.
    """
    param_group_dict = param_set[class_group]
    class_keys = list(param_group_dict.keys())

    # Enforce single class
    if enforce_single_class and len(class_keys) > 1:
        raise ValueError(f"Only one class is allowed in the {class_group} group when 'enforce_single_class' is True.")

    if len(class_keys) > 1:
        class_key_map = {cls.__name__: cls for cls in class_keys}
        chosen_class_str = trial.suggest_categorical(f'{class_group}_class', list(class_key_map.keys()))
        chosen_class = class_key_map[chosen_class_str]
    else:
        chosen_class = class_keys[0]

    # Collect parameters for the chosen class
    if pass_through_kwargs:
        class_params = pass_through_kwargs
    else:
        class_params = {}
    for key, values in param_group_dict[chosen_class].items():
        if isinstance(values, list):
            if len(values) > 1:
                if isinstance(values[-1], SearchMethod):
                    search_method = values[-1]
                    values.pop()
                else:
                    search_method = SearchMethod.CATEGORICAL
                if len(class_keys) > 1:
                    unique_key = f"{chosen_class_str}_{key}"
                    param_value = __create_optuna_suggest(trial, unique_key, values, search_method)
                else:
                    param_value = __create_optuna_suggest(trial, key, values, search_method)
                class_params[key] = param_value
            elif len(values) == 1:
                class_params[key] = values[0]
        else:
            class_params[key] = values  # Directly assign non-iterable values

    __validate_params(chosen_class, class_params)
    return chosen_class(**class_params)

def __create_optuna_suggest(trial, name, values, method):
    if method == SearchMethod.CATEGORICAL:
        return trial.suggest_categorical(name, values)
    elif method == SearchMethod.UNIFORM:
        return trial.suggest_uniform(name, min(values), max(values))
    elif method == SearchMethod.LOG_UNIFORM:
        return trial.suggest_loguniform(name, min(values), max(values))

def __get_direction(param_set):
    try:
        solver_dict = param_set['solver']
    except:
        ValueError(f"param_set must include solver class with \"direction\" entry.")
    for _, v in solver_dict.items():
        try:
            return v['direction']
        except:
            ValueError(f"param_set must include solver class with \"direction\" entry.")

def __validate_params(chosen_class, class_params):
    valid_params = dict(inspect.signature(chosen_class.__init__).parameters)
    valid_params.pop('kwargs', None)
    valid_keys = set(valid_params.keys())
    provided_keys = set(class_params.keys())

    # Check for any invalid parameters
    invalid_keys = provided_keys - valid_keys
    if invalid_keys:
        raise ValueError(f"Invalid parameters for {chosen_class.__name__}: {invalid_keys}")

    # Optionally, check for missing required parameters
    required_params = {name for name, param in valid_params.items()
                       if param.default == inspect.Parameter.empty and name != 'self'}
    missing_params = required_params - provided_keys
    if missing_params:
        raise ValueError(f"Missing required parameters for {chosen_class.__name__}: {missing_params}")
