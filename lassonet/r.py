from dataclasses import asdict
import torch
from .interfaces import (
    lassonet_path as _lassonet_path,
    LassoNetClassifier,
    LassoNetRegressor,
)


def lassonet_path(*args, **kwargs):
    def convert_item(item):
        item = asdict(item)
        item["state_dict"] = {k: v.numpy() for k, v in item["state_dict"].items()}
        item["selected"] = item["selected"].numpy()
        return item

    return list(map(convert_item, _lassonet_path(*args, **kwargs)))


def lassonet_eval(X, task, state_dict, **kwargs):
    if task == "classification":
        model = LassoNetClassifier(**kwargs)
    elif task == "regression":
        model = LassoNetRegressor(**kwargs)
    else:
        raise ValueError('task must be "classification" or "regression"')
    state_dict = {k: torch.tensor(v) for k, v in state_dict.items()}
    model.load(state_dict)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    else:
        return model.predict(X)
