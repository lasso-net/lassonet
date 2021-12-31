import numpy as np
import dataclasses
from pathlib import Path
import json
import requests
import sys
from urllib.parse import quote
from appdirs import user_config_dir

from .utils import query_yes_no, machine_identifier


DEFAULT_CONFIG = dict(
    autolog=False, identifier=None, endpoint="https://log.lassonet.ml"
)
config_path = Path(user_config_dir("lassonet")) / "config.json"
config_path.parent.mkdir(parents=True, exist_ok=True)
if not config_path.exists():
    with open(config_path, "w") as f:
        json.dump(DEFAULT_CONFIG, f)


def get_config(attr=None):
    with open(config_path) as f:
        config = json.load(f)
    if attr is None:
        return config
    return config.get(attr)


def configure(config=None):
    if config is None:
        config = get_config()
        config["autolog"] = query_yes_no("Activate automatic logging?")
        config["identifier"] = (
            input(
                "Use custom identifier? (leave empty to use a machine identifier)\n",
            )
            or None
        )
    with open(config_path, "w") as f:
        json.dump(config, f)


def dump_model(model):
    dump = model.get_params()  # exploit sklearn's features
    if dump["optim"] is not None:
        dump["optim"] = str(dump["optim"])
    if dump["device"] is not None:
        dump["device"] = dump["device"].type
    del dump["random_state"]
    del dump["torch_seed"]
    return dump


def footprint(X, y):
    return np.cov(np.concatenate((X.cpu().numpy(), y.cpu().numpy()), axis=1).T)


def convert_history(hist):
    ans = []
    for it in hist:
        it = dataclasses.asdict(it)
        del it["state_dict"]
        it["selected"] = it["selected"].numpy().tolist()
        ans.append(it)
    return ans


def is_dev():
    """
    Detect if there is a .git folder (indicates local development)
    """
    return (Path(__file__).parents[1] / ".git").exists()


def identifier():
    ans = get_config("identifier")
    if ans is None:
        return machine_identifier()
    return ans


def upload(model, data, hist, online_logging=False):
    """
    data : tuple (X, y)
    """
    from . import __version__  # avoid circular import

    if not (get_config("autolog") or online_logging):
        return
    experiment = online_logging if isinstance(online_logging, str) else ""
    mid = str(identifier())
    log = dict(
        version=__version__,
        model=dump_model(model),
        data_footprint=footprint(*model._cast_input(*data)).tolist(),
        history=convert_history(hist),
        dev=is_dev(),
        identifier=mid,
        experiment=experiment,
    )
    endpoint = get_config("endpoint")
    r = requests.post(
        endpoint + "/log/new",
        json=log
        # gzip spares >50% of bandwidth, causes
        # causes `Can not decode content-encoding: gzip`
        # on server
        # data=zlib.compress(json.dumps(log).encode()),
        # headers={"content-encoding": "gzip"},
    )
    print(r)
    if not r.ok:
        print("Could not log, got status code", r.status_code, file=sys.stderr)
    else:
        id_ = r.json()["id"]
        print(f"Successfully uploaded to {endpoint}/log/{id_}")
        if experiment:
            print(
                f"See other logs for {experiment} at "
                f"{endpoint}/log/exp/{quote(experiment)})"
            )
        print(f"See all your logs at {endpoint}/log/id/{quote(mid)}")
