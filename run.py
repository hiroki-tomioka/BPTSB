import sys
import torch
import argparse
import time
import json
import yaml

sys.path.append("src/")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to a config file")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--gpu", type=int, default=-1, help="if -1, only CPU will be used")
parser.add_argument(
    "--load", type=str, default=None, help="path to an existing result folder"
)
parser.add_argument(
    "--load_file_suffix",
    type=str,
    default=None,
    help="suffix of the file to load; ex. if 'final', agent_final.pickle and simulenv_final.pickle will be loaded",
)
parser.add_argument(
    "--memo",
    type=str,
    default=None,
    help="this will be the suffix for the result folder",
)

args = parser.parse_args()


def load_pretraining_config(path, paramsList):
    """Load pretraining parameters to be reused

    Args:
        path (str): Pretraining file path
        paramsList (list): Parameters to be reused

    Returns:
        dict: Loaded parameters to be reused
    """
    pretraining_config_paths = []
    while True:
        with open(path + "params.json", encoding="utf-8") as f:
            _data = json.load(f)
        pretraining_config_paths.append(_data["config"])
        if _data["load"] is None:
            break
        else:
            path = _data["load"]
    data = {}
    for pretraining_config_path in reversed(pretraining_config_paths):
        with open(pretraining_config_path, encoding="utf-8") as f:
            _data = yaml.safe_load(f)
        data.update(_data)
    pretraining_config = {}
    for param in paramsList:
        if param in data:
            pretraining_config[param] = data[param]
    return pretraining_config


# load config file
with open(args.config, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# load pretrining config file
if args.load is not None:
    reuse_params = ["timeStep", "tau", "shape", "size", "nodes", "neighbours"]
    if config["task"] == "MNIST":
        reuse_params.extend(["pixel"])
    elif config["task"] in ["expandedNARMA", "memory"]:
        reuse_params.extend(["inputScale"])
    elif config["task"] == "Lissajous":
        reuse_params.extend(
            [
                "inputScale",
                "lissajous1",
                "lissajous2",
                "wind",
                "generatorNum",
                "updateTimeScale",
                "feedbackSprings",
                "generatorDist",
                "reservoirStateType",
            ]
        )
    elif config["task"] == "locomotion":
        reuse_params.extend(
            [
                "inputScale",
                "wind",
                "generatorNum",
                "updateTimeScale",
                "feedbackSprings",
                "generatorDist",
                "reservoirStateType",
            ]
        )
    for param in reuse_params:
        if param in config.keys():
            reuse_params = [p for p in reuse_params if p != param]
    pretraining_config = load_pretraining_config(args.load, reuse_params)
    config.update(pretraining_config)

assert config["task"] in [
    "MNIST",
    "expandedNARMA",
    "memory",
    "Lissajous",
    "locomotion",
]
assert config["brain"] in ["LIL", "MLP", "CNN", "SWG", "FL"]
assert config["shape"] in ["DoubleCircles", "MultiCircles", "CaterpillarRobot"]

if "lissajous2" in config:
    if len(config["lissajous2"]) == 4:
        lissajous2 = config["lissajous2"]
    else:
        lissajous2 = None

if "generatorNum" in config:
    generatorNum = config["generatorNum"]
elif "generatorDist" in config:
    assert (config["nodes"] - 1) % 2 == 0
    if "all" in config["generatorDist"]:
        generatorNum = [-1]
    else:
        generatorNum = []
        if "ex" in config["generatorDist"]:
            generatorNum.extend([i for i in range(config["nodes"] - 1)])
        if "mid" in config["generatorDist"]:
            generatorNum.extend(
                [config["nodes"] - 1]
                + [i for i in range(config["nodes"], config["nodes"] * 2 - 4, 2)]
            )
        if "in" in config["generatorDist"]:
            generatorNum.extend(
                [i for i in range(config["nodes"] + 1, config["nodes"] * 2 - 3, 2)]
                + [config["nodes"] * 2 - 3]
            )
        generatorNum.sort()

torch.autograd.set_detect_anomaly(True)
if args.gpu == -1 or not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{args.gpu}")
print("device:", device)


if __name__ == "__main__":

    start_time = time.time()

    if config["task"] == "MNIST":
        from experiment_batch import *
    else:
        from experiment import *

    # general settings
    e = Experiment(
        simTime_=config["simTime"],
        simTimeStep_=config["timeStep"],
        tau_=config["tau"],
        shape_=config["shape"],
        initSize_=config["size"],
        noNodes_=config["nodes"],
        mass_=1,
        noNeighbours_=config["neighbours"],
        seed_=args.seed,
        device_=device,
        loadFolder_=args.load,
        loadFileSuffix_=args.load_file_suffix,
        params_=args.__dict__,
        memo_=args.memo,
    )

    # individual experiment
    if config["task"] == "MNIST":
        if config["experiment"] == "train":
            e.MNIST_training(
                physicalOutput=True,
                brain=config["brain"],
                pixelNum=config["pixel"],
                char=config["char"],
                optim=config["optim"],
                lr_spring=config["lr_spring"],
                lr_damping=config["lr_damping"],
                lr_restlength=config["lr_restlength"],
                lr_readoutlayer=0,
                lr_brain=config["lr_brain"],
                dataSize=config["dataSize"],
                testDataSize=config["testDataSize"],
                epoch=config["epoch"],
                fixedImages=config["fixedImages"],
                batchSize=config["batchSize"],
            )
        elif config["experiment"] == "labelmap":
            e.MNIST_labelmap(
                physicalOutput=True,
                brain=config["brain"],
                pixelNum=config["pixel"],
                char=config["char"],
                testDataSize=config["testDataSize"],
                fixedImages=config["fixedImages"],
                deltaScale=config["deltaScale"],
            )
        elif config["experiment"] == "PCA":
            e.MNIST_PCA(
                brain=config["brain"],
                pixelNum=config["pixel"],
                char=config["char"],
                testDataSize=config["testDataSize"],
                fixedImages=config["fixedImages"],
            )
        else:
            raise ValueError("Experiment in config file is inappropriate")
    elif config["task"] == "expandedNARMA":
        if config["experiment"] == "train":
            e.timeseries_training(
                dirClass="expandedNARMA/delta",
                delay=config["delay"],
                physicalOutput=True,
                epoch=config["epoch"],
                inputScale=config["inputScale"],
                brain=config["brain"],
                optim=config["optim"],
                lr_spring=config["lr_spring"],
                lr_damping=config["lr_damping"],
                lr_restlength=config["lr_restlength"],
                lr_readoutlayer=0,
                lr_brain=config["lr_brain"],
            )
        else:
            raise ValueError("Experiment in config file is inappropriate")
    elif config["task"] == "memory":
        if config["experiment"] == "train":
            e.timeseries_training(
                dirClass="memoryTask/delta",
                delay=config["delay"],
                physicalOutput=True,
                epoch=config["epoch"],
                inputScale=config["inputScale"],
                brain=config["brain"],
                optim=config["optim"],
                lr_spring=config["lr_spring"],
                lr_damping=config["lr_damping"],
                lr_restlength=config["lr_restlength"],
                lr_readoutlayer=0,
                lr_brain=config["lr_brain"],
                cumulative=config["cumulative"],
            )
        else:
            raise ValueError("Experiment in config file is inappropriate")
    elif config["task"] == "Lissajous":
        if config["experiment"] == "SWG":
            e.Lissajous_training(
                updateTimeScale=config["updateTimeScale"],
                optim=config["optim"],
                lr_spring=config["lr_spring"],
                lr_damping=config["lr_damping"],
                lr_restlength=config["lr_restlength"],
                lr_amplitude=config["lr_amplitude"],
                lr_omega=0,
                lr_phase=config["lr_phase"],
                generatorNum=generatorNum,
                inputScale=config["inputScale"],
                wind=config["wind"],
                lissajous1=config["lissajous1"],
                lissajous2=lissajous2,
                noise=config["noise"],
                generatorDist=config["generatorDist"],
            )
        elif config["experiment"] == "closedloop":
            e.Lissajous_closedloop(
                updateTimeScale=config["updateTimeScale"],
                generatorNum=generatorNum,
                wind=config["wind"],
                lissajous1=config["lissajous1"],
                lissajous2=lissajous2,
                feedbackSprings=config["feedbackSprings"],
                reservoirStateType=config["reservoirStateType"],
                noise=config["noise"],
            )
        elif config["experiment"] == "perturbation":
            e.Lissajous_perturbation(
                updateTimeScale=config["updateTimeScale"],
                lissajous1=config["lissajous1"],
                feedbackSprings=config["feedbackSprings"],
                reservoirStateType=config["reservoirStateType"],
            )
        elif config["experiment"] == "switching-dynamics":
            e.Lissajous_switching_dynamics(
                generatorNum=generatorNum,
                inputScale=config["inputScale"],
                wind=config["wind"],
                lissajous1=config["lissajous1"],
                lissajous2=lissajous2,
            )
        else:
            raise ValueError("Experiment in config file is inappropriate")
    elif config["task"] == "locomotion":
        if config["experiment"] == "SWG":
            e.locomotion_training(
                updateTimeScale=config["updateTimeScale"],
                optim=config["optim"],
                lr_spring=config["lr_spring"],
                lr_damping=config["lr_damping"],
                lr_restlength=config["lr_restlength"],
                lr_amplitude=config["lr_amplitude"],
                lr_omega=0,
                lr_phase=config["lr_phase"],
                generatorNum=generatorNum,
                inputScale=config["inputScale"],
                wind=config["wind"],
            )
        elif config["experiment"] == "closedloop":
            e.locomotion_closedloop(
                generatorNum=generatorNum,
                wind=config["wind"],
                feedbackSprings=config["feedbackSprings"],
                reservoirStateType=config["reservoirStateType"],
                noise=config["noise"],
            )
        elif config["experiment"] == "perturbation":
            e.locomotion_perturbation(
                updateTimeScale=config["updateTimeScale"],
                feedbackSprings=config["feedbackSprings"],
                reservoirStateType=config["reservoirStateType"],
            )
        elif config["experiment"] == "switching-dynamics":
            e.locomotion_switching_dynamics(
                updateTimeScale=config["updateTimeScale"],
                generatorNum=generatorNum,
                inputScale=config["inputScale"],
                wind=config["wind"],
            )
        else:
            raise ValueError("Experiment in config file is inappropriate")
    else:
        raise ValueError("Task in config file is inappropriate")

    end_time = time.time()
    print("elapsed time:", end_time - start_time)
