import os
import torch
import csv
import json
import random
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker


"""Optimizers"""
OPTIM_DICT = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
}


def isCallable(obj):
    """Returns boolean whether object can be called (like a function)"""
    return hasattr(obj, "__call__")


def mkdir_p(path):
    """Method to create a directory only if it does not already exists"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        raise


class SpaceList(object):
    """Class which allows easy bundeling of x,y and eventualy z positions"""

    __array_priority__ = 1000  # assure numpy calls with SpaceList multiplication are handled by SpaceList

    def __init__(self, *args):
        """Create a vector instance, no copy of data is created!!!"""
        if len(args) == 0:
            self.dim = 0
        if len(args) == 1:
            assert (
                args[0].ndim == 2
            ), "if only one argument is used a matrix must be given"
            assert args[0].shape[0] in (2, 3)
            self.dim = args[0].shape[0]
            self.matrix = args[0]
        elif len(args) == 2:
            assert args[0].ndim == 1, "each element must be an numpy array"
            assert args[1].ndim == 1, "each element must be an numpy array"
            self.matrix = torch.stack((args[0], args[1]), 0)
            self.dim = 2
        elif len(args) == 3:
            assert args[0].ndim == 1, "each element must be an numpy array"
            assert args[1].ndim == 1, "each element must be an numpy array"
            assert args[2].ndim == 1, "each element must be an numpy array"
            self.dim = 3
            self.matrix = torch.stack((args[0], args[1], args[2]), 0)
        else:
            raise (NotImplementedError)

    def __getattribute__(self, name):
        if name == "x":
            return self.matrix[0, :]
        elif name == "y":
            return self.matrix[1, :]
        elif name == "shape":
            return self.matrix.shape
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "x":
            self.matrix[0, :] = value
        elif name == "y":
            self.matrix[1, :] = value
        else:
            super(SpaceList, self).__setattr__(name, value)

    def __add__(left, right):
        return SpaceList(left.matrix + right.matrix)

    def __iadd__(self, right):
        self.matrix = self.matrix + right.matrix
        return self

    def __mul__(self, other):
        """Multipication with SpaceList, np array or scalar"""
        if hasattr(other, "matrix"):
            return SpaceList(self.matrix * other.matrix)
        return SpaceList(self.matrix * other)

    __rmul__ = __mul__  # make no distiction between left and right multiplication

    def __truediv__(self, other):
        """Division with SpaceList, np array or scalar"""
        if hasattr(other, "matrix"):
            return SpaceList(self.matrix / other.matrix)
        return SpaceList(self.matrix / other)

    def __imul__(self, other):
        try:
            self.matrix = self.matrix * other.matrix
        except AttributeError:
            self.matrix = self.matrix * other
        return self

    def copy(self):
        """Create a copy of a vector"""
        mat = self.matrix.clone()
        return SpaceList(mat)

    def __str__(self):
        string = "X: " + str(self.x) + " \n Y:" + str(self.y)
        if self.dim == 3:
            string += "\n Z: " + str(self.z)
        return string

    def get(self, nodenumber):
        return torch.tensor(self.matrix[:, nodenumber])

    def getnoNodes(self):
        return torch.numel(self.matrix[0])

    def getDifference(self):
        """Returns 2 or 3 matrices with the element on position (i,j) gives the difference between xi and xj"""
        difx = self.x[:, None] - self.x[None, :]
        dify = self.y[:, None] - self.y[None, :]
        return difx, dify

    def getArray(self):
        """Returns an array with all the matrix values"""
        return torch.reshape(self.matrix, (1, -1))

    def ground(pos, speed):
        for i in range(pos.getnoNodes()):
            if pos.matrix[1, i] < 0:  # Y component goes below the ground plane
                pos.matrix[1, i] = 0
                speed.matrix[:, i] = 0


class SpaceListBatch(SpaceList):
    """Class which allows easy bundeling of x,y and eventualy z positions"""

    __array_priority__ = 1000  # assure numpy calls with SpaceList multiplication are handled by SpaceList

    def __init__(self, *args):
        """Create a vector instance, no copy of data is created!!!"""
        if len(args) == 0:
            self.dim = 0
        if len(args) == 1:
            assert (
                args[0].ndim == 3
            ), "if only one argument is used a matrix must be given"
            assert args[0][0].shape[0] in (2, 3)
            self.dim = args[0][0].shape[0]
            self.matrix = args[0]
        elif len(args) == 2:
            assert args[0].ndim == 2, "each element must be an numpy array"
            assert args[1].ndim == 2, "each element must be an numpy array"
            self.matrix = torch.stack((args[0], args[1]), 1)
            self.dim = 2
        elif len(args) == 3:
            assert args[0].ndim == 2, "each element must be an numpy array"
            assert args[1].ndim == 2, "each element must be an numpy array"
            assert args[2].ndim == 2, "each element must be an numpy array"
            self.dim = 3
            self.matrix = torch.stack((args[0], args[1], args[2]), 0)
        else:
            raise (NotImplementedError)

    def __getattribute__(self, name):
        if name == "x":
            return self.matrix[:, 0, :]
        elif name == "y":
            return self.matrix[:, 1, :]
        elif name == "shape":
            return self.matrix.shape
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "x":
            self.matrix[:, 0, :] = value
        elif name == "y":
            self.matrix[:, 1, :] = value
        else:
            super(SpaceListBatch, self).__setattr__(name, value)

    def __add__(left, right):
        return SpaceListBatch(left.matrix + right.matrix)

    def __iadd__(self, right):
        self.matrix = self.matrix + right.matrix
        return self

    def __mul__(self, other):
        """Multipication with SpaceListBatch, np array or scalar"""
        if hasattr(other, "matrix"):
            return SpaceListBatch(self.matrix * other.matrix)
        return SpaceListBatch(self.matrix * other)

    __rmul__ = __mul__  # make no distiction between left and right multiplication

    def __truediv__(self, other):
        """Division with SpaceListBatch, np array or scalar"""
        if hasattr(other, "matrix"):
            return SpaceListBatch(self.matrix / other.matrix)
        return SpaceListBatch(self.matrix / other)

    def copy(self):
        """Create a copy of a vector"""
        mat = self.matrix.clone()
        return SpaceListBatch(mat)

    def getnoNodes(self):
        return torch.numel(self.matrix[0][0])

    def getDifference(self):
        """Returns 2 or 3 matrices with the element on position (i,j) gives the difference between xi and xj"""
        difx = self.x[:, :, None] - self.x[:, None, :]
        dify = self.y[:, :, None] - self.y[:, None, :]
        return difx, dify

    def ground(pos, speed):
        for i in range(pos.getnoNodes()):
            if pos.matrix[:, 1, i] < 0:  # Y component goes below the ground plane
                pos.matrix[:, 1, i] = 0
                speed.matrix[:, :, i] = 0


def plotTrajectory(target, traj, path, color="red", noFrame=False):
    plt.clf()
    plt.close()
    plt.figure(figsize=(5, 5))
    plt.tight_layout()
    if target is not None:
        plt.plot(
            target[0], target[1], color="black", linestyle="dashed", label="target"
        )
    plt.plot(traj[0], traj[1], color=color, label="trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    if noFrame:
        plt.gca().axis("off")
    else:
        plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plotTrajectories(target, trajs, colors, labels, path):
    plt.clf()
    plt.close()
    plt.figure(figsize=(5, 5))
    plt.tight_layout()
    plt.plot(target[0], target[1], color="black", label="target")
    for traj, color, label in zip(trajs, colors, labels):
        plt.plot(traj[0], traj[1], color=color, label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plotTargetImage(targetImg, path):
    """Save an MNIST image"""
    plt.clf()
    plt.close()
    plt.imshow(targetImg, cmap="gray")
    plt.gca().axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plotPca(data, fileSaveDir):
    """Perform PCA and save the results"""
    onlyPosData, plotData = [], []
    for datum in data:
        if len(datum) == 4:
            onlyPosData.append([datum[0].squeeze(), datum[1].squeeze()])
            plotData.append([datum[2], datum[3]])
        elif len(datum) == 3:
            onlyPosData.append([datum[0][0], datum[0][1]])
            plotData.append([datum[1], datum[2]])
    onlyPosData = np.stack(onlyPosData, 0)
    onlyPosData = onlyPosData.reshape(onlyPosData.shape[0], -1)
    pca = PCA(n_components=2)
    pca.fit(onlyPosData)
    pcaData = pca.transform(onlyPosData)
    labels = []
    for pcaDatum, plotDatum in zip(pcaData, plotData):
        plt.scatter(
            pcaDatum[0],
            pcaDatum[1],
            label=plotDatum[0] if not plotDatum[0] in labels else None,
            color=plotDatum[1],
            alpha=0.5,
        )
        if not plotDatum[0] in labels:
            labels.append(plotDatum[0])
    plt.grid(True)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(fileSaveDir + ".png", bbox_inches="tight")
    plt.clf()
    plt.close()
    return pcaData, pca.explained_variance_ratio_


def plotLabelMap(labelArray, xaxis, yaxis, fileSaveDir):
    cmap = plt.get_cmap("tab10")
    fileName = fileSaveDir + ".png"
    plt.xticks(
        [i for i in range(0, len(xaxis), int(len(xaxis) / 5))],
        xaxis[:: int(len(xaxis) / 5)],
    )
    plt.yticks(
        [i for i in range(0, len(yaxis), int(len(yaxis) / 5))],
        yaxis[:: int(len(yaxis) / 5)],
    )
    plt.xlabel("delta x")
    plt.ylabel("delta y")
    plt.imshow(labelArray, cmap=cmap, vmin=-0.5, vmax=9.5, origin="lower")
    plt.colorbar(ticks=mticker.MultipleLocator(base=1))
    plt.savefig(fileName, bbox_inches="tight")
    plt.clf()
    plt.close()


def compareMNISTtrajectory(trajectories, targets, lenList, criterion):
    """Calculate which character the trajectory is most similar to"""
    allErrors, minErrors, trajLabels = [], [], []
    for trajectory in trajectories:
        minError, trajLabel = 1, 0
        errors = []
        for label in range(10):
            target = targets[label][:, : lenList[label]]
            traj = trajectory[:, : lenList[label]]
            error = criterion(traj, target).item()
            if error < minError:
                minError = error
                trajLabel = label
            errors.append(error)
        allErrors.append(errors)
        minErrors.append(minError)
        trajLabels.append(trajLabel)
    return allErrors, minErrors, trajLabels


def character_trajectory(maxSize):
    """Load the predefined target character trajectories"""
    targetList = []
    simStepList = []
    filePath = "datasets/MNIST/target_trajectory/csv/"
    for i in range(10):
        df = pd.read_csv(filePath + f"{i}.csv", index_col=0)
        simStepList.append(len(df))
        X0, Xmin, Xmax, Y0, Ymin, Ymax = (
            df.x[0],
            min(df.x),
            max(df.x),
            df.y[0],
            min(df.y),
            max(df.y),
        )
        scalingParam = max(Xmax - X0, X0 - Xmin, Ymax - Y0, Y0 - Ymin)
        x_trajectory = [j * maxSize / scalingParam for j in list(df.x - X0)]
        y_trajectory = [j * maxSize / scalingParam for j in list(df.y - Y0)]
        targetList.append(torch.tensor([x_trajectory, y_trajectory]))
    return targetList, simStepList


def MNIST_train_dataset(char, size=-1):
    """Load MNIST training dataset"""
    train_samples = []
    train_dataset = datasets.MNIST(
        "datasets", train=True, transform=transforms.ToTensor(), download=True
    )
    if (char == -1) and (size != -1):
        appended_nums = [0] * 10
        num_of_each_char = int(size / 10)
    appended_nums = [0] * 10
    for data in train_dataset:
        if char == -1:
            if size == -1:
                train_samples.append(data)
            else:
                if appended_nums[data[1]] < num_of_each_char:
                    train_samples.append(data)
                    appended_nums[data[1]] += 1
        elif char == 10:
            if (data[1] == 0) or (data[1] == 1):
                train_samples.append(data)
        elif data[1] == char:
            train_samples.append(data)
        if (size != -1) and (len(train_samples) == size):
            break
    del train_dataset
    random.shuffle(train_samples)
    return train_samples


def MNIST_test_samples(num=-1):
    """Load MNIST testing dataset"""
    test_samples = []
    test_dataset = datasets.MNIST(
        "datasets", train=False, transform=transforms.ToTensor(), download=True
    )
    if num == -1:
        for data in test_dataset:
            test_samples.append(data)
    else:
        for h in range(10):
            for _ in range(num):
                while True:
                    sample = test_dataset[random.randint(0, len(test_dataset) - 1)]
                    if sample[1] == h:
                        test_samples.append(sample)
                        break
    del test_dataset
    return test_samples


def LissajousTimeSeries(
    simTime, simTimeStep, networkSize, offset=0, A=0, B=0, a=1, b=2, delta=0
):
    """Generate timeseries for Lissajous curve"""
    if (A == 0) and (B == 0):
        A = networkSize / 20
        B = networkSize / 20
    t = torch.tensor(
        [i * simTimeStep * 2 * np.pi for i in range(0, int(simTime / simTimeStep))]
    )
    x = A * torch.cos(a * t) + offset
    y = B * torch.sin(b * t + delta)
    xy = torch.cat((x.unsqueeze(0), y.unsqueeze(0)), dim=0).T.unsqueeze(2)
    return xy


def npy2npz(npyList, dir, npzName, transpose=False):
    """Save npz file based on npy files"""
    arrayList = []
    for npyFile in npyList:
        if transpose:
            arrayList.append(np.load(npyFile).T)
        else:
            arrayList.append(np.load(npyFile))
    concatenatedData = np.concatenate(arrayList, 0)
    np.savez_compressed(dir + npzName + ".npz", data=concatenatedData)
    del concatenatedData, arrayList
    for npyFile in npyList:
        os.remove(npyFile)


def img2deltaPos(imgs, pixelNum, agent, brain, deltaPosMask):
    """Calculate the displacement of mass point based on brain function"""
    deltaPoss = []
    for img in imgs:
        img = (
            transforms.functional.resize(img[0], (pixelNum, pixelNum))
            .reshape(pixelNum, pixelNum)
            .to(torch.float64)
            .to(agent.device)
        )
        if brain == "MLP":
            imgInput = torch.flatten(img).unsqueeze(0)
            deltaPos = agent.brain(imgInput).reshape(2, -1)
        elif brain == "CNN":
            imgInput = img.unsqueeze(0)
            deltaPos = agent.brain(imgInput).reshape(2, -1)
        else:
            imgInput = torch.flatten(img).unsqueeze(0)
            deltaPos = torch.mm(imgInput, agent.inputLayer).reshape(2, -1)
        deltaPoss.append(deltaPos)
    deltaPos = torch.stack(deltaPoss, 0).to(agent.device)
    deltaPos = torch.mul(deltaPosMask.to(agent.device), deltaPos)
    return deltaPos


def generateDeltaPos(noNodes, nodeNum, xs, ys, device, deltaPosMask):
    """Calculate the displacement of mass point based on designated values"""
    deltaPoss = []
    for deltaX in xs:
        for deltaY in ys:
            deltaPosPiece = torch.zeros((2, noNodes))
            deltaPosPiece[0, nodeNum] = deltaX
            deltaPosPiece[1, nodeNum] = deltaY
            deltaPoss.append(deltaPosPiece)
    deltaPos = torch.stack(deltaPoss, 0).to(device)
    deltaPos = torch.mul(deltaPosMask.to(device), deltaPos)
    return deltaPos


def saveCSV(data, path, iterable=True):
    with open(path, "w") as f:
        writer = csv.writer(f)
        if iterable:
            writer.writerows(data)
        else:
            writer.writerow(data)


def saveNPY(data, path):
    np.save(path, data)


def save_params_as_json(data, path):
    with open(path + "params.json", mode="w") as f:
        json.dump(data, f, indent=2)


def concatenate_files(suffix):
    stacked_data = None
    fileList = sorted(glob.glob(suffix + "*.npy"))
    for file in fileList:
        data = np.load(file)
        if stacked_data is None:
            stacked_data = data
        else:
            stacked_data = np.hstack([stacked_data, data])
    return stacked_data


def plotBifurcationDiagram(data, forces, ylabel, path):
    """Save bifurcation diagram"""
    plt.rcParams["font.size"] = 20
    plt.clf()
    plt.close()
    for force, nodeData in zip(forces, data):
        dataNum = nodeData.shape[0]
        x = [force for _ in range(dataNum)]
        plt.scatter(x, nodeData, c="black", s=1)
    plt.xlabel("gain of external force")
    plt.ylabel(ylabel)
    plt.savefig(path + ".png", bbox_inches="tight")
    plt.close()


def plotBifurcationDiagram2ax(data, forces, path, vlines):
    """Save bifurcation diagram with 2 axes"""
    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    for force, nodeData in zip(forces, data[0]):
        dataNum = nodeData.shape[0]
        x = [force for _ in range(dataNum)]
        ax1.scatter(x, nodeData, c="black", s=1)
    ax1.set_ylabel("x")
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    for force, nodeData in zip(forces, data[1]):
        dataNum = nodeData.shape[0]
        x = [force for _ in range(dataNum)]
        ax2.scatter(x, nodeData, c="black", s=1)
    ax2.set_xlabel("wind")
    ax2.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(path + ".png", bbox_inches="tight")
    plt.clf()
    plt.close()
    if vlines is not None:
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        for force, nodeData in zip(forces, data[0]):
            dataNum = nodeData.shape[0]
            x = [force for _ in range(dataNum)]
            ax1.scatter(x, nodeData, c="black", s=1)
        for x in vlines:
            ax1.axvline(x=x, linewidth=1, color="tab:purple")
        ax1.axvline(x=0, linewidth=1, color="tab:blue")
        ax1.axvline(x=1, linewidth=1, color="tab:orange")
        ax1.set_ylabel("x")
        ax1.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        for force, nodeData in zip(forces, data[1]):
            dataNum = nodeData.shape[0]
            x = [force for _ in range(dataNum)]
            ax2.scatter(x, nodeData, c="black", s=1)
        for x in vlines:
            ax2.axvline(x=x, linewidth=1, color="tab:purple")
        ax2.axvline(x=0, linewidth=1, color="tab:blue")
        ax2.axvline(x=1, linewidth=1, color="tab:orange")
        ax2.set_xlabel("wind")
        ax2.set_ylabel("y")
        plt.tight_layout()
        plt.savefig(path + "_vlines.png", bbox_inches="tight")
        plt.clf()
        plt.close()


def plotSpeed(speed, wind, target, path):
    """Plot locomotion speed according to wind"""
    plt.rcParams["font.size"] = 20
    plt.figure(figsize=(6, 5))
    mean = speed.mean(axis=1)
    std = speed.std(axis=1)
    plt.plot(wind, target, c="black", lw=2, label="target speed")
    plt.plot(wind, mean, c="r", lw=2, label="actual speed")
    plt.fill_between(wind, mean - std, mean + std, label=None, color="r", alpha=0.3)
    plt.grid(axis="y")
    plt.legend(loc="best")
    plt.xlabel("wind")
    plt.ylabel("speed")
    plt.tight_layout()
    plt.savefig(path + ".png", bbox_inches="tight")
    plt.clf()
    plt.close()


def plotLines(
    data,
    xlabel,
    ylabel,
    title,
    file_name,
    marker=True,
    vline=None,
    noFrame=False,
):
    plt.rcParams["font.size"] = 15
    if noFrame:
        plt.figure(figsize=(10, 5))
    else:
        plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    for datum in data:
        if marker:
            plt.plot(datum[0], datum[1], label=datum[2], marker="o", color=datum[3])
        else:
            plt.plot(datum[0], datum[1], label=datum[2], color=datum[3])
    if vline is not None:
        if type(vline) == list:
            for x in vline:
                plt.axvline(x=x, linewidth=4)
        else:
            plt.axvline(x=vline, linewidth=4)
    plt.tight_layout()
    if noFrame:
        plt.gca().axis("off")
    else:
        plt.legend(loc="best")
    plt.savefig(file_name, bbox_inches="tight")
    plt.clf()
    plt.close()


def plotTrajectory_timeseries(target, traj, simtime, path):
    plt.clf()
    plt.close()
    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect("equal", adjustable="datalim")
    t = np.linspace(0, simtime, traj.shape[1])
    plt.plot(target[0], target[1], color="black", label="target")
    plt.scatter(traj[0], traj[1], c=t, cmap=cm.rainbow, lw=0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    colorbar = plt.colorbar()
    colorbar.set_label("time [sec]")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plotLines2ax(xs, ys, ylabels, path, boundary=None):
    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    plt.xlabel("time [sec]")
    cmap = plt.get_cmap("tab10")
    colors = [cmap(0), cmap(2)]
    for datum in xs:
        ax1.plot(datum[0], datum[1], label=datum[2], color=datum[3])
    ax1.set_ylabel(ylabels[0])
    ax1.grid()
    if boundary is not None:
        for i, end in enumerate(boundary):
            if i == 0:
                start = end
                continue
            ax1.axvspan(start, end, color=colors[i % 2], alpha=0.3)
            start = end
    for datum in ys:
        ax2.plot(datum[0], datum[1], label=datum[2], color=datum[3])
    ax2.set_ylabel(ylabels[1])
    ax2.grid()
    if boundary is not None:
        for i, end in enumerate(boundary):
            if i == 0:
                start = end
                continue
            ax2.axvspan(start, end, color=colors[i % 2], alpha=0.3)
            start = end
    plt.tight_layout()
    plt.savefig(path + ".png", bbox_inches="tight")
    plt.clf()
    plt.close()


def generate_expandedNARMA_target(length, order, rseed=0):
    """Generate the target timeseries of expanded NARMA"""
    ys = [0] * order
    if rseed is not None:
        np.random.seed(seed=rseed)
    _us = np.random.uniform(-1, 1, length)
    us = 0.5 * (_us + 1)
    k = order
    while k < length:
        y_k = ys[k - 1] * 0.3 + 0.1
        for i in range(1, order + 1):
            y_k += ((us[k - i] ** 3) - (us[k - i] ** 4)) * 1 + (
                (ys[k - i] ** 3) - (ys[k - i] ** 4)
            ) * 0.2
        ys.append(y_k)
        k += 1
    ys = np.array(ys).astype(np.float32)
    return _us, ys


def generate_delay(length, order, rseed=0, cumulative=False):
    """Generate the target timeseries of memory task"""
    ys = [0] * order
    if rseed is not None:
        np.random.seed(seed=rseed)
    _us = np.random.uniform(-1, 1, length)
    us = 0.5 * (_us + 1)
    k = order
    while k < length:
        if cumulative:
            y_k = 0
            for _order in range(1, order + 1):
                y_k += us[k - _order]
            y_k /= order
        else:
            y_k = us[k - order]
        ys.append(y_k)
        k += 1
    ys = np.array(ys).astype(np.float32)
    return _us, ys


def MSE(pred, target):
    return ((pred - target) ** 2).mean()


def print_current_params(agent, iteration, loss, brain_type, params_history):
    """Print current tunable parameters"""
    if brain_type == "LIL":
        print(
            "{}: loss = {:.06f}, Mean(spring) = {:.06f}, "
            "Mean(damping) = {:.06f}, Mean(restLength) = {:.06f}, "
            "Mean(inputLayer) = {:06f}, Mean(readoutLayer) = {:06f}".format(
                iteration,
                loss,
                torch.sum(agent.morph.spring).item()
                / torch.sum(agent.morph.spring != 0).item(),
                torch.sum(agent.morph.damping).item()
                / torch.sum(agent.morph.damping != 0).item(),
                torch.sum(agent.morph.restLength).item()
                / torch.sum(agent.morph.restLength != 0).item(),
                torch.mean(agent.inputLayer).item(),
                torch.mean(agent.readoutLayer).item(),
            )
        )
        if type(params_history) == list:
            params_history.append(
                [
                    torch.sum(agent.morph.spring).item()
                    / torch.sum(agent.morph.spring != 0).item(),
                    torch.sum(agent.morph.damping).item()
                    / torch.sum(agent.morph.damping != 0).item(),
                    torch.sum(agent.morph.restLength).item()
                    / torch.sum(agent.morph.restLength != 0).item(),
                    torch.mean(agent.inputLayer).item(),
                    torch.mean(agent.readoutLayer).item(),
                ]
            )
    elif (brain_type == "MLP") or (brain_type == "CNN"):
        print(
            "{}: loss = {:.06f}, Mean(spring) = {:.06f}, "
            "Mean(damping) = {:.06f}, Mean(restLength) = {:.06f}, "
            "Mean(inputLayer) = {:06f}, Mean(readoutLayer) = {:06f}, "
            "Mean(brain.fc1) = {:06f}, Mean(brain.fc2) = {:06f}, "
            "Mean(brain.fc3) = {:06f}".format(
                iteration,
                loss,
                torch.sum(agent.morph.spring).item()
                / torch.sum(agent.morph.spring != 0).item(),
                torch.sum(agent.morph.damping).item()
                / torch.sum(agent.morph.damping != 0).item(),
                torch.sum(agent.morph.restLength).item()
                / torch.sum(agent.morph.restLength != 0).item(),
                torch.mean(agent.inputLayer).item(),
                torch.mean(agent.readoutLayer).item(),
                torch.mean(agent.brain.fc1.weight).item(),
                torch.mean(agent.brain.fc2.weight).item(),
                torch.mean(agent.brain.fc3.weight).item(),
            )
        )
        if type(params_history) == list:
            params_history.append(
                [
                    torch.sum(agent.morph.spring).item()
                    / torch.sum(agent.morph.spring != 0).item(),
                    torch.sum(agent.morph.damping).item()
                    / torch.sum(agent.morph.damping != 0).item(),
                    torch.sum(agent.morph.restLength).item()
                    / torch.sum(agent.morph.restLength != 0).item(),
                    torch.mean(agent.inputLayer).item(),
                    torch.mean(agent.readoutLayer).item(),
                    torch.mean(agent.brain.fc1.weight).item(),
                    torch.mean(agent.brain.fc2.weight).item(),
                    torch.mean(agent.brain.fc3.weight).item(),
                ]
            )
    elif (brain_type == "SWG") or (brain_type == "FL"):
        print(
            "{}: loss = {:.06f}, Mean(spring) = {:.06f}, "
            "Mean(damping) = {:.06f}, Mean(restLength) = {:.06f}, "
            "Mean(inputLayer) = {:06f}, Mean(amplitude) = {:.06f}, "
            "Mean(omega) = {:06f}, Mean(phase) = {:06f}".format(
                iteration,
                loss,
                torch.sum(agent.morph.spring).item()
                / torch.sum(agent.morph.spring != 0).item(),
                torch.sum(agent.morph.damping).item()
                / torch.sum(agent.morph.damping != 0).item(),
                torch.sum(agent.morph.restLength).item()
                / torch.sum(agent.morph.restLength != 0).item(),
                torch.mean(agent.inputLayer).item(),
                torch.sum(agent.control.amplitude).item()
                / torch.sum(agent.control.amplitude != 0).item(),
                torch.sum(agent.control.omega).item()
                / torch.sum(agent.control.omega != 0).item(),
                torch.sum(agent.control.phase).item()
                / torch.sum(agent.control.phase != 0).item(),
            )
        )
        if type(params_history) == list:
            params_history.append(
                [
                    torch.sum(agent.morph.spring).item()
                    / torch.sum(agent.morph.spring != 0).item(),
                    torch.sum(agent.morph.damping).item()
                    / torch.sum(agent.morph.damping != 0).item(),
                    torch.sum(agent.morph.restLength).item()
                    / torch.sum(agent.morph.restLength != 0).item(),
                    torch.mean(agent.inputLayer).item(),
                    torch.sum(agent.control.amplitude).item()
                    / torch.sum(agent.control.amplitude != 0).item(),
                    torch.sum(agent.control.omega).item()
                    / torch.sum(agent.control.omega != 0).item(),
                    torch.sum(agent.control.phase).item()
                    / torch.sum(agent.control.phase != 0).item(),
                ]
            )
    return params_history


class Readout:
    """Class for readout function"""

    def __init__(self, dim_x, dim_out, rseed=0):
        torch.manual_seed(seed=rseed)
        self.w_out = torch.randn(dim_out, dim_x)

    def __call__(self, x):
        return self.w_out @ x

    def set_weight(self, w_out_opt):
        self.w_out = w_out_opt


class Ridge_opt:
    """Class for ridge regression"""

    def __init__(self, beta, device):
        self.X = []
        self.Y = []
        self.test_X = []
        self.beta = beta
        self.device = device
        self.df = 0

    def add_data(self, vec_x, vec_y, test=False):
        vec_x = vec_x.view(-1, 1)
        vec_y = vec_y.view(-1, 1)
        if not test:
            self.X.append(vec_x)
            self.Y.append(vec_y)
        else:
            self.test_X.append(vec_x)

    def solve(self):
        self.X = torch.cat(
            (
                torch.ones((1, self.X.shape[1]), dtype=torch.float64).to(self.device),
                self.X,
            ),
            dim=0,
        )
        df = self.set_ridge_parameter()
        self.df = df
        X_XT = self.X @ self.X.T
        Y_XT = self.Y @ self.X.T
        X_pseudo_inv = torch.linalg.pinv(
            X_XT + self.beta * torch.eye(X_XT.shape[0]).to(self.device)
        )
        w_out_opt = Y_XT @ X_pseudo_inv
        return w_out_opt, np.array([df, self.beta])

    def set_ridge_parameter(self):
        X_XT = self.X @ self.X.T
        eigs = torch.linalg.eigvals(X_XT)
        betas = []
        for df in range(1, self.X.shape[0] + 1):
            best_beta = 0
            min_diff = 10000
            for order in range(15):
                beta_order = 1 / (10**order)
                for b in range(1, 10):
                    beta = b * beta_order
                    form = -df
                    for eig in eigs:
                        form += eig / (eig + beta)
                    if (form.item().imag == 0.0) and (abs(form.item().real) < min_diff):
                        best_beta = beta
                        min_diff = abs(form.item().real)
            betas.append(best_beta)
        AICs = self.AIC(betas)
        if not np.all(np.isnan(AICs)):
            min_index = np.nanargmin(AICs)
            self.beta = betas[min_index]
        else:
            min_index = -1
        return min_index + 1

    def AIC(self, betas):
        M = self.X.shape[1]
        X_XT = self.X @ self.X.T
        Y_XT = self.Y @ self.X.T
        AICs = []
        for i, beta in enumerate(betas):
            df = i + 1
            X_pseudo_inv = torch.linalg.pinv(
                X_XT + beta * torch.eye(X_XT.shape[0]).to(self.device)
            )
            w_out_opt = Y_XT @ X_pseudo_inv
            y_hat = w_out_opt @ self.X
            RSS = (y_hat - self.Y).squeeze()
            AICs.append(((M * torch.log(torch.sum(RSS))) + (2 * df)).item())
        return np.array(AICs)
