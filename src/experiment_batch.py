import os
import math
import datetime
import copy
import random
import subprocess
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Process, cpu_count
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib as mpl

from agent_batch import *
from simulate import *
from utils import *

mpl.rcParams["agg.path.chunksize"] = 100000


class Experiment(object):
    """Class to perform standard experiments"""

    def __init__(
        self,
        simTime_=30,
        simTimeStep_=0.01,
        tau_=10,
        shape_="DoubleCircles",
        initSize_=4,
        noNodes_=33,
        mass_=1,
        noNeighbours_=3,
        seed_=None,
        device_=torch.device("cpu"),
        loadFolder_=None,
        loadFileSuffix_=None,
        params_={},
        memo_=None,
    ):
        """Initialize Experiment object

        Args:
            simTime_ (int, optional): Simulation time. Defaults to 30.
            simTimeStep_ (float, optional): Time step. Defaults to 0.01.
            tau_ (int, optional): Time scaling variable. Defaults to 10.
            shape_ (str, optional): Shape of MSDN. Defaults to "DoubleCircles".
            initSize_ (int, optional): Size of MSDN. Defaults to 4.
            noNodes_ (int, optional): Number of mass points. Defaults to 33.
            mass_ (int, optional): Mass. Defaults to 1.
            noNeighbours_ (int, optional): Number of neighbours to which a spring connection is made. Defaults to 3.
            seed_ (_type_, optional): Random seed. Defaults to None.
            device_ (_type_, optional): Device. Defaults to torch.device("cpu").
            loadFolder_ (_type_, optional): Path to an existing result folder. Defaults to None.
            loadFileSuffix_ (_type_, optional): Suffix of the file to load. Defaults to None.
            params_ (dict, optional): Designated parameters. Defaults to {}.
            memo_ (_type_, optional): This will be the suffix for the result folder. Defaults to None.
        """

        self.simTime = simTime_
        self.simTimeStep = simTimeStep_
        self.tau = tau_
        self.shape = shape_
        self.initSize = initSize_
        self.noNodes = noNodes_
        self.mass = mass_
        self.noNeighbours = noNeighbours_
        self.seed = seed_
        self.device = device_
        self.loadFolder = loadFolder_
        self.loadFileSuffix = loadFileSuffix_
        self.params = params_
        self.memo = memo_

        self.timeStamp = str(datetime.date.today())
        self.figDirPre = "results/"
        self.figDirSuf = "/"
        self.scaleSimtimeTau = self.simTime / self.tau
        self.savePosType = "tensor"

        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

    def initSet(
        self,
        initSize,
        tau,
        deltaPos=None,
        readoutDim=1,
        centerFlag=False,
        _simtime=None,
        pixXpix=25,
        brain="LIL",
        seed=None,
        controlType="no",
        generatorNum=[-1],
        amplitude=0.8,
        ground=False,
        wind=None,
        closedloop=False,
        fixedExternalForce=None,
        noise=0,
    ):
        """Initial experimental settings"""
        env = HardEnvironment(ground=ground, wind=wind, device=self.device)
        if _simtime is None:
            simTime = int(self.scaleSimtimeTau * tau)
        else:
            simTime = _simtime
        if seed is None:
            seed = self.seed
        if type(simTime) == torch.Tensor:
            simTime = simTime.detach().numpy()
        if type(tau) == torch.Tensor:
            tau = tau.item()
        morph = SpringMorphology(
            mass=self.mass,
            noNodes=self.noNodes,
            noNeighbours=self.noNeighbours,
            environment=env,
            shape=self.shape,
            initSize=initSize,
            deltaPos=deltaPos,
            seed=seed,
            batchSize=self.batchSize,
            device=self.device,
        )
        if controlType == "no":
            control = NoControl(
                morph,
                simTime,
                self.simTimeStep,
                tau,
                batchSize=self.batchSize,
                device=self.device,
            )
        elif controlType == "sine":
            control = SineControl(
                morph,
                simTime,
                self.simTimeStep,
                tau,
                device=self.device,
                amplitude=amplitude,
                generatorNum=generatorNum,
            )
        elif controlType == "free":
            control = FreeControl(
                morph,
                simTime,
                self.simTimeStep,
                tau,
                device=self.device,
                amplitude=amplitude,
                generatorNum=generatorNum,
            )
        agent = Agent(
            morph,
            control,
            state=None,
            deltaPos=deltaPos,
            seed=seed,
            readoutDim=readoutDim,
            centerFlag=centerFlag,
            centralPointOutput=self.centralPointOutput,
            savePosType=self.savePosType,
            pixXpix=pixXpix,
            brain=brain,
            closedloop=closedloop,
            fixedExternalForce=fixedExternalForce,
            noise=noise,
            device=self.device,
        )
        plotter = Plotter(plot=False, plotCycle=10, pauseTime=0.1)
        simulenv = SimulationEnvironment(
            timeStep=self.simTimeStep,
            simulationLength=int(simTime / self.simTimeStep),
            plot=plotter,
        )
        return agent, simulenv

    def loadObjects(self, path=None):
        """Load objects (agent and simulenv)"""
        assert self.loadFolder is not None
        assert self.loadFileSuffix is not None
        if path is None:
            with open(
                self.loadFolder + f"agent_{self.loadFileSuffix}.pickle", mode="rb"
            ) as f:
                agent = pickle.load(f)
            with open(
                self.loadFolder + f"simulenv_{self.loadFileSuffix}.pickle", mode="rb"
            ) as f:
                simulenv = pickle.load(f)
        else:
            with open(path + f"agent_{self.loadFileSuffix}.pickle", mode="rb") as f:
                agent = pickle.load(f)
            with open(path + f"simulenv_{self.loadFileSuffix}.pickle", mode="rb") as f:
                simulenv = pickle.load(f)
        self.initSize = agent.morph.initSize
        self.tau = agent.control.tau
        if hasattr(agent.control, "inputScale"):
            self.inputScale = agent.control.inputScale
        agent.resetAgent(morph=agent.morph)
        if simulenv.plot.colored_comp == []:
            simulenv.plot.colored_comp = [[], []]
        return agent, simulenv

    def saveObjects(self, agent, simulenv, fileSaveDir, tag):
        """Save objects (agent and simulenv)"""
        with open(fileSaveDir + f"agent_{tag}.pickle", mode="wb") as f:
            pickle.dump(agent, f)
        with open(fileSaveDir + f"simulenv_{tag}.pickle", mode="wb") as f:
            pickle.dump(simulenv, f)

    def make_fileSaveDir(self, name, suffix):
        """Make folder to save results"""
        fileSaveDir = name
        if suffix is not None:
            if self.memo is None:
                self.memo = suffix
            else:
                self.memo = suffix + "_" + self.memo
        if self.memo is None:
            fileSaveDir += "/"
        else:
            fileSaveDir += "_" + self.memo + "/"
        if not os.path.exists(fileSaveDir):
            os.makedirs(fileSaveDir)
        return fileSaveDir

    def shape_tunableParameters(
        self,
        agent,
        readoutLayer=False,
        restLength=False,
        sineControl=False,
        shape=True,
        inputLayerClamp=False,
    ):
        """Shape tunable parameters"""
        shaped_parameters = []
        if shape:
            _spring = agent.morph.SymmetricMatrix(agent.morph.spring)
            _spring = torch.where(_spring < -0.0001, 0.1, _spring).detach().clone()
            _damping = agent.morph.SymmetricMatrix(agent.morph.damping)
            _damping = torch.where(_damping < -0.0001, 0.1, _damping).detach().clone()
            shaped_parameters.extend((_spring, _damping))
        else:
            _spring = agent.morph.spring.detach().clone()
            _damping = agent.morph.damping.detach().clone()
            shaped_parameters.extend((_spring, _damping))
        if inputLayerClamp:
            if self.shape == "MultiCircles":
                _inputLayer = (
                    torch.clamp(input=agent.inputLayer, min=-0.01, max=0.01)
                    .detach()
                    .clone()
                )
            elif self.shape == "DoubleCircles":
                _inputLayer = (
                    torch.clamp(input=agent.inputLayer, min=-0.1, max=0.1)
                    .detach()
                    .clone()
                )
        else:
            _inputLayer = agent.inputLayer.detach().clone()
        shaped_parameters.append(_inputLayer)
        if readoutLayer:
            _readoutLayer = agent.readoutLayer.detach().clone()
            shaped_parameters.append(_readoutLayer)
        if restLength:
            _restLength = agent.morph.SymmetricMatrix(agent.morph.restLength)
            _restLength = (
                torch.where(_restLength < -0.0001, 0.1, _restLength).detach().clone()
            )
            shaped_parameters.append(_restLength)
        if sineControl:
            _amplitude = agent.morph.SymmetricMatrix(agent.control.amplitude)
            _amplitude = (
                torch.where(_amplitude < -0.0001, 0.1, _amplitude).detach().clone()
            )
            shaped_parameters.append(_amplitude)
            _omega = agent.morph.SymmetricMatrix(agent.control.omega)
            _omega = torch.where(_omega < -0.0001, 0.1, _omega).detach().clone()
            shaped_parameters.append(_omega)
            _phase = agent.morph.SymmetricMatrix(agent.control.phase)
            shaped_parameters.append(_phase)
        return shaped_parameters

    def calc_states(self, agent):
        """Calculate states to read out in MNIST task"""
        trainStates = None
        if self.centralPointOutput:
            raise NotImplementedError
        else:
            nodesHis = torch.permute(
                torch.stack(agent.state.movableNodes_history, dim=0), (1, 2, 3, 0)
            ).reshape([self.batchSize, 2, -1])
        xpos = nodesHis[:, 0]
        ypos = nodesHis[:, 1]
        trainStates = torch.cat(
            (
                torch.ones((self.batchSize, 1)).to(self.device),
                nodesHis.reshape([self.batchSize, -1]),
            ),
            dim=1,
        ).to(torch.float64)
        return xpos, ypos, trainStates

    def setControl_forMNIST(
        self, physicalOutput, simStepList, data, agent, pixelNum, simulenv
    ):
        """Set simulation settings in MNIST task"""
        if physicalOutput:
            self.simTime = simStepList[data[0][1]] * self.simTimeStep
        agent.control.setSimTime(self.simTime, self.simTimeStep, self.tau)
        agent.control.generateMNISTtimeseries(
            inputScale=0,
            pixelNum=pixelNum,
        )
        simulenv.setSimulationLength(agent.control.simulLen)
        return agent, simulenv

    def set_requiresGrads(self, agent, requires_grad_flag, generator=False):
        """Define whether to retain gradients or not"""
        if isinstance(agent.morph, SpringMorphology):
            agent.morph.spring.requires_grad = requires_grad_flag
            agent.morph.damping.requires_grad = requires_grad_flag
        agent.morph.restLength.requires_grad = requires_grad_flag
        agent.inputLayer.requires_grad = requires_grad_flag
        agent.readoutLayer.requires_grad = requires_grad_flag
        if agent.brain is not None:
            for brain_layer in agent.brain.parameters():
                brain_layer.requires_grad = requires_grad_flag
        if generator:
            agent.control.amplitude.requires_grad = requires_grad_flag
            agent.control.omega.requires_grad = requires_grad_flag
            agent.control.phase.requires_grad = requires_grad_flag
        return agent

    def MNIST_training(
        self,
        physicalOutput,
        brain,
        pixelNum,
        char,
        optim,
        lr_spring,
        lr_damping,
        lr_restlength,
        lr_readoutlayer,
        lr_brain,
        dataSize,
        testDataSize,
        epoch,
        fixedImages=-1,
        batchSize=50,
    ):
        """MNIST experiment

        Args:
            physicalOutput (bool): Whether output is physicalized or not
            brain (str): Brain type
            pixelNum (int): Number of pixels on each side of the MNIST image
            char (int): Character to use for training and testing
            optim (str): Optimizer
            lr_spring (float): Learning rate of spring constant of MSDN
            lr_damping (float): Learning rate of damping coefficient of MSDN
            lr_restlength (float): Learning rate of rest length of MSDN
            lr_readoutlayer (float): Learning rate of readout layer
            lr_brain (float): Learning rate of FNN parameters
            dataSize (int): Number of data to use for training
            testDataSize (int): Number of data to use for testing
            epoch (int): Training epoch
            fixedImages (int, optional): Whether to train and test only on specific MNIST images. Defaults to -1.
            batchSize (int, optional): Batch size. Defaults to 50.
        """
        if fixedImages != -1:
            assert char == -1
            assert fixedImages % 10 == 0
        self.centralPointOutput = physicalOutput
        self.batchSize = batchSize
        deltaPosMask = torch.ones((2, self.noNodes))
        if self.shape == "MultiCircles":
            assert math.sqrt(self.noNodes - 1).is_integer()
            sqrtNodes = int(math.sqrt(self.noNodes - 1))
            deltaPosMask[:, : 4 * sqrtNodes - 4] = torch.zeros((2, 4 * sqrtNodes - 4))
            deltaPosMask[:, -1] = torch.zeros(2)
        elif self.shape == "DoubleCircles":
            deltaPosMask[:, : int((self.noNodes - 1) / 2)] = torch.zeros(
                (2, int((self.noNodes - 1) / 2))
            )
            deltaPosMask[:, -1] = torch.zeros(2)
        deltaPosMasks = [deltaPosMask for _ in range(self.batchSize)]
        deltaPosMask = torch.stack(deltaPosMasks, 0)
        self.figDirSuf += f"{brain}_{self.shape}/"
        if self.loadFolder is not None:
            fileSaveDir = self.make_fileSaveDir(
                self.loadFolder + "/additionalTraining/" + self.timeStamp,
                self.loadFileSuffix,
            )
        elif physicalOutput:
            fileSaveDir = self.make_fileSaveDir(
                self.figDirPre
                + "MNIST/physicalOutput/"
                + self.figDirSuf
                + self.timeStamp
                + f"_shape{self.shape}_nodes{self.noNodes}_batch{self.batchSize}"
                f"_tau{self.tau}_fixedImages{fixedImages}_seed{self.seed}_char{char}",
                None,
            )
        else:
            fileSaveDir = self.make_fileSaveDir(
                self.figDirPre
                + "MNIST/readout/"
                + self.figDirSuf
                + self.timeStamp
                + f"_shape{self.shape}_nodes{self.noNodes}_batch{self.batchSize}"
                f"_simtime{self.simTime}_tau{self.tau}_epoch{epoch}"
                f"_pixel{pixelNum}_seed{self.seed}",
                None,
            )
        save_params_as_json(self.params, fileSaveDir)
        loss_history, params_history = [], []
        requires_grad_flag = True
        timeScale = 3
        if testDataSize != -1:
            testDataSize = int(testDataSize / 10)
        test_dataset = MNIST_test_samples(testDataSize)
        if physicalOutput:
            criterion = nn.MSELoss()
            uneven_length_target, simStepList = character_trajectory(self.initSize / 8)
            uneven_length_target = list(
                map(lambda l: l[:, ::timeScale], uneven_length_target)
            )
            simStepList = list(map(lambda l: -(-l // timeScale), simStepList))
            maxSimStep = max(simStepList)
            target = [
                torch.zeros((2, maxSimStep), dtype=torch.float64)
                for _ in range(len(uneven_length_target))
            ]
            for i, targ in enumerate(uneven_length_target):
                target[i][:, : targ.shape[1]] = targ
            sampleNum = 3
            test_dataset = MNIST_test_samples(num=sampleNum)
        else:
            criterion = nn.CrossEntropyLoss()
            prob_history, simStepList = [], []
        if self.loadFolder is not None:
            agent, simulenv = self.loadObjects()
            agent.control.setSimTime(self.simTime, self.simTimeStep, self.tau)
        elif physicalOutput:
            agent, simulenv = self.initSet(
                self.initSize,
                self.tau,
                deltaPos=torch.zeros((self.batchSize, 2, self.noNodes)).to(self.device),
                readoutDim=10,
                centerFlag=True,
                pixXpix=pixelNum**2,
                brain=brain,
            )
        else:
            agent, simulenv = self.initSet(
                self.initSize,
                self.tau,
                readoutDim=10,
                centerFlag=False,
                pixXpix=pixelNum**2,
                brain=brain,
            )
        agent.setBatchSize(self.batchSize)
        _spring_prev = copy.deepcopy(agent.morph.spring.detach())
        _damping_prev = copy.deepcopy(agent.morph.damping.detach())
        _restLength_prev = copy.deepcopy(agent.morph.restLength.detach())
        if brain == "LIL":
            _inputLayer_prev = copy.deepcopy(agent.inputLayer.detach())
            params_label = [
                "spring",
                "damping",
                "restLength",
                "inputLayer(mean)",
                "inputLayer(max)",
                "readoutLayer(mean)",
                "readoutLayer(max)",
            ]
        elif (brain == "MLP") or (brain == "CNN"):
            for brain_layer in agent.brain.parameters():
                brain_layer.detach()
            _brain_prev = copy.deepcopy(agent.brain)
            params_label = [
                "spring",
                "damping",
                "restLength",
                "inputLayer(mean)",
                "inputLayer(max)",
                "readoutLayer(mean)",
                "readoutLayer(max)",
                "brain(fc1)",
                "brain(fc2)",
                "brain(fc3)",
            ]
        limit_to_reduce_lr = 5
        countdown_to_reduce_lr = [limit_to_reduce_lr, False]
        lr_changes = []

        for k in range(epoch):
            if physicalOutput:
                if fixedImages != -1:
                    if self.loadFolder is None:
                        train_dataset = MNIST_train_dataset(char, size=fixedImages)
                    else:
                        with open(
                            self.loadFolder + "trainSamples.pickle", mode="rb"
                        ) as f:
                            train_dataset = pickle.load(f)
                            random.shuffle(train_dataset)
                    test_dataset = train_dataset.copy()
                    if k == 0:
                        with open(fileSaveDir + "trainSamples.pickle", "wb") as f:
                            pickle.dump(train_dataset, f)
                else:
                    train_dataset = MNIST_train_dataset(char, size=dataSize)
            else:
                train_dataset = MNIST_train_dataset(char=-1, size=dataSize)
            num_of_batches = len(train_dataset) // self.batchSize
            print("Number of batches:", num_of_batches, ", Batch size:", self.batchSize)
            assert num_of_batches > 0
            for i in range(num_of_batches):
                if (i != 0) or (k != 0):
                    agent.morph.spring = _spring
                    agent.morph.damping = _damping
                    agent.morph.restLength = _restLength
                    if not physicalOutput:
                        agent.readoutLayer = _readoutLayer
                    if brain == "LIL":
                        agent.inputLayer = _inputLayer
                    else:
                        agent.brain = _brain
                agent.morph.spring.requires_grad = requires_grad_flag
                agent.morph.damping.requires_grad = requires_grad_flag
                agent.morph.restLength.requires_grad = requires_grad_flag
                if brain == "LIL":
                    agent.inputLayer.requires_grad = requires_grad_flag
                else:
                    for brain_layer in agent.brain.parameters():
                        brain_layer.requires_grad = requires_grad_flag

                if len(train_dataset) == 0:
                    break
                data = []
                for _ in range(self.batchSize):
                    data.append(train_dataset.pop(0))
                deltaPos = img2deltaPos(data, pixelNum, agent, brain, deltaPosMask)
                agent.setdeltaPos(deltaPos)
                agent, simulenv = self.setControl_forMNIST(
                    physicalOutput,
                    [maxSimStep for _ in range(len(simStepList))],
                    data,
                    agent,
                    pixelNum,
                    simulenv,
                )
                if not physicalOutput:
                    if (i == 0) and (k == 0):
                        if self.loadFolder is None:
                            if self.centralPointOutput:
                                agent.normalReadoutLayer(
                                    int(agent.control.simulLen // (self.tau) * 2),
                                    self.seed,
                                )
                            else:
                                agent.normalReadoutLayer(
                                    int(
                                        agent.control.simulLen
                                        // (self.tau)
                                        * 2
                                        * agent.movableNodes.shape[-1]
                                    ),
                                    self.seed,
                                )
                        _readoutLayer_prev = copy.deepcopy(agent.readoutLayer.detach())
                    agent.readoutLayer.requires_grad = requires_grad_flag

                params_lr_list = [
                    {"params": agent.morph.spring, "lr": lr_spring},
                    {"params": agent.morph.damping, "lr": lr_damping},
                    {"params": agent.morph.restLength, "lr": lr_restlength},
                ]
                if physicalOutput:
                    predicts = torch.zeros((self.batchSize, 2, maxSimStep))
                    targets = torch.zeros((self.batchSize, 2, maxSimStep))
                else:
                    params_lr_list.extend(
                        [{"params": agent.readoutLayer, "lr": lr_readoutlayer}]
                    )
                    predicts = torch.zeros((self.batchSize, 10), dtype=torch.float64)
                    targets = torch.zeros((self.batchSize), dtype=torch.long)
                if brain == "LIL":
                    params_lr_list.append({"params": agent.inputLayer, "lr": lr_brain})
                else:
                    params_lr_list.append(
                        {"params": agent.brain.parameters(), "lr": lr_brain}
                    )
                optimizer = OPTIM_DICT[optim](params_lr_list)

                simulenv.plot.setMovie(False)
                VerletSimulationBatch(simulenv, agent).runSimulation()

                if physicalOutput:
                    centerNodeHis = torch.permute(
                        torch.stack(agent.state.pos_history, dim=0), (2, 1, 0, 3)
                    )[:, :, :, -1]
                    batched_target = []
                    for jj in range(self.batchSize):
                        label = data[jj][1]
                        centerNodeHis[jj, :, simStepList[label] :] = torch.zeros(
                            (
                                1,
                                centerNodeHis.shape[1],
                                maxSimStep - simStepList[label],
                            ),
                            dtype=torch.float64,
                        )
                        batched_target.append(target[label])
                    agent.resetAgent(morph=agent.morph)
                    predicts = centerNodeHis
                    targets = torch.stack(batched_target, 0).to(self.device)
                else:
                    _, _, trainStates = self.calc_states(agent)
                    train_pred = trainStates @ agent.readoutLayer
                    agent.resetAgent(morph=agent.morph)
                    predicts = train_pred
                    targets = torch.tensor([data[jj][1] for jj in range(len(data))]).to(
                        self.device
                    )

                if (i + k * num_of_batches) % 100 == 0:
                    agent.control.target, agent.control.us = None, None
                    self.saveObjects(
                        agent, simulenv, fileSaveDir, i + k * num_of_batches
                    )

                optimizer.zero_grad()
                loss = criterion(predicts, targets)
                params_history = print_current_params(
                    agent, i + k * num_of_batches, loss.item(), brain, params_history
                )
                try:
                    loss.backward(retain_graph=True)
                except RuntimeError:
                    loss_history.append(loss_history[-1])
                    params_history[-1] = params_history[-2]
                    countdown_to_reduce_lr[0] -= 1
                    countdown_to_reduce_lr[1] = False
                    _spring, _damping, _inputLayer, _restLength = (
                        _spring_prev,
                        _damping_prev,
                        _inputLayer_prev,
                        _restLength_prev,
                    )
                    if not physicalOutput:
                        _readoutLayer = _readoutLayer_prev
                    if brain == "LIL":
                        _inputLayer = _inputLayer_prev
                    else:
                        _brain = _brain_prev
                    if countdown_to_reduce_lr[0] == 0:
                        lr_spring /= 10
                        lr_damping /= 10
                        lr_restlength /= 10
                        lr_readoutlayer /= 10
                        lr_brain /= 10
                        countdown_to_reduce_lr = [limit_to_reduce_lr, False]
                        lr_changes.append(
                            [
                                i + k * num_of_batches,
                                lr_spring,
                                lr_damping,
                                lr_restlength,
                                lr_readoutlayer,
                                lr_brain,
                            ]
                        )
                    continue
                if len(loss_history) > 0:
                    if (
                        (len(loss_history) < 5)
                        and (loss.item() > loss_history[-1] * 10)
                    ) or (
                        (len(loss_history) >= 5)
                        and (
                            loss.item()
                            > sum(loss_history[-5:]) / len(loss_history[-5:]) * 10
                        )
                    ):
                        loss_history.append(loss_history[-1])
                        params_history[-1] = params_history[-2]
                        countdown_to_reduce_lr[0] -= 1
                        countdown_to_reduce_lr[1] = False
                        _spring, _damping, _restLength = (
                            _spring_prev,
                            _damping_prev,
                            _restLength_prev,
                        )
                        if not physicalOutput:
                            _readoutLayer = _readoutLayer_prev
                        if brain == "LIL":
                            _inputLayer = _inputLayer_prev
                        else:
                            _brain = _brain_prev
                        if countdown_to_reduce_lr[0] == 0:
                            lr_spring /= 10
                            lr_damping /= 10
                            lr_restlength /= 10
                            lr_readoutlayer /= 10
                            lr_brain /= 10
                            countdown_to_reduce_lr = [limit_to_reduce_lr, False]
                            lr_changes.append(
                                [
                                    i + k * num_of_batches,
                                    lr_spring,
                                    lr_damping,
                                    lr_restlength,
                                    lr_readoutlayer,
                                    lr_brain,
                                ]
                            )
                        continue
                loss_history.append(loss.item())
                if countdown_to_reduce_lr[0] != limit_to_reduce_lr:
                    if not countdown_to_reduce_lr[1]:
                        countdown_to_reduce_lr[1] = True
                    else:
                        countdown_to_reduce_lr = [limit_to_reduce_lr, False]
                if (i != 0) or (k != 0):
                    _spring_prev = copy.deepcopy(agent.morph.spring.detach())
                    _damping_prev = copy.deepcopy(agent.morph.damping.detach())
                    _restLength_prev = copy.deepcopy(agent.morph.restLength.detach())
                    if not physicalOutput:
                        _readoutLayer_prev = copy.deepcopy(agent.readoutLayer.detach())
                    if brain == "LIL":
                        _inputLayer_prev = copy.deepcopy(agent.inputLayer.detach())
                    else:
                        for brain_layer in agent.brain.parameters():
                            brain_layer.detach()
                        _brain_prev = copy.deepcopy(agent.brain)
                if (i != num_of_batches - 1) or (k != epoch - 1):
                    optimizer.step()
                with torch.no_grad():
                    if brain != "LIL":
                        _brain = agent.brain
                    if physicalOutput:
                        [_spring, _damping, _inputLayer, _restLength] = (
                            self.shape_tunableParameters(
                                agent, None, restLength=True, inputLayerClamp=False
                            )
                        )
                    else:
                        [_spring, _damping, _inputLayer, _readoutLayer, _restLength] = (
                            self.shape_tunableParameters(
                                agent,
                                None,
                                readoutLayer=True,
                                restLength=True,
                                inputLayerClamp=False,
                            )
                        )
                agent.resetAgent()
                if (i == num_of_batches - 1) and (k == epoch - 1):
                    agent.control.target, agent.control.us = None, None
                    self.saveObjects(agent, simulenv, fileSaveDir, "final")

        with torch.no_grad():
            epoch_array = np.arange(0, epoch * num_of_batches, 1)
            plotLines(
                [[epoch_array, loss_history, None, "tab:blue"]],
                "epoch",
                "loss",
                None,
                fileSaveDir + "loss.png",
                marker=False,
            )
            testNum = len(test_dataset)
            num_of_batches_for_test = testNum // self.batchSize
            if physicalOutput:
                sampleNums = [0] * 10
                sample_pos_histories = [[] for _ in range(10)]
                self.batchSize = 1
                agent.setBatchSize(self.batchSize)
                for q in range(len(test_dataset)):
                    data = [test_dataset[q]]
                    label = data[0][1]
                    deltaPos = img2deltaPos(
                        data,
                        pixelNum,
                        agent,
                        brain,
                        deltaPosMask[0].unsqueeze(0),
                    )
                    agent.setdeltaPos(deltaPos)
                    simulenv.plot.setMovie(
                        fileSaveDir,
                        f"{label}-{sampleNums[label]}",
                        only_central_red=True,
                        trajectory=True,
                    )
                    simulenv.plot.setRange(
                        self.initSize / 2 * 1.5,
                        -self.initSize / 2 * 1.5,
                        self.initSize / 2 * 1.5,
                        -self.initSize / 2 * 1.5,
                    )
                    agent, simulenv = self.setControl_forMNIST(
                        physicalOutput,
                        [simStepList[label] for _ in range(len(simStepList))],
                        data,
                        agent,
                        pixelNum,
                        simulenv,
                    )
                    VerletSimulationBatch(simulenv, agent).runSimulation()
                    target_traj = target[label][:, : simStepList[label]]
                    centerNodeHis = torch.permute(
                        torch.stack(agent.state.pos_history, dim=0), (2, 1, 0, 3)
                    )[0, :, : simStepList[label], -1]
                    plotTrajectory(
                        target_traj,
                        centerNodeHis.cpu(),
                        fileSaveDir + f"centerNode_{label}-{sampleNums[label]}.png",
                        noFrame=True,
                    )
                    print(
                        "{:.7f}".format(
                            criterion(centerNodeHis.cpu(), target_traj).item()
                        ),
                        f"centerNode_{label}-{sampleNums[label]}.png",
                    )
                    target_img = transforms.functional.resize(
                        data[0][0], (pixelNum, pixelNum)
                    ).reshape(pixelNum, pixelNum)
                    plotTargetImage(
                        target_img,
                        fileSaveDir + f"target_{label}-{sampleNums[label]}.png",
                    )
                    pos_history = (
                        torch.permute(
                            torch.stack(agent.state.pos_history, dim=0), (2, 1, 0, 3)
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    sample_pos_histories[label].append(pos_history[0])
                    if q != len(test_dataset) - 1:
                        agent.resetAgent(agent.morph)
                    sampleNums[label] += 1
                for jj, sample_pos_history in enumerate(sample_pos_histories):
                    sample_pos_histories[jj] = np.stack(sample_pos_history, 0)
                np.savez_compressed(
                    fileSaveDir + f"position.npz",
                    label0=sample_pos_histories[0],
                    label1=sample_pos_histories[1],
                    label2=sample_pos_histories[2],
                    label3=sample_pos_histories[3],
                    label4=sample_pos_histories[4],
                    label5=sample_pos_histories[5],
                    label6=sample_pos_histories[6],
                    label7=sample_pos_histories[7],
                    label8=sample_pos_histories[8],
                    label9=sample_pos_histories[9],
                )
            else:
                acc = 0
                test_preds = []
                data_for_movie = []
                data_for_movie_is_saved = [False] * 10
                for q in range(num_of_batches_for_test):
                    data = []
                    for _ in range(self.batchSize):
                        _data = test_dataset.pop(0)
                        data.append(_data)
                        if not data_for_movie_is_saved[_data[1]]:
                            data_for_movie.append(_data)
                            data_for_movie_is_saved[_data[1]] = True
                    deltaPos = img2deltaPos(
                        data,
                        pixelNum,
                        agent,
                        brain,
                        deltaPosMask,
                    )
                    agent.setdeltaPos(deltaPos)
                    agent, simulenv = self.setControl_forMNIST(
                        physicalOutput, simStepList, data, agent, pixelNum, simulenv
                    )
                    VerletSimulationBatch(simulenv, agent).runSimulation()
                    _, _, testStates = self.calc_states(agent)
                    test_pred = F.softmax(testStates @ agent.readoutLayer, dim=1)
                    class_ans = torch.argmax(test_pred, dim=1)
                    class_labels = torch.tensor(
                        [data[jj][1] for jj in range(self.batchSize)]
                    )
                    test_preds.append(
                        torch.cat(
                            (
                                class_labels.unsqueeze(-1),
                                class_ans.unsqueeze(-1),
                                test_pred,
                            ),
                            1,
                        )
                    )
                    acc += torch.where(class_ans == class_labels)[0].shape[0]
                    if q != testNum - 1:
                        agent.resetAgent(morph=agent.morph)
                    else:
                        print("")
                test_preds = torch.stack(test_preds, 0).reshape(
                    -1, test_preds[0].shape[-1]
                )

                self.batchSize = 1
                agent.setBatchSize(self.batchSize)
                for q in range(len(data_for_movie)):
                    data = [data_for_movie[q]]
                    deltaPos = img2deltaPos(
                        data,
                        pixelNum,
                        agent,
                        brain,
                        deltaPosMask[0].unsqueeze(0),
                    )
                    agent.setdeltaPos(deltaPos)
                    simulenv.plot.setMovie(
                        fileSaveDir, data[0][1], only_central_red=False
                    )
                    simulenv.plot.setRange(
                        self.initSize / 2 * 1.2,
                        -self.initSize / 2 * 1.2,
                        self.initSize / 2 * 1.2,
                        -self.initSize / 2 * 1.2,
                    )
                    agent, simulenv = self.setControl_forMNIST(
                        physicalOutput, simStepList, data, agent, pixelNum, simulenv
                    )
                    VerletSimulationBatch(simulenv, agent).runSimulation()

        saveNPY(np.array(loss_history), fileSaveDir + "loss.npy")
        if len(lr_changes) > 0:
            saveNPY(
                np.array(lr_changes),
                fileSaveDir + f"lr_changes_{len(lr_changes)}.npy",
            )
        cmap = plt.get_cmap("tab10")
        params_history = np.stack(params_history, 0).T
        for i in range(params_history.shape[0]):
            results = [[epoch_array, params_history[i], params_label[i], cmap(i)]]
            plotLines(
                results,
                "epoch",
                None,
                None,
                fileSaveDir + f"params_change_{params_label[i]}.png",
                marker=False,
            )
        params_change_dict = {
            f"{params_label[i]}": params_history[i]
            for i in range(params_history.shape[0])
        }
        np.savez_compressed(fileSaveDir + "params_change.npz", **params_change_dict)
        if not physicalOutput:
            acc /= testNum
            print("Accuracy:", acc)
            saveNPY(np.array(test_preds), fileSaveDir + f"prediction_acc{acc}.npy")
            saveNPY(np.array(prob_history), fileSaveDir + "probability.npy")

        print("The results were saved in ", fileSaveDir)

    def labelmap_process(
        self,
        agent,
        simulenv,
        data,
        target,
        num,
        nodeNum,
        xs,
        ys,
        mask,
        defaultDeltaPos,
        maxSimStep,
        simStepList,
        pixelNum,
        fileSaveDir,
    ):
        """Individual process to make label maps

        Args:
            agent (Agent): Agent
            simulenv (SimulationEnvironment): Simulation environment
            data (list): MNIST data with smallest error
            target (list): Target trajectories
            num (int): Character (0 to 9)
            nodeNum (int): Id of mass point
            xs (list): Displacements in x direction
            ys (list): Displacements in y direction
            mask (torch.tensor): Mask for mass point displacement
            defaultDeltaPos (torch.tensor): Displacement determined by brain function
            maxSimStep (int): Maximum simulation step per character
            simStepList (list): Simulation steps per character
            pixelNum (int): Number of pixels on each side of the MNIST image
            fileSaveDir (str): Path to result folder
        """
        allAllErrors, allTrajLabels, allMinErrors = [], [], []
        for column in ys:
            deltaPos = generateDeltaPos(
                self.noNodes, nodeNum, xs, [column], self.device, mask
            )
            agent.setdeltaPos(
                torch.stack([defaultDeltaPos[num] for _ in range(self.batchSize)], 0)
                + deltaPos
            )
            agent, simulenv = self.setControl_forMNIST(
                True,
                [maxSimStep for _ in range(len(simStepList))],
                [data[num] for _ in range(self.batchSize)],
                agent,
                pixelNum,
                simulenv,
            )
            VerletSimulationBatch(simulenv, agent).runSimulation()
            centerNodeHis = torch.permute(
                torch.stack(agent.state.pos_history, dim=0), (2, 1, 0, 3)
            )
            allErrors, minErrors, trajLabels = compareMNISTtrajectory(
                centerNodeHis[:, :, :, -1], target, simStepList, nn.MSELoss()
            )
            allAllErrors.append(allErrors)
            allMinErrors.append(minErrors)
            allTrajLabels.append(trajLabels)
            agent.resetAgent(morph=agent.morph)
        allAllErrors = np.array(allAllErrors)
        allMinErrors = np.array(allMinErrors)
        allTrajLabels = np.array(allTrajLabels)
        plotLabelMap(
            allTrajLabels,
            xs,
            ys,
            fileSaveDir + f"labelmap_num{num}_node{nodeNum}",
        )
        np.savez_compressed(
            fileSaveDir + f"labelmap_num{num}_node{nodeNum}.npz",
            allError=allAllErrors,
            minError=allMinErrors,
            label=allTrajLabels,
        )
        print(f"Labelmap: num={num}, node={nodeNum} is completed")

    def MNIST_labelmap(
        self,
        physicalOutput,
        brain,
        pixelNum,
        char,
        testDataSize,
        fixedImages=-1,
        deltaScale=100,
    ):
        """Make label maps

        Args:
            physicalOutput (bool): Whether output is physicalized or not
            brain (str): Brain type
            pixelNum (int): Number of pixels on each side of the MNIST image
            char (int): Character to use for training and testing
            testDataSize (int): Number of data to use for testing
            fixedImages (int, optional): Whether to train and test only on specific MNIST images. Defaults to -1.
            deltaScale (int, optional): Number of pixels on each side of label map. Defaults to 100.
        """
        assert self.loadFolder is not None
        self.centralPointOutput = physicalOutput
        self.batchSize = 1
        criterion = nn.MSELoss()
        if fixedImages != -1:
            assert char == -1
            assert fixedImages % 10 == 0
        deltaPosMaskPiece = torch.ones((2, self.noNodes))
        if self.shape == "MultiCircles":
            assert math.sqrt(self.noNodes - 1).is_integer()
            sqrtNodes = int(math.sqrt(self.noNodes - 1))
            deltaPosMaskPiece[:, : 4 * sqrtNodes - 4] = torch.zeros(
                (2, 4 * sqrtNodes - 4)
            )
            deltaPosMaskPiece[:, -1] = torch.zeros(2)
        elif self.shape == "DoubleCircles":
            deltaPosMaskPiece[:, : int((self.noNodes - 1) / 2)] = torch.zeros(
                (2, int((self.noNodes - 1) / 2))
            )
            deltaPosMaskPiece[:, -1] = torch.zeros(2)
        deltaPosMasks = [deltaPosMaskPiece for _ in range(self.batchSize)]
        deltaPosMask = torch.stack(deltaPosMasks, 0)
        cmap = plt.get_cmap("tab10")
        if fixedImages != -1:
            added_suffix = f"fixedImages{fixedImages}_" + self.loadFileSuffix
        elif testDataSize != -1:
            added_suffix = f"testDataSize{testDataSize}_" + self.loadFileSuffix
        else:
            added_suffix = self.loadFileSuffix
        fileSaveDir = self.make_fileSaveDir(
            self.loadFolder + "/analysis/labelmap/" + self.timeStamp, added_suffix
        )
        save_params_as_json(self.params, fileSaveDir)
        timeScale = 3
        if testDataSize != -1:
            testDataSize = int(testDataSize / 10)
        test_dataset = MNIST_test_samples(testDataSize)
        if physicalOutput:
            uneven_length_target, simStepList = character_trajectory(self.initSize / 8)
            uneven_length_target = list(
                map(lambda l: l[:, ::timeScale], uneven_length_target)
            )
            simStepList = list(map(lambda l: -(-l // timeScale), simStepList))
            maxSimStep = max(simStepList)
            target = [
                torch.zeros((2, maxSimStep), dtype=torch.float64)
                for _ in range(len(uneven_length_target))
            ]
            for i, targ in enumerate(uneven_length_target):
                target[i][:, : targ.shape[1]] = targ
            test_dataset = MNIST_test_samples(num=3)
        agent, simulenv = self.loadObjects()
        agent = self.set_requiresGrads(agent, False, generator=False)

        if physicalOutput:
            bestInputs = [None for _ in range(10)]
            initPosData = []
            averageError = 0
            test_dataset = MNIST_test_samples(testDataSize)
            test_dataset = sorted(test_dataset, key=lambda x: x[1])
            agent.setBatchSize(self.batchSize)
            testNum = len(test_dataset)
            num_of_batches = testNum // self.batchSize
            for _ in range(num_of_batches):
                data = []
                labels = []
                for _ in range(self.batchSize):
                    data.append(test_dataset.pop(0))
                    labels.append(data[-1][1])
                deltaPos = img2deltaPos(
                    data,
                    pixelNum,
                    agent,
                    brain,
                    deltaPosMask,
                )
                agent.setdeltaPos(deltaPos)
                for initpos_x, initpos_y, label in zip(
                    agent.state.pos.x.gather(1, agent.state.movableNodes)
                    .detach()
                    .numpy(),
                    agent.state.pos.y.gather(1, agent.state.movableNodes)
                    .detach()
                    .numpy(),
                    labels,
                ):
                    initPosDatum = [initpos_x, initpos_y, label, cmap(label)]
                    initPosData.append(initPosDatum)
                agent, simulenv = self.setControl_forMNIST(
                    physicalOutput,
                    [maxSimStep for _ in range(len(simStepList))],
                    data,
                    agent,
                    pixelNum,
                    simulenv,
                )
                VerletSimulationBatch(simulenv, agent).runSimulation()
                trimed_centerNodeHis = torch.permute(
                    torch.stack(agent.state.pos_history, dim=0), (2, 1, 0, 3)
                )[:, :, :, -1]
                for jj in range(self.batchSize):
                    label = data[jj][1]
                    trimed_centerNodeHis[jj, :, simStepList[label] :] = torch.zeros(
                        (
                            1,
                            trimed_centerNodeHis.shape[1],
                            maxSimStep - simStepList[label],
                        ),
                        dtype=torch.float64,
                    )
                    error = criterion(
                        trimed_centerNodeHis[jj, :, : simStepList[label]],
                        target[label][:, : simStepList[label]],
                    ).item()
                    averageError += error
                    if fixedImages == -1:
                        if (bestInputs[label] is None) or (
                            error < bestInputs[label][1]
                        ):
                            bestInputs[label] = [data[jj], error]
                agent.resetAgent(morph=agent.morph)
            averageError /= testNum
            print("Average error: ", averageError)
            if testDataSize != -1:
                testDataSize *= 10
            data, defaultDeltaPos = [], []
            for bestInput in bestInputs:
                data.append(bestInput[0])
                defaultDeltaPos.append(
                    img2deltaPos(
                        [bestInput[0]],
                        pixelNum,
                        agent,
                        brain,
                        deltaPosMaskPiece.unsqueeze(0),
                    )[0]
                )
            defaultDeltaPos = torch.stack(defaultDeltaPos, 0)
            labelmap_search_width = self.initSize
            labelmap_search_delta = labelmap_search_width / deltaScale
            labelmap_search_num_x = labelmap_search_num_y = [
                round(j * labelmap_search_delta, 2)
                for j in range(
                    round(-labelmap_search_width / 2 / labelmap_search_delta),
                    round(
                        (labelmap_search_width / 2 + labelmap_search_delta)
                        / labelmap_search_delta
                    ),
                )
            ]
            self.batchSize = len(labelmap_search_num_x)
            agent.setBatchSize(self.batchSize)
            deltaPosMask = torch.stack(
                [deltaPosMaskPiece for _ in range(self.batchSize)], 0
            )
            mp.set_start_method("spawn", force=True)
            for nodeNum in range(self.noNodes):
                if deltaPosMaskPiece[0, nodeNum] == 0:
                    continue
                processes = []
                for num in range(10):
                    p = Process(
                        target=self.labelmap_process,
                        args=(
                            agent,
                            simulenv,
                            data,
                            target,
                            num,
                            nodeNum,
                            labelmap_search_num_x,
                            labelmap_search_num_y,
                            deltaPosMask,
                            defaultDeltaPos,
                            maxSimStep,
                            simStepList,
                            pixelNum,
                            fileSaveDir,
                        ),
                    )
                    p.start()
                    processes.append(p)
                    _pid = p.pid
                    affinity_out = subprocess.check_output(
                        ["taskset", "-p", "-c", str(_pid)], text=True
                    )
                    target_index = affinity_out.find(": ")
                    affinity_list = affinity_out[target_index + 2 : -1]
                    good_affinity = f"0-{cpu_count()-1}"
                    if affinity_list != good_affinity:
                        os.system(
                            "taskset -p -c %d-%d %d" % (0, cpu_count() - 1, p.pid)
                        )
                for p in processes:
                    p.join()
        else:
            initPosData = []
            self.batchSize = 1
            agent.setBatchSize(self.batchSize)
            test_dataset = MNIST_test_samples(testDataSize)
            test_dataset = sorted(test_dataset, key=lambda x: x[1])
            testNum = len(test_dataset)
            num_of_batches = testNum // self.batchSize
            acc = 0
            for _ in range(num_of_batches):
                data = []
                labels = []
                for _ in range(self.batchSize):
                    data.append(test_dataset.pop(0))
                    labels.append(data[-1][1])
                deltaPos = img2deltaPos(data, pixelNum, agent, brain, deltaPosMask)
                agent.setdeltaPos(deltaPos)
                for initpos_x, initpos_y, label in zip(
                    agent.state.pos.x.gather(1, agent.state.movableNodes)
                    .detach()
                    .numpy(),
                    agent.state.pos.y.gather(1, agent.state.movableNodes)
                    .detach()
                    .numpy(),
                    labels,
                ):
                    initPosDatum = [initpos_x, initpos_y, label, cmap(label)]
                    initPosData.append(initPosDatum)
                agent, simulenv = self.setControl_forMNIST(
                    physicalOutput, [], data, agent, pixelNum, simulenv
                )
                VerletSimulation(simulenv, agent).runSimulation()
                _, _, testStates = self.calc_states(agent)
                test_pred = F.softmax(testStates @ agent.readoutLayer, dim=1)
                class_ans = torch.argmax(test_pred, dim=1)
                class_labels = torch.tensor(
                    [data[jj][1] for jj in range(self.batchSize)]
                )
                acc += torch.where(class_ans == class_labels)[0].shape[0]
                agent.resetAgent(morph=agent.morph)
            acc /= testNum
            print("Accuracy: ", acc)

        print("The results were saved in ", fileSaveDir)

    def MNIST_PCA(
        self,
        brain,
        pixelNum,
        char,
        testDataSize,
        fixedImages=-1,
    ):
        """PCA

        Args:
            brain (str): Brain type
            pixelNum (int): Number of pixels on each side of the MNIST image
            char (int): Character to use for training and testing
            testDataSize (int): Number of data to use for testing
            fixedImages (int, optional): Whether to train and test only on specific MNIST images. Defaults to -1.
        """
        assert self.loadFolder != "None"
        self.batchSize = 1
        self.centralPointOutput = True
        self.batchSize = 1
        criterion = nn.MSELoss()
        if fixedImages != -1:
            assert char == -1
            assert fixedImages % 10 == 0
        deltaPosMaskPiece = torch.ones((2, self.noNodes))
        if self.shape == "MultiCircles":
            assert math.sqrt(self.noNodes - 1).is_integer()
            sqrtNodes = int(math.sqrt(self.noNodes - 1))
            deltaPosMaskPiece[:, : 4 * sqrtNodes - 4] = torch.zeros(
                (2, 4 * sqrtNodes - 4)
            )
            deltaPosMaskPiece[:, -1] = torch.zeros(2)
        elif self.shape == "DoubleCircles":
            deltaPosMaskPiece[:, : int((self.noNodes - 1) / 2)] = torch.zeros(
                (2, int((self.noNodes - 1) / 2))
            )
            deltaPosMaskPiece[:, -1] = torch.zeros(2)
        deltaPosMasks = [deltaPosMaskPiece for _ in range(self.batchSize)]
        deltaPosMask = torch.stack(deltaPosMasks, 0)
        cmap = plt.get_cmap("tab10")
        fileSaveDir = self.make_fileSaveDir(
            self.loadFolder + "/analysis/PCA/" + self.timeStamp, self.loadFileSuffix
        )
        save_params_as_json(self.params, fileSaveDir)
        timeScale = 3
        if testDataSize != -1:
            testDataSize = int(testDataSize / 10)
        test_dataset = MNIST_test_samples(testDataSize)
        uneven_length_target, simStepList = character_trajectory(self.initSize / 8)
        uneven_length_target = list(
            map(lambda l: l[:, ::timeScale], uneven_length_target)
        )
        simStepList = list(map(lambda l: -(-l // timeScale), simStepList))
        maxSimStep = max(simStepList)
        target = [
            torch.zeros((2, maxSimStep), dtype=torch.float64)
            for _ in range(len(uneven_length_target))
        ]
        for i, targ in enumerate(uneven_length_target):
            target[i][:, : targ.shape[1]] = targ
        test_dataset = MNIST_test_samples(num=3)
        agent, simulenv = self.loadObjects()
        agent = self.set_requiresGrads(agent, False, generator=False)
        best_pos_histories = [[] for _ in range(10)]

        bestInputs = [None for _ in range(10)]
        initPosData = []
        allData = []
        averageError = 0
        errors = [[] for _ in range(10)]
        test_dataset = MNIST_test_samples(testDataSize)
        test_dataset = sorted(test_dataset, key=lambda x: x[1])
        agent.setBatchSize(self.batchSize)
        testNum = len(test_dataset)
        num_of_batches = testNum // self.batchSize
        for _ in range(num_of_batches):
            data = []
            labels = []
            for _ in range(self.batchSize):
                data.append(test_dataset.pop(0))
                labels.append(data[-1][1])
            deltaPos = img2deltaPos(
                data,
                pixelNum,
                agent,
                brain,
                deltaPosMask,
            )
            agent.setdeltaPos(deltaPos)
            for initpos_x, initpos_y, label in zip(
                agent.state.pos.x.gather(1, agent.state.movableNodes).detach().numpy(),
                agent.state.pos.y.gather(1, agent.state.movableNodes).detach().numpy(),
                labels,
            ):
                initPosDatum = [initpos_x, initpos_y, label, cmap(label)]
                initPosData.append(initPosDatum)
            agent, simulenv = self.setControl_forMNIST(
                True,
                [maxSimStep for _ in range(len(simStepList))],
                data,
                agent,
                pixelNum,
                simulenv,
            )
            VerletSimulationBatch(simulenv, agent).runSimulation()
            target_traj = target[label][:, : simStepList[label]]
            trimed_nodeHis = torch.permute(
                torch.stack(agent.state.pos_history, dim=0), (2, 1, 0, 3)
            )
            allData.append([trimed_nodeHis, label, cmap(label)])
            trimed_centerNodeHis = torch.permute(
                torch.stack(agent.state.pos_history, dim=0), (2, 1, 0, 3)
            )[:, :, :, -1]
            for jj in range(self.batchSize):
                label = data[jj][1]
                trimed_centerNodeHis[jj, :, simStepList[label] :] = torch.zeros(
                    (
                        1,
                        trimed_centerNodeHis.shape[1],
                        maxSimStep - simStepList[label],
                    ),
                    dtype=torch.float64,
                )
                error = criterion(
                    trimed_centerNodeHis[jj, :, : simStepList[label]],
                    target[label][:, : simStepList[label]],
                ).item()
                averageError += error
                errors[label].append(error)
                if fixedImages == -1:
                    if (bestInputs[label] is None) or (error < bestInputs[label][1]):
                        bestInputs[label] = [data[jj], error]
                        plotTrajectory(
                            target_traj,
                            trimed_centerNodeHis[jj, :, : simStepList[label]]
                            .detach()
                            .cpu(),
                            fileSaveDir + f"centerNode_best_{label}.png",
                            noFrame=True,
                        )
                        pos_history = (
                            torch.permute(
                                torch.stack(agent.state.pos_history, dim=0),
                                (2, 1, 0, 3),
                            )
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        best_pos_histories[label] = [pos_history[0]]
            agent.resetAgent(morph=agent.morph)
        averageError /= testNum
        print("Average error: ", averageError)
        minLen = 100000
        for er in errors:
            if len(er) < minLen:
                minLen = len(er)
        for ii in range(10):
            errors[ii] = errors[ii][:minLen]
        saveNPY(np.array(errors), fileSaveDir + f"errors.npy")
        errorsMean = np.mean(errors, axis=1)
        errorsMin = np.min(errors, axis=1)
        saveNPY(
            np.stack([errorsMean, errorsMin]),
            fileSaveDir + "errors_mean_min.npy",
        )
        np.savez_compressed(
            fileSaveDir + f"best_position_test.npz",
            label0=best_pos_histories[0],
            label1=best_pos_histories[1],
            label2=best_pos_histories[2],
            label3=best_pos_histories[3],
            label4=best_pos_histories[4],
            label5=best_pos_histories[5],
            label6=best_pos_histories[6],
            label7=best_pos_histories[7],
            label8=best_pos_histories[8],
            label9=best_pos_histories[9],
        )
        if testDataSize != -1:
            testDataSize *= 10
        with open(fileSaveDir + f"PCA_test_initPosData.pickle", mode="wb") as fo:
            pickle.dump(initPosData, fo)
        with open(fileSaveDir + f"allData_for_pca.pickle", mode="wb") as fo:
            pickle.dump(allData, fo)
        allData_initPos = [
            [_data[0][0, :, 0, agent.state.movableNodes[0]].detach().numpy()]
            + _data[1:]
            for _data in allData
        ]
        pcaData_initPos, contrib_initPos = plotPca(
            allData_initPos,
            fileSaveDir + f"PCA_testData{int(testDataSize)}_initPos",
        )
        saveNPY(
            pcaData_initPos,
            fileSaveDir + f"PCA_testData{int(testDataSize)}_initPos.npy",
        )
        saveNPY(
            contrib_initPos,
            fileSaveDir + f"PCA_testImages_contribution_ratio_initPos.npy",
        )
        if self.noNodes == 37:
            allData_initPosInner = [
                [_data[0][0, :, 0, [32, 33, 34, 35]].detach().numpy()] + _data[1:]
                for _data in allData
            ]
            pcaData_initPosInner, contrib_initPosInner = plotPca(
                allData_initPosInner,
                fileSaveDir + f"PCA_testData{int(testDataSize)}_initPosInner",
            )
            saveNPY(
                pcaData_initPosInner,
                fileSaveDir + f"PCA_testData{int(testDataSize)}_initPosInner.npy",
            )
            saveNPY(
                contrib_initPosInner,
                fileSaveDir + f"PCA_testImages_contribution_ratio_initPosInner.npy",
            )
            allData_initPosOuter = [
                [
                    _data[0][
                        0,
                        :,
                        0,
                        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                    ]
                    .detach()
                    .numpy()
                ]
                + _data[1:]
                for _data in allData
            ]
            pcaData_initPosOuter, contrib_initPosOuter = plotPca(
                allData_initPosOuter,
                fileSaveDir + f"PCA_testData{int(testDataSize)}_initPosOuter",
            )
            saveNPY(
                pcaData_initPosOuter,
                fileSaveDir + f"PCA_testData{int(testDataSize)}_initPosOuter.npy",
            )
            saveNPY(
                contrib_initPosOuter,
                fileSaveDir + f"PCA_testImages_contribution_ratio_initPosOuter.npy",
            )
        allData_traj = [
            [_data[0][0, :, :, -1].detach().numpy()] + _data[1:] for _data in allData
        ]
        pcaData_traj, contrib_traj = plotPca(
            allData_traj, fileSaveDir + f"PCA_testData{int(testDataSize)}_traj"
        )
        saveNPY(
            pcaData_traj,
            fileSaveDir + f"PCA_testData{int(testDataSize)}_traj.npy",
        )
        saveNPY(
            contrib_traj,
            fileSaveDir + f"PCA_testImages_contribution_ratio_traj.npy",
        )
        pcaData, contrib = plotPca(
            initPosData, fileSaveDir + f"PCA_testData{int(testDataSize)}"
        )
        saveNPY(
            pcaData,
            fileSaveDir
            + f"PCA_testData{int(testDataSize)}_averageError{averageError}.npy",
        )
        saveNPY(contrib, fileSaveDir + f"PCA_testImages_contribution_ratio.npy")
        best_pos_histories = [[] for _ in range(10)]
        bestInputs = [None for _ in range(10)]

        print("The results were saved in ", fileSaveDir)
