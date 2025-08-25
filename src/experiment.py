import os
import math
import datetime
import copy
import random
import pickle
import subprocess
import numpy as np
from scipy.signal import argrelmax, argrelextrema
from sklearn.neighbors import KernelDensity

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Manager, Process, cpu_count

import matplotlib.pyplot as plt
import matplotlib as mpl

from agent import *
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
        """_summary_

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
        self.centralPointOutput = False

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
        bias=False,
        brain="LIL",
        seed=None,
        controlType="no",
        generatorNum=[-1],
        amplitude=0.8,
        ground=False,
        wind=None,
        closedloop=False,
        noise=0,
    ):
        """Initial experimental settings"""
        if ground:
            env = SoftEnvironment(ground=ground, wind=wind, device=self.device)
        else:
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
            device=self.device,
        )
        if controlType == "no":
            control = NoControl(
                morph, simTime, self.simTimeStep, tau, device=self.device
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
            bias=bias,
            brain=brain,
            closedloop=closedloop,
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

    def loadLossMean(self):
        """Load mean loss during training"""
        data = np.load(self.loadFolder + "loss.npy")
        lossMean = np.mean(data[:, int(data.shape[1] / 2) :], 1)
        return lossMean

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

    def loadAgent(
        self,
        suffix,
        path=None,
        keepStates=False,
    ):
        """Load agent"""
        assert self.loadFolder is not None
        assert suffix is not None
        if path is None:
            with open(self.loadFolder + f"agent_{suffix}.pickle", mode="rb") as f:
                agent = pickle.load(f)
        else:
            with open(path + f"agent_{suffix}.pickle", mode="rb") as f:
                agent = pickle.load(f)
        self.initSize = agent.morph.initSize
        self.tau = agent.control.tau
        if hasattr(agent.control, "inputScale"):
            self.inputScale = agent.control.inputScale
        agent.resetAgent(morph=agent.morph, keepStates=keepStates)
        if keepStates:
            agent.state.pos.matrix = agent.state.pos.matrix.detach()
            agent.state.speed.matrix = agent.state.speed.matrix.detach()
        return agent

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
                if suffix[0] == "/":
                    self.memo += suffix
                else:
                    self.memo = suffix + "_" + self.memo
        if self.memo is None:
            fileSaveDir += "/"
        else:
            if self.memo[0] == "/":
                fileSaveDir += self.memo + "/"
            else:
                fileSaveDir += "_" + self.memo + "/"
        if not os.path.exists(fileSaveDir):
            os.makedirs(fileSaveDir)
        return fileSaveDir

    def check_taskLength(self, agent, train_pred, test_pred):
        """Check data length in timeseries emulation task"""
        trainTaskLen = int(agent.control.trainLen / agent.control.tau)
        testTaskLen = int(agent.control.testLen / agent.control.tau)
        assert (
            trainTaskLen == train_pred.shape[0]
        ), f"trainTaskLen={trainTaskLen}, train_pred.shape[0]={train_pred.shape[0]}"
        assert (
            testTaskLen == test_pred.shape[0]
        ), f"testTaskLen={testTaskLen}, test_pred.shape[0]={test_pred.shape[0]}"
        return trainTaskLen, testTaskLen

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

    def ridgeRegression(self, agent):
        """Perform ridge regression"""
        readout = Readout(agent.reservoirStatesDim + 1, 1)
        w_out_opt, df_ridgeparam = agent.optimizer_ridge.solve()
        readout.set_weight(w_out_opt)
        train_pred = readout(agent.optimizer_ridge.X).squeeze()
        data_for_test_pred = torch.cat(
            (
                torch.ones((1, agent.optimizer_ridge.test_X.shape[1])),
                agent.optimizer_ridge.test_X,
            ),
            dim=0,
        )
        test_pred = readout(data_for_test_pred).squeeze()
        pred = torch.cat((train_pred, test_pred), dim=0)
        return train_pred, test_pred, pred, w_out_opt, df_ridgeparam

    def timeseries_training(
        self,
        dirClass,
        delay,
        physicalOutput,
        epoch,
        inputScale,
        brain,
        optim,
        lr_spring,
        lr_damping,
        lr_restlength,
        lr_readoutlayer,
        lr_brain,
        cumulative=False,
    ):
        """Timeseries emulation experiment

        Args:
            dirClass (str): Target timeseries type
            delay (int): Step delay
            physicalOutput (bool): Whether output is physicalized or not
            epoch (int): Training epoch
            inputScale (int): Input value scaling
            brain (str): Brain type
            optim (str): Optimizer
            lr_spring (float): Learning rate of spring constant of MSDN
            lr_damping (float): Learning rate of damping coefficient of MSDN
            lr_restlength (float): Learning rate of rest length of MSDN
            lr_readoutlayer (float): Learning rate of readout layer
            lr_brain (float): Learning rate of FNN parameters
            cumulative (bool, optional): Whether to set the target output as the cumulative value of past inputs in memory task. Defaults to False.
        """
        if physicalOutput:
            self.centralPointOutput = True
            algClass = "/physicalOutput/"
            readout = False
        else:
            algClass = "/non-physicalized/BP/"
            readout = True
        self.inputScale = inputScale
        if cumulative:
            algClass = "_cumulative" + algClass
        self.figDirSuf += f"{brain}_{self.shape}/"
        if self.loadFolder is not None:
            fileSaveDir = self.make_fileSaveDir(
                self.loadFolder + "/additionalTraining/" + self.timeStamp,
                self.loadFileSuffix,
            )
        else:
            fileSaveDir = self.make_fileSaveDir(
                self.figDirPre
                + dirClass
                + str(delay)
                + algClass
                + self.figDirSuf
                + self.timeStamp
                + f"_shape{self.shape}_nodes{self.noNodes}_tau{self.tau}_seed{self.seed}",
                self.loadFileSuffix,
            )
        save_params_as_json(self.params, fileSaveDir)
        if brain == "LIL":
            params_label = [
                "spring",
                "damping",
                "restLength",
                "inputLayer",
                "readoutLayer",
            ]
        elif (brain == "MLP") or (brain == "CNN"):
            params_label = [
                "spring",
                "damping",
                "restLength",
                "inputLayer",
                "readoutLayer",
                "brain(conv1)",
                "brain(conv2)",
                "brain(fc)",
            ]
        criterion = nn.MSELoss()
        loss_history, params_history = [], []
        requires_grad_flag = True
        limit_to_reduce_lr = 5
        countdown_to_reduce_lr = [limit_to_reduce_lr, False]
        lr_changes = []

        for i in range(epoch + 3):
            if self.loadFolder is not None:
                agent, simulenv = self.loadObjects()
                agent.control.setSimTime(self.simTime, self.simTimeStep, self.tau)
            else:
                agent, simulenv = self.initSet(
                    self.initSize,
                    self.tau,
                    _simtime=self.simTime,
                    centerFlag=True,
                    brain=brain,
                )
            agent.state.resetSavePosType(self.savePosType)
            if i != 0:
                agent.morph.spring = _spring
                agent.morph.damping = _damping
                agent.morph.restLength = _restLength
                if readout:
                    agent.readoutLayer = _readoutLayer
                if brain == "LIL":
                    agent.inputLayer = _inputLayer
                else:
                    agent.brain = _brain
            else:
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
            agent.morph.spring.requires_grad = requires_grad_flag
            agent.morph.damping.requires_grad = requires_grad_flag
            agent.morph.restLength.requires_grad = requires_grad_flag
            if readout:
                agent.readoutLayer.requires_grad = requires_grad_flag
            if brain == "LIL":
                agent.inputLayer.requires_grad = requires_grad_flag
            else:
                for brain_layer in agent.brain.parameters():
                    brain_layer.requires_grad = requires_grad_flag

            simulenv.plot.setMovie(False)
            if i == epoch + 1:
                simulenv.plot.setMovie(
                    fileSaveDir,
                    only_central_red=physicalOutput,
                    removeWashout=True,
                )
                simulenv.plot.setRange(Xmax, Xmin, Ymax, Ymin)
            if (i == 0) or (i >= epoch - 1) or (i % 100 == 0):
                if "memoryTask" in dirClass:
                    agent.control.setMemoryTask(
                        self.tau,
                        delay,
                        seed=10,
                        inputScale=self.inputScale,
                        cumulative=cumulative,
                    )
                elif "expandedNARMA" in dirClass:
                    agent.control.setExpandedNARMA(
                        self.tau, delay, seed=10, inputScale=self.inputScale
                    )
            else:
                if "memoryTask" in dirClass:
                    agent.control.setMemoryTask(
                        self.tau,
                        delay,
                        seed=random.randint(0, 1000),
                        inputScale=self.inputScale,
                        cumulative=cumulative,
                    )
                elif "expandedNARMA" in dirClass:
                    agent.control.setExpandedNARMA(
                        self.tau,
                        delay,
                        seed=random.randint(0, 1000),
                        inputScale=self.inputScale,
                    )
            simulenv.setSimulationLength(agent.control.us.shape[0])

            params_lr_list = [
                {"params": agent.morph.spring, "lr": lr_spring},
                {"params": agent.morph.damping, "lr": lr_damping},
                {"params": agent.morph.restLength, "lr": lr_restlength},
            ]
            if readout:
                params_lr_list.append(
                    {"params": agent.readoutLayer, "lr": lr_readoutlayer}
                )
            if brain == "LIL":
                params_lr_list.append({"params": agent.inputLayer, "lr": lr_brain})
            else:
                params_lr_list.append(
                    {"params": agent.brain.parameters(), "lr": lr_brain}
                )
            optimizer = OPTIM_DICT[optim](params_lr_list)

            VerletSimulation(simulenv, agent).runSimulation()
            nodesHis = torch.permute(
                torch.stack(agent.state.movableNodes_history, 0), (1, 2, 0)
            ).to(torch.float64)

            if physicalOutput:
                if self.shape == "Lattice":
                    nodeIndex = int((nodesHis.shape[1] - 1) / 2)
                else:
                    nodeIndex = -1
                train_pred = nodesHis[
                    1,
                    nodeIndex,
                    agent.control.washoutLen
                    // self.tau : (agent.control.washoutLen + agent.control.trainLen)
                    // self.tau,
                ]
                test_pred = nodesHis[
                    1,
                    nodeIndex,
                    (agent.control.washoutLen + agent.control.trainLen) // self.tau :,
                ]
            else:
                nodesHis = nodesHis.reshape([-1, nodesHis.shape[-1]])
                trainData = nodesHis[
                    :,
                    agent.control.washoutLen
                    // self.tau : (agent.control.washoutLen + agent.control.trainLen)
                    // self.tau,
                ]
                testData = nodesHis[
                    :, (agent.control.washoutLen + agent.control.trainLen) // self.tau :
                ]
                trainStates = torch.cat(
                    (torch.ones((1, trainData.shape[-1])).to(self.device), trainData),
                    dim=0,
                )
                testStates = torch.cat(
                    (torch.ones((1, testData.shape[-1])).to(self.device), testData),
                    dim=0,
                )
                if agent.readoutLayer.dtype == torch.float32:
                    trainStates = trainStates.to(torch.float32)
                    testStates = testStates.to(torch.float32)
                if agent.readoutLayer.shape[1] != trainStates.shape[0]:
                    train_pred = (agent.readoutLayer.T @ trainStates).squeeze()
                    test_pred = (agent.readoutLayer.T @ testStates).squeeze()
                else:
                    train_pred = (agent.readoutLayer @ trainStates).squeeze()
                    test_pred = (agent.readoutLayer @ testStates).squeeze()

            trainTaskLen, testTaskLen = self.check_taskLength(
                agent, train_pred, test_pred
            )
            mse = MSE(test_pred, agent.control.target[-testTaskLen:]).item()
            train_preds = train_pred.to(torch.float64)
            train_targets = agent.control.target[
                -(trainTaskLen + testTaskLen) : -testTaskLen
            ].to(torch.float64)
            if i == epoch:
                pos_his = (
                    torch.stack(agent.state.pos_history, dim=0).cpu().detach().numpy()
                )
            agent.resetAgent()
            if i == epoch:
                agent.control.target, agent.control.us = None, None
                self.saveObjects(agent, simulenv, fileSaveDir, "final")
            if i <= epoch:
                if i <= 10:
                    agent.control.target, agent.control.us = None, None
                    self.saveObjects(agent, simulenv, fileSaveDir, i)
                elif (i <= 100) and (i % 10 == 0):
                    agent.control.target, agent.control.us = None, None
                    self.saveObjects(agent, simulenv, fileSaveDir, i)
                elif i % 100 == 0:
                    agent.control.target, agent.control.us = None, None
                    self.saveObjects(agent, simulenv, fileSaveDir, i)

            optimizer.zero_grad()
            loss = criterion(train_preds, train_targets)
            params_history = print_current_params(
                agent, i, loss.item(), brain, params_history
            )
            if i < epoch - 1:
                try:
                    loss.backward(retain_graph=True)
                except RuntimeError:
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
                                i,
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
                                    i,
                                    lr_spring,
                                    lr_damping,
                                    lr_restlength,
                                    lr_readoutlayer,
                                    lr_brain,
                                ]
                            )
                        continue
                if countdown_to_reduce_lr[0] != limit_to_reduce_lr:
                    if not countdown_to_reduce_lr[1]:
                        countdown_to_reduce_lr[1] = True
                    else:
                        countdown_to_reduce_lr = [limit_to_reduce_lr, False]
                if i != 0:
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
                optimizer.step()
                with torch.no_grad():
                    if brain != "LIL":
                        _brain = agent.brain
                    if not readout:
                        [_spring, _damping, _inputLayer, _restLength] = (
                            self.shape_tunableParameters(
                                agent,
                                readoutLayer=False,
                                restLength=True,
                            )
                        )
                    else:
                        [_spring, _damping, _inputLayer, _readoutLayer, _restLength] = (
                            self.shape_tunableParameters(
                                agent,
                                readoutLayer=True,
                                restLength=True,
                            )
                        )
            elif i == epoch - 1:
                with torch.no_grad():
                    if brain != "LIL":
                        _brain = agent.brain
                    if not readout:
                        [_spring, _damping, _inputLayer, _restLength] = (
                            self.shape_tunableParameters(
                                agent,
                                readoutLayer=False,
                                restLength=True,
                                shape=False,
                            )
                        )
                    else:
                        [
                            _spring,
                            _damping,
                            _inputLayer,
                            _readoutLayer,
                            _restLength,
                        ] = self.shape_tunableParameters(
                            agent,
                            readoutLayer=True,
                            restLength=True,
                            shape=False,
                        )
                requires_grad_flag = False
            elif i == epoch:
                Xmax, Xmin, Ymax, Ymin = (
                    np.max(pos_his[:, 0, :]),
                    np.min(pos_his[:, 0, :]),
                    np.max(pos_his[:, 1, :]),
                    np.min(pos_his[:, 1, :]),
                )
                requires_grad_flag = False
            if i < epoch:
                loss_history.append(loss.item())
            if i == epoch - 1:
                if self.simTime > 15:
                    self.simTime /= 10
                    agent.control.setSimTime(self.simTime, self.simTimeStep, self.tau)
            elif i == epoch + 1:
                if self.simTime > 1.5:
                    self.simTime *= 100
                else:
                    self.simTime *= 10
                agent.control.setSimTime(self.simTime, self.simTimeStep, self.tau)

        pred = torch.cat((train_pred, test_pred), dim=0)
        epoch_array = np.arange(0, epoch, 1)
        plotLines(
            [[epoch_array, loss_history, None, "tab:blue"]],
            "epoch",
            "loss",
            None,
            fileSaveDir + f"loss_{loss.item()}_when_simTime_is_{self.simTime}.png",
            marker=False,
        )
        target_array = agent.control.target.cpu().detach().numpy()
        t_task_array = np.arange(
            0, agent.control.simulLen * self.simTimeStep, self.simTimeStep * self.tau
        )
        results = [
            [
                t_task_array[-(trainTaskLen + testTaskLen) :],
                target_array[-(trainTaskLen + testTaskLen) :],
                "target",
                "black",
            ],
            [
                t_task_array[-(trainTaskLen + testTaskLen) :],
                pred.cpu().detach().numpy(),
                "output",
                "tab:red",
            ],
        ]
        plotLines(
            results,
            "t",
            None,
            "MSE={}".format(mse),
            fileSaveDir + "trainANDtest.png",
            marker=False,
        )
        results_test = [
            [
                t_task_array[-testTaskLen:],
                target_array[-testTaskLen:],
                "target",
                "black",
            ],
            [
                t_task_array[-testTaskLen:],
                test_pred.cpu().detach().numpy(),
                "output",
                "tab:red",
            ],
        ]
        plotLines(
            results_test,
            "t",
            None,
            "MSE={}".format(mse),
            fileSaveDir + "test.png",
            marker=False,
        )
        if len(lr_changes) > 0:
            saveNPY(
                np.array(lr_changes),
                fileSaveDir + f"lr_changes_{len(lr_changes)}.npy",
            )
        cmap = plt.get_cmap("tab10")
        params_history = np.stack(params_history, 0).T
        for i in range(params_history.shape[0]):
            results = [
                [
                    epoch_array,
                    params_history[i][: epoch_array.shape[0]],
                    params_label[i],
                    cmap(i),
                ]
            ]
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
        np.savez_compressed(
            fileSaveDir + "input_target_prediction.npz",
            input=agent.control.us.cpu().detach().numpy(),
            target=target_array,
            prediction=pred.cpu().detach().numpy(),
        )
        saveNPY(np.array(loss_history), fileSaveDir + "loss.npy")
        saveNPY(
            np.array(
                [
                    [
                        agent.morph.initSize,
                        agent.control.tau,
                        agent.control.inputScale,
                    ]
                ]
            ),
            fileSaveDir + "initSize_tau_inputScale.npy",
        )

        print("The results were saved in ", fileSaveDir)

    def Lissajous_training(
        self,
        updateTimeScale,
        optim,
        lr_spring,
        lr_damping,
        lr_restlength,
        lr_amplitude,
        lr_omega,
        lr_phase,
        generatorNum,
        inputScale,
        wind=False,
        lissajous1=[1, 2, 0, 1],
        lissajous2=None,
        noise=0,
        generatorDist=None,
    ):
        """Lissajous drawing training

        Args:
            updateTimeScale (int): Number of steps to update parameters
            optim (str): Optimizer
            lr_spring (float): Learning rate of spring constant of MSDN
            lr_damping (float): Learning rate of damping coefficient of MSDN
            lr_restlength (float): Learning rate of rest length of MSDN
            lr_amplitude (float): Learning rate of amplitude of SWG
            lr_omega (float): Learning rate of omega of SWG
            lr_phase (float): Learning rate of phase of SWG
            generatorNum (list): Id of springs modulated by SWG
            inputScale (int): Input value scaling
            wind (bool, optional): Whether to introduce wind in the experiment. Defaults to False.
            lissajous1 (list, optional): First target Lissajous curve's parameters. Defaults to [1, 2, 0, 1].
            lissajous2 (list, optional): Second target Lissajous curve's parameters. Defaults to None.
            noise (int, optional): Standard deviation of noise added to feedback information. Defaults to 0.
            generatorDist (_type_, optional): Distribution of springs modulated by SWG. Defaults to None.
        """
        self.centralPointOutput = True
        [lissajous_a1, lissajous_b1, lissajous_delta_n1, lissajous_delta_d1] = [
            int(s) for s in lissajous1
        ]
        if lissajous2 is not None:
            [lissajous_a2, lissajous_b2, lissajous_delta_n2, lissajous_delta_d2] = [
                int(s) for s in lissajous2
            ]
        assert self.simTime % 10 == 0
        trainingSteps = int(self.simTime * 9 / 10 / self.simTimeStep / updateTimeScale)
        evalSteps = int(self.simTime / 10 / self.simTimeStep / updateTimeScale)
        epoch = trainingSteps + evalSteps
        self.inputScale = inputScale
        lissajous_targets = []
        lissajous_targets.append(
            LissajousTimeSeries(
                self.simTime,
                self.simTimeStep,
                self.initSize,
                a=lissajous_a1,
                b=lissajous_b1,
                delta=np.pi * lissajous_delta_n1 / lissajous_delta_d1,
            )
        )
        if lissajous2 is not None:
            target_x_offset = self.initSize / 20 if wind else 0
            lissajous_targets.append(
                LissajousTimeSeries(
                    self.simTime,
                    self.simTimeStep,
                    self.initSize,
                    target_x_offset,
                    a=lissajous_a2,
                    b=lissajous_b2,
                    delta=np.pi * lissajous_delta_n2 / lissajous_delta_d2,
                )
            )
        _inputScale = self.inputScale
        criterion = nn.MSELoss()
        if self.loadFolder is not None:
            agent, simulenv = self.loadObjects()
            fileSaveDir = self.make_fileSaveDir(
                self.loadFolder + "/additionalTraining/" + self.timeStamp,
                self.loadFileSuffix,
            )
            self.inputScale = agent.control.inputScale = _inputScale
        else:
            agent, simulenv = self.initSet(
                self.initSize,
                self.tau,
                _simtime=self.simTime,
                centerFlag=True,
                bias=True,
                controlType="sine",
                generatorNum=generatorNum,
                amplitude=0.5,
                noise=noise,
            )
            genDist = "-".join([key for key in generatorDist])
            if lissajous2 is None:
                fileSaveDir = self.make_fileSaveDir(
                    self.figDirPre
                    + "Lissajous/"
                    + self.figDirSuf
                    + f"{lissajous_a1}-{lissajous_b1}-{lissajous_delta_n1}-{lissajous_delta_d1}/"
                    + self.timeStamp
                    + f"_shape{self.shape}_nodes{self.noNodes}_updateTimeScale{updateTimeScale}"
                    f"_seed{self.seed}_{genDist}",
                    None,
                )
            else:
                if wind:
                    self.figDirSuf += "wind/"
                else:
                    self.figDirSuf += "inputLayer/"
                fileSaveDir = self.make_fileSaveDir(
                    self.figDirPre
                    + "Lissajous/"
                    + self.figDirSuf
                    + f"{lissajous_a1}-{lissajous_b1}-{lissajous_delta_n1}-{lissajous_delta_d1}_"
                    + f"{lissajous_a2}-{lissajous_b2}-{lissajous_delta_n2}-{lissajous_delta_d2}/"
                    + self.timeStamp
                    + f"_shape{self.shape}_nodes{self.noNodes}_updateTimeScale{updateTimeScale}"
                    f"_seed{self.seed}_{genDist}",
                    None,
                )
        save_params_as_json(self.params, fileSaveDir)
        params_label = [
            "spring",
            "damping",
            "restLength",
            "inputLayer",
            "amplitude",
            "omega",
            "phase",
        ]
        simulenv.plot.setMovie(False)
        iterationNumber = 0
        loss_history, params_history = [], []
        if (
            (lr_spring == 0)
            and (lr_damping == 0)
            and (lr_restlength == 0)
            and (lr_amplitude == 0)
            and (lr_omega == 0)
            and (lr_phase == 0)
        ):
            requires_grad_flag = False
        else:
            requires_grad_flag = True
        Xmax, Xmin, Ymax, Ymin = 0, 0, 0, 0
        limit_to_reduce_lr = 5
        countdown_to_reduce_lr = [limit_to_reduce_lr, False]
        lr_changes = []
        current_target_index = 0

        for i in range(epoch):
            target_trajectory = []
            if lissajous2 is None:
                target_ind, wind_val = 0, 0
                if i == epoch - 10:
                    simulenv.plot.setMovie(
                        fileSaveDir,
                        only_central_red=True,
                        colored_comp=[generatorNum, []],
                    )
                    simulenv.plot.setRange(Xmax, Xmin, Ymax, Ymin)
            else:
                if i <= epoch - 200 - 40:
                    if i % 100 < 50:
                        target_ind, wind_val = 0, 0
                    else:
                        target_ind, wind_val = 1, 15
                elif i < epoch - 200:
                    if (i < epoch - 200 - 30) or (
                        (i >= epoch - 200 - 20) and (i < epoch - 200 - 10)
                    ):
                        target_ind, wind_val = 0, 0
                    else:
                        target_ind, wind_val = 1, 15
                else:
                    if i < epoch - 100:
                        target_ind, wind_val = 0, 0
                    else:
                        target_ind, wind_val = 1, 15
                if (epoch > 1000) and (
                    (i == int(trainingSteps * 0.8)) or (i == int(trainingSteps * 0.9))
                ):
                    lr_spring /= 10
                    lr_damping /= 10
                    lr_restlength /= 10
                    lr_amplitude /= 10
                    lr_omega /= 10
                    lr_phase /= 10
                if i == epoch - 200 - 20:
                    simulenv.plot.setMovie(
                        fileSaveDir,
                        only_central_red=True,
                        colored_comp=[generatorNum, []],
                        text=True,
                    )
                    simulenv.plot.setRange(Xmax, Xmin, Ymax, Ymin)
                elif i == epoch - 200:
                    verletSim.simulEnv.end()
                    simulenv.plot.setMovie(False)
            lissajous_target = lissajous_targets[target_ind]
            agent.state.resetSavePosType(self.savePosType)
            if i >= trainingSteps:
                requires_grad_flag = False
            if i != 0:
                if i < trainingSteps:
                    agent.morph.spring = _spring
                    agent.morph.damping = _damping
                    agent.morph.restLength = _restLength
                    agent.inputLayer = _inputLayer
                    agent.control.amplitude = _amplitude
                    agent.control.omega = _omega
                    agent.control.phase = _phase
                elif i == trainingSteps:
                    agent.morph.spring = agent.morph.spring
                    agent.morph.damping = agent.morph.damping
                    agent.morph.restLength = agent.morph.restLength
                    agent.inputLayer = agent.inputLayer
            agent.morph.spring.requires_grad = requires_grad_flag
            agent.morph.damping.requires_grad = requires_grad_flag
            agent.morph.restLength.requires_grad = requires_grad_flag
            agent.inputLayer.requires_grad = requires_grad_flag
            agent.control.amplitude.requires_grad = requires_grad_flag
            agent.control.omega.requires_grad = requires_grad_flag
            agent.control.phase.requires_grad = requires_grad_flag

            params_lr_list = [
                {"params": agent.morph.spring, "lr": lr_spring},
                {"params": agent.morph.damping, "lr": lr_damping},
                {"params": agent.morph.restLength, "lr": lr_restlength},
                {"params": agent.control.amplitude, "lr": lr_amplitude},
                {"params": agent.control.omega, "lr": lr_omega},
                {"params": agent.control.phase, "lr": lr_phase},
            ]
            optimizer = OPTIM_DICT[optim](params_lr_list)

            exForceTiming = torch.randint(updateTimeScale, (1, 1)).item()

            for j in range(updateTimeScale):
                if i == 0 and j == 0:
                    central_position = torch.zeros((2, 1))
                if (j == exForceTiming) and (lissajous2 is not None):
                    current_target_index = target_ind
                    if wind:
                        agent.morph.environment.setWind(int(wind_val * self.inputScale))
                    else:
                        agent.morph.environment.setWind(None)
                target_trajectory.append(
                    lissajous_targets[current_target_index][
                        updateTimeScale * i + j, :
                    ].squeeze()
                )
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber
                if (
                    (lissajous2 is None) and (i >= epoch - 20) and (i < epoch - 10)
                ) or (
                    (lissajous2 is not None)
                    and (i >= epoch - 200 - 40)
                    and (i < epoch - 200 - 20)
                ):
                    pos_his = (
                        torch.stack(agent.state.pos_history, dim=0)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    Xmax, Xmin, Ymax, Ymin = (
                        max(Xmax, np.max(pos_his[:, 0, :])),
                        min(Xmin, np.min(pos_his[:, 0, :])),
                        max(Ymax, np.max(pos_his[:, 1, :])),
                        min(Ymin, np.min(pos_his[:, 1, :])),
                    )
                if j == updateTimeScale - 1:
                    central_position = (
                        torch.stack(agent.state.centerNode_history, dim=0)
                        .reshape([-1, 4])
                        .t()[:2]
                        .to(torch.float32)
                    )
                    target_trajectory = torch.stack(target_trajectory, 0).T.to(
                        self.device
                    )

            agent.resetAgent()
            agent.control.target, agent.control.us = None, None
            if (i % 1000 == 0) and (i < trainingSteps):
                self.saveObjects(agent, simulenv, fileSaveDir, i)
            if i == trainingSteps - 1:
                self.saveObjects(agent, simulenv, fileSaveDir, "final")

            optimizer.zero_grad()
            loss = criterion(central_position, target_trajectory)
            if i % int(epoch / 10) == 0:
                central_position_history = central_position.cpu().detach()
            else:
                central_position_history = torch.cat(
                    (central_position_history, central_position.cpu().detach()), 1
                )
            if i % int(epoch / 10) == int(epoch / 10) - 1:
                saveNPY(
                    central_position_history.cpu().detach().numpy(),
                    fileSaveDir + f"trajectory_{int(i//(epoch/10))}.npy",
                )
                central_position_history = torch.zeros(
                    2, int(lissajous_target.shape[0] / 10)
                )
            if torch.sum(agent.control.amplitude != 0).item() != 0:
                params_history = print_current_params(
                    agent, i, loss.item(), "SWG", params_history
                )
            if i < trainingSteps:
                if requires_grad_flag:
                    try:
                        loss.backward(retain_graph=True)
                    except RuntimeError:
                        return
                    loss_history.append(loss.item())
                    if countdown_to_reduce_lr[0] != limit_to_reduce_lr:
                        if not countdown_to_reduce_lr[1]:
                            countdown_to_reduce_lr[1] = True
                        else:
                            countdown_to_reduce_lr = [limit_to_reduce_lr, False]
                    del loss
                    if i < trainingSteps - 1:
                        optimizer.step()
                else:
                    loss_history.append(loss.item())
                with torch.no_grad():
                    agent.state.pos = agent.state.pos.copy()
                    agent.state.speed = agent.state.speed.copy()
                    [
                        _spring,
                        _damping,
                        _inputLayer,
                        _restLength,
                        _amplitude,
                        _omega,
                        _phase,
                    ] = self.shape_tunableParameters(
                        agent, restLength=True, sineControl=True
                    )
            else:
                loss_history.append(loss.item())

        verletSim.endSimulation()
        epoch_array = np.arange(0, trainingSteps + evalSteps, 1)
        plotLines(
            [[epoch_array, loss_history, None, "tab:blue"]],
            "epoch",
            "loss",
            None,
            fileSaveDir + "loss.png",
            marker=False,
        )
        all_central_position_history = concatenate_files(fileSaveDir + "trajectory_")
        saveNPY(
            lissajous_target.cpu().detach().numpy(),
            fileSaveDir + "target_trajectory.npy",
        )
        saveNPY(np.array(loss_history), fileSaveDir + "loss.npy")
        trajs = [
            np.load(fileSaveDir + "trajectory_0.npy"),
            np.load(fileSaveDir + "trajectory_2.npy"),
            np.load(fileSaveDir + "trajectory_5.npy"),
            np.load(fileSaveDir + "trajectory_9.npy"),
        ]
        colors = ["red", "orange", "green", "blue"]
        labels = ["initial", "training(mid)", "training(final)", "test"]
        plotTrajectories(
            lissajous_target.squeeze().T.cpu().detach().numpy(),
            trajs,
            colors,
            labels,
            fileSaveDir + "trajectories.png",
        )
        colors = ["blue"]
        labels = ["output"]
        if lissajous2 is None:
            trajs = [all_central_position_history[:, -10 * updateTimeScale :]]
            plotTrajectories(
                lissajous_targets[0][-10 * updateTimeScale :]
                .squeeze()
                .T.cpu()
                .detach()
                .numpy(),
                trajs,
                colors,
                labels,
                fileSaveDir
                + f"trajectory_{sum(loss_history[-5:])/len(loss_history[-5:])}.png",
            )
        else:
            trajs = [
                all_central_position_history[
                    :, -120 * updateTimeScale : -100 * updateTimeScale
                ]
            ]
            plotTrajectories(
                lissajous_targets[0][-120 * updateTimeScale : -100 * updateTimeScale]
                .squeeze()
                .T.cpu()
                .detach()
                .numpy(),
                trajs,
                colors,
                labels,
                fileSaveDir + "trajectory_firstHalf.png",
            )
            trajs = [all_central_position_history[:, -20 * updateTimeScale :]]
            plotTrajectories(
                lissajous_targets[1][-20 * updateTimeScale :]
                .squeeze()
                .T.cpu()
                .detach()
                .numpy(),
                trajs,
                colors,
                labels,
                fileSaveDir + "trajectory_latterHalf.png",
            )
        if len(lr_changes) > 0:
            saveNPY(
                np.array(lr_changes),
                fileSaveDir + f"lr_changes_{len(lr_changes)}.npy",
            )
        cmap = plt.get_cmap("tab10")
        if self.loadFolder is None:
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
        npy2npz(
            [fileSaveDir + f"trajectory_{i}.npy" for i in range(10)],
            fileSaveDir,
            "trajectory",
            transpose=True,
        )

        print("The results were saved in ", fileSaveDir)

    def Lissajous_closedloop(
        self,
        updateTimeScale,
        generatorNum,
        wind,
        lissajous1,
        lissajous2,
        feedbackSprings,
        reservoirStateType="springLength",
        noise=0.08,
    ):
        """Realize closed-loop control for Lissajous drawing

        Args:
            updateTimeScale (int): Number of steps to update parameters
            generatorNum (list): Id of springs modulated by SWG
            wind (bool): Whether to introduce wind in the experiment
            lissajous1 (list): First target Lissajous curve's parameters
            lissajous2 (list): Second target Lissajous curve's parameters
            feedbackSprings (list): Distribution of springs used for feedback
            reservoirStateType (str, optional): Feedback information. Defaults to "springLength".
            noise (float, optional): Standard deviation of noise added to feedback information. Defaults to 0.08.
        """
        assert self.simTime % 10 == 0
        assert self.loadFolder is not None
        self.centralPointOutput = True
        simulLen = int(self.simTime / self.simTimeStep)
        [lissajous_a1, lissajous_b1, lissajous_delta_n1, lissajous_delta_d1] = [
            int(s) for s in lissajous1
        ]
        if lissajous2 is not None:
            [lissajous_a2, lissajous_b2, lissajous_delta_n2, lissajous_delta_d2] = [
                int(s) for s in lissajous2
            ]
        lissajous_targets = []
        lissajous_targets.append(
            LissajousTimeSeries(
                self.simTime,
                self.simTimeStep,
                self.initSize,
                a=lissajous_a1,
                b=lissajous_b1,
                delta=np.pi * lissajous_delta_n1 / lissajous_delta_d1,
            )
        )
        epoch = 1
        text_in_movie = False
        if lissajous2 is not None:
            target_x_offset = self.initSize / 20 if wind else 0
            lissajous_targets.append(
                LissajousTimeSeries(
                    self.simTime,
                    self.simTimeStep,
                    self.initSize,
                    target_x_offset,
                    a=lissajous_a2,
                    b=lissajous_b2,
                    delta=np.pi * lissajous_delta_n2 / lissajous_delta_d2,
                )
            )
            text_in_movie = True
        agent, simulenv = self.loadObjects()
        agent = self.set_requiresGrads(agent, False, generator=True)
        suffix = self.loadFileSuffix
        suffix += "_" + "-".join([key for key in feedbackSprings])
        suffix += f"_noise{noise}"
        fileSaveDir = self.make_fileSaveDir(
            self.loadFolder + "/closedloop/" + self.timeStamp, suffix
        )
        save_params_as_json(self.params, fileSaveDir)
        criterion = nn.MSELoss()
        iterationNumber = 0
        central_position_history = []
        targetComp = []
        w_opt_list = []
        Xmax, Xmin, Ymax, Ymin = 0, 0, 0, 0

        for i in range(-1, epoch):
            simulenv.plot.setMovie(False)
            if i == -1:
                target_ind, wind_val = 0, 0
            elif lissajous2 is None:
                target_ind, wind_val = 0, 0
            lissajous_target = lissajous_targets[target_ind]
            if wind:
                agent.morph.environment.setWind(int(wind_val * self.inputScale))
            else:
                agent.morph.environment.setWind(None)
            agent.state.resetSavePosType(self.savePosType)

            for j in range(simulLen):
                if lissajous2 is None:
                    target_ind, wind_val = 0, 0
                else:
                    if int(j / updateTimeScale) % 30 < 15:
                        target_ind, wind_val = 0, 0
                    else:
                        target_ind, wind_val = 1, 15
                if wind:
                    agent.morph.environment.setWind(int(wind_val * self.inputScale))
                else:
                    agent.morph.environment.setWind(None)
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber
                if i >= 0:
                    modulationFactor = (
                        agent.control.modulationFactor(agent.state, agent.brain) - 1.0
                    )
                    modulationFactor = modulationFactor[
                        np.nonzero(np.triu(agent.morph.connections))
                    ]
                    targetComp.append(modulationFactor)
                if i == -1:
                    pos_his = (
                        torch.stack(agent.state.pos_history, dim=0)
                        .to("cpu")
                        .detach()
                        .numpy()
                    )
                    Xmax, Xmin, Ymax, Ymin = (
                        max(Xmax, np.max(pos_his[:, 0, :])),
                        min(Xmin, np.min(pos_his[:, 0, :])),
                        max(Ymax, np.max(pos_his[:, 1, :])),
                        min(Ymin, np.min(pos_his[:, 1, :])),
                    )
                if j == simulLen - 1:
                    central_position = (
                        torch.stack(agent.state.centerNode_history, dim=0)
                        .reshape([-1, 4])
                        .t()[:2]
                    )

            loss = criterion(central_position, lissajous_target.squeeze().T)
            _ = print_current_params(agent, i, loss.item(), "SWG", None)

        verletSim.endSimulation()
        agent.resetAgent()
        targetComp = torch.stack(targetComp, dim=0)

        period_per_replacement = self.simTime
        if reservoirStateType == "massPosition":
            reservoirStatesDim = agent.movableNodes.shape[0] * 2 + 1
        elif reservoirStateType == "springLength":
            reservoirStatesDim = agent.morph.noComps + 1
        is_replaced = [False for _ in range(agent.morph.noComps)]
        if generatorNum != [-1]:
            for i in range(len(is_replaced)):
                if not i in generatorNum:
                    is_replaced[i] = True
        epoch = len(feedbackSprings)
        replaceNum = []
        for replace in feedbackSprings:
            if replace == "all":
                replaceNum.append([i for i in range(agent.morph.noComps)])
            elif replace == "ex":
                replaceNum.append([i for i in range(self.noNodes - 1)])
            elif replace == "mid":
                replaceNum.append(
                    [self.noNodes - 1]
                    + [i for i in range(self.noNodes, self.noNodes * 2 - 4, 2)]
                )
            elif replace == "in":
                replaceNum.append(
                    [i for i in range(self.noNodes + 1, self.noNodes * 2 - 3, 2)]
                    + [self.noNodes * 2 - 3]
                )
            elif (replace == "exmid") or (replace == "midex"):
                replaceNum.append(
                    [i for i in range(self.noNodes - 1)]
                    + [self.noNodes - 1]
                    + [i for i in range(self.noNodes, self.noNodes * 2 - 4, 2)]
                )
            elif (replace == "exin") or (replace == "inex"):
                replaceNum.append(
                    [i for i in range(self.noNodes - 1)]
                    + [i for i in range(self.noNodes + 1, self.noNodes * 2 - 3, 2)]
                    + [self.noNodes * 2 - 3]
                )
            elif (replace == "midin") or (replace == "inmid"):
                replaceNum.append(
                    [self.noNodes - 1]
                    + [i for i in range(self.noNodes, self.noNodes * 2 - 4, 2)]
                    + [i for i in range(self.noNodes + 1, self.noNodes * 2 - 3, 2)]
                    + [self.noNodes * 2 - 3]
                )
            else:
                replaceOtherThan = int(replace)
                appendList = [i for i in range(agent.morph.noComps)]
                appendList.remove(replaceOtherThan)
                replaceNum.append(appendList)
        compNum = -1
        allNodesPos = []
        central_position_history = []
        losses = []
        lissajous_target = (
            lissajous_targets[0]
            .squeeze()
            .T[:, : period_per_replacement * updateTimeScale]
        )
        agent.control.initFeedbackLoop(agent.morph.noNodes, reservoirStatesDim)
        simulenv.plot.colored_comp[0] = generatorNum

        for i in range(-1, epoch):
            nodesPos, reservoirStates = [], []
            _mses_with_ridgeRegression = []
            losses_in_epoch = []
            for j in range(period_per_replacement * updateTimeScale):
                if lissajous2 is None:
                    target_ind, wind_val = 0, 0
                else:
                    if int(j / updateTimeScale) % 30 < 15:
                        target_ind, wind_val = 0, 0
                    else:
                        target_ind, wind_val = 1, 15
                if j == (period_per_replacement - 22) * updateTimeScale:
                    simulenv.plot.setMovie(
                        fileSaveDir,
                        num=i,
                        only_central_red=True,
                        text=text_in_movie,
                        colored_comp=simulenv.plot.colored_comp,
                    )
                    simulenv.plot.setRange(Xmax, Xmin, Ymax, Ymin)
                elif j == (period_per_replacement - 2) * updateTimeScale:
                    verletSim.simulEnv.end()
                    plt.clf()
                    plt.close()
                    simulenv.plot.setMovie(False)
                    _agent, _simulenv = copy.deepcopy(agent), copy.deepcopy(simulenv)
                    _simulenv.plot.lightenVar()
                    _simulenv.plot.setFigure()
                    _agent.control.target, _agent.control.us = None, None
                    _agent.resetAgent()
                    self.saveObjects(_agent, _simulenv, fileSaveDir, i)
                    if i != epoch - 1:
                        simulenv.plot.setMovie(
                            fileSaveDir,
                            num=f"{i}to{i+1}",
                            only_central_red=True,
                            text=text_in_movie,
                            colored_comp=simulenv.plot.colored_comp,
                        )
                        simulenv.plot.setRange(Xmax, Xmin, Ymax, Ymin)
                    else:
                        simulenv.plot.setMovie(False)
                elif i >= 0:
                    if j == 0:
                        simulenv.plot.setFigure()
                    elif j == 2 * updateTimeScale:
                        verletSim.simulEnv.end()
                        plt.clf()
                        plt.close()
                        simulenv.plot.setMovie(False)
                lissajous_target = (
                    lissajous_targets[target_ind]
                    .squeeze()
                    .T[:, : period_per_replacement * updateTimeScale]
                )
                if wind:
                    agent.morph.environment.setWind(int(wind_val * self.inputScale))
                else:
                    agent.morph.environment.setWind(None)
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber
                if j % updateTimeScale == updateTimeScale - 1:
                    central_position = (
                        torch.stack(agent.state.centerNode_history, dim=0)
                        .reshape([-1, 4])
                        .t()[:2]
                    )
                    _losses = []
                    for _step in range(101):
                        loss = criterion(
                            central_position[:, -updateTimeScale:],
                            lissajous_target[:, _step : _step + updateTimeScale],
                        )
                        _losses.append(loss.item())
                    loss = min(_losses)
                    losses_in_epoch.append(loss)
                nodesPosData = torch.cat(
                    (
                        torch.flatten(agent.state.pos.x.cpu()).gather(
                            0, agent.movableNodes
                        ),
                        torch.flatten(agent.state.pos.y.cpu()).gather(
                            0, agent.movableNodes
                        ),
                    )
                )
                nodesPos.append(nodesPosData)
                if reservoirStateType == "massPosition":
                    reservoirState = torch.cat(
                        (
                            torch.ones(1, dtype=torch.float64),
                            torch.flatten(agent.state.pos.x.cpu()).gather(
                                0, agent.movableNodes
                            ),
                            torch.flatten(agent.state.pos.y.cpu()).gather(
                                0, agent.movableNodes
                            ),
                        )
                    )
                elif reservoirStateType == "springLength":
                    difx, dify = agent.state.pos.getDifference()
                    difxy = torch.sqrt(difx**2 + dify**2 + 1e-16) + torch.eye(
                        agent.morph.noNodes
                    )
                    actualLength = difxy * agent.morph.connections
                    reservoirState = torch.cat(
                        (
                            torch.ones(1, dtype=torch.float64),
                            actualLength[np.nonzero(np.triu(actualLength))],
                        )
                    )
                noiseTensor = torch.normal(
                    mean=0, std=noise, size=(reservoirState.shape)
                )
                reservoirStates.append(reservoirState[1:] + noiseTensor[1:])
                modulationFactor = (
                    agent.control.modulationFactor(agent.state, agent.brain) - 1.0
                )
                modulationFactor = modulationFactor[
                    np.nonzero(np.triu(agent.morph.connections))
                ]
                if i != -1:
                    if j == 0:
                        for compNum in replaceNum[i]:
                            is_replaced[compNum] = True
                            print(f"==== No.{compNum} comp is replaced ====")
                            agent.control.setFeedbackLayers(
                                np.nonzero(np.triu(agent.morph.connections))[0][
                                    compNum
                                ],
                                np.nonzero(np.triu(agent.morph.connections))[1][
                                    compNum
                                ],
                                w_opts[compNum],
                            )
                            simulenv.plot.colored_comp[1].append(compNum)
                    agent.control.setFeedbackFactor(reservoirState)
            central_position = (
                torch.stack(agent.state.centerNode_history, dim=0)
                .reshape([-1, 4])
                .t()[:2]
            )
            agent.state.centerNode_history = []
            central_position_history.append(central_position)
            losses.append(losses_in_epoch)

            nodesPos = torch.stack(nodesPos, dim=0)
            allNodesPos.append(nodesPos.T)
            reservoirStates = torch.stack(reservoirStates, dim=0)
            if i != -1:
                replacing_dynamics_after = nodesPos[: 5 * updateTimeScale, :]
                replacing_dynamics = torch.cat(
                    (replacing_dynamics_before, replacing_dynamics_after), 0
                )
                movableNodeNum = int(nodesPos.shape[1] / 2)
                t_array = np.arange(
                    -replacing_dynamics.shape[0] / 2 * self.simTimeStep,
                    replacing_dynamics.shape[0] / 2 * self.simTimeStep,
                    self.simTimeStep,
                )
                dynamics_x = [
                    [t_array, replacing_dynamics[:, n].detach().cpu(), None, "black"]
                    for n in range(movableNodeNum)
                ]
                dynamics_y = [
                    [
                        t_array,
                        replacing_dynamics[:, movableNodeNum + n].detach().cpu(),
                        None,
                        "black",
                    ]
                    for n in range(movableNodeNum)
                ]
                dynamics_x[-1][-1] = dynamics_y[-1][-1] = "tab:red"
                plotLines(
                    dynamics_x,
                    "t",
                    "x",
                    None,
                    fileSaveDir + f"dynamics_x_{i-1}to{i}.png",
                    False,
                    vline=0,
                )
                plt.clf()
                plt.close()
                plotLines(
                    dynamics_y,
                    "t",
                    "y",
                    None,
                    fileSaveDir + f"dynamics_y_{i-1}to{i}.png",
                    False,
                    vline=0,
                )
                plt.clf()
                plt.close()
            replacing_dynamics_before = nodesPos[-5 * updateTimeScale :, :]
            df_ridgeparam_tmp = []
            for idx, target in enumerate(targetComp.T):
                if is_replaced[idx]:
                    _mses_with_ridgeRegression.append(1)
                    w_opt_list.append(
                        torch.zeros(reservoirStatesDim, dtype=torch.float64)
                    )
                    df_ridgeparam_tmp.append(np.zeros(2))
                else:
                    agent.optimizer_ridge.X = reservoirStates.T[
                        :, int(simulLen / 5) : int(simulLen * 4 / 5)
                    ]
                    agent.optimizer_ridge.test_X = reservoirStates.T[
                        :, int(simulLen * 4 / 5) :
                    ]
                    agent.optimizer_ridge.Y = target[
                        int(simulLen / 5) : int(simulLen * 4 / 5)
                    ]
                    trainStates, testStates, pred, w_opt, df_ridgeparam = (
                        self.ridgeRegression(agent)
                    )
                    mse = MSE(testStates, target[int(simulLen * 4 / 5) :]).item()
                    _mses_with_ridgeRegression.append(mse)
                    w_opt_list.append(w_opt)
                    df_ridgeparam_tmp.append(df_ridgeparam)
                    print(
                        f"{i+2}/{epoch+1}:  MSE when No.{idx} comp is replaced: {mse}"
                    )
            w_opts = torch.stack(w_opt_list, dim=0)
            w_opt_list = []

        verletSim.endSimulation()
        plt.clf()
        plt.close()
        agent.resetAgent()

        if lissajous2 is None:
            trajs = central_position_history
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i) for i in range(len(trajs))]
            replacedPart = []
            for i in range(len(feedbackSprings)):
                replacedPart.append("-".join([key for key in feedbackSprings[: i + 1]]))
            labels = ["SWG"] + [f"Closed loop ({part})" for part in replacedPart]
            plotTrajectories(
                lissajous_targets[0]
                .squeeze()
                .T.cpu()
                .detach()
                .numpy()[:, : period_per_replacement * updateTimeScale],
                trajs,
                colors,
                labels,
                fileSaveDir + "trajectories.png",
            )

        agent.control.target, agent.control.us = None, None
        self.saveObjects(agent, simulenv, fileSaveDir, "final")
        losses = np.array(losses)
        saveNPY(losses, fileSaveDir + "loss.npy")
        time_array = np.arange(0, losses.flatten().shape[0], 1)
        lossFigName = "loss"
        for i in range(losses.shape[0]):
            averageLoss = np.mean(losses[i])
            lossFigName += "_{:.06f}".format(averageLoss)
        plotLines(
            [[time_array, losses.flatten(), None, "tab:blue"]],
            "time [sec]",
            "MSE",
            None,
            fileSaveDir + lossFigName + ".png",
            marker=False,
        )
        saveNPY(losses, fileSaveDir + "loss.npy")
        allNodesPos = torch.stack(allNodesPos)
        allNodesPosX = allNodesPos[:, : int(allNodesPos.shape[1] / 2), :]
        allNodesPosY = allNodesPos[:, int(allNodesPos.shape[1] / 2) :, :]
        np.savez_compressed(
            fileSaveDir + "dynamics.npz",
            x=allNodesPosX.detach().cpu().numpy(),
            y=allNodesPosY.detach().cpu().numpy(),
        )

        print("The results were saved in ", fileSaveDir)

    def Lissajous_perturbation_process(
        self,
        simTime,
        suffix,
        seed_initpos_num,
        updateTimeScale,
        lissajous_target,
        error_threshold,
        loss_baseline,
        fileSaveDir,
        all_losses,
        all_central_positions,
        all_return_rate,
        reservoirStateType,
    ):
        """Individual process to check robustness for Lissajous drawing

        Args:
            simTime (int): Simulation time
            suffix (str): Suffix of the file to load
            seed_initpos_num (int): Range of random seed to determine initial mass point positions
            updateTimeScale (int): Number of steps to update parameters
            lissajous_target (torch.tensor): Target Lissajous trajectories
            error_threshold (float): Threshold for successful return
            loss_baseline (np.array): Baseline loss of the system driven by SWG
            fileSaveDir (str): Path to result folder
            all_losses (list): List for storing losses
            all_central_positions (list): List for storing positions of the central mass point
            all_return_rate (list): List for storing return rates
            reservoirStateType (str): Feedback information
        """
        agent = self.loadAgent(suffix, keepStates=True)
        _, simulenv = self.initSet(
            self.initSize,
            self.tau,
            _simtime=self.simTime,
            centerFlag=True,
            bias=True,
            controlType="sine",
            generatorNum=[-1],
            amplitude=0.5,
            noise=0.08,
        )
        (
            losses_for_each_suffix,
            central_positions_for_each_suffix,
            return_rate_for_each_suffix,
        ) = ([], [], [])
        for std in range(0, 11):
            std /= 10
            losses_for_each_std, central_positions_for_each_std = [], []
            success, fail = 0, 0

            for random_seed in range(seed_initpos_num):
                agent = self.loadAgent(suffix, keepStates=True)
                loss_history, central_position_history = [], None
                iterationNumber = 0
                if self.shape == "MultiCircles":
                    assert math.sqrt(self.noNodes - 1).is_integer()
                    pixelNum = int(math.sqrt(self.noNodes - 1))
                    deltaPosMask = torch.ones((2, self.noNodes))
                    deltaPosMask[:, : 4 * pixelNum - 4] = torch.zeros(
                        (2, 4 * pixelNum - 4)
                    )
                elif self.shape == "DoubleCircles":
                    deltaPosMask = torch.zeros(2, self.noNodes).to(self.device)
                    deltaPosMask[:, int((self.noNodes - 1) / 2) :] = torch.ones(
                        2, int((self.noNodes + 1) / 2)
                    ).to(self.device)
                torch.manual_seed(random_seed)
                deltaPos = torch.normal(mean=0, std=std, size=(2, self.noNodes))
                deltaPos = torch.mul(deltaPosMask, deltaPos)
                agent.setdeltaPos(deltaPos)
                for i in range(simTime):
                    agent.state.resetSavePosType(self.savePosType)
                    for _ in range(updateTimeScale):
                        agent.control.set_for_limitcycle(
                            agent.morph, self.inputScale, None
                        )
                        verletSim = VerletSimulation(simulenv, agent)
                        verletSim.iterationNumber = iterationNumber
                        verletSim.runSimulationOneStep()
                        iterationNumber = verletSim.iterationNumber
                        if reservoirStateType == "massPosition":
                            reservoirState = torch.cat(
                                (
                                    torch.ones(1, dtype=torch.float64),
                                    torch.flatten(agent.state.pos.x.cpu()).gather(
                                        0, agent.movableNodes
                                    ),
                                    torch.flatten(agent.state.pos.y.cpu()).gather(
                                        0, agent.movableNodes
                                    ),
                                )
                            )
                        elif reservoirStateType == "springLength":
                            difx, dify = agent.state.pos.getDifference()
                            difxy = torch.sqrt(
                                difx**2 + dify**2 + 1e-16
                            ) + torch.eye(agent.morph.noNodes)
                            actualLength = difxy * agent.morph.connections
                            reservoirState = torch.cat(
                                (
                                    torch.ones(1, dtype=torch.float64),
                                    actualLength[np.nonzero(np.triu(actualLength))],
                                )
                            )
                        agent.control.setFeedbackFactor(reservoirState)

                    central_position = (
                        torch.stack(agent.state.centerNode_history, dim=0)
                        .reshape([-1, 4])
                        .t()[:2]
                    )
                    if central_position_history is None:
                        central_position_history = central_position
                    else:
                        central_position_history = torch.cat(
                            (central_position_history, central_position), 1
                        )
                    _losses = []
                    for _step in range(101):
                        loss = nn.MSELoss()(
                            central_position,
                            lissajous_target[_step : _step + updateTimeScale, :]
                            .squeeze()
                            .T,
                        )
                        _losses.append(loss.item())
                    loss = min(_losses)
                    loss_history.append(loss)
                    if i == simTime - 1:
                        print(
                            "suffix: {} | std: {} | seed: {} | loss = {:.06f}".format(
                                suffix,
                                std,
                                random_seed,
                                sum(loss_history[-3:]) / len(loss_history[-3:]),
                            )
                        )

                verletSim.endSimulation()
                agent.resetAgent()
                plt.clf()
                plt.close()
                loss_history_last3 = loss_history[-3:]
                if (
                    sum(loss_history_last3) / len(loss_history_last3)
                    < loss_baseline[suffix + 1] * error_threshold
                ):
                    success += 1
                    if success <= 3:
                        plotTrajectory_timeseries(
                            lissajous_target.squeeze().T.cpu().detach().numpy(),
                            central_position_history,
                            self.simTime,
                            fileSaveDir
                            + f"successful_trajectory_{suffix}_std{std}_seed{random_seed}.png",
                        )
                else:
                    fail += 1
                    if fail <= 3:
                        plotTrajectory_timeseries(
                            lissajous_target.squeeze().T.cpu().detach().numpy(),
                            central_position_history,
                            self.simTime,
                            fileSaveDir
                            + f"failed_trajectory_{suffix}_std{std}_seed{random_seed}.png",
                        )
                losses_for_each_std.append(loss_history)
                central_positions_for_each_std.append(central_position_history)
                plt.clf()
                plt.close()

            losses_for_each_suffix.append(losses_for_each_std)
            central_positions_for_each_suffix.append(
                torch.stack(central_positions_for_each_std, 0)
            )
            return_rate_for_each_suffix.append(success / seed_initpos_num * 100)

        all_losses[suffix + 1] = losses_for_each_suffix
        all_central_positions[suffix + 1] = torch.stack(
            central_positions_for_each_suffix, 0
        ).numpy()
        all_return_rate[suffix + 1] = return_rate_for_each_suffix

    def Lissajous_perturbation(
        self,
        updateTimeScale,
        lissajous1,
        feedbackSprings,
        reservoirStateType,
    ):
        """Check robustness for Lissajous drawing

        Args:
            updateTimeScale (int): Number of steps to update parameters
            lissajous1 (list): irst target Lissajous curve's parameters
            feedbackSprings (list): Distribution of springs used for feedback
            reservoirStateType (str): Feedback information
        """
        self.centralPointOutput = True
        [lissajous_a1, lissajous_b1, lissajous_delta_n1, lissajous_delta_d1] = [
            int(s) for s in lissajous1
        ]
        loss_baseline = self.loadLossMean()
        lissajous_target = LissajousTimeSeries(
            self.simTime,
            self.simTimeStep,
            self.initSize,
            a=lissajous_a1,
            b=lissajous_b1,
            delta=np.pi * lissajous_delta_n1 / lissajous_delta_d1,
        )
        fileSaveDir = self.make_fileSaveDir(
            self.loadFolder + "/analysis/" + self.timeStamp, None
        )
        save_params_as_json(self.params, fileSaveDir)
        seed_initpos_num = 10 #100
        error_threshold = 2.0

        agentNum = 0
        while True:
            if os.path.isfile(self.loadFolder + f"agent_{agentNum-1}.pickle"):
                agentNum += 1
            else:
                break

        processes = []
        manager = Manager()
        all_losses = manager.list([None for _ in range(agentNum)])
        all_central_positions = manager.list([None for _ in range(agentNum)])
        all_return_rate = manager.list([None for _ in range(agentNum)])

        mp.set_start_method("spawn", force=True)
        for suffix in range(-1, agentNum - 1):
            assert os.path.isfile(self.loadFolder + f"agent_{suffix}.pickle")
            p = Process(
                target=self.Lissajous_perturbation_process,
                args=(
                    self.simTime,
                    suffix,
                    seed_initpos_num,
                    updateTimeScale,
                    lissajous_target,
                    error_threshold,
                    loss_baseline,
                    fileSaveDir,
                    all_losses,
                    all_central_positions,
                    all_return_rate,
                    reservoirStateType,
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
                os.system("taskset -p -c %d-%d %d" % (0, cpu_count() - 1, p.pid))

        for p in processes:
            p.join()

        all_losses = np.array(all_losses)
        all_mean_losses, clustered_losses = [], []
        for losses in all_losses:
            mean_losses = np.mean(losses[:, :, -3:], axis=-1, keepdims=True).flatten()
            minval, maxval = mean_losses.min(), mean_losses.max()
            kde = KernelDensity(kernel="gaussian", bandwidth=0.01).fit(
                mean_losses.reshape(-1, 1)
            )
            s = np.linspace(minval, maxval, 1024)
            e = kde.score_samples(s.reshape(-1, 1))
            mi = argrelextrema(e, np.less)[0]
            mean_losses_class = copy.deepcopy(mean_losses)
            for i in range(0, len(s[mi])):
                mean_losses_class = np.where(
                    (mean_losses >= minval) & (mean_losses <= s[mi][i]),
                    i,
                    mean_losses_class,
                )
                minval = s[mi][i]
            mean_losses_class = np.where(
                mean_losses >= minval, len(s[mi]), mean_losses_class
            )
            mean_losses = mean_losses.reshape(losses.shape[:-1])
            all_mean_losses.append(mean_losses)
            mean_losses_class = mean_losses_class.reshape(losses.shape[:-1])
            clustered_losses.append(mean_losses_class)
        all_mean_losses = np.stack(all_mean_losses, 0)
        clustered_losses = np.stack(clustered_losses, 0)
        clusterNum = int(np.max(clustered_losses)) + 1
        saveNPY(
            np.stack([all_mean_losses, clustered_losses], 0),
            fileSaveDir + f"clustered_losses_{clusterNum}clusters.npy",
        )
        all_central_positions = np.stack(all_central_positions, 0)
        saveNPY(np.array(all_return_rate), fileSaveDir + "return_rate.npy")
        std_list = [i / 10 for i in range(0, len(all_return_rate[0]))]
        cmap = plt.get_cmap("tab10")
        replacedPart = []
        for i in range(len(feedbackSprings)):
            replacedPart.append("-".join([key for key in feedbackSprings[: i + 1]]))
        labels = ["SWG"] + [f"Closed loop ({part})" for part in replacedPart]
        result = [
            [std_list, all_return_rate[j], labels[j], cmap(j)]
            for j in range(len(feedbackSprings) + 1)
        ]
        plotLines(
            result,
            "std",
            "Return rate",
            None,
            fileSaveDir + "return_rate.png",
            marker=False,
        )
        np.savez_compressed(
            fileSaveDir + "target_losses_centralPositions.npz",
            target=lissajous_target.cpu().detach().numpy(),
            losses=all_losses,
            centralPositions=all_central_positions,
        )

        print("The results were saved in ", fileSaveDir)

    def Lissajous_switching_dynamics(
        self, generatorNum, inputScale, wind, lissajous1, lissajous2
    ):
        """Analyze the dynamics in switching Lissajous curves

        Args:
            generatorNum (list): Id of springs modulated by SWG
            inputScale (int): Input value scaling
            wind (bool): Whether to introduce wind in the experiment
            lissajous1 (list): First target Lissajous curve's parameters
            lissajous2 (list): Second target Lissajous curve's parameters
        """
        self.centralPointOutput = True
        [lissajous_a1, lissajous_b1, lissajous_delta_n1, lissajous_delta_d1] = [
            int(s) for s in lissajous1
        ]
        if lissajous2 is not None:
            [lissajous_a2, lissajous_b2, lissajous_delta_n2, lissajous_delta_d2] = [
                int(s) for s in lissajous2
            ]
        self.inputScale = inputScale
        washout_time = 10
        fixed_force_time = 5
        repetition = 5
        fixed_force_time_long = 100
        self.simTime = epoch = (
            washout_time
            + (fixed_force_time * 3 * repetition)
            + (3 * fixed_force_time_long)
        )
        updateTimeScale = 100
        lissajous_targets = []
        lissajous_targets.append(
            LissajousTimeSeries(
                self.simTime,
                self.simTimeStep,
                self.initSize,
                a=lissajous_a1,
                b=lissajous_b1,
                delta=np.pi * lissajous_delta_n1 / lissajous_delta_d1,
            )
        )
        if lissajous2 is not None:
            target_x_offset = self.initSize / 20 if wind else 0
            lissajous_targets.append(
                LissajousTimeSeries(
                    self.simTime,
                    self.simTimeStep,
                    self.initSize,
                    target_x_offset,
                    a=lissajous_a2,
                    b=lissajous_b2,
                    delta=np.pi * lissajous_delta_n2 / lissajous_delta_d2,
                )
            )
        lissajous_targets = torch.stack(lissajous_targets, 0)
        criterion = nn.MSELoss()
        agent, simulenv = self.loadObjects()
        fileSaveDir = self.make_fileSaveDir(
            self.loadFolder + "/analysis/" + self.timeStamp, self.loadFileSuffix
        )
        iterationNumber = 0
        loss_history = []
        agent = self.set_requiresGrads(agent, False, generator=True)
        Xmax, Xmin, Ymax, Ymin = 0, 0, 0, 0
        simulenv.plot.colored_comp[0] = generatorNum
        movableNodes_history = None

        ## trained behavior
        for i in range(epoch):
            if i < washout_time:
                target_ind, wind_val = 0, 0
            else:
                if i == washout_time + (fixed_force_time * 2 * (repetition - 2)):
                    simulenv.plot.setMovie(
                        fileSaveDir,
                        num="trained_switching",
                        only_central_red=True,
                        colored_comp=simulenv.plot.colored_comp,
                        text=True,
                        trajectory=True,
                    )
                    simulenv.plot.setRange(Xmax, Xmin, Ymax, Ymin)
                    movableNodes_history = []
                if i == washout_time + (fixed_force_time * 2 * repetition):
                    verletSim.simulEnv.end()
                    simulenv.plot.setMovie(False)
                if i < washout_time + (fixed_force_time * 2 * repetition):
                    if (i - washout_time) % (fixed_force_time * 2) < fixed_force_time:
                        target_ind, wind_val = 0, 0
                    else:
                        target_ind, wind_val = 1, 15
                else:
                    if (i - (washout_time + (fixed_force_time * 2 * repetition))) % (
                        fixed_force_time_long * 2
                    ) < fixed_force_time_long:
                        target_ind, wind_val = 0, 0
                    else:
                        target_ind, wind_val = 1, 15
            lissajous_target = lissajous_targets[target_ind]
            if wind:
                agent.morph.environment.setWind(wind_val * self.inputScale)
            else:
                agent.morph.environment.setWind(None)
            agent.state.resetSavePosType(self.savePosType)

            for j in range(updateTimeScale):
                if i == 0 and j == 0:
                    central_position = torch.zeros((2, 1))
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber
                if movableNodes_history is not None:
                    movableNodes = torch.cat(
                        (
                            torch.flatten(agent.state.pos.x.cpu())
                            .gather(0, agent.movableNodes)
                            .unsqueeze(0),
                            torch.flatten(agent.state.pos.y.cpu())
                            .gather(0, agent.movableNodes)
                            .unsqueeze(0),
                        ),
                        0,
                    )
                    movableNodes_history.append(movableNodes)
                if i == washout_time - 1:
                    pos_his = (
                        torch.stack(agent.state.pos_history, dim=0).detach().numpy()
                    )
                    Xmax, Xmin, Ymax, Ymin = (
                        max(Xmax, np.max(pos_his[:, 0, :])),
                        min(Xmin, np.min(pos_his[:, 0, :])),
                        max(Ymax, np.max(pos_his[:, 1, :])),
                        min(Ymin, np.min(pos_his[:, 1, :])),
                    )
                if j == updateTimeScale - 1:
                    central_position = (
                        torch.stack(agent.state.centerNode_history, dim=0)
                        .reshape([-1, 4])
                        .t()[:2]
                        .to(torch.float64)
                    )

            loss = criterion(
                central_position,
                lissajous_target[updateTimeScale * i : updateTimeScale * (i + 1), :]
                .squeeze()
                .T,
            )
            if i == 0:
                central_position_history = central_position.cpu().detach()
            else:
                central_position_history = torch.cat(
                    (central_position_history, central_position.cpu().detach()), 1
                )
            if i == epoch - 1:
                saveNPY(
                    central_position_history.cpu().detach().numpy(),
                    fileSaveDir + "trajectory.npy",
                )
            _ = print_current_params(agent, i, loss.item(), "SWG", None)
            loss_history.append(loss.item())

        verletSim.endSimulation()
        movableNodes_history = torch.stack(movableNodes_history, 0)
        epoch_array = np.arange(0, epoch, 1)
        plotLines(
            [[epoch_array, loss_history, None, "tab:blue"]],
            "epoch",
            "loss",
            None,
            fileSaveDir + "loss.png",
            marker=False,
        )
        all_central_position_history = central_position_history
        save_params_as_json(self.params, fileSaveDir)
        saveNPY(np.array(loss_history), fileSaveDir + "loss.npy")
        t_array = np.arange(
            washout_time + (fixed_force_time * 2 * (repetition - 2)),
            washout_time + (fixed_force_time * 2 * repetition),
            self.simTimeStep,
        )
        colors = ["blue"]
        labels = ["output"]
        trajs = [
            all_central_position_history[
                :,
                -(fixed_force_time_long + 10)
                * updateTimeScale : -fixed_force_time_long
                * updateTimeScale,
            ]
        ]
        plotTrajectories(
            lissajous_targets[0][
                -(fixed_force_time_long + 10)
                * updateTimeScale : -fixed_force_time_long
                * updateTimeScale
            ]
            .squeeze()
            .T.cpu()
            .detach()
            .numpy(),
            trajs,
            colors,
            labels,
            fileSaveDir + "trajectory_firstHalf.png",
        )
        trajs = [all_central_position_history[:, -10 * updateTimeScale :]]
        plotTrajectories(
            lissajous_targets[1][-10 * updateTimeScale :]
            .squeeze()
            .T.cpu()
            .detach()
            .numpy(),
            trajs,
            colors,
            labels,
            fileSaveDir + "trajectory_latterHalf.png",
        )
        dynamics_x = [
            [
                t_array,
                movableNodes_history[: t_array.shape[0], 0, n].detach().cpu(),
                None,
                "black",
            ]
            for n in range(movableNodes_history.shape[-1])
        ]
        dynamics_y = [
            [
                t_array,
                movableNodes_history[: t_array.shape[0], 1, n].detach().cpu(),
                None,
                "black",
            ]
            for n in range(movableNodes_history.shape[-1])
        ]
        dynamics_x[-1][-1] = dynamics_y[-1][-1] = "tab:red"
        boundary = [
            washout_time + (fixed_force_time * 2 * (repetition - 2)),
            washout_time + (fixed_force_time * 2 * (repetition - 2)) + fixed_force_time,
            washout_time
            + (fixed_force_time * 2 * (repetition - 2))
            + (fixed_force_time * 2),
            washout_time
            + (fixed_force_time * 2 * (repetition - 2))
            + (fixed_force_time * 3),
            washout_time
            + (fixed_force_time * 2 * (repetition - 2))
            + (fixed_force_time * 4),
        ]
        plotLines2ax(
            dynamics_x,
            dynamics_y,
            ["x", "y"],
            fileSaveDir + "switching_dynamics",
            boundary=boundary,
        )
        np.savez_compressed(
            fileSaveDir + "trained_dynamics.npz",
            target=lissajous_targets.squeeze().cpu().detach().numpy(),
            central=all_central_position_history.detach().cpu().numpy(),
            movable=movableNodes_history.detach().cpu().numpy(),
        )

        simulenv.plot.setMovie(False)
        (
            movableNodes_history_for_movie,
            movableNodes_history,
            movableNodes_history_up,
            movableNodes_history_down,
        ) = (None, None, None, None)
        fixed_force_time = 300
        gradation_scale = 100
        fixed_force_time_in_gradation_for_movie = 1
        force_gradation_time_for_movie = fixed_force_time_in_gradation_for_movie * (
            gradation_scale * 3 + 1
        )
        fixed_force_time_in_gradation = 300
        force_gradation_time = fixed_force_time_in_gradation * (gradation_scale * 3 + 1)
        self.simTime = epoch = (
            (washout_time * 3)
            + fixed_force_time
            + force_gradation_time_for_movie
            + force_gradation_time
        )
        gradationForces = np.arange(-1, 2 + (1 / gradation_scale), 1 / gradation_scale)
        gradationCount = 0
        plotPeriod = 100
        lissajous_targets = []
        lissajous_targets.append(
            LissajousTimeSeries(
                self.simTime,
                self.simTimeStep,
                self.initSize,
                a=lissajous_a1,
                b=lissajous_b1,
                delta=np.pi * lissajous_delta_n1 / lissajous_delta_d1,
            )
        )
        if lissajous2 is not None:
            target_x_offset = self.initSize / 20 if wind else 0
            lissajous_targets.append(
                LissajousTimeSeries(
                    self.simTime,
                    self.simTimeStep,
                    self.initSize,
                    target_x_offset,
                    a=lissajous_a2,
                    b=lissajous_b2,
                    delta=np.pi * lissajous_delta_n2 / lissajous_delta_d2,
                )
            )

        ## untrained behavior (half of trained force & gradation force)
        for i in range(epoch):
            if i < washout_time:
                target_ind, wind_val = 0, 0
            else:
                if i == washout_time:
                    simulenv.plot.setMovie(False)
                    target_ind, wind_val = 1, 15 / 2
                    print("----------- washout -> untrained behavior -----------")
                elif i == washout_time + fixed_force_time:
                    target_ind, wind_val = (0, -15 - (15 / gradation_scale))
                    print("----------- untrained behavior -> washout -----------")
                elif (i >= (washout_time * 2) + fixed_force_time) and (
                    i
                    < (washout_time * 2)
                    + fixed_force_time
                    + force_gradation_time_for_movie
                ):
                    if gradationCount == 0:
                        simulenv.plot.setMovie(
                            fileSaveDir,
                            num="gradation",
                            only_central_red=True,
                            colored_comp=simulenv.plot.colored_comp,
                            text=True,
                            trajectory=True,
                        )
                        simulenv.plot.setRange(Xmax, Xmin, Ymax, Ymin)
                        movableNodes_history_for_movie = []
                        print("----------- washout -> gradation (short) -----------")
                    if gradationCount % fixed_force_time_in_gradation_for_movie == 0:
                        wind_val += 15 / gradation_scale
                    gradationCount += 1
                elif (
                    i
                    == (washout_time * 2)
                    + fixed_force_time
                    + force_gradation_time_for_movie
                ):
                    verletSim.simulEnv.end()
                    plt.clf()
                    plt.close()
                    simulenv.plot.setMovie(False)
                    target_ind, wind_val = (0, -15 - (15 / gradation_scale))
                    gradationCount = 0
                    print("----------- gradation (short) -> washout -----------")
                elif (
                    i
                    >= (washout_time * 3)
                    + fixed_force_time
                    + force_gradation_time_for_movie
                ):
                    if gradationCount == 0:
                        movableNodes_history, movableNodes_history_per_force = (
                            [],
                            [],
                        )
                        print("----------- washout -> gradation (long) -----------")
                    if gradationCount % fixed_force_time_in_gradation == 0:
                        wind_val += 15 / gradation_scale
                        movableNodes_history_per_force = []
                    gradationCount += 1
            lissajous_target = lissajous_targets[target_ind]
            if wind:
                agent.morph.environment.setWind(wind_val * self.inputScale)
            else:
                agent.morph.environment.setWind(None)
            agent.state.resetSavePosType(self.savePosType)

            for j in range(updateTimeScale):
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber
                movableNodes = torch.cat(
                    (
                        torch.flatten(agent.state.pos.x.cpu())
                        .gather(0, agent.movableNodes)
                        .unsqueeze(0),
                        torch.flatten(agent.state.pos.y.cpu())
                        .gather(0, agent.movableNodes)
                        .unsqueeze(0),
                    ),
                    0,
                )
                if (movableNodes_history is not None) and (
                    i
                    >= (washout_time * 3)
                    + fixed_force_time
                    + force_gradation_time_for_movie
                ):
                    movableNodes_history_per_force.append(movableNodes)
                    if len(movableNodes_history_per_force) == int(
                        fixed_force_time_in_gradation / self.simTimeStep
                    ):
                        movableNodes_history.append(
                            torch.stack(movableNodes_history_per_force, 0)
                        )
                elif (movableNodes_history_for_movie is not None) and (
                    i
                    < (washout_time * 2)
                    + fixed_force_time
                    + force_gradation_time_for_movie
                ):
                    movableNodes_history_for_movie.append(movableNodes)
                if movableNodes_history_down is not None:
                    movableNodes_history_down.append(movableNodes)
                elif (movableNodes_history_up is not None) and (
                    i < washout_time + fixed_force_time * 3 + force_gradation_time
                ):
                    movableNodes_history_up.append(movableNodes)
                if i == washout_time - 1:
                    pos_his = (
                        torch.stack(agent.state.pos_history, dim=0).detach().numpy()
                    )
                    Xmax, Xmin, Ymax, Ymin = (
                        max(Xmax, np.max(pos_his[:, 0, :])),
                        min(Xmin, np.min(pos_his[:, 0, :])),
                        max(Ymax, np.max(pos_his[:, 1, :])),
                        min(Ymin, np.min(pos_his[:, 1, :])),
                    )

            central_position = (
                torch.stack(agent.state.centerNode_history, dim=0)
                .reshape([-1, 4])
                .t()[:2]
                .to(torch.float64)
            )
            loss = criterion(
                central_position,
                lissajous_target[updateTimeScale * i : updateTimeScale * (i + 1), :]
                .squeeze()
                .T,
            )
            if i == 0:
                central_position_history = central_position.cpu().detach()
            else:
                central_position_history = torch.cat(
                    (central_position_history, central_position.cpu().detach()), 1
                )
            if (
                i
                >= (washout_time * 3)
                + fixed_force_time
                + force_gradation_time_for_movie
            ):
                if (
                    int(
                        i
                        - (
                            (washout_time * 3)
                            + fixed_force_time
                            + force_gradation_time_for_movie
                        )
                    )
                    % int(fixed_force_time_in_gradation)
                    == int(fixed_force_time_in_gradation) - 1
                ):
                    traj = central_position_history[
                        :, int(-plotPeriod / self.simTimeStep) :
                    ]
                    gain = round(
                        -1
                        + (
                            (gradationCount / fixed_force_time_in_gradation - 1)
                            / gradation_scale
                        ),
                        2,
                    )
                    plotTrajectory(
                        None, traj, fileSaveDir + f"trajectory_gain{gain}.png"
                    )
            _ = print_current_params(agent, i, loss.item(), "SWG", None)
            loss_history.append(loss.item())

        verletSim.endSimulation()
        movableNodes_history_for_movie = torch.stack(movableNodes_history_for_movie, 0)
        movableNodes_history = torch.stack(movableNodes_history, 0)
        epoch_array = np.arange(0, epoch, 1)
        all_central_position_history = central_position_history
        t_array = np.arange(0, force_gradation_time_for_movie, self.simTimeStep)
        colors = ["blue"]
        labels = ["output"]
        dynamics_x = [
            [
                t_array,
                movableNodes_history_for_movie[:, 0, n].detach().cpu(),
                None,
                "black",
            ]
            for n in range(movableNodes_history_for_movie.shape[-1])
        ]
        dynamics_y = [
            [
                t_array,
                movableNodes_history_for_movie[:, 1, n].detach().cpu(),
                None,
                "black",
            ]
            for n in range(movableNodes_history_for_movie.shape[-1])
        ]
        dynamics_x[-1][-1] = dynamics_y[-1][-1] = "tab:red"
        plotLines(
            dynamics_x,
            "t",
            "x",
            None,
            fileSaveDir + "dynamics_x_gradation.png",
            False,
        )
        plotLines(
            dynamics_y,
            "t",
            "y",
            None,
            fileSaveDir + "dynamics_y_gradation.png",
            False,
        )
        np.savez_compressed(
            fileSaveDir + "gradation_dynamics.npz",
            central=all_central_position_history.detach().cpu().numpy(),
            movable=movableNodes_history.detach().cpu().numpy(),
        )
        localMax = []
        for axis in range(2):
            for n, nodeNum in enumerate(agent.state.movableNodes):
                for f in range(movableNodes_history.shape[0]):
                    data = (
                        movableNodes_history[
                            f, int(-plotPeriod / self.simTimeStep) :, axis, n
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    localMax.append(data[argrelmax(data, order=3)])
                if axis == 0:
                    plotBifurcationDiagram(
                        localMax,
                        gradationForces,
                        "x",
                        fileSaveDir + f"bifurcation_diagram_node{nodeNum}_x",
                    )
                else:
                    plotBifurcationDiagram(
                        localMax,
                        gradationForces,
                        "y",
                        fileSaveDir + f"bifurcation_diagram_node{nodeNum}_y",
                    )
                localMax = []

        localMax = []
        gradation_positions = movableNodes_history.detach().cpu().numpy()
        for n in range(gradation_positions.shape[-1]):
            if n != gradation_positions.shape[-1] - 1:
                continue
            localMaxs = []
            for axis in range(2):
                for f in range(gradation_positions.shape[0]):
                    data = gradation_positions[
                        f, int(-plotPeriod / self.simTimeStep) :, axis, n
                    ]
                    localMax.append(data[argrelmax(data, order=3)])
                nodeNum = n + int((self.noNodes - 1) / 2)
                localMaxs.append(localMax)
                localMax = []
            vlines = None
            plotBifurcationDiagram2ax(
                localMaxs,
                gradationForces,
                fileSaveDir + f"bifurcation_diagram_node{nodeNum}",
                vlines,
            )

        print("The results were saved in ", fileSaveDir)

    def locomotion_training(
        self,
        updateTimeScale,
        optim,
        lr_spring,
        lr_damping,
        lr_restlength,
        lr_amplitude,
        lr_omega,
        lr_phase,
        generatorNum,
        inputScale,
        wind=False,
    ):
        """Locomotion experiment

        Args:
            updateTimeScale (int): Number of steps to update parameters
            optim (str): Optimizer
            lr_spring (float): Learning rate of spring constant of MSDN
            lr_damping (float): Learning rate of damping coefficient of MSDN
            lr_restlength (float): Learning rate of rest length of MSDN
            lr_amplitude (float): Learning rate of amplitude of SWG
            lr_omega (float): Learning rate of omega of SWG
            lr_phase (float): Learning rate of phase of SWG
            generatorNum (list): Id of springs modulated by SWG
            inputScale (int): Input value scaling
            wind (bool, optional): Whether to introduce wind in the experiment. Defaults to False.
        """
        targetType = "forward"
        if not wind:
            assert inputScale == 0
        _inputScale = self.inputScale = inputScale
        assert self.tau >= 100
        assert int(self.simTime / self.simTimeStep) / updateTimeScale % 10 == 0
        trainingSteps = int(self.simTime * 9 / 10 / self.simTimeStep / updateTimeScale)
        evalSteps = int(self.simTime / 10 / self.simTimeStep / updateTimeScale)
        epoch = trainingSteps + evalSteps
        criterion = nn.MSELoss()
        if self.loadFolder is not None:
            agent = self.loadAgent(self.loadFileSuffix, keepStates=True)
            _, simulenv = self.initSet(
                self.initSize,
                self.tau,
                _simtime=self.simTime,
                centerFlag=True,
                bias=True,
                controlType="sine",
                generatorNum=generatorNum,
                amplitude=0.4,
                ground=True,
            )
            suffix = self.loadFileSuffix
            if wind:
                suffix += "_wind"
            fileSaveDir = self.make_fileSaveDir(
                self.loadFolder + "/additionalTraining/" + self.timeStamp, suffix
            )
            self.inputScale = agent.control.inputScale = _inputScale
        else:
            agent, simulenv = self.initSet(
                self.initSize,
                self.tau,
                _simtime=self.simTime,
                centerFlag=True,
                bias=True,
                controlType="sine",
                generatorNum=generatorNum,
                amplitude=0.4,
                ground=True,
            )
            fileSaveDir = self.make_fileSaveDir(
                self.figDirPre
                + "locomotion/"
                + self.figDirSuf
                + self.timeStamp
                + f"_shape{self.shape}_nodes{self.noNodes}_"
                f"updateTimeScale{updateTimeScale}_seed{self.seed}_{targetType}",
                None,
            )
        save_params_as_json(self.params, fileSaveDir)
        params_label = [
            "spring",
            "damping",
            "restLength",
            "inputLayer",
            "amplitude",
            "omega",
            "phase",
        ]
        initXpos = torch.mean(agent.state.pos.x).item()
        if agent.morph.shape == "CaterpillarRobot":
            frameWidth = agent.morph.initSize / 2 * agent.morph.noNodes
        simulenv.plot.setMovie(False)
        iterationNumber = 0
        speed_history, target_speed_history = [], []
        loss_history, params_history = [], []
        all_speed_history_list, all_target_speed_history_list = [], []
        if wind:
            wind_list = []
        # allSpringLen, allNodesPos, allNodesSpeed = [], [], []
        requires_grad_flag = False
        final_target_speed = 0

        for i in range(-100, epoch):
            target_speed = []
            agent.state.resetSavePosType(self.savePosType)
            if i >= trainingSteps:
                requires_grad_flag = False
            if not wind:
                if epoch > 10:
                    if i == -10:
                        simulenv.plot.setMovie(
                            fileSaveDir,
                            num="before",
                            colored_comp=[generatorNum, []],
                            all_black=True,
                            text=True,
                            axis=True,
                            xScale=xScale,
                        )
                    elif i == 0:
                        verletSim.simulEnv.end()
                        plt.clf()
                        plt.close()
                        simulenv.plot.setMovie(False)
                        if (
                            (lr_spring == 0)
                            and (lr_damping == 0)
                            and (lr_restlength == 0)
                            and (lr_amplitude == 0)
                            and (lr_omega == 0)
                            and (lr_phase == 0)
                        ):
                            requires_grad_flag = False
                        else:
                            requires_grad_flag = True
                    elif i == epoch - 10:
                        simulenv.plot.setMovie(
                            fileSaveDir,
                            num="after",
                            colored_comp=[generatorNum, []],
                            all_black=True,
                            text=True,
                            axis=True,
                            xScale=xScale,
                        )
            else:
                if i == 0:
                    if (
                        (lr_spring == 0)
                        and (lr_damping == 0)
                        and (lr_restlength == 0)
                        and (lr_amplitude == 0)
                        and (lr_omega == 0)
                        and (lr_phase == 0)
                    ):
                        requires_grad_flag = False
                    else:
                        requires_grad_flag = True
                if i < 0:
                    wind_val = 0
                elif i <= epoch - 200 - 100:
                    if i % 100 < 50:
                        wind_val = 0
                    else:
                        wind_val = -15
                elif i < epoch - 200:
                    if (i < epoch - 200 - 50) or (i >= epoch - 200 - 20):
                        wind_val = 0
                    else:
                        wind_val = -15
                else:
                    if i < epoch - 100:
                        wind_val = 0
                    else:
                        wind_val = -15
                if (epoch > 1000) and (
                    (i == int(trainingSteps * 0.8)) or (i == int(trainingSteps * 0.9))
                ):
                    lr_spring /= 10
                    lr_damping /= 10
                    lr_restlength /= 10
                    lr_amplitude /= 10
                    lr_omega /= 10
                    lr_phase /= 10
                if i == epoch - 200 - 60:
                    print("----------- recording started -----------")
                    simulenv.plot.setMovie(
                        fileSaveDir,
                        colored_comp=[generatorNum, []],
                        all_black=True,
                        text=True,
                        axis=True,
                        xScale=xScale,
                    )
                elif i == epoch - 200:
                    print("----------- recording finished -----------")
                    verletSim.simulEnv.end()
                    simulenv.plot.setMovie(False)
                elif i == epoch - 120:
                    simulenv.plot.setMovie(
                        fileSaveDir,
                        num="wo",
                        plotCycle=10,
                        colored_comp=[generatorNum, []],
                        all_black=True,
                        text=True,
                        axis=True,
                        xScale=xScale,
                    )
                elif i == epoch - 110:
                    verletSim.simulEnv.end()
                    simulenv.plot.setMovie(False)
                elif i == epoch - 20:
                    simulenv.plot.setMovie(
                        fileSaveDir,
                        num="w",
                        plotCycle=10,
                        colored_comp=[generatorNum, []],
                        all_black=True,
                        text=True,
                        axis=True,
                        xScale=xScale,
                    )
                elif i == epoch - 10:
                    verletSim.simulEnv.end()
                    simulenv.plot.setMovie(False)
                if wind:
                    agent.morph.environment.setWind(int(wind_val * self.inputScale))
                else:
                    agent.morph.environment.setWind(None)
            if i == -100:
                if self.loadFolder is None:
                    agent.inputLayer *= self.inputScale
            else:
                if i < trainingSteps:
                    agent.morph.spring = _spring
                    agent.morph.damping = _damping
                    agent.morph.restLength = _restLength
                    agent.inputLayer = _inputLayer
                    agent.control.amplitude = _amplitude
                    agent.control.omega = _omega
                    agent.control.phase = _phase
                elif i == trainingSteps:
                    agent.morph.spring = agent.morph.spring
                    agent.morph.damping = agent.morph.damping
                    agent.morph.restLength = agent.morph.restLength
                    agent.inputLayer = agent.inputLayer
            agent.morph.spring.requires_grad = requires_grad_flag
            agent.morph.damping.requires_grad = requires_grad_flag
            agent.morph.restLength.requires_grad = requires_grad_flag
            agent.inputLayer.requires_grad = requires_grad_flag
            agent.control.amplitude.requires_grad = requires_grad_flag
            agent.control.omega.requires_grad = requires_grad_flag
            agent.control.phase.requires_grad = requires_grad_flag

            params_lr_list = [
                {"params": agent.morph.spring, "lr": lr_spring},
                {"params": agent.morph.damping, "lr": lr_damping},
                {"params": agent.morph.restLength, "lr": lr_restlength},
                {"params": agent.control.amplitude, "lr": lr_amplitude},
                {"params": agent.control.omega, "lr": lr_omega},
                {"params": agent.control.phase, "lr": lr_phase},
            ]
            optimizer = OPTIM_DICT[optim](params_lr_list)

            for _ in range(updateTimeScale):
                if targetType == "forward":
                    if (i >= 0) and (i < trainingSteps):
                        target_speed.append(final_target_speed + self.simTimeStep / 500)
                        final_target_speed = target_speed[-1]
                    elif i >= trainingSteps:
                        target_speed.append(final_target_speed)
                    else:
                        target_speed.append(0)
                elif targetType == "stay":
                    target_speed.append(0)
                if lr_spring == 0:
                    difx, dify = agent.state.pos.getDifference()
                    difxy = torch.sqrt(difx**2 + dify**2 + 1e-16) + torch.eye(
                        agent.morph.noNodes
                    )
                    actualLength = difxy * agent.morph.connections
                    allSpringLen.append(actualLength[np.nonzero(np.triu(actualLength))])
                simulenv.plot.setRangeIndiv(
                    torch.mean(agent.state.pos.x).item() + frameWidth,
                    torch.mean(agent.state.pos.x).item() - frameWidth,
                    frameWidth * 1.5,
                    -frameWidth * 0.5,
                )
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber

            agent_speed = (
                torch.mean(torch.stack(agent.state.vel_history, 0)[:, 0, :], 1)
                .view(-1, self.tau)
                .mean(1)
            )
            target_speed = (
                torch.tensor(target_speed, dtype=torch.float64)
                .view(-1, self.tau)
                .mean(1)
            ).to(self.device)
            # if lr_spring == 0:
            #     allNodesPos.append(torch.stack(agent.state.pos_history, 0))
            #     allNodesSpeed.append(torch.stack(agent.state.vel_history, 0))
            if i == -11:
                xScale, _ = str((initXpos // 10) * 10).split(".")
                xScale = float(xScale)
                initXspeed = torch.mean(agent_speed).item()

            agent.resetAgent(agent.morph, keepTime=True, keepStates=True)
            agent.control.target, agent.control.us = None, None
            if (i >= 0) and (i % 50 == 0) and (i < trainingSteps):
                self.saveObjects(agent, simulenv, fileSaveDir, i)
            if i == trainingSteps - 1:
                self.saveObjects(agent, simulenv, fileSaveDir, "final")

            optimizer.zero_grad()
            loss = criterion(agent_speed, target_speed)
            if i >= 0:
                speed_history.append(agent_speed.cpu().detach())
                all_speed_history_list.append(agent_speed.cpu().detach())
                target_speed_history.append(target_speed.cpu().detach())
                all_target_speed_history_list.append(target_speed.cpu().detach())
                if wind:
                    for _ in range(int(updateTimeScale * self.simTimeStep)):
                        wind_list.append(wind_val)
                if i % int(epoch / 10) == int(epoch / 10) - 1:
                    saveNPY(
                        np.stack(speed_history, 0).flatten(),
                        fileSaveDir + f"speed_{int(i//(epoch/10))}.npy",
                    )
                    saveNPY(
                        np.stack(target_speed_history, 0).flatten(),
                        fileSaveDir + f"target_{int(i//(epoch/10))}.npy",
                    )
                    speed_history, target_speed_history = [], []
            if torch.sum(agent.control.amplitude != 0) > 0:
                _ = print_current_params(agent, i, loss.item(), "SWG", None)
            else:
                print(
                    "{}: loss = {:.06f}, speed = {:.06f}".format(
                        i, loss.item(), torch.mean(agent_speed)
                    )
                )
            if i >= 0:
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
                loss_history.append(loss.item())
                if (i % 10 == 0) and (i > 0):
                    epoch_array = np.arange(0, i + 1, 1)
                    plotLines(
                        [[epoch_array, loss_history, None, "tab:blue"]],
                        "epoch",
                        "loss",
                        None,
                        fileSaveDir + "loss.png",
                        marker=False,
                    )
                    saveNPY(np.array(loss_history), fileSaveDir + "loss.npy")
                    step_array = np.arange(
                        0,
                        self.simTime * (i + 1) / epoch,
                        self.simTimeStep * self.tau,
                    )
                    all_speed_history = torch.stack(all_speed_history_list, 0).flatten()
                    all_target_speed_history = torch.stack(
                        all_target_speed_history_list, 0
                    ).flatten()
                    results = [
                        [step_array, all_speed_history, "speed", "red"],
                        [step_array, all_target_speed_history, "target", "black"],
                    ]
                    plotLines(
                        results,
                        "t",
                        "speed",
                        None,
                        fileSaveDir + "speed.png",
                        marker=False,
                    )
                    if wind:
                        results = [[step_array, wind_list, "wind", "tab:green"]]
                        plotLines(
                            results,
                            "t",
                            "wind",
                            None,
                            fileSaveDir + "wind.png",
                            marker=False,
                        )
                        saveNPY(np.array(wind_list), fileSaveDir + "wind.npy")
                    saveNPY(all_speed_history.numpy(), fileSaveDir + "_speed.npy")
                    saveNPY(
                        all_target_speed_history.numpy(),
                        fileSaveDir + "_target_speed.npy",
                    )
                    cmap = plt.get_cmap("tab10")
                    params_his = np.stack(params_history, 0).T
                    for i in range(params_his.shape[0]):
                        results = [
                            [epoch_array, params_his[i], params_label[i], cmap(i)]
                        ]
                        plotLines(
                            results,
                            "epoch",
                            None,
                            None,
                            fileSaveDir + f"params_change_{params_label[i]}.png",
                            marker=False,
                        )
                    params_change_dict = {
                        f"{params_label[i]}": params_his[i]
                        for i in range(params_his.shape[0])
                    }
                    np.savez_compressed(
                        fileSaveDir + "params_change.npz", **params_change_dict
                    )
            if i < trainingSteps:
                if requires_grad_flag:
                    loss.backward()
                    del loss
                    if i < trainingSteps - 1:
                        optimizer.step()
                with torch.no_grad():
                    agent.state.pos = agent.state.pos.copy()
                    agent.state.speed = agent.state.speed.copy()
                    [
                        _spring,
                        _damping,
                        _inputLayer,
                        _restLength,
                        _amplitude,
                        _omega,
                        _phase,
                    ] = self.shape_tunableParameters(
                        agent, restLength=True, sineControl=True
                    )

        verletSim.endSimulation()
        epoch_array = np.arange(0, epoch, 1)
        plotLines(
            [[epoch_array, loss_history, None, "tab:blue"]],
            "epoch",
            "loss",
            None,
            fileSaveDir + "loss.png",
            marker=False,
        )
        step_array = np.arange(0, self.simTime, self.simTimeStep * self.tau)
        all_speed_history = concatenate_files(fileSaveDir + "speed_")
        all_target_speed_history = concatenate_files(fileSaveDir + "target_")
        results = [
            [step_array, all_speed_history, "speed", "red"],
            [step_array, all_target_speed_history, "target", "black"],
        ]
        plotLines(
            results,
            "t",
            "speed",
            None,
            fileSaveDir + "speed.png",
            marker=False,
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
        saveNPY(np.array(loss_history), fileSaveDir + "loss.npy")
        npy2npz(
            [fileSaveDir + f"speed_{i}.npy" for i in range(10)],
            fileSaveDir,
            "speed",
            transpose=True,
        )
        npy2npz(
            [fileSaveDir + f"target_{i}.npy" for i in range(10)],
            fileSaveDir,
            "target",
            transpose=True,
        )

        print("The results were saved in ", fileSaveDir)

    def locomotion_closedloop(
        self,
        generatorNum,
        wind,
        feedbackSprings=["all"],
        reservoirStateType="springLength",
        noise=0.2,
    ):
        """Realize closed-loop control for locomotion

        Args:
            generatorNum (list): Id of springs modulated by SWG
            wind (bool, optional): Whether to introduce wind in the experiment. Defaults to False.
            feedbackSprings (list): Distribution of springs used for feedback. Defaults to ["all"].
            reservoirStateType (str, optional): Feedback information. Defaults to "springLength".
            noise (float, optional): Standard deviation of noise added to feedback information. Defaults to 0.2.
        """
        assert self.simTime % 10 == 0
        assert self.loadFolder is not None
        simulLen = int(self.simTime / self.simTimeStep)
        updateTimeScale = 100
        epoch = 1
        agent = self.loadAgent(self.loadFileSuffix, keepStates=True)
        _, simulenv = self.initSet(
            self.initSize,
            self.tau,
            _simtime=self.simTime,
            centerFlag=True,
            bias=True,
            controlType="sine",
            generatorNum=generatorNum,
            amplitude=0.0,
            ground=True,
        )
        agent = self.set_requiresGrads(agent, False, generator=True)
        suffix = self.loadFileSuffix
        suffix += "_" + "-".join([key for key in feedbackSprings])
        suffix += f"_noise{noise}"
        fileSaveDir = self.make_fileSaveDir(
            self.loadFolder + "/closedloop/" + self.timeStamp, suffix
        )
        save_params_as_json(self.params, fileSaveDir)
        initXpos = torch.mean(agent.state.pos.x).item()
        if agent.morph.shape == "CaterpillarRobot":
            frameWidth = agent.morph.initSize / 2 * agent.morph.noNodes
        criterion = nn.MSELoss()
        iterationNumber = 0
        targetComp = []
        w_opt_list = []

        for i in range(-2, epoch):
            target_speed = []
            simulenv.plot.setMovie(False)
            if i < 0:
                wind_val = 0
            if wind:
                agent.morph.environment.setWind(int(wind_val * self.inputScale))
            else:
                agent.morph.environment.setWind(None)
            agent.state.resetSavePosType(self.savePosType)

            for j in range(simulLen):
                if i == -2:
                    target_speed.append(0)
                else:
                    target_speed.append(initXspeed)
                simulenv.plot.setRangeIndiv(
                    torch.mean(agent.state.pos.x).item() + frameWidth,
                    torch.mean(agent.state.pos.x).item() - frameWidth,
                    frameWidth * 1.5,
                    -frameWidth * 0.5,
                )
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber
                if i >= 0:
                    modulationFactor = (
                        agent.control.modulationFactor(agent.state, agent.brain) - 1.0
                    )
                    modulationFactor = modulationFactor[
                        np.nonzero(np.triu(agent.morph.connections))
                    ]
                    targetComp.append(modulationFactor)
                if i == -1:
                    pos_his = (
                        torch.stack(agent.state.pos_history, dim=0)
                        .to("cpu")
                        .detach()
                        .numpy()
                    )

            agent_speed = (
                torch.mean(torch.stack(agent.state.vel_history, 0)[:, 0, :], 1)
                .view(-1, self.tau)
                .mean(1)
            )
            target_speed = (
                torch.tensor(target_speed, dtype=torch.float64)
                .view(-1, self.tau)
                .mean(1)
            )
            if i == -2:
                xScale, _ = str((initXpos // 10) * 10).split(".")
                xScale = float(xScale)
                initXspeed = torch.mean(agent_speed).item()
            agent.resetAgent(agent.morph, keepTime=True, keepStates=True)

            loss = criterion(agent_speed, target_speed)
            _ = print_current_params(agent, i, loss.item(), "SWG", None)

        verletSim.endSimulation()
        agent.resetAgent()
        targetComp = torch.stack(targetComp, dim=0)

        period_per_replacement = self.simTime
        if reservoirStateType == "massPosition":
            reservoirStatesDim = agent.movableNodes.shape[0] * 2 + 1
        elif reservoirStateType == "massSpeed":
            reservoirStatesDim = agent.movableNodes.shape[0] * 2 + 1
        elif reservoirStateType == "massAcceleration":
            reservoirStatesDim = agent.movableNodes.shape[0] * 2 + 1
        elif reservoirStateType == "springLength":
            reservoirStatesDim = agent.morph.noComps + 1
        is_replaced = [False for _ in range(agent.morph.noComps)]
        if generatorNum != [-1]:
            for i in range(len(is_replaced)):
                if not i in generatorNum:
                    is_replaced[i] = True
        epoch = 1
        if feedbackSprings[0] == "all":
            replaceNum = [[i for i in range(agent.morph.noComps)]]
        elif feedbackSprings[0] == "half":
            replaceNum = [
                random.sample(
                    [i for i in range(agent.morph.noComps)],
                    int(agent.morph.noComps / 2),
                )
            ]
        elif feedbackSprings[0] == "quarter":
            replaceNum = [
                random.sample(
                    [i for i in range(agent.morph.noComps)],
                    int(agent.morph.noComps / 4),
                )
            ]
        else:
            replaceNum = [[]]
        compNum = -1
        allNodesPos, allSpringLen = [], []
        losses = []
        agent.control.initFeedbackLoop(agent.morph.noNodes, reservoirStatesDim)
        simulenv.plot.colored_comp[0] = generatorNum

        for i in range(-1, epoch):
            nodesPos, reservoirStates, target_speeds = [], [], []
            _mses_with_ridgeRegression = []
            for j in range(period_per_replacement * updateTimeScale):
                target_speeds.append(initXspeed)
                simulenv.plot.setRangeIndiv(
                    torch.mean(agent.state.pos.x).item() + frameWidth,
                    torch.mean(agent.state.pos.x).item() - frameWidth,
                    frameWidth * 1.5,
                    -frameWidth * 0.5,
                )
                if j == (period_per_replacement - 12) * updateTimeScale:
                    simulenv.plot.setMovie(
                        fileSaveDir,
                        num=i,
                        all_black=True,
                        text=True,
                        axis=True,
                        colored_comp=simulenv.plot.colored_comp,
                        xScale=xScale,
                    )
                elif j == (period_per_replacement - 2) * updateTimeScale:
                    verletSim.simulEnv.end()
                    plt.clf()
                    plt.close()
                    simulenv.plot.setMovie(False)
                    _agent, _simulenv = copy.deepcopy(agent), copy.deepcopy(simulenv)
                    _simulenv.plot.lightenVar()
                    _simulenv.plot.setFigure()
                    _agent.control.target, _agent.control.us = None, None
                    _agent.resetAgent(agent.morph, keepTime=True, keepStates=True)
                    self.saveObjects(_agent, _simulenv, fileSaveDir, i)
                    if i != epoch - 1:
                        simulenv.plot.setMovie(
                            fileSaveDir,
                            num=f"{i}to{i+1}",
                            all_black=True,
                            text=True,
                            colored_comp=simulenv.plot.colored_comp,
                            axis=True,
                            xScale=xScale,
                        )
                    else:
                        simulenv.plot.setMovie(False)
                elif i >= 0:
                    if j == 0:
                        simulenv.plot.setFigure()
                    elif j == 10 * updateTimeScale:
                        verletSim.simulEnv.end()
                        plt.clf()
                        plt.close()
                        simulenv.plot.setMovie(False)
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber
                nodesPosData = torch.cat(
                    (
                        torch.flatten(agent.state.pos.x.cpu()).gather(
                            0, agent.movableNodes
                        ),
                        torch.flatten(agent.state.pos.y.cpu()).gather(
                            0, agent.movableNodes
                        ),
                    )
                )
                nodesPos.append(nodesPosData)
                if reservoirStateType == "massPosition":
                    xPoss = torch.flatten(agent.state.pos.x.cpu()).gather(
                        0, agent.movableNodes
                    ) - torch.mean(agent.state.pos.x.cpu())
                    yPoss = torch.flatten(agent.state.pos.y.cpu()).gather(
                        0, agent.movableNodes
                    ) - torch.mean(agent.state.pos.y.cpu())
                    reservoirState = torch.cat(
                        (torch.ones(1, dtype=torch.float64), xPoss, yPoss)
                    )
                elif reservoirStateType == "massSpeed":
                    xSpeeds = torch.flatten(agent.state.speed.x.cpu()).gather(
                        0, agent.movableNodes
                    )
                    ySpeeds = torch.flatten(agent.state.speed.y.cpu()).gather(
                        0, agent.movableNodes
                    )
                    reservoirState = torch.cat(
                        (torch.ones(1, dtype=torch.float64), xSpeeds, ySpeeds)
                    )
                elif reservoirStateType == "massAcceleration":
                    xAccs = torch.flatten(agent.state.acceleration.x.cpu()).gather(
                        0, agent.movableNodes
                    )
                    yAccs = torch.flatten(agent.state.acceleration.y.cpu()).gather(
                        0, agent.movableNodes
                    )
                    reservoirState = torch.cat(
                        (torch.ones(1, dtype=torch.float64), xAccs, yAccs)
                    )
                elif reservoirStateType == "springLength":
                    difx, dify = agent.state.pos.getDifference()
                    difxy = torch.sqrt(difx**2 + dify**2 + 1e-16) + torch.eye(
                        agent.morph.noNodes
                    )
                    actualLength = difxy * agent.morph.connections
                    reservoirState = torch.cat(
                        (
                            torch.ones(1, dtype=torch.float64),
                            actualLength[np.nonzero(np.triu(actualLength))],
                        )
                    )
                    # allSpringLen.append(actualLength[np.nonzero(np.triu(actualLength))])
                noiseTensor = torch.normal(
                    mean=0, std=noise, size=(reservoirState.shape)
                )
                reservoirStates.append(reservoirState[1:] + noiseTensor[1:])
                modulationFactor = (
                    agent.control.modulationFactor(agent.state, agent.brain) - 1.0
                )
                modulationFactor = modulationFactor[
                    np.nonzero(np.triu(agent.morph.connections))
                ]
                if i != -1:
                    if j == 0:
                        for compNum in replaceNum[i]:
                            is_replaced[compNum] = True
                            print(f"==== No.{compNum} comp is replaced ====")
                            agent.control.setFeedbackLayers(
                                np.nonzero(np.triu(agent.morph.connections))[0][
                                    compNum
                                ],
                                np.nonzero(np.triu(agent.morph.connections))[1][
                                    compNum
                                ],
                                w_opts[compNum],
                            )
                            simulenv.plot.colored_comp[1].append(compNum)
                    agent.control.setFeedbackFactor(reservoirState)

            nodesPos = torch.stack(nodesPos, dim=0)
            allNodesPos.append(nodesPos.T)
            reservoirStates = torch.stack(reservoirStates, dim=0)
            if i != -1:
                replacing_dynamics_after = nodesPos[: 5 * updateTimeScale, :]
                replacing_dynamics = torch.cat(
                    (replacing_dynamics_before, replacing_dynamics_after), 0
                )
                movableNodeNum = int(nodesPos.shape[1] / 2)
                t_array = np.arange(
                    -replacing_dynamics.shape[0] / 2 * self.simTimeStep,
                    replacing_dynamics.shape[0] / 2 * self.simTimeStep,
                    self.simTimeStep,
                )
                dynamics_x = [
                    [t_array, replacing_dynamics[:, n].detach().cpu(), None, "black"]
                    for n in range(movableNodeNum)
                ]
                dynamics_y = [
                    [
                        t_array,
                        replacing_dynamics[:, movableNodeNum + n].detach().cpu(),
                        None,
                        "black",
                    ]
                    for n in range(movableNodeNum)
                ]
                dynamics_x[-1][-1] = dynamics_y[-1][-1] = "tab:red"
                plotLines(
                    dynamics_x,
                    "t",
                    "x",
                    None,
                    fileSaveDir + f"dynamics_x_{i-1}to{i}.png",
                    False,
                    vline=0,
                )
                plt.clf()
                plt.close()
                plotLines(
                    dynamics_y,
                    "t",
                    "y",
                    None,
                    fileSaveDir + f"dynamics_y_{i-1}to{i}.png",
                    False,
                    vline=0,
                )
                plt.clf()
                plt.close()
            replacing_dynamics_before = nodesPos[-5 * updateTimeScale :, :]
            df_ridgeparam_tmp = []
            for idx, target in enumerate(targetComp.T):
                if is_replaced[idx]:
                    _mses_with_ridgeRegression.append(1)
                    w_opt_list.append(
                        torch.zeros(reservoirStatesDim, dtype=torch.float64)
                    )
                    df_ridgeparam_tmp.append(np.zeros(2))
                else:
                    agent.optimizer_ridge.X = reservoirStates.T[
                        :, int(simulLen / 5) : int(simulLen * 4 / 5)
                    ]
                    agent.optimizer_ridge.test_X = reservoirStates.T[
                        :, int(simulLen * 4 / 5) :
                    ]
                    agent.optimizer_ridge.Y = target[
                        int(simulLen / 5) : int(simulLen * 4 / 5)
                    ]
                    trainStates, testStates, pred, w_opt, df_ridgeparam = (
                        self.ridgeRegression(agent)
                    )
                    mse = MSE(testStates, target[int(simulLen * 4 / 5) :]).item()
                    _mses_with_ridgeRegression.append(mse)
                    w_opt_list.append(w_opt)
                    df_ridgeparam_tmp.append(df_ridgeparam)
                    print(
                        f"{i+2}/{epoch+1}:  MSE when No.{idx} comp is replaced: {mse}"
                    )
            w_opts = torch.stack(w_opt_list, dim=0)
            w_opt_list = []

        verletSim.endSimulation()
        agent_speed = (
            torch.mean(torch.stack(agent.state.vel_history, 0)[:, 0, :], 1)
            .view(-1, self.tau)
            .mean(1)
        )
        plt.clf()
        plt.close()
        agent.resetAgent(agent.morph, keepTime=True, keepStates=True)

        agent.control.target, agent.control.us = None, None
        self.saveObjects(agent, simulenv, fileSaveDir, "final")
        step_array = np.arange(0, agent_speed.shape[0], self.simTimeStep * self.tau)
        results = [[step_array, agent_speed, "speed", "red"]]
        plotLines(
            results,
            "t",
            "speed",
            None,
            fileSaveDir + "speed.png",
            marker=False,
        )
        saveNPY(agent_speed, fileSaveDir + "speed.npy")
        saveNPY(losses, fileSaveDir + "loss.npy")
        allNodesPos = torch.stack(allNodesPos)
        allNodesPosX = allNodesPos[:, : int(allNodesPos.shape[1] / 2), :]
        allNodesPosY = allNodesPos[:, int(allNodesPos.shape[1] / 2) :, :]
        # allSpringLen = torch.stack(allSpringLen, 0).T.cpu().detach().numpy()
        # saveNPY(allSpringLen, fileSaveDir + "springLength.npy")
        np.savez_compressed(
            fileSaveDir + "dynamics.npz",
            x=allNodesPosX.detach().cpu().numpy(),
            y=allNodesPosY.detach().cpu().numpy(),
        )

        print("The results were saved in ", fileSaveDir)

    def locomotion_perturbation_process(
        self,
        simTime,
        suffix,
        seed_initpos_num,
        updateTimeScale,
        error_threshold,
        all_losses,
        all_speed,
        all_return_rate,
        reservoirStateType,
    ):
        """Individual process to check robustness in locomotion

        Args:
            simTime (_type_): Simulation time
            suffix (str): Suffix of the file to load
            seed_initpos_num (int): Range of random seed to determine initial mass point positions
            updateTimeScale (int): Number of steps to update parameters
            error_threshold (float): Threshold for successful return
            all_losses (list): List for storing losses
            all_speed (list): List for storing speed
            all_return_rate (list): List for storing return rates
            reservoirStateType (str): Feedback information
        """
        agent = self.loadAgent(suffix, keepStates=True)
        final_target_speed = torch.mean(agent.state.speed.x).item()
        agent.control.inputScale = self.inputScale
        initXpos = torch.mean(agent.state.pos.x).item()
        xScale, _ = str((initXpos // 10) * 10).split(".")
        xScale = float(xScale)
        posScale = (
            torch.max(agent.state.pos.x).item() - torch.min(agent.state.pos.x).item()
        )
        speedScale = (
            torch.max(agent.state.speed.x).item()
            - torch.min(agent.state.speed.x).item()
        )
        _, simulenv = self.initSet(
            self.initSize,
            self.tau,
            _simtime=self.simTime,
            centerFlag=True,
            bias=True,
            controlType="sine",
            generatorNum=[-1],
            amplitude=0.4,
            ground=True,
        )
        losses_for_each_suffix, speed_for_each_suffix, return_rate_for_each_suffix = (
            [],
            [],
            [],
        )
        for std in range(0, 11):
            std /= 10
            losses_for_each_std, speed_for_each_std = [], []
            success, fail = 0, 0

            for random_seed in range(seed_initpos_num):
                agent = self.loadAgent(suffix, keepStates=True)
                agent.control.inputScale = self.inputScale
                loss_history, speed_history = [], None
                iterationNumber = 0
                if agent.morph.shape == "CaterpillarRobot":
                    frameWidth = agent.morph.initSize / 2 * agent.morph.noNodes
                torch.manual_seed(random_seed)
                deltaPos = torch.normal(
                    mean=0, std=std * posScale / 8, size=(2, self.noNodes)
                )
                deltaSpeed = torch.normal(
                    mean=0, std=std * speedScale, size=(2, self.noNodes)
                )
                agent.state.pos.x += deltaPos[0]
                agent.state.pos.y += deltaPos[1]
                agent.state.speed.x += deltaSpeed[0]
                agent.state.speed.y += deltaSpeed[1]
                for i in range(simTime):
                    target_speed = []
                    agent.state.resetSavePosType(self.savePosType)
                    if i == 0:
                        simulenv.plot.setMovie(False)
                    for _ in range(updateTimeScale):
                        target_speed.append(final_target_speed)
                        difx, dify = agent.state.pos.getDifference()
                        difxy = torch.sqrt(difx**2 + dify**2 + 1e-16) + torch.eye(
                            agent.morph.noNodes
                        )
                        actualLength = difxy * agent.morph.connections
                        reservoirState = torch.cat(
                            (
                                torch.ones(1, dtype=torch.float64),
                                actualLength[np.nonzero(np.triu(actualLength))],
                            )
                        )
                        agent.control.setFeedbackFactor(reservoirState)
                        simulenv.plot.setRangeIndiv(
                            torch.mean(agent.state.pos.x).item() + frameWidth,
                            torch.mean(agent.state.pos.x).item() - frameWidth,
                            frameWidth * 1.5,
                            -frameWidth * 0.5,
                        )
                        agent.control.set_for_limitcycle(
                            agent.morph, self.inputScale, None
                        )
                        verletSim = VerletSimulation(simulenv, agent)
                        verletSim.iterationNumber = iterationNumber
                        verletSim.runSimulationOneStep()
                        iterationNumber = verletSim.iterationNumber
                        if reservoirStateType == "massPosition":
                            reservoirState = torch.cat(
                                (
                                    torch.ones(1, dtype=torch.float64),
                                    torch.flatten(agent.state.pos.x.cpu()).gather(
                                        0, agent.movableNodes
                                    ),
                                    torch.flatten(agent.state.pos.y.cpu()).gather(
                                        0, agent.movableNodes
                                    ),
                                )
                            )
                        elif reservoirStateType == "springLength":
                            difx, dify = agent.state.pos.getDifference()
                            difxy = torch.sqrt(
                                difx**2 + dify**2 + 1e-16
                            ) + torch.eye(agent.morph.noNodes)
                            actualLength = difxy * agent.morph.connections
                            reservoirState = torch.cat(
                                (
                                    torch.ones(1, dtype=torch.float64),
                                    actualLength[np.nonzero(np.triu(actualLength))],
                                )
                            )
                        agent.control.setFeedbackFactor(reservoirState)

                    agent_speed = (
                        torch.mean(torch.stack(agent.state.vel_history, 0)[:, 0, :], 1)
                        .view(-1, self.tau)
                        .mean(1)
                    )
                    target_speed = (
                        torch.tensor(target_speed, dtype=torch.float64)
                        .view(-1, self.tau)
                        .mean(1)
                    )
                    if speed_history is None:
                        speed_history = agent_speed
                    else:
                        speed_history = torch.cat((speed_history, agent_speed), 0)
                    _losses = []
                    for _ in range(101):
                        loss = nn.MSELoss()(agent_speed, target_speed)
                        _losses.append(loss.item())
                    loss = min(_losses)
                    loss_history.append(loss)
                    if i == simTime - 1:
                        print(
                            "suffix: {} | std: {} | seed: {} | loss = {:.06f}".format(
                                suffix,
                                std,
                                random_seed,
                                sum(loss_history[-3:]) / len(loss_history[-3:]),
                            )
                        )

                verletSim.endSimulation()
                agent.resetAgent()
                plt.clf()
                plt.close()
                speed_history_last30 = speed_history[-30:]
                if (
                    sum(speed_history_last30) / len(speed_history_last30)
                    > final_target_speed * error_threshold
                ):
                    success += 1
                else:
                    fail += 1
                losses_for_each_std.append(loss_history)
                speed_for_each_std.append(speed_history)
                plt.clf()
                plt.close()

            losses_for_each_suffix.append(losses_for_each_std)
            speed_for_each_suffix.append(torch.stack(speed_for_each_std, 0))
            return_rate_for_each_suffix.append(success / seed_initpos_num * 100)

        all_losses[suffix + 1] = losses_for_each_suffix
        all_speed[suffix + 1] = torch.stack(speed_for_each_suffix, 0).numpy()
        all_return_rate[suffix + 1] = return_rate_for_each_suffix

    def locomotion_perturbation(
        self,
        updateTimeScale,
        feedbackSprings,
        reservoirStateType,
    ):
        """Check robustness in locomotion

        Args:
            updateTimeScale (int): Number of steps to update parameters
            feedbackSprings (list): Distribution of springs used for feedback
            reservoirStateType (str): Feedback information
        """
        fileSaveDir = self.make_fileSaveDir(
            self.loadFolder + "/analysis/" + self.timeStamp, None
        )
        save_params_as_json(self.params, fileSaveDir)
        seed_initpos_num = 10 #100
        error_threshold = 0.7

        agentNum = 0
        while True:
            if os.path.isfile(self.loadFolder + f"agent_{agentNum-1}.pickle"):
                agentNum += 1
            else:
                break

        processes = []
        manager = Manager()
        all_losses = manager.list([None for _ in range(agentNum)])
        all_speed = manager.list([None for _ in range(agentNum)])
        all_return_rate = manager.list([None for _ in range(agentNum)])

        mp.set_start_method("spawn", force=True)
        for suffix in range(-1, agentNum - 1):
            assert os.path.isfile(self.loadFolder + f"agent_{suffix}.pickle")
            p = Process(
                target=self.locomotion_perturbation_process,
                args=(
                    self.simTime,
                    suffix,
                    seed_initpos_num,
                    updateTimeScale,
                    error_threshold,
                    all_losses,
                    all_speed,
                    all_return_rate,
                    reservoirStateType,
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
                os.system("taskset -p -c %d-%d %d" % (0, cpu_count() - 1, p.pid))

        for p in processes:
            p.join()

        all_losses = np.array(all_losses)

        all_speed = np.stack(all_speed, 0)
        saveNPY(np.array(all_return_rate), fileSaveDir + "return_rate.npy")
        std_list = [i / 10 for i in range(0, len(all_return_rate[0]))]
        cmap = plt.get_cmap("tab10")
        replacedPart = []
        for i in range(len(feedbackSprings)):
            replacedPart.append("-".join([key for key in feedbackSprings[: i + 1]]))
        labels = ["SWG"] + [f"Closed loop ({part})" for part in replacedPart]
        result = [
            [std_list, all_return_rate[j], labels[j], cmap(j)]
            for j in range(len(feedbackSprings) + 1)
        ]
        plotLines(
            result,
            "std",
            "Return rate",
            None,
            fileSaveDir + "return_rate.png",
            marker=False,
        )
        np.savez_compressed(
            fileSaveDir + "losses_speed.npz", losses=all_losses, speed=all_speed
        )

        print("The results were saved in ", fileSaveDir)

    def locomotion_switching_dynamics(
        self, updateTimeScale, generatorNum, inputScale, wind
    ):
        """Analyze the dynamics in switching locomotion

        Args:
            updateTimeScale (int): Number of steps to update parameters
            generatorNum (list): Id of springs modulated by SWG
            inputScale (int): Input value scaling
            wind (bool): Whether to introduce wind in the experiment
        """
        self.inputScale = inputScale
        washout_time = 30
        fixed_force_time = 10
        repetition = 5
        fixed_force_time_long = 10  ## 100
        self.simTime = epoch = (
            washout_time
            + (fixed_force_time * 2 * repetition)
            + (2 * fixed_force_time_long)
        )
        criterion = nn.MSELoss()
        agent = self.loadAgent(self.loadFileSuffix, keepStates=True)
        _, simulenv = self.initSet(
            self.initSize,
            self.tau,
            _simtime=self.simTime,
            centerFlag=True,
            bias=True,
            controlType="sine",
            generatorNum=generatorNum,
            amplitude=0.4,
            ground=True,
        )
        fileSaveDir = self.make_fileSaveDir(
            self.loadFolder + "/analysis/" + self.timeStamp, self.loadFileSuffix
        )
        initXpos = torch.mean(agent.state.pos.x).item()
        if agent.morph.shape == "CaterpillarRobot":
            frameWidth = agent.morph.initSize / 2 * agent.morph.noNodes
        simulenv.plot.setMovie(False)
        iterationNumber = 0
        loss_history = []
        agent = self.set_requiresGrads(agent, False, generator=True)
        simulenv.plot.colored_comp[0] = generatorNum
        movableNodes_history = None

        ## trained behavior
        for i in range(epoch):
            target_speed = []
            if i < washout_time:
                wind_val = 0
            else:
                if i == washout_time + (fixed_force_time * 2 * (repetition - 2)):
                    movableNodes_history = []
                if i == washout_time + (fixed_force_time * 2 * repetition):
                    verletSim.simulEnv.end()
                    simulenv.plot.setMovie(False)
                if i < washout_time + (fixed_force_time * 2 * repetition):
                    if (i - washout_time) % (fixed_force_time * 2) < fixed_force_time:
                        wind_val = 0
                    else:
                        wind_val = -15
                else:
                    if (i - (washout_time + (fixed_force_time * 2 * repetition))) % (
                        fixed_force_time_long * 2
                    ) < fixed_force_time_long:
                        wind_val = 0
                    else:
                        wind_val = -15
            if wind:
                agent.morph.environment.setWind(wind_val * self.inputScale)
            else:
                agent.morph.environment.setWind(None)
            agent.state.resetSavePosType(self.savePosType)

            for _ in range(updateTimeScale):
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber
                if movableNodes_history is not None:
                    movableNodes = torch.cat(
                        (
                            torch.flatten(agent.state.pos.x.cpu())
                            .gather(0, agent.movableNodes)
                            .unsqueeze(0),
                            torch.flatten(agent.state.pos.y.cpu())
                            .gather(0, agent.movableNodes)
                            .unsqueeze(0),
                        ),
                        0,
                    )
                    movableNodes_history.append(movableNodes)
                if i >= washout_time:
                    target_speed.append(initXspeed)
                else:
                    target_speed.append(0)
                simulenv.plot.setRangeIndiv(
                    torch.mean(agent.state.pos.x).item() + frameWidth,
                    torch.mean(agent.state.pos.x).item() - frameWidth,
                    frameWidth * 1.5,
                    -frameWidth * 0.5,
                )
            if i == washout_time - 1:
                xScale, _ = str((initXpos // 10) * 10).split(".")
                xScale = float(xScale)
                initXspeed = torch.mean(agent_speed).item()

            agent_speed = (
                torch.mean(torch.stack(agent.state.vel_history, 0)[:, 0, :], 1)
                .view(-1, self.tau)
                .mean(1)
            )
            target_speed = (
                torch.tensor(target_speed, dtype=torch.float64)
                .view(-1, self.tau)
                .mean(1)
            )

            loss = criterion(agent_speed, target_speed)
            _ = print_current_params(agent, i, loss.item(), "SWG", None)
            loss_history.append(loss.item())

        verletSim.endSimulation()
        movableNodes_history = torch.stack(movableNodes_history, 0)
        epoch_array = np.arange(0, epoch, 1)
        plotLines(
            [[epoch_array, loss_history, None, "tab:blue"]],
            "epoch",
            "loss",
            None,
            fileSaveDir + "loss.png",
            marker=False,
        )
        save_params_as_json(self.params, fileSaveDir)
        saveNPY(np.array(loss_history), fileSaveDir + "loss.npy")

        simulenv.plot.setMovie(False)
        movableNodes_history = None
        agent_speed_bifurcation = []
        fixed_force_time = 300
        gradation_scale = 100
        fixed_force_time_in_gradation = 300
        force_gradation_time = fixed_force_time_in_gradation * (gradation_scale * 3 + 1)
        mid_washout_time = washout_time * (gradation_scale * 3)
        self.simTime = epoch = washout_time + force_gradation_time + mid_washout_time
        gradationForce = []
        stackFlag = False
        agent_speeds = []

        ## untrained behavior (half of trained force & gradation force)
        for i in range(epoch):
            target_speed = []
            if i < washout_time:
                wind_val = 0
            else:
                if i == washout_time:
                    print("----------- washout -> untrained behavior -----------")
                    wind_val = wind_val_prev = -30
                elif (i - washout_time) % (
                    fixed_force_time_in_gradation + washout_time
                ) == fixed_force_time_in_gradation:
                    wind_val_prev = wind_val
                    wind_val = 0
                    agent = self.loadAgent(self.loadFileSuffix, keepStates=True)
                    _, simulenv = self.initSet(
                        self.initSize,
                        self.tau,
                        _simtime=self.simTime,
                        centerFlag=True,
                        bias=True,
                        controlType="sine",
                        generatorNum=generatorNum,
                        amplitude=0.4,
                        ground=True,
                    )
                    agent = self.set_requiresGrads(agent, False, generator=True)
                elif (i - washout_time) % (
                    fixed_force_time_in_gradation + washout_time
                ) == 0:
                    wind_val = wind_val_prev + 15 / gradation_scale
            if wind:
                agent.morph.environment.setWind(wind_val * self.inputScale)
            else:
                agent.morph.environment.setWind(None)
            agent.state.resetSavePosType(self.savePosType)

            for _ in range(updateTimeScale):
                target_speed.append(initXspeed)
                agent.control.set_for_limitcycle(agent.morph, self.inputScale, None)
                verletSim = VerletSimulation(simulenv, agent)
                verletSim.iterationNumber = iterationNumber
                verletSim.runSimulationOneStep()
                iterationNumber = verletSim.iterationNumber
                movableNodes = torch.cat(
                    (
                        torch.flatten(agent.state.pos.x.cpu())
                        .gather(0, agent.movableNodes)
                        .unsqueeze(0),
                        torch.flatten(agent.state.pos.y.cpu())
                        .gather(0, agent.movableNodes)
                        .unsqueeze(0),
                    ),
                    0,
                )
            agent_speed = (
                torch.mean(torch.stack(agent.state.vel_history, 0)[:, 0, :], 1)
                .view(-1, self.tau)
                .mean(1)
            )
            x_speed = torch.mean(torch.stack(agent.state.vel_history, 0)[:, 0, :], 1)
            target_speed = (
                torch.tensor(target_speed, dtype=torch.float64)
                .view(-1, self.tau)
                .mean(1)
            )
            if i >= washout_time:
                if (
                    (i - washout_time) % (fixed_force_time_in_gradation + washout_time)
                    >= fixed_force_time_in_gradation - 100
                ) and (
                    (i - washout_time) % (fixed_force_time_in_gradation + washout_time)
                    < fixed_force_time_in_gradation
                ):
                    if (i - washout_time) % (
                        fixed_force_time_in_gradation + washout_time
                    ) == fixed_force_time_in_gradation - 1:
                        stackFlag = True
                    agent_speeds.extend(x_speed)
            loss = criterion(agent_speed, target_speed)
            _ = print_current_params(agent, i, loss.item(), "SWG", None)
            loss_history.append(loss.item())
            if (len(agent_speeds) > 0) and stackFlag:
                agent_speeds = torch.tensor(agent_speeds)
                agent_speed_bifurcation.append(agent_speeds)
                gradationForce.append(wind_val / 15)
                stackFlag = False
                agent_speeds = []

        verletSim.endSimulation()
        agent_speed_bifurcation = torch.stack(agent_speed_bifurcation, 0)
        np.savez_compressed(
            fileSaveDir + "agent_speed_bifurcation.npz",
            speed=agent_speed_bifurcation,
            forces=np.array(gradationForce),
        )
        agent_speed_bifurcation = agent_speed_bifurcation[:, -100:]
        wind = np.linspace(-2.0, 1.0, agent_speed_bifurcation.shape[0])
        target_speed = [
            target_speed[0] for _ in range(agent_speed_bifurcation.shape[0])
        ]
        plotSpeed(agent_speed_bifurcation, wind, target_speed, fileSaveDir + "speed")

        print("The results were saved in ", fileSaveDir)
