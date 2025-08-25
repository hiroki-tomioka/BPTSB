import datetime
import itertools
import os
import glob
import numpy as np
from PIL import Image
from collections import deque

from matplotlib.mlab import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.animation import ArtistAnimation

from utils import *


class Plotter(object):
    """Create instance with plotting properties"""

    def __init__(
        self,
        border=2,
        plotCycle=10,
        startPlot=0,
        text=False,
        plot=False,
        pauseTime=0.00001,
        movieName="out",
        color=True,
        delete=True,
        only_central_red=False,
        colored_comp=[[], []],
        saveFirstImg=None,
        trajectory=False,
        colored_points=[],
        all_black=None,
        node_colors=None,
        axis=False,
        xScale=None,
        removeWashout=0,
    ):
        self.plot = plot
        self.color = color
        self.delete = delete
        self.first_it = True
        self.tmpDir = None
        self.saveFileName = None
        self.only_central_is_red = only_central_red
        self.colored_comp = colored_comp
        self.all_black = all_black
        self.node_colors = node_colors
        self.colored_comp = colored_comp
        self.saveFirstImg = saveFirstImg
        self.trajectory = trajectory
        if self.trajectory:
            self.centralPosX, self.centralPosY = deque([], maxlen=20), deque(
                [], maxlen=20
            )
        self.colored_points = colored_points
        self.axis = axis
        self.xScale = xScale
        self.removeWashout = removeWashout

        if plot:
            self.border = border
            self.plotCycle = plotCycle
            self.startPlot = startPlot

            self.pauseTime = pauseTime
            self.fig = plt.subplots(
                figsize=(3, 3),
                dpi=300,
            )
            if text:
                self.text = plt.text(
                    2.0, 2.2, "test", ha="center", va="center", size="medium"
                )
            else:
                self.text = None
            self.init = True  # plot in initial modus
            xlist = []
            ylist = []
            (self.plt_line,) = plt.plot(xlist, ylist)
            # init drawing
            self.fileList = []
            self.fps = 30
            self.frame = 0
            if os.name == "posix":
                self.tmpDir = f"tmp/{datetime.datetime.now()}/"
                mkdir_p(self.tmpDir)
                self.IMGname = self.tmpDir + "tmp%04d.png"
            else:
                self.IMGname = f"tmp/{datetime.datetime.now()}/_tmp%04d.png"
            self.movieName = movieName
            if color:
                jet = cm = plt.get_cmap("hsv")
                cNorm = colors.Normalize(vmin=0.5, vmax=1.5)
                self.colorMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    def setMovie(
        self,
        key,
        num=None,
        plotCycle=10,
        only_central_red=False,
        colored_comp=[[], []],
        text=False,
        saveFirstImg=None,
        trajectory=False,
        axis=False,
        colored_points=[],
        all_black=False,
        node_colors=None,
        xScale=None,
        removeWashout=0,
    ):
        """Set properties of instance for movies"""
        if key != False:
            self.fileSaveDir = key
            self.movieNum = num
            self.__init__(
                text=text,
                plot=True,
                plotCycle=plotCycle,
                pauseTime=0.1,
                only_central_red=only_central_red,
                colored_comp=colored_comp,
                saveFirstImg=saveFirstImg,
                trajectory=trajectory,
                colored_points=colored_points,
                all_black=all_black,
                node_colors=node_colors,
                axis=axis,
                xScale=xScale,
                removeWashout=removeWashout,
            )
        else:
            self.fileSaveDir = None
            self.plot = False

    def _construct_plot_lines(self, xpos, ypos, connections):
        """Update properties of instance based on x and y coordinates and connections matrix"""
        if self.init:
            self.init = False
            plt.ylim(-self.border, self.border + 1.2 * np.max(ypos))
            self.xplotwidth = max(xpos) + self.border - (min(xpos) - self.border)

        xlist = []
        ylist = []
        for i, j in itertools.product(range(len(xpos)), range(len(ypos))):
            if connections[i, j]:
                xlist.append(xpos[i])
                xlist.append(xpos[j])
                xlist.append(None)
                ylist.append(ypos[i])
                ylist.append(ypos[j])
                ylist.append(None)

        return xlist, ylist

    def update(self, agent, iterationCount=0):
        self.draw(agent, iterationCount)

    def draw(self, agent, iterationCount):
        plt.rcParams["font.size"] = 10
        """Draw a plot of the agent parameters"""
        # draw agent
        if self.plot:
            if type(agent.morph.initSize) == torch.Tensor:
                self.border = agent.morph.initSize.item() * 0.1
            else:
                self.border = agent.morph.initSize * 0.1
            if iterationCount % self.plotCycle == self.startPlot % self.plotCycle:
                xpos, ypos, connections = agent._getAgentPos2D()
                if xpos.ndim > 1:
                    xpos = xpos[0]
                    ypos = ypos[0]
                xpos = xpos.cpu().detach().numpy()
                ypos = ypos.cpu().detach().numpy()
                if self.color:
                    plt.cla()
                    if self.text is not None:
                        width = height = self.Xmax - self.Xmin + 2 * self.border
                        self.text = plt.text(
                            self.Xmax + self.border - width * 0.05,
                            self.Ymax + self.border - height * 0.05,
                            "",
                            ha="right",
                            va="top",
                            size=10,
                        )
                    if self.init:
                        self.init = False
                    plt.ylim(self.Ymin - self.border, self.border + self.Ymax)
                    self.xplotwidth = (
                        max(xpos) + self.border - (min(xpos) - self.border)
                    )

                    compNum = 0
                    _connections = np.triu(connections)
                    for i, j in itertools.product(range(len(xpos)), range(len(ypos))):
                        if _connections[i, j]:
                            if (compNum in self.colored_comp[0]) or (
                                self.colored_comp[0] == [-1]
                            ):
                                plt.plot(
                                    [xpos[i], xpos[j]], [ypos[i], ypos[j]], color="b"
                                )
                            else:
                                plt.plot(
                                    [xpos[i], xpos[j]], [ypos[i], ypos[j]], color="0.5"
                                )
                            if compNum in self.colored_comp[1]:
                                plt.plot(
                                    [xpos[i], xpos[j]], [ypos[i], ypos[j]], color="g"
                                )
                            compNum += 1

                else:
                    xlist, ylist = self._construct_plot_lines(xpos, ypos, connections)
                    self.plt_line.set_xdata(xlist)
                    self.plt_line.set_ydata(ylist)
                plt.xlim(self.Xmin - self.border, self.border + self.Xmax)
                if agent.movableNodes.ndim == 1:
                    movableList = agent.movableNodes.tolist()
                else:
                    movableList = agent.movableNodes.tolist()[0]
                fixed_nodes = list(
                    set([i for i in range(agent.morph.noNodes)]) - set(movableList)
                )
                for k in fixed_nodes:
                    plt.plot(xpos[k], ypos[k], "wo", markersize=5, markeredgecolor="k")
                for k in movableList:
                    plt.plot(xpos[k], ypos[k], "ko", markersize=3)
                if self.trajectory:
                    self.centralPosX.append(xpos[-1])
                    self.centralPosY.append(ypos[-1])
                    plt.plot(
                        self.centralPosX,
                        self.centralPosY,
                        "ro",
                        markersize=3,
                        alpha=0.5,
                    )
                if self.only_central_is_red:
                    red_nodes = [-1]
                elif self.all_black:
                    red_nodes = []
                else:
                    red_nodes = agent.movableNodes.tolist()
                for k in red_nodes:
                    plt.plot(xpos[k], ypos[k], "ro", markersize=3)
                for k in self.colored_points:
                    plt.plot(xpos[k], ypos[k], "mo", markersize=3)
                if self.node_colors is not None:
                    for node_color in self.node_colors:
                        if (len(node_color) < 3) or node_color[2] == "all-order":
                            plt.plot(
                                xpos[node_color[0]],
                                ypos[node_color[0]],
                                "o",
                                color=node_color[1],
                                markersize=5,
                            )
                        else:
                            movableList = agent.movableNodes.tolist()
                            plt.plot(
                                xpos[movableList[node_color[0]]],
                                ypos[movableList[node_color[0]]],
                                "o",
                                color=node_color[1],
                                markersize=5,
                            )
                if agent.morph.environment.ground:
                    plt.axhline(0, color="black")
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                if self.xScale is not None:
                    import matplotlib as mpl

                    plt.gca().xaxis.set_major_formatter(
                        mpl.ticker.ScalarFormatter(self.xScale, useMathText=True)
                    )

                if self.text is not None:
                    self.text.set_text(agent.printState())
                self.frame += 1
                fname = self.IMGname % self.frame
                if not self.axis:
                    plt.gca().axis("off")
                plt.savefig(fname, bbox_inches="tight")
                self.fileList.append(fname)

    def end(self):
        if self.plot:
            _frameList = sorted(glob.glob(self.tmpDir + "*.png"))
            if self.removeWashout > 0:
                _frameList = _frameList[self.removeWashout :]
            plt.clf()
            plt.close()
            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)
            ax.axis("off")
            ax.set_aspect("equal")
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            frameList = []
            for i, frame in enumerate(_frameList):
                frameList.append([plt.imshow(Image.open(frame))])
                if (i == 0) and (self.saveFirstImg is not None):
                    plt.savefig(
                        self.fileSaveDir + f"initPosition_{self.saveFirstImg}.png"
                    )
            anim = ArtistAnimation(fig, frameList, interval=100, repeat=False)

            anim.save(self.fileSaveDir + f"movie_{self.movieNum}.gif")
            plt.clf()
            plt.close()

            if self.delete:
                for fname in self.fileList:
                    os.remove(fname)
                os.removedirs(self.tmpDir)
            self.fileList = []
            self.frame = 0
            self.tmpDir = None

    def setRange(self, Xmax, Xmin, Ymax, Ymin):
        """Set screen range"""
        maxNum = max(abs(Xmax), abs(Xmin), abs(Ymax), abs(Ymin))
        self.Xmax, self.Xmin, self.Ymax, self.Ymin = (
            abs(maxNum),
            -abs(maxNum),
            abs(maxNum),
            -abs(maxNum),
        )

    def setRangeIndiv(self, Xmax, Xmin, Ymax, Ymin):
        """Set screen range"""
        self.Xmax, self.Xmin, self.Ymax, self.Ymin = Xmax, Xmin, Ymax, Ymin

    def setFigure(self):
        plt.clf()
        plt.close()
        self.fig = plt.subplots(
            figsize=(3, 3),
            dpi=300,
        )

    def lightenVar(self):
        self.fig, self.IMGname, self.movieName, self.colorMap = None, None, None, None


class SimulationEnvironment(object):
    """Class with general Parameters for Simulations but not bound to a specific agent"""

    def __init__(
        self,
        timeStep=0.005,
        simulationLength=10000,
        plot=Plotter(),
        verlet=True,
    ):
        self.timeStep = timeStep  # time step size
        self.plot = plot  # plotting
        assert isinstance(simulationLength, int), "simulation length should be integer"
        self.simulationLength = simulationLength  # number of iterations
        self.verlet = verlet

    def setSimulationLength(self, simulationLength):
        self.simulationLength = simulationLength

    def end(self):
        return self.plot.end()


class Simulation(object):
    """class to run and store simulation runs, for better results the
    child class VerletSimulation is advised"""

    def __init__(self, simulEnv, agent):
        self.simulEnv = simulEnv
        self.agent = agent
        self.initState = agent.getState()
        self.endState = None
        self.iterationNumber = 0
        self.atimes = []
        self.vtimes = []

    def simulateStep(self, tau):
        """Euler integration for a single time step"""
        simStep = round(self.agent.state.currentTime * 100)
        A = self.agent.computeAcceleration(tau, simStep)
        V = self.agent.getVelocity()
        self.iterationNumber += 1
        return self.agent.changeState(self.simulEnv.timeStep, V, A)

    def runSimulation(self):
        """Runs a simulation over a number of iterations"""
        for _ in range(self.simulEnv.simulationLength):
            self.simulateStep(self.agent.control.tau)
            self.simulEnv.plot.update(self.agent, self.iterationNumber)
        self.simulEnv.end()
        self.endState = self.agent.getState()
        return

    def runSimulationOneStep(self):
        self.simulateStep(self.agent.control.tau)
        self.simulEnv.plot.update(self.agent, self.iterationNumber)

    def endSimulation(self):
        self.simulEnv.end()
        self.endState = self.agent.getState()
        return


class VerletSimulation(Simulation):
    """Use the Verlet algorithm to obtain more accurate simulations"""

    def __init__(self, simulEnv, agent, reset=False):
        if reset:
            agent.reset()
        super(VerletSimulation, self).__init__(simulEnv, agent)
        self.Aold = SpaceList(torch.zeros_like(agent.getVelocity().matrix))

    def simulateStep(self, tau):
        V = self.agent.getVelocity()
        timeStep = self.simulEnv.timeStep
        self.Aold = self.agent.changeStateVerlet(timeStep, V, self.Aold, tau=tau)
        self.iterationNumber += 1
        return self.Aold


class VerletSimulationBatch(Simulation):
    """Use the Verlet algorithm to obtain more accurate simulations"""

    def __init__(self, simulEnv, agent, reset=False):
        if reset:
            agent.reset()
        super(VerletSimulationBatch, self).__init__(simulEnv, agent)
        self.Aold = SpaceListBatch(torch.zeros_like(agent.getVelocity().matrix))

    def simulateStep(self, tau):
        V = self.agent.getVelocity()
        timeStep = self.simulEnv.timeStep
        self.Aold = self.agent.changeStateVerlet(timeStep, V, self.Aold, tau=tau)
        self.iterationNumber += 1
        return self.Aold
