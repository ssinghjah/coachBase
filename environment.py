import sys

#sys.path.append('/root/aihetnets/Common/')
import pickle,copy
import gym, time
from rl_coach.environments.environment import Environment, LevelSelection, EnvironmentParameters
from rl_coach.spaces import DiscreteActionSpace, StateSpace, VectorObservationSpace, BoxActionSpace
from rl_coach.filters.filter import NoOutputFilter, NoInputFilter
from oct2py import octave
from rl_coach.base_parameters import VisualizationParameters
from typing import Union
import numpy as np
from gym import spaces
#from utils import ncr, saveArrayToFile, saveObj, makeDir
from enum import IntEnum
import uav360Env

MyEnvironmentOutputFilter = NoOutputFilter()
MyEnvironmentInputFilter = NoInputFilter()

class MyEnvironmentParameters(EnvironmentParameters):
    def __init__(self, level=None, seed=None, frame_skip=4, human_control=False, resultsFolder=""):
        super().__init__(level=level)
        #print("In UavHetNetFull From Coach's directory")
        self.default_input_filter = MyEnvironmentInputFilter()
        self.default_output_filter = MyEnvironmentOutputFilter()
        self.resultsFolder = resultsFolder

    @property
    def path(self):
        return 'rl_coach.environments.myEnvironment:MyEnvironment'

# Environment
class MyEnvironment(Environment):
    def __init__(self,
                 level: LevelSelection,
                 seed: int,
                 frame_skip: int,
                 human_control: bool,
                 custom_reward_threshold: Union[int, float],
                 visualization_parameters: VisualizationParameters,
                 resultsFolder: str,
                 **kwargs):
        super().__init__(visualization_parameters=VisualizationParameters(),
                         level=None,
                         frame_skip=4,
                         seed=None,
                         human_control=False,
                         custom_reward_threshold=None,
                         **kwargs)
        #self.rootPath = '/root/aihetnets'
        self.visualization_parameters = VisualizationParameters()
        #octave.addpath(self.rootPath + '/Common/')
        #self.realizationNumber = 0
        #self.realization = octave.loadRealization(self.realizationNumber)
        #self.realization.Nue = int(self.realization.Nue)
        #self.realization.Nmbs = int(self.realization.Nm)
        #self.realization.NumUAVs = int(self.realization.NumUAVs)
        #self.Nalpha = self.realization.AlphaRange.size
        #self.Nbeta = self.realization.BetaRange.size
        #self.Nrho = self.realization.RhoRange.size
        #self.NrhoPrime = self.realization.RhoPrimeRange.size
        #self.Ncre = self.realization.REBRange.size
        #self.REBRange = np.array([1, 3, 15])
        self._resetRealization()
        #print("RealizationNumber:" + str(self.realizationNumber))
        self.MAX_STEPS = 500
        self.bestReward = -sys.maxsize
        self.t = 0
        #self.bestRealization = None
        #self.bestRewards = []
        #Rho = 10.0 ** (35 / 10.0)
        #RhoPrime = 10.0 ** (-5 / 10.0)

        #numActionVariables = self.realization.Nmbs*3 + self.realization.NumUAVs*4
        #print([self.realization.REBRange[0][0], self.realization.RhoPrimeRange[0][0], 0, 0])
        #actionVariableLows = np.tile([int(self.realization.AlphaRange[0][0]), int(self.realization.BetaRange[0][0]), self.realization.RhoRange[0][0]], self.realization.Nmbs)
        #actionVariableLows = np.append(actionVariableLows, np.tile([self.realization.REBRange[0][0], self.realization.RhoPrimeRange[0][0], 0, 0], self.realization.NumUAVs))

        #actionVariableHighs = np.tile([self.realization.AlphaRange[0][-1], self.realization.BetaRange[0][-1], self.realization.RhoRange[0][-1]], self.realization.Nmbs)
        #actionVariableHighs = np.append(actionVariableHighs, np.tile([self.realization.REBRange[0][-1], self.realization.RhoPrimeRange[0][-1], self.realization.XMax, self.realization.YMax], self.realization.NumUAVs))
        numActionVariables =
        actionVariableLows =
        actionVariableHighs = 
        self.action_space = BoxActionSpace(shape=numActionVariables, low=actionVariableLows, high=actionVariableHighs)

        # TODO: State space should be updated to have a vector of ue and uav locations
        self.state_space = StateSpace({
            "measurements": VectorObservationSpace(self.realization.Nue + self.realization.NumUAVs, measurements_names=["ueLocations", "uavLocations"])})

        self.state = {}
        #self.resultsFolderPath = resultsFolder
        #makeDir(self.resultsFolderPath)
        

    def _resetRealization(self):
        self.t = 0
        '''
        self.realization.Xuav = [[]]
        self.realization.Yuav = [[]]
        self.realization.REB = [[]]
        self.realization.RhoPrime = [[]]
        for uabsNum in range(self.realization.NumUAVs):
            self.realization.Xuav[0].append(self.realization.StepSize / 2.0)
            self.realization.Yuav[0].append(self.realization.StepSize / 2.0)
            self.realization.REB[0].append(self.realization.REBRange[0][0])
            self.realization.RhoPrime[0].append(self.realization.RhoPrimeRange[0][0])
            
        self.realization.Alpha = [[]]
        self.realization.Beta = [[]]
        self.realization.Rho = [[]]
        for mbsNum in range(self.realization.Nmbs):
            self.realization.Alpha[0].append(self.realization.AlphaRange[0][0])
            self.realization.Beta[0].append(self.realization.BetaRange[0][0])
            self.realization.Rho[0].append(self.realization.RhoRange[0][0])
            
        self.realizationNumber += 1

        Xue = pickle.load(
            open(self.rootPath + '/Realizations/' + str(self.realization.Nue) + 'UE/Xue' + str(self.realizationNumber) + '.pkl',
                 'rb'))
        Yue = pickle.load(
            open(self.rootPath + '/Realizations/' + str(self.realization.Nue) + 'UE/Yue' + str(self.realizationNumber) + '.pkl',
                 'rb'))
        
        Xm = pickle.load(
            open(self.rootPath + '/Realizations/' + str(self.realization.Nmbs) + 'MBS/Xm' + str(self.realizationNumber) + '.pkl', 'rb'))
        Ym = pickle.load(
            open(self.rootPath + '/Realizations/' + str(self.realization.Nmbs) + 'MBS/Ym' + str(self.realizationNumber) + '.pkl', 'rb'))

        self.realization.Xue = np.array(Xue)
        self.realization.Yue = np.array(Yue)

        self.realization.Xm = np.array(Xm)
        self.realization.Ym = np.array(Ym)

        self.numSteps = 0
        self.done = False
        self.reward = 0
        self._update_state()

        self.bestReward = -sys.maxsize
        self.bestRealization = None
        '''
        
    def _update_state(self):
        # TODO: UE State and UAV States should be arrays
        # stateArray = []
        stateArray = uav360Env.getStateArray(self.t)
        # SNR, thetaP, phiP
        self.state = {"measurements": stateArray}


    # Get action of UABS as input, update the state accordingly
    def _step(self, action):
        thetaHPrime = action[0]
        phiNPrime = action[1]
        phiSPrime = action[2]
        qpI = action[3]
        qpO = action[4]
        
        # update self.reward as per action
        # update self.done depending on the scenario
        self.reward = uav360.calculateQoE(self.t, action)
        self.t += 1
        self.done = uav360.isDone(self.t)
        
        '''
        actionIndex = 0
        for mbsNum in range(self.realization.Nmbs):
            self.realization.Alpha[0][mbsNum] = action[actionIndex]
            actionIndex += 1
            self.realization.Beta[0][mbsNum] = action[actionIndex]
            actionIndex += 1
            self.realization.Rho[0][mbsNum] = action[actionIndex]
            actionIndex += 1
            
            
        for uabsNum in range(self.realization.NumUAVs):
            self.realization.REB[uabsNum] = action[actionIndex]
            actionIndex += 1
            self.realization.RhoPrime[uabsNum] = action[actionIndex]
            actionIndex += 1
            self.realization.Xuav[0][uabsNum] = action[actionIndex]
            actionIndex += 1
            self.realization.Yuav[0][uabsNum] = action[actionIndex]
            actionIndex += 1
            
#        input(self.realization)
        reward = 0
        done = False
        metric = float(octave.CalculateReward(self.realization))
        self.reward = metric
        
        if self.reward > self.bestReward:
            self.bestReward = self.reward
            self.bestRealization = copy.copy(self.realization)
            saveObj(self.bestRealization, self.resultsFolderPath + '/bestRealization' + str(self.realizationNumber) + ".pkl")

        self.numSteps += 1
        self.done = self.numSteps > self.MAX_STEPS
        if self.done:
            print("Best Reward:" + str(self.bestReward) + ", realization number:" + str(self.realizationNumber))
            self.bestRewards.append(self.bestReward)
            print("Best realization: " + str(self.bestRealization))
            saveArrayToFile(self.bestRewards, self.resultsFolderPath + '/rewards.csv')
        '''
        self._update_state()

    def _take_action(self, action):
        self._step(action)

    def reset(self):
        self._resetRealization()
        return self.state

    def _restart_environment_episode(self, force_environment_reset=False):
        self.reset()

    def get_rendered_image(self):
        pass

    def _render(self):
        pass

    def set_realization(self, realization):
        
        self.realization = realization

    def _get_obs(self):
        return self.state

    # Return values of cre, location(deltaX and deltaY), and rho(in dB)
    def _action_decoder_uabs(self, action):
        '''
        location = action % self.Nlocations
        xDelta = self.movementDecoder[location][0]*self.realization.StepSize;
        yDelta = self.movementDecoder[location][1]*self.realization.StepSize;
        return xDelta, yDelta

        xDelta = self.movementDecoder[action] * self.realization.StepSize;
        yDelta = 0;
        return xDelta, yDelta
        '''


from rl_coach.environments.environment import EnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter


# Parameters
class MyEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()

    @property
    def path(self):
        return 'environments.myEnvironment:MyEnvironment'
