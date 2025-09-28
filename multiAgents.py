# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()  # 剩余食物
        foodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        Capsules = successorGameState.getCapsules()

        score = currentGameState.getScore()

        # 需要考虑 food ghost 胶囊 当前得分，通过给这些赋予参数或者权重来决定最后的得分
        "*** YOUR CODE HERE ***"
        #         当前得分情况
        if successorGameState.getNumFood() < currentGameState.getNumFood():
            score += 100

        #         food情况
        if foodList:
            food_dis = min(manhattanDistance(newPos, food_position) for food_position in foodList)
            #         防止part_food出现0的情况
            score += 10 / (food_dis + 1)

        #         ghost情况
        for gState in successorGameState.getGhostStates():
            gpos = gState.getPosition()
            dist = manhattanDistance(newPos, gpos)
            if gState.scaredTimer > 0:
                score += 20 / (dist + 1)
            else:
                if dist <= 1:
                    score -= 200
                else:
                    score -= 10 / (dist + 1)

        #         胶囊
        if Capsules:
            cap_dist = min(manhattanDistance(newPos, pos) for pos in Capsules)
            score += 5 / (cap_dist + 1)

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def minimax(state, agentIndex, pacmanMovesSoFar):
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), None)

            if pacmanMovesSoFar == self.depth:
                return (self.evaluationFunction(state), None)

            legalActions = state.getLegalActions(agentIndex)
            if len(legalActions) == 0:
                return (self.evaluationFunction(state), None)

            # pacman
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)

                    nextAgent = (agentIndex + 1) % numAgents
                    nextPacmanMoves = pacmanMovesSoFar
                    if nextAgent == 0:
                        nextPacmanMoves += 1

                    val, _ = minimax(successor, nextAgent, nextPacmanMoves)
                    if val > bestValue:
                        bestValue = val
                        bestAction = action
                return (bestValue, bestAction)
            #             ghost
            else:
                worstValue = float("inf")
                worstAction = None
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    nextAgent = (agentIndex + 1) % numAgents
                    nextPacmanMoves = pacmanMovesSoFar
                    if nextAgent == 0:
                        nextPacmanMoves += 1

                    val, _ = minimax(successor, nextAgent, nextPacmanMoves)
                    if val < worstValue:
                        worstValue = val
                        worstAction = action

                return (worstValue, None)

        value, action = minimax(gameState, 0, 0)
        return action if action is not None else Directions.STOP


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def alphabeta(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if len(legalActions) == 0:
                return self.evaluationFunction(state)

            #             pacman
            if agentIndex == 0:
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(1, depth, successor, alpha, beta))
                    if value > beta:
                        return value;
                    alpha = max(alpha, value)
                return value

            #             ghost
            else:
                value = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth
                if agentIndex == numAgents - 1:
                    nextAgent = 0
                    nextDepth += 1
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(nextAgent, nextDepth, successor, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        alpha = float('-inf')
        beta = float('inf')
        bestValue = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(1, 0, successor, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            if bestValue > beta:
                return bestAction
            alpha = max(alpha, bestValue)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state,depth,agentIndex):
            #游戏结束判断
            if depth==self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legalActions=state.getLegalActions(agentIndex)

            # for pacman
            if agentIndex==0:
                return max(expectimax(state.generateSuccessor(0,action),depth,1) for action in legalActions)

            # for ghost
            else:
                nextAgent=agentIndex+1

                if nextAgent==state.getNumAgents():
                    nextdepth=depth+1
                    nextAgent=0

                else:
                    nextdepth=depth

                legalActions=[a for a in legalActions if a != Directions.STOP]
                if not legalActions:
                    return self.evaluationFunction(state)

                values=[]
                for action in legalActions:
                    try:
                        successor = state.generateSuccessor(agentIndex, action)
                        values.append(expectimax(successor, nextdepth, nextAgent))
                    except Exception:
                        # 跳过非法动作
                        continue

                return sum(values)/len(values) if values else self.evaluationFunction(state)

        bestAction=max(gameState.getLegalActions(0),key=lambda action:expectimax(gameState.generateSuccessor(0,action),0,1))
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <
    该评价函数结合了当前分数，食物距离，鬼魂距离，和能量豆距离，pacman会倾向于靠近食物和能量豆，避开危险的
    鬼魂，但是当鬼魂处于scared情况，则靠经鬼魂，这样可以让pacman在有限深度下更加激进的吃豆子，同时避开鬼
    魂
    >
    """
    "*** YOUR CODE HERE ***"
    foodList=currentGameState.getFood().asList()
    pacmanPos=currentGameState.getPacmanPosition()
    score=currentGameState.getScore()
    capsules=currentGameState.getCapsules()
    ghosts=currentGameState.getGhostStates()

#     food
    from util import manhattanDistance
    foodDistance=[manhattanDistance(pacmanPos,foodPos)for foodPos in foodList]
    if foodDistance:
        foodMinDis=min(foodDistance)
        score+=1.0/foodMinDis
    else:
        score+=0

#     ghost
    for ghost in ghosts:
        ghostDis=manhattanDistance(pacmanPos,ghost.getPosition())
        if ghost.scaredTimer>0:
            score+=10.0/ghostDis
        else:
            if ghostDis<=1:
                score-=100
            else:
                score-=2.0/ghostDis

#     capsules
    capsulesDis=[manhattanDistance(pacmanPos,capsuleDis)for capsuleDis in capsules]
    if capsulesDis:
        capsuleMinDis=min(capsulesDis)
        score+=5.0/capsuleMinDis
    else:
        score+=0

    return score


# Abbreviation
better = betterEvaluationFunction
