# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    fringe=Stack()
    start_state=problem.getStartState()

    fringe.push((start_state,[]))
    visited=set()

    while not fringe.isEmpty():
        state,actions=fringe.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)
            for next_state,action,cost in problem.getSuccessors(state):
                if next_state not in visited:
                    fringe.push((next_state,actions + [action]))

    return []



    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    queue = Queue()
    start_state=problem.getStartState()
    queue.push((start_state,[]))
    visited=set()

    while not queue.isEmpty():
        state,actions=queue.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)
            for next_state,action,cost in problem.getSuccessors(state):
                queue.push((next_state,actions+[action]))

    return []
    # util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    pq=PriorityQueue()
    start_state=problem.getStartState()
    pq.push((start_state,[],0),0)
    visited=set()

    while not pq.isEmpty():
        state,actions,costs=pq.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)
            for next_state,action,cost in problem.getSuccessors(state):
                # pq.push((next_state,actions+[action]),costs+cost)
                # pq.push((next_state,actions+[action],costs+cost),costs+cost)
                pq.update((next_state,actions+[action],costs+cost),costs+cost)

    return []

    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueueWithFunction
    pqf=PriorityQueueWithFunction(lambda x:x[2]+heuristic(x[0],problem))
    start_state=problem.getStartState()
    pqf.push((start_state,[],0))
    visited= {}

    while not pqf.isEmpty():
        state,actions,costs=pqf.pop()
        if problem.isGoalState(state):
            return actions
        if state not in visited or costs<visited[state]:
            visited[state]=costs
            for next_state,action,cost in problem.getSuccessors(state):
                pqf.push((next_state,actions+[action],costs+cost))

    return []
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
