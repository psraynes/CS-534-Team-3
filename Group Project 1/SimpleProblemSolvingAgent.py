# A Simple Problem Solving Agent
# Takes a graph and can fine a path between 2 points via
# either Greedy Best-First Search or A*
# Basic architecture taken from Artificial Intelligence: A Modern Approach reference code
# Repo located at https://github.com/aimacode/aima-python

from graph import Graph, UndirectedGraph
from search import best_first_graph_search, astar_search, Node


class SimpleProblemSolvingAgent:
    """
    Abstract framework for a problem-solving agent, modified to
    be a graph searching agent.
    """

    def __init__(self, initial_state={}, search_type=best_first_graph_search):
        """State is a dictionary containing the current and goal locations in the graph,
        seq is the list of locations required to get to a particular state from the initial
        state(root). Search_type is the type of search we want to perform. This
        can be changed after initialization."""
        self.state = initial_state
        self.seq = []
        self.search_type = search_type

    def __call__(self, percept):
        """Formulate a goal and problem, then search for a sequence of actions
        to solve it. Returns a list of locations representing the path"""
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq

    def update_state(self, state, percept):
        # For our use case, this will be setting the goal of the search
        raise NotImplementedError

    def formulate_goal(self, state):
        # For our use case, this will be reading the goal of the search
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        # Problem created here will be a GraphProblem
        raise NotImplementedError

    def search(self, problem):
        results = self.search_type(problem, problem.h)
        # Result will be a Node representing the end of the path or None if no path
        if results is None:
            return None
        else:
            path = []
            for result in results.path():
                path.append(result)

    def set_search_type(self, search_type):
        if search_type == best_first_graph_search or search_type == astar_search:
            self.search_type = search_type
        else:
            raise NotImplementedError
        