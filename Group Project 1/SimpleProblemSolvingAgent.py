# A Simple Problem Solving Agent
# Takes a graph and can fine a path between 2 points via
# either Greedy Best-First Search or A*
# Basic architecture taken from Artificial Intelligence: A Modern Approach reference code
# Repo located at https://github.com/aimacode/aima-python

from graph import Graph, UndirectedGraph
from search import best_first_graph_search, astar_search


class SimpleProblemSolvingAgent:
    """
    [Figure 3.1]
    Abstract framework for a problem-solving agent.
    """

    def __init__(self, initial_state=None, search_type=best_first_graph_search):
        """State is an abstract representation of the state
        of the world, and seq is the list of actions required
        to get to a particular state from the initial state(root).
        Search_type is the type of search we want to perform. This
        can be changed after initialization."""
        self.state = initial_state
        self.seq = []
        self.search_type = search_type

    def __call__(self, percept):
        """[Figure 3.1] Formulate a goal and problem, then
        search for a sequence of actions to solve it."""
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, state, percept):
        raise NotImplementedError

    def formulate_goal(self, state):
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        raise NotImplementedError

    def search(self, problem):
        return self.search_type(problem)

    def set_search_type(self, search_type):
        if (search_type == best_first_graph_search or search_type == astar_search):
            self.search_type = search_type
        else:
            raise NotImplementedError
        