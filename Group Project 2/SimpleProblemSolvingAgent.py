# A Simple Problem Solving Agent
# Takes a graph and can fine a path between 2 points via
# either Greedy Best-First Search or A*
# Basic architecture taken from Artificial Intelligence: A Modern Approach reference code
# Repo located at https://github.com/aimacode/aima-python

from graph import Graph, UndirectedGraph, GraphProblem
from search import best_first_graph_search, astar_search, Node, hill_climbing, simulated_annealing, exp_schedule


class SimpleProblemSolvingAgent:
    """
    Abstract framework for a problem-solving agent, modified to
    be a graph searching agent.
    """

    def __init__(self, initial_state=None, search_type=best_first_graph_search):
        """State is a dictionary containing the graph, the current and the goal locations in the graph,
        seq is the list of locations required to get to a particular state from the initial
        state(root). Search_type is the type of search we want to perform. This
        can be changed after initialization."""
        self.state = initial_state
        self.seq = []
        self.search_type = search_type

    def __call__(self, percept):
        """Formulate a goal and problem, then search for a sequence of actions
        to solve it. Returns a list of locations representing the path"""
        self.update_state(percept)
        if not self.seq:
            self.formulate_goal()
            problem = self.formulate_problem()
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq

    def update_state(self, percept):
        # For our use case, this will be setting the goal of the search
        self.state["goal"] = percept
        return self.state

    def formulate_goal(self):
        # For our use case, this will be reading the goal of the search
        return self.state.get("goal")

    def formulate_problem(self):
        # Problem created combines all the previous information and passes it all to the searches
        problem = GraphProblem(self.state.get("initial"), self.state.get("goal"), self.state.get("graph"))
        return problem

    def search(self, problem):
        results = self.search_type(problem, problem.h)
        # Result will be a list of Nodes representing the path or None if no path
        if results is None:
            return None
        else:
            # print(results)
            # print(results.path())
            return results.path()

    def set_search_type(self, search_type):
        # Added hill_climbing and simulated_annealing
        if (search_type == best_first_graph_search or search_type == astar_search
                or search_type == hill_climbing or search_type == simulated_annealing):
            self.search_type = search_type
        else:
            # We only have Best-First,A* implemented, Hill Climbing, and
            # Simulated Annealing. Any other search type would cause errors
            raise NotImplementedError
    
    def reset(self):
        # Reset the current sequence to start fresh
        self.seq = []
        
    def total_path_cost(self):
        # Return the total path cost.
        if self.seq:
            # Each node contains the path cost to get to them, so read that value from the last node
            return self.seq[-1].path_cost          
        else:
            # There is no path found, so return 0
            return 0
        