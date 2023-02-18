# Graph object and related classes.
# Majority of this code is pulled from the Artificial Intelligence: A Modern Approach reference code
# Repo located at https://github.com/aimacode/aima-python

from util import *
from re import split


class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)


class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        """The path cost will be the current cost + the link distance"""
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf
    
    def value(self, state):
        return self.h(state)


def loadGraphFromFile(file_path):
    file = open(file_path,'r')
    
    section_header = ""
    location_dict = {}
    # We assume for now that the paths are bidirectional, making this an undirected graph
    return_graph = UndirectedGraph()
    
    for line in file:
        if line.endswith(':\n'):
            # Lines that end with : are section headers and determine how we process the next section
            # Strip the last 2 characters (:\n) and treat that as our section header
            section_header = line[:-2]
            
        elif line.isspace():
            # We ignore empty lines
            continue
        
        elif section_header == "Locations":
            # These are the locations and their coordinates, store them in a dict
            # This Regex matches anything in the form "text = (number,number)"
            # The split call will return a list containing the text and two numbers
            split_location_data = split(r"(\w+)\s*=\s*\((\d+),(\d+)\)", line)
            
            if len(split_location_data) == 5:
                location_dict[split_location_data[1]] = (int(split_location_data[2]), int(split_location_data[3]))
                
            else:
                # The split call is exptected to return a list containing exactly 5 elements:
                # And empty string, the location name, the X coordinate, the Y coordinate, a second empty string
                # Anything else is malformed
                print("ERROR Loading File, Malformed Location: " + line)
                break
            
        elif section_header == "Paths":
            # These are the paths between locations, add them directly to the graph
            # This Regex matches anything in the form "text,text = number"
            # The split call will return a list containing the two texts and a number
            split_path_data = split(r"(\w+)(?:,(\w+))?\s*=\s*(\d+)", line)
            
            if len(split_path_data) == 5:
                return_graph.connect(split_path_data[1], split_path_data[2], int(split_path_data[3]))
                
            else:
                # The split call is exptected to return a list containing exactly 5 elements:
                # And empty string, the first location, the second location, the distance, a second empty string
                # Anything else is malformed
                print("ERROR Loading File, Malformed Path: " + line)
                break
        
        else:
            print("ERROR Loading File, Unrecognized Section Header: " + section_header)
            break
            
    return_graph.locations = location_dict
    
    return return_graph


# Romania map represented as a graph. Used as reference data
romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))
romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))
