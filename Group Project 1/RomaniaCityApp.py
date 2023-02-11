# Main Romania City App

from SimpleProblemSolvingAgent import SimpleProblemSolvingAgent, astar_search
from graph import romania_map


def main():
    # Read the cities from the user
    starting_city = input("Please enter the name of your starting city: ")
    destination_city = input("Please enter the name of your desired destination city: ")

    # Create the SPSA object from the map loaded earlier and the starting city
    state_dict = {"graph": romania_map, "initial": starting_city}
    spsa = SimpleProblemSolvingAgent(state_dict)

    # By default, the SPSA object uses best-first search. Call it by passing the desitination
    best_first_path = spsa(destination_city)
    print(best_first_path)

    # Change to A* search and call it
    spsa.set_search_type(astar_search)
    astar_path = spsa(destination_city)
    print(astar_path)
    return 0


if __name__ == "__main__":
    main()