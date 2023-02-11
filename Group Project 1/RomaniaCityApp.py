# Main Romania City App

from SimpleProblemSolvingAgent import SimpleProblemSolvingAgent, astar_search
from graph import romania_map


def main():

    while True:
        valid_inputs = False

        while not valid_inputs:
            # Read the cities from the user, and determines if they are valid via the assignment criterion
            starting_city = input("Please enter the name of your starting city: ")
            destination_city = input("Please enter the name of your desired destination city: ")
            if starting_city == destination_city:
                print("Your starting location cannot be your destination.")
            elif starting_city not in romania_map.locations:
                print("Your starting city does not exist, please check your spelling.")
            elif destination_city not in romania_map.locations:
                print("Your destination city does not exist, please check your spelling.")
            else:
                valid_inputs = True

        # Create the SPSA object from the map loaded earlier and the starting city
        state_dict = {"graph": romania_map, "initial": starting_city}
        spsa = SimpleProblemSolvingAgent(state_dict)

        # By default, the SPSA object uses best-first search. Call it by passing the destination
        best_first_path = spsa(destination_city)
        best_cost =spsa.total_path_cost()
        print("Best First Path " + str(best_first_path) + "with a cost of " + str(best_cost))

        # Reset the sequence in the SPSA, Change to A* search and call it
        spsa.reset()
        spsa.set_search_type(astar_search)
        astar_path = spsa(destination_city)
        acost = spsa.total_path_cost()
        print("A* Path " + str(astar_path) + "with a cost of " + str(acost))

        want_to_go_again = input("Would you like to keep using this application? Y/N: ")
        if want_to_go_again == "Y":
            continue
        else:
            break


if __name__ == "__main__":
    main()
