# Main Romania City App

from SimpleProblemSolvingAgent import SimpleProblemSolvingAgent, astar_search
from graph import romania_map


def main():
    # Load map from text file here
    while True:
        while True:
            # Read the cities from the user, and determines if they are valid
            while True:
                starting_city = input("Please enter the name of your starting city: ")
                
                # Starting city is valid if it exists on our map
                if starting_city not in romania_map.locations:
                    print("Your starting city does not exist, please check your spelling.")
                    continue
                else:
                    break
            
            while True:
                destination_city = input("Please enter the name of your desired destination city: ")
                
                # Destination city is valid if it exists on our map
                if destination_city not in romania_map.locations:
                    print("Your destination city does not exist, please check your spelling.")
                    continue
                else:
                    break
            
            # Starting and Destination cities cannot match
            if starting_city == destination_city:
                print("Your starting location cannot be your destination.")
            else:
                break

        print("\n")
        
        # Create the SPSA object from the map loaded earlier and the starting city
        state_dict = {"graph": romania_map, "initial": starting_city}
        spsa = SimpleProblemSolvingAgent(state_dict)

        # By default, the SPSA object uses best-first search. Call it by passing the destination
        best_first_path = spsa(destination_city)
        print("Best First Search: Total Cost = " + str(spsa.total_path_cost()))
        print(str(best_first_path) + "\n")

        # Reset the path in the SPSA, Change to A* search and call it
        spsa.reset()
        spsa.set_search_type(astar_search)
        astar_path = spsa(destination_city)
        print("A* Search: Total Cost = " + str(spsa.total_path_cost()))
        print(str(astar_path) + "\n")

        # Query if the user wants to keep searching this map
        want_to_go_again = input("Would you like to continue to search the map? Y/N: ")
        if want_to_go_again.casefold() == "y":
            continue
        else:
            print("Thank You for Using Our App!")
            break


if __name__ == "__main__":
    main()
