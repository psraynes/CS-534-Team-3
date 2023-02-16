# Main Romania City App

from SimpleProblemSolvingAgent import SimpleProblemSolvingAgent, astar_search
from graph import loadGraphFromFile

def main():
    
    load_successful = False
    while not load_successful:
        # Ask the user for the location of the map text file and then load it
        # Performed in a loop to verify that the user has provided a valid file
        map_file = input("Please enter the file location of a map.txt file: ")
        try:
            loaded_map = loadGraphFromFile(map_file)
        except FileNotFoundError:
            print("""The provided file does not exist, please double check your spelling and try again.
If the problem persists, verify that you have the correct file path.
""")
        else:
            load_successful = True
    
    while True:
        # Reading the inputs from the user is done in a while loop to allow us to 
        # ask them to correct any invalid data
        while True:
            # Read the cities from the user, and determines if they are valid
            while True:
                starting_city = input("Please enter the name of your starting city: ")
                
                # Starting city is valid if it exists on our map
                if starting_city not in loaded_map.locations:
                    print("Your starting city does not exist, please check your spelling.")
                    continue
                else:
                    break
            
            while True:
                destination_city = input("Please enter the name of your desired destination city: ")
                
                # Destination city is valid if it exists on our map
                if destination_city not in loaded_map.locations:
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
        state_dict = {"graph": loaded_map, "initial": starting_city}
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
            # If they do, we restart this loop
            continue
        else:
            print("Thank You for Using Our App!")
            break


if __name__ == "__main__":
    main()
