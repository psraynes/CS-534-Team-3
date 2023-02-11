# Main Romania City App

from SimpleProblemSolvingAgent import SimpleProblemSolvingAgent
from graph import romania_map


def main():
    starting_city = input("Please enter the name of your starting city: ")
    destination_city = input("Please enter the name of your desired destination city: ")

    spsa = SimpleProblemSolvingAgent( romania_map, starting_city)

    desired_path = spsa(destination_city)
    print(desired_path)
    return 0


if __name__ == "__main__":
    main()