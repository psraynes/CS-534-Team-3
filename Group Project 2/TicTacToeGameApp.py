# Main TicTacToe App

import re
from TicTacToeClass import TicTacToe
from game import random_player, query_player, min_max_player, alpha_beta_player


def main():
    while True:

        while True:
            # Query the user for the size of the board
            size = input("Please input a board size in the form HxV:")
            size_list = re.split("[x|X]", size)

            # Sizes are valid if there are exactly 2, they are both numbers, and are both greater than 3
            if len(size_list) != 2:
                print("Invalid number of sizes.")
                continue
            elif not size_list[0].isnumeric():
                print("First size is not a number.")
                continue
            elif not size_list[1].isnumeric():
                print("Second size is not a number.")
                continue
            elif int(size_list[0]) < 3 or int(size_list[1]) < 3:
                print("Sizes must be greater than or equal to 3.")
                continue
            else:
                break
        
        print("""Player Types:
1) Random
2) User
3) Min Max Search
4) Alpha Beta Search""")

        while True:
            player_type_x = input("Please select the player type for X:")

            # Player type is valid if it is a number between 1 and 4
            if not player_type_x.isnumeric() or not 1 <= int(player_type_x) <= 4:
                print("Unrecognized player type")
                continue
            else:
                break

        while True:
            player_type_o = input("Please select the player type for O:")

            # Player type is valid if it is a number between 1 and 4
            if not player_type_o.isnumeric() or not 1 <= int(player_type_o) <= 4:
                print("Unrecognized player type")
                continue
            else:
                break
        
        player_types = [random_player, query_player, min_max_player, alpha_beta_player]
        
        print()
        
        # Create the TicTacToe object
        # The size and player types are specified by the user
        ttt = TicTacToe(int(size_list[0]), int(size_list[1]))
        result = ttt.play_game(player_types[int(player_type_x) - 1], player_types[int(player_type_o) - 1])
        
        # Display Results
        print()
        if result == 1:
            print("X Wins!\n")
        elif result == -1:
            print("O Wins!\n")
        else:
            print("Draw!\n")
        
        # Query if the user wants to keep searching this map
        want_to_go_again = input("Would you like to continue to play Tic-Tac-Toe? Y/N: ")
        if want_to_go_again.casefold() == "y":
            # If they do, we restart this loop
            continue
        else:
            print("Thank You for Using Our App!")
            break
    

if __name__ == "__main__":
    main()
    