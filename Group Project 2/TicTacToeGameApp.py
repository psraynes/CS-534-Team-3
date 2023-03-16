# Main TicTacToe App

from TicTacToeClass import TicTacToe
from game import random_player, query_player, min_max_player, alpha_beta_player


def main():
    while True:
        # Query the user for the size of the board
        size = input("Please input a board size in the form HxV:")
        size_list = size.split("x")
        
        # TODO: Verify that the input size is valid aka numbers and >3
        
        print("""Player Types:
1) Random
2) User
3) Min Max Search
4) Alpha Beta Search""")
        player_type_X = input("Please select the player type for X:")
        player_type_O = input("Please select the player type for O:")
        
        player_types = [random_player, query_player, min_max_player, alpha_beta_player]
        
        # TODO: Verify that the input player types are valid aka numbers and 1-4
        
        print()
        
        # Create the TicTacToe object
        # The size and player types are specified by the user
        ttt = TicTacToe(int(size_list[0]), int(size_list[1]))
        result = ttt.play_game(player_types[int(player_type_X) - 1], player_types[int(player_type_O) - 1])
        
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
    