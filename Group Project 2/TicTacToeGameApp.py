# Main TicTacToe App

from TicTacToeClass import TicTacToe
from game import random_player, query_player


def main():
    while True:
        # Query the user for the size of the board
        size = input("Please input a board size in the form HxV:")
        size_list = size.split("x")
        
        # TODO: Verify that the input size is valid aka numbers and >3
        
        print()
        
        # Create the TicTacToe object
        ttt = TicTacToe(int(size_list[0]), int(size_list[1]))
        result = ttt.play_game(random_player, random_player)
        
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
    