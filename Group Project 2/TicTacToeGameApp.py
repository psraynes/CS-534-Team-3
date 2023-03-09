# Main TicTacToe App

from TicTacToeClass import TicTacToe
from game import random_player, query_player


def main():
    ttt = TicTacToe()
    ttt.play_game(random_player, query_player)
    

if __name__ == "__main__":
    main()
    