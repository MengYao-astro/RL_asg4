import numpy as np

from collections import OrderedDict
from trueskill import Rating, rate_1vs1
from .HexBoard import HexBoard
from .Player import Player, Alpha_Beta, MCTS, A0_Player
import sys
from tqdm import tqdm


class Game():
    def __init__(self, size=7):
        self.size = size #default boardsize

    def start(self):
        print("""
        "Welcome to Hex by Christiaan and Matthijs"
            Options:
            1. play a game
            2. play a tournament
            3. change board size (current: {})
            4. exit

        """.format(self.size))
        choice = int(input())
        if choice == 1:
            players = self._select_players(2)
            self.play_game(players)
        elif choice == 2:
            players = self._select_players( int(input("how many players in the tournament? .. ")))
            self.tournament( int(input("how many games?")), players)
        elif choice == 3:
            new_size = int(input("What size?"))
            self._set_size(new_size)
        elif choice == 4:
            exit()
        else:
            print("invalid option")
        self.start()

    def _select_players(self, n_players):
        players = []
        search_time = None
        for i in range(n_players):
            print("""
                Adding player {}, choose from:
                1. human player
                2. alpha-beta with search depth 3 and random evaluation 
                3. alpha-beta with search depth 3 and Dijkstra evaluation 
                4. alpha-beta with search depth 4 and Dijkstra evaluation 
                5. alpha-beta with iterative deepening and transposition tables and Dijkstra evaluation 
                6. mcts
                7. Alpha zero self play
            """.format(i+1))
            choice = int(input("choice: "))
            if choice == 1:
                players.append(Player(is_human=True))
            elif choice == 2:
                players.append(Player(is_human=False, ai=Alpha_Beta(heuristic="random", depth=3)))
            elif choice == 3:
                players.append(Player(is_human=False, ai=Alpha_Beta(heuristic="dijkstra", depth=3)))
            elif choice == 4:
                players.append(Player(is_human=False, ai=Alpha_Beta(heuristic="dijkstra", depth=4)))
            elif choice == 5:
                if search_time == None:
                    search_time = float(input("How much time can the ai player spend on their turn? (s): "))
                players.append(Player(is_human=False, ai=Alpha_Beta(heuristic="dijkstra", id=True, max_time=search_time)))
            elif choice == 6:
                if search_time == None:
                    search_time = float(input("How much time can the ai player spend on their turn? (s): "))
                players.append(Player(is_human=False, ai=MCTS(max_time=search_time))) 
            elif choice == 7:
                players.append(A0_Player(self.size))
            else:
                print("Invalid choice, defaulting to random evaluation, search depth 3")
                players.append(Player(is_human=False, heuristic="random", depth=3))
            
        return players


    def _set_size(self, size):
        self.size = size
    
    def tournament(self, n_rounds, players):
        """
        Function to determine the rating of a set of players over a given amount of rounds. To determine their rating, they play rounds of matches against random opponents after which their TrueSkill ranking is updated according to the results. The mean and standard deviations of their ratings is saved and returned. Every player plays 2 games each round. 
        """
        ratings = [ Rating() for i in range(len(players))]
        
        mean, std = np.zeros( (len(players), n_rounds+1)), np.zeros((len(players), n_rounds+1))
        mean[:,0], std[:,0] = 25, 25/3
        for n in tqdm(range(n_rounds)):
            print(n/n_rounds)
            order = np.random.permutation(len(players))
            for i in range(len(players)):
            
                p1, p2 = order[i], order[(i+1)%len(players)]
                outcome = self.play_game((players[p1], players[p2]) )
                if outcome==0:
                    ratings[p1], ratings[p2] = rate_1vs1(ratings[p1], ratings[p2])
                elif outcome==1:
                    ratings[p2], ratings[p1] = rate_1vs1(ratings[p2], ratings[p1])
                elif outcome ==2:
                    ratings[p1], ratings[p2] = rate_1vs1(ratings[p1], ratings[p2], drawn=True)

                players[p1].reset()
                players[p2].reset()

            for i in range(len(players)):
                mean[i, n+1], std[i, n+1] = ratings[i].mu, ratings[i].sigma
        
        print(ratings)
        return mean, std


    def play_game(self, players):
        board = HexBoard(self.size)

        p1, p2 = players
        p1.set_color(board.BLUE)
        p2.set_color(board.RED)

        # board.print()

        #starting player
        color = board.BLUE
        while not board.is_game_over():

            if color == board.BLUE:
                p1.move(board)
            else:
                p2.move(board)
            color = board.get_opposite_color(color)
            # board.print()

        if board.check_win(board.BLUE):
            print("blue wins")
            return 0
        elif board.check_win(board.RED):
            print("red wins")
            return 1
        else:
            print("draw")
            return 2