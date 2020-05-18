import numpy as np
import time 
from collections import OrderedDict
import random
import logging
from .HexGame import HexGame 
from .keras.NNet import NNetWrapper as NNet
from .HexBoard import HexBoard
from utils import *

import sys
sys.path.append('..')
from MCTS import MCTS as MCTS_A0

class OrderedDefaultDict(OrderedDict):
    """ 
    Class for priority queue for the Dijkstra algrorithm. 
    Default distance to a node is infinity. 
    Nodes can be ordered by priority
    """
    def __missing__(self, key):
        return np.inf

class Player():
    """
    Player class for hex. 
    - If human player, allows for placement of pieces.
    - If AI player, stores the evaluation method, search depth and whether transposition tables (tt)
    and iterative deepening are used (id). 
    """
    def __init__(self, is_human, ai=None ):
        self.is_human = is_human
        if self.is_human == True:
            self.move = self._human_move
        else:
            self.ai = ai
            self.move = self.ai.move

    def reset(self):
        if not self.is_human:
            self.ai.reset()

    def set_color(self, color):
        self.color = color
        if not self.is_human:
            self.ai.set_color(self.color)

    def _human_move(self, board):
        '''
        Function that allows for human player input. 
        Only allows input as integer coordinates given in de format: x,y
        '''
        move = input("Please enter the x,y coordinates of your next move: ")
        coordinates = move.split(",")
        coordinates = (int(coordinates[0]), int(coordinates[1]))
        while (not board.is_valid(coordinates)) or (not board.is_empty(coordinates)):
            move = input("Place not emtpy or not valid! The coordinates must fall within the dimensions of the table. \n"\
                         f"For example, for a {board.size} by {board.size} board we could give the coordinates: 0,{board.size-1} \n"\
                         "Try again: ")
            coordinates = move.split(",")
            coordinates = (int(coordinates[0]), int(coordinates[1]))

        board.place(coordinates, self.color) 

class Alpha_Beta():
    """class for implementation of the alpha-beta algorithm with iterative deepening and transposition tables"""
    def __init__(self, heuristic="random", depth=4, id=False, max_time=None):
        assert (heuristic in ["random", "dijkstra"]), "heuristic must be in: ['random', 'dijkstra']"
        assert (type(depth) is int), "depth must be an integer"

        if id:
            self.move = self._ai_move_tt_id
            self.max_time = max_time
        else: 
            self.move = self._ai_move
        self.depth = depth
        self.tt = {}
    
        if heuristic == "random":
            self._evalfunction = self._random_eval
        elif heuristic == "dijkstra":
            self._evalfunction = self._dijkstra_eval

    def set_color(self, color):
        self.color = color

    def reset(self):
        self.tt = {}

    def _ai_move(self, board, debug=False):
        best_move, score = self._alpha_beta(board, self.depth, -np.inf, np.inf, self.color, debug=debug)
        if debug:
            print("DEBUG:", "Best move is {} with value {}".format(best_move, score))
        board.place(best_move, self.color)
    
    def _ai_move_tt_id(self, board):
        best_move, score = self._iterative_deepening(board, max_time =self.max_time)
        board.place(best_move, self.color)

    def _alpha_beta(self, board, depth, alpha, beta, color, transposition_table=False, debug=False):
        """
        A function implementing the alpha beta search algorithm recursively. 
        """
        best_move = ''

        if depth == 0 or board.is_game_over(): 
            return best_move, self._evalfunction(board)

        if transposition_table:
            # Checking if board state is in transposition table
            board_state = frozenset(board.board.items())
            if (board_state,depth) in self.tt:
                best_move, best_score = self.tt[(board_state,depth)]
                if debug:
                    board.print()
                    print("DEBUG: found state, best_move {}, best score {}".format( best_move,best_score))
                return best_move, best_score

        if color == self.color:
            best_score = -np.inf
            for possible_move in board.get_move_list():
                
                board.place(possible_move, self.color)
                _, score = self._alpha_beta(board, depth -1, alpha, beta, board.get_opposite_color(self.color), transposition_table=transposition_table, debug=debug) # next move for opposite player
                if debug:
                    board.print()
                    print("DEBUG:", "depth = {}, Score for this move is {}".format(depth, score))
                board.undo_move(possible_move)
                
                if score >= best_score:
                    best_score = score
                    best_move = possible_move
                
                alpha = max(alpha, best_score)

                if alpha > beta: #no point in looking further, NB this is larger instead of larger/equal -> see report
                    if debug:
                        #continue when debugging
                        print("DEBUG: this branch is pruned. Alpha is {}, Beta is {}".format(alpha, beta))
                    else:
                        break

        else:
            best_score = np.inf
            for possible_move in board.get_move_list(): 
                board.place(possible_move, board.get_opposite_color(self.color))
                _, score = self._alpha_beta(board, depth -1, alpha, beta, self.color, transposition_table=transposition_table, debug=debug) # next move for this player
                if debug:
                    board.print()
                    print("DEBUG:", "depth = {}, Score for this move is {}".format(depth, score))
                board.undo_move(possible_move)

                if score <= best_score:
                    best_score = score
                    best_move  = possible_move
                beta = min(beta, best_score)

                if alpha > beta: #no point in looking further, NB this is larger instead of larger/equal -> see report
                    if debug:
                        #continue when debugging
                        print("DEBUG: this branch is pruned. Alpha is {}, Beta is {}".format(alpha, beta))
                    else:
                        break 

        if transposition_table:
            # Store to transposition table
            self.tt[(board_state,depth)] = best_move, best_score

        return best_move, best_score

    def _iterative_deepening(self, board, max_time=.5):
        '''
        Function which starts with a search depth of 1 and iteratively deepends the alpha-beta search
        until the maximum amount of time allowed for the move is up. 
        '''
        start_time = time.time()
        d = 1
        best_move, best_score = (), -np.inf
        last_time = start_time
        while True:
            move, score = self._alpha_beta(board, d, -np.inf, np.inf, self.color, transposition_table=True) #determines score for deeper level
            if (time.time() - last_time) + (time.time() - start_time) > max_time:
                if (time.time() - start_time > max_time):
                    d -= 1
                break
            elif  (time.time() - start_time < max_time):
                last_time = time.time()
                best_score = score
                best_move = move
                d += 1
            else:
                d -= 1 #correct to the last succesful search
                break
        print(d)
        return best_move, best_score

    def _random_eval(self, board):
        return np.random.randint(-board.size, board.size)
    
    def _dijkstra_eval(self, board):
        return  self._dijkstra_distance(board, board.get_opposite_color(self.color)) - self._dijkstra_distance(board, self.color)

    def _dijkstra_distance(self, board, color):
        """
        Function which calculates the Dijkstra's shortest path from one side of the board to the other.
        """
        if color == board.BLUE:
            sx, sy = -1, 0
        elif color == board.RED:
            sx, sy = 0, -1

        queue = OrderedDefaultDict()
        queue[(sx,sy)] = 0 #queue stores location and distance

        visited = {}

        while len(queue) > 0:
            coordinates = list(queue.keys())[0]

            neighbours = board.get_dijkstra_neighbors(coordinates, color)
            if ((board.size,0) in neighbours and color == board.BLUE) or \
               ((0,board.size) in neighbours and color == board.RED):
                return queue[coordinates]

            for i in neighbours:
                if i not in visited:
                    if board.get_dijkstra_color(i) == color:
                        queue[i] = min(queue[i], queue[coordinates]) #if it's the same colour, free walk
                    elif board.get_dijkstra_color(i) == board.get_opposite_color(color):
                        pass #don't add the node to the queue
                    else:
                        queue[i] = min(queue[i], queue[coordinates]+1) #if it's emtpy, add 1 to the path

            visited[coordinates] = queue[coordinates] #add finished node to visisted, store distance
            del queue[coordinates] #added all neigbours, don't need these coordinates anymore

            queue = OrderedDefaultDict( sorted(queue.items(), key=lambda x:x[1]))
        return np.inf

class Node():
    """class for Nodes used in the graphs for the MCTS algorithm"""
    def __init__(self, board, color, C_p=2):
        
        self.wi = 0
        self.n = 0

        self.C_p = C_p

        self.move = None
        self.parent_node = None
        self.color = color
        self.child_nodes = []
        self.untried_moves = board.get_move_list()
        self.UCT = self._calc_UCT()

    def add_child(self, move, state):
        """function for adding a child to a node"""
        child = Node(state, state.get_opposite_color(self.color), C_p=self.C_p) #next move always has opposite color
        child.parent_node = self #current node is the parent
        child.move = move
        self.child_nodes.append(child)
        self.untried_moves.remove(move)

        return child
        
    def _calc_UCT(self):
        if self.n == 0: #if never visited, the UCT is unkown
            return np.inf
        if self.parent_node is None: #edge case for the root node which has no parent
            return np.inf
        return self.wi/self.n + self.C_p * (np.log(self.parent_node.n)/self.n)**0.5
    
    def update(self, result):
        self.wi += result #+1 for win, -1 for loss, 0 for draw  
        self.n += 1

    def UCT_select_child(self):
        """selects the child with the highest UCT score"""
        max_UCT = -np.inf
        best_node = self.child_nodes[0]
        for child in self.child_nodes:
            child.UCT = child._calc_UCT() #update new UCT value
            if child.UCT > max_UCT:
                max_UCT = child.UCT
                best_node = child
        
        return best_node
    
    def print_tree(self):
        """prints the information in the root node and its first children"""
        self._print_leaf(0)
        for child in self.child_nodes:
            child._print_leaf(1)
        logging.debug("")

    def _print_leaf(self, depth):
        """prints the information stored in the node"""
        logging.debug(f"move: {self.move}, depth {depth}, UCT: {self._calc_UCT():5.3f}, w: {self.wi:2}, n: {self.n}")

class MCTS():
    """class for the MCTS AI functions"""
    def __init__(self, max_iter=None, max_time=None, C_p=2):
        self.max_iter = max_iter
        self.max_time = max_time
        self.C_p = C_p

    def set_color(self, color):
        self.color = color

    def reset(self):
        pass

    def move(self, board):
        ai = MCTS()
        if self.max_iter is not None:
            best_move = self._MCTS(board, self.color, max_iter=self.max_iter, C_p=self.C_p)
        elif self.max_time is not None:
            best_move = self._MCTS(board, self.color, max_time=self.max_time, C_p=self.C_p)
        else:
            best_move = self._MCTS(board, self.color, max_iter=1000, C_p=self.C_p)
        board.place(best_move, self.color)        

    def _MCTS(self, board, color, max_iter=np.inf, max_time=np.inf, C_p=2):
        rootnode = Node(board, board.get_opposite_color(color), C_p=C_p)
        rootnode.n, rootnode.move = 1, "root  "
        start_time = time.time()
        i = 0
        while (i < max_iter) and (time.time() - start_time < max_time):
            logging.debug(f"iteration {i}")
            node = rootnode
            state = board.clone()

            #select
            while node.untried_moves == [] and node.child_nodes != []:
                logging.debug(f"untried moves: {node.untried_moves}")
                node = node.UCT_select_child()
                state.place(node.move, node.color)
            
            logging.debug("selected state:")
            state.print(level='debug')

            #expand
            if node.untried_moves != []:
                logging.debug(f"untried moves: {node.untried_moves}")
                move = random.choice(node.untried_moves)
                state.place(move, state.get_opposite_color(node.color))
                node = node.add_child(move, state)
                logging.debug(f"expanding {move} with color {node.color} ")
                
            
            logging.debug("expanded state:")
            state.print(level="debug")

            #playout
            color = node.color
            while state.get_move_list() != []:
                color = state.get_opposite_color(color)
                m = random.choice(state.get_move_list())
                logging.debug(f"random move: {m} node color: {node.color} color: {color}" )
                state.place(m , color)   
            
            logging.debug("state after rollout")
            state.print(level="debug")

            #backpropagate
            if state.check_win(state.get_opposite_color(rootnode.color)):
                result = 1
            elif state.check_win(rootnode.color):
                result = -1
            else: 
                result = 0

            while node != None:
                node.update(result)
                node = node.parent_node
            
            i += 1
            rootnode.print_tree()
        best_child = np.argmax([child._calc_UCT() for child in rootnode.child_nodes])
        return [child.move for child in rootnode.child_nodes][best_child]

class A0_Player():
    """
    Alpha-zero player class for hex. 
    """
    def __init__(self, n, load_folder="temp", load_name="temp"):
        self.game = HexGame(n)
        n1 = NNet(self.game)
        n1.load_checkpoint(load_folder,load_name) # TODO Make most recent
        
        args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        mcts1 = MCTS_A0(self.game, n1, args1)
        self.n1p = lambda x, player: np.argmax(mcts1.getActionProb(x, temp=0, player=player))

    def reset(self):
        pass
        
    def set_color(self, color):
        self.color = color
        if self.color == 1: #blue
            self.player = 1
        elif self.color == 2: #red
            self.player = -1

    def move(self, board):
        canonicalBoard = self.game.getCanonicalForm(board, self.player)
        action = self.n1p(canonicalBoard, self.player)
        newBoard, next_player = self.game.getNextState(canonicalBoard, self.player, action)
        board.board = newBoard.board
        board.game_over = newBoard.game_over