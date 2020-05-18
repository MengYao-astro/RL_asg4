from __future__ import print_function
from .HexBoard import HexBoard
from Game import Game
import sys
import numpy as np

class HexGame(Game):

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = HexBoard(self.n)
        return b

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n

    def getNextState(self, canonicalBoard, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        move = np.unravel_index(action, self.getBoardSize())
        assert type(canonicalBoard) == np.ndarray, "not giving a canonical Board"
        assert canonicalBoard[move] == 0, f"Picking an occupied space {move}"
        canonicalBoard[move] = 1
        board = self.convertCanonical(canonicalBoard, player)

        return (board, -player)

    def getValidMoves(self, canonicalBoard, player):
        # return a fixed size binary vector
        return (canonicalBoard == 0).flatten()

    def getGameEnded(self, board, player):
        if type(board) == np.ndarray: # a canonical board is passed through
            board = self.convertCanonical(board, player) 

        if not board.is_game_over():
            return 0
        elif board.check_win(board.BLUE):
            return 1 * player  # blue is player 1
        elif board.check_win(board.RED):
            return -1 * player # red is player -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        canonicalBoard = np.zeros(self.getBoardSize())

        for c in board.board:
            x, y = c
            if board.board[c] == board.BLUE :
                canonicalBoard[x,y] = 1
            elif board.board[c] == board.RED :
                canonicalBoard[x,y] = -1
            else:
                canonicalBoard[x,y] = 0
        
        if player == -1:
            canonicalBoard = canonicalBoard.T # make sure they always move in same direction
        # return state if player==1, else return -state if player==-1
        return player*canonicalBoard

    def convertCanonical(self, canonicalBoard, player):
        # convert the canonical board in a new hexBoard
        hexBoard = HexBoard(self.n)
        canonicalBoard *= player
        if player == -1:
            canonicalBoard = canonicalBoard.T # undo transformation
        for x in range(canonicalBoard.shape[0]):
            for y in range(canonicalBoard.shape[1]):
                if canonicalBoard[x,y] == 1:
                    color = hexBoard.BLUE
                elif canonicalBoard[x,y] == -1:
                    color = hexBoard.RED
                else:
                    color = hexBoard.EMPTY
                hexBoard.board[x,y] = color
        
        if hexBoard.check_win(hexBoard.BLUE) or hexBoard.check_win(hexBoard.RED):
            hexBoard.game_over = True          
        return hexBoard

    def getSymmetries(self, canonicalBoard, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2)  # no 1 for pass
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []
        l += [(canonicalBoard, pi)] # 1 symmetries
        # l += [(np.rot90(canonicalBoard, k=2), np.rot90(pi,k=2))] # 1 symmetries

        return l

    def stringRepresentation(self, canonicalBoard):
        return canonicalBoard.tostring()

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

    def getScore(self, board, player):
        if type(board) == np.ndarray: # a canonical board is passed through
            board = self.convertCanonical(board, player) 

        if player == 1:
            color = board.BLUE
        elif player == -1:
            color = board.RED
        return _dijkstra_distance(board, board.get_opposite_color(color)) - _dijkstra_distance(board, color)   


    @staticmethod
    def display(board):
        board.print()
