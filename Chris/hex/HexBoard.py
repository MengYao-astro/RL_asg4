'''
Reinforcement learning assignment 1
Names: Matthijs Mars, Christiaan van Buchem
'''
import logging

class HexBoard:
    '''
    Hexboard class
    '''

    BLUE = 1
    RED = 2
    EMPTY = 3
    
    def __init__(self, board_size):
        self.board = {}
        self.size = board_size
        self.game_over = False
    
        for x in range(board_size):
            for y in range (board_size):
                self.board[x,y] = HexBoard.EMPTY
  
    def is_game_over(self):
        return self.game_over
  
    def is_empty(self, coordinates):
        return self.board[coordinates] == HexBoard.EMPTY
  
    def is_valid(self, coordinates):
        """checks if move is possible"""
        try:
            self.board[coordinates]
            return True
        except:
            return False

    def clone(self):
        new_board = HexBoard(self.size)
        for x in range(self.size):
            for y in range (self.size):
                new_board.board[x,y] = self.board[x,y]
        new_board.game_over = self.game_over #necessary?
        return new_board

    def is_color(self, coordinates, color):
        return self.board[coordinates] == color
  
    def get_color(self, coordinates):
        if coordinates == (-1,-1):
            return HexBoard.EMPTY
        return self.board[coordinates]
  
    def place(self, coordinates, color):
        if not self.game_over and self.board[coordinates] == HexBoard.EMPTY:
            self.board[coordinates] = color
        if self.check_win(HexBoard.RED) or self.check_win(HexBoard.BLUE):
            self.game_over = True

    def get_opposite_color(self, current_color):
        if current_color == HexBoard.BLUE:
            return HexBoard.RED
        return HexBoard.BLUE
  
    def get_neighbors(self, coordinates):
        (cx,cy) = coordinates
        neighbors = []
        if cx-1>=0:   neighbors.append((cx-1,cy))
        if cx+1<self.size: neighbors.append((cx+1,cy))
        if cx-1>=0    and cy+1<=self.size-1: neighbors.append((cx-1,cy+1))
        if cx+1<self.size  and cy-1>=0: neighbors.append((cx+1,cy-1))
        if cy+1<self.size: neighbors.append((cx,cy+1))
        if cy-1>=0:   neighbors.append((cx,cy-1))
        return neighbors
  
    def get_dijkstra_neighbors(self, coordinates, color):
        """
        Function which returns the neigbouring coordinates of a tile. If the tile is on the border, it will return the coordinate of the ending node (board.size, 0) and (0, board.size) 
        """
        nx,ny = coordinates
        
        if   (color == HexBoard.BLUE) and (nx == -1): #start position
            return [(0, i) for i in range(self.size)] #link to first nodes

        elif (color == HexBoard.RED)  and (ny == -1):
            return [(i, 0) for i in range(self.size)]
        
        neighbors = self.get_neighbors(coordinates)

        if   (color == HexBoard.BLUE) and (nx == self.size-1):
            neighbors.append((self.size, 0))
        elif (color == HexBoard.RED)  and (ny == self.size-1):
            neighbors.append((0, self.size))
        
        return neighbors

    def get_dijkstra_color(self, coordinates):
        """
        Function to get the color of a tile, in combination with the edge tiles (off the board) for the dijkstra algorithm
        """
        nx, ny = coordinates
        if nx == self.size:
            return self.BLUE
        elif ny == self.size:
            return self.RED
        else:
            return self.get_color(coordinates)


    def border(self, color, move):
        (nx, ny) = move
        return (color == HexBoard.BLUE and nx == self.size-1) or (color == HexBoard.RED and ny == self.size-1)
  
    def traverse(self, color, move, visited):
        """check if we can move to another hex"""
        if not self.is_color(move, color) or (move in visited and visited[move]): 
            return False
        if self.border(color, move): 
            return True
    
        visited[move] = True
    
        for n in self.get_neighbors(move):
            if self.traverse(color, n, visited): 
                return True
        return False

    def get_move_list(self):
        """
        Function which returns the coordinates of possible moves (empty tiles)
        """
        if self.game_over:
            return []
        
        move_list = []
        for m in self.board:
            if self.is_empty(m):
                move_list.append(m)
        return move_list

    def undo_move(self, coordinates):
        """
        Undoes the move by setting the tile at coordinates to be empty. Also makes sure the game is not over when the undone move was a game ending move.
        """
        self.board[coordinates] = self.EMPTY
        self.game_over = False
        
    def check_win(self, color):
        '''
        Checks win condition for the given colour.
        '''
        for i in range(self.size):
            if color == HexBoard.BLUE: 
                move = (0,i)
            else: 
                move = (i,0)
            
            if self.traverse(color, move, {}):
                return True
        return False
  
    def print(self, level="print"):
        '''
        Prints the current state of the board.
        '''
        if level == "info":
            logger = logging.info
        elif level == "debug":
            logger = logging.debug
        else:
            logger = print
        
        board_string = "\n"
        board_string += "   "
        for y in range(self.size):
            board_string += chr(y+ord('a')) + " "
        board_string += "\n" 
        board_string += (" ----------------------- \n")
        for y in range(self.size):
            board_string += y*" " + str(y) + " \\ "
            for x in range(self.size):
                piece = self.board[x,y]
                if piece == HexBoard.BLUE:  board_string += "b "
                elif piece == HexBoard.RED: board_string += "r "
                else:
                    if x==self.size:
                        board_string += "-"
                    else:
                        board_string += "- "
            board_string += "\\" + (10-y)*" " + "\n"
        board_string += "   ----------------------- \n"

        logger(board_string)
