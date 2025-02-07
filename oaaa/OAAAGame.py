from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np
import ctypes
from ctypes import c_int, c_float, c_void_p, c_bool, c_char_p, POINTER, Structure

class BoardSize(Structure):
    _fields_ = [("x", c_int), ("y", c_int)]

class NextState(Structure):
    _fields_ = [("board", c_void_p), ("player", c_int)]

class OAAAGame(Game):
    def __init__(self, size1):
        # Load the compiled Odin library
        self.lib = ctypes.CDLL('../oaaa/build/liboaaa.so')  # or .dll on Windows
        # Set up function signatures
        print('self.lib')
        self.lib.get_init_board.restype = c_void_p
        self.lib.get_board_size.restype = BoardSize
        self.lib.get_action_size.restype = c_int

        self.lib.get_next_state.argtypes = [c_void_p, c_int, c_int]
        self.lib.get_next_state.restype = NextState

        self.lib.get_valid_moves.argtypes = [c_void_p, c_int]
        self.lib.get_valid_moves.restype = POINTER(c_bool)

        self.lib.get_game_ended.argtypes = [c_void_p, c_int]
        self.lib.get_game_ended.restype = c_float

        self.lib.get_canonical_form.argtypes = [c_void_p, c_int]
        self.lib.get_canonical_form.restype = c_void_p

        self.lib.board_ptr_to_f32_array.argtypes = [c_void_p]
        self.lib.board_ptr_to_f32_array.restype = POINTER(c_float)

        self.lib.get_string_representation.argtypes = [c_void_p]
        self.lib.get_string_representation.restype = c_char_p
        
    def getInitBoard(self):
        return self.lib.get_init_board()
        
    def getBoardSize(self):
        boardsize = self.lib.get_board_size()
        return boardsize.x, boardsize.y
        
    def getActionSize(self) -> c_int:
        return self.lib.get_action_size()
        
    def getNextState(self, board, player, action):
        nextState = self.lib.get_next_state(board, player, action)
        return nextState.board, nextState.player
        
    def getGameEnded(self, canonicalBoard, player) -> c_float:
        return self.lib.get_game_ended(canonicalBoard, player)
        
    def getValidMoves(self, board, player):
        moves_ptr = self.lib.get_valid_moves(board, player)
        size = self.getActionSize()
        # Convert the C boolean array to numpy array of bools
        return np.ctypeslib.as_array(moves_ptr, shape=(size,)).astype(np.int64)
        
    def getCanonicalForm(self, board, player):
        canonicalBoard_ptr = self.lib.get_canonical_form(board, player)
        # board_size = self.getBoardSize()
        # x, y = board_size.width, board_size.height
        # # Convert the C array to numpy array
        return canonicalBoard_ptr
        # return board
        
    def getSymmetries(self, canonicalBoard, pi):
        # mirror, rotational
        # If no meaningful symmetries exist in OAAA
        board_nparray = self.board_ptr_to_np_array(canonicalBoard)
        return [(board_nparray, pi)]
        
    def stringRepresentation(self, board):
        # if not isinstance(board, (c_void_p, int)):
        #     raise TypeError(f"Expected board to be a pointer (c_void_p or int), got {type(board)}")
        # if isinstance(board, int):
        #     board = c_void_p(board)
        # string_rep = self.lib.get_string_representation(board)
        # return string_rep.decode('utf-8')

        x, y = self.getBoardSize()
        # Convert the C array to numpy array
        board_array = np.ctypeslib.as_array(board, shape=(x, y))
        return str(board_array.tobytes())

    def board_ptr_to_np_array(self, board):
        board_f32_array = self.lib.board_ptr_to_f32_array(board)
        x, y = self.getBoardSize()
        # # Convert the C array to numpy array
        return np.ctypeslib.as_array(board_f32_array, shape=(x, y))