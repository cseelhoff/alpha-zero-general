from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np
import ctypes
from ctypes import c_int, c_float, c_void_p, c_bool, POINTER

class OAAAGame(Game):
    def __init__(self):
        # Load the compiled Odin library
        self.lib = ctypes.CDLL('./liboaaa.so')  # or .dll on Windows
        # Set up function signatures
        self.lib.get_init_board.restype = c_void_p
        self.lib.get_board_size.restype = (c_int, c_int)
        self.lib.get_action_size.restype = c_int
        self.lib.get_next_state.argtypes = [c_void_p, c_int, c_int]
        self.lib.get_next_state.restype = (c_void_p, c_int)

        self.lib.get_valid_moves.argtypes = [c_void_p, c_int]
        self.lib.get_valid_moves.restype = POINTER(c_bool)

        self.lib.get_game_ended.argtypes = [c_void_p, c_int]
        self.lib.get_game_ended.restype = c_float

        self.lib.get_canonical_form.argtypes = [c_void_p, c_int]
        self.lib.get_canonical_form.restype = POINTER(c_float)

        self.lib.get_string_representation.argtypes = [c_void_p]
        self.lib.get_string_representation.restype = c_char_p
        
    def getInitBoard(self):
        return self.lib.get_init_board()
        
    def getBoardSize(self):
        return self.lib.get_board_size()
        
    def getActionSize(self) -> c_int:
        return self.lib.get_action_size()
        
    def getNextState(self, board, player, action):
        return self.lib.get_next_state(board, player, action)
        
    def getGameEnded(self, board, player) -> c_float:
        return self.lib.get_game_ended(board, player)
        
    def getValidMoves(self, board, player) -> np.ndarray:
        moves_ptr = self.lib.get_valid_moves(board, player)
        size = self.getActionSize()
        # Convert the C boolean array to numpy array of bools
        return np.ctypeslib.as_array(moves_ptr, shape=(size,)).astype(bool)
        
    def getCanonicalForm(self, board, player)-> np.ndarray:
        canonicalBoard_ptr = self.lib.get_canonical_form(board, player)
        x, y = self.getBoardSize()
        # Convert the C array to numpy array
        return np.ctypeslib.as_array(canonicalBoard_ptr, shape=(x, y))
        
    def getSymmetries(self, board, pi):
        # mirror, rotational
        # If no meaningful symmetries exist in OAAA
        return [(board, pi)]
        
    def stringRepresentation(self, board) -> string:
        return self.lib.get_string_representation(board).decode('utf-8')
