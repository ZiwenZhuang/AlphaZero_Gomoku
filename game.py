# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""
    
    ### 8 counter-clockwise directions
    directions = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]]

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

        # 0 for no check, 
        # 1 for check current move but not keypoints, 2 for check recursively
        # positive option for removing balance breakers out of self.availables
        # negative option for check forbiddens when judging
        self.forbidden_check_level = kwargs.get('forbidden_check_level', 0)

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.start_player = start_player
        self.balance_breaker_violated = False
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        if self.forbidden_check_level > 0 \
            and self.current_player is self.players[self.start_player]:
            for m in self.availables: # TODO: This for loop can be in parallel
                if self.check_forbidden(m): self.availables.remove(m)

        if self.forbidden_check_level < 0 \
            and self.current_player is self.players[self.start_player]:
            self.balance_breaker_violated = self.check_forbidden(move)

        # switch player and record last_move
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        if self.balance_breaker_violated:
            winner = (self.players[0] if self.start_player == self.players[1]
                else self.players[1]
            )
            return True, winner

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def check_keypoint_forbidden(self, x, y, dir_i: int, adjsame: int):
        """ A wrapper to check keypoints based on caller's move location.
        """
        if abs(self.forbidden_check_level) < 2:
            return False

        self.states[self.location_to_move([x, y])] = self.current_player

        dir_x, dir_y = self.directions[dir_i]
        keypoint_move = self.location_to_move([x+dir_x*adjsame, y+dir_y*adjsame])
        result = self.check_forbidden(keypoint_move)

        self.states.pop(self.location_to_move([x, y]))
        return result

    def check_forbidden(self, move):
        """ check whether current move is a balance breaker. If forbidden, return True
        Implementation referring to https://blog.csdn.net/JkSparkle/article/details/822873
        """
        loc_x, loc_y = self.move_to_location(move)

        ### 8 for each direction
        adjsame = np.zeros((8,), dtype= np.uint8) # the number of current player directly adjacent to this point
        adjempty = np.zeros((8,), dtype= np.uint8) # the number of empty space behind adjsame
        jumpsame = np.zeros((8,), dtype= np.uint8) # the number of current player behind adjempty
        jumpempty = np.zeros((8,), dtype= np.uint8) # the number of empty space behind jumpsame
        jumpjumpsame = np.zeros((8,), dtype= np.uint8) # the number of current player behind jumpempty

        ### Then search along all 8 directions
        for i, (x_i, y_i) in enumerate(self.directions):
            x_, y_ = loc_x+x_i, loc_y+y_i
            while (x_ >= 0 and x_ < self.width and y_ >= 0 and y_ < self.height \
                and self.states.get(self.location_to_move([x_, y_]), None) is self.current_player):
                x_ += x_i; y_ += y_i; adjsame[i] += 1
            while (x_ >= 0 and x_ < self.width and y_ >= 0 and y_ < self.height \
                and self.states.get(self.location_to_move([x_, y_]), None) is None):
                x_ += x_i; y_ += y_i; adjempty[i] += 1
            while (x_ >= 0 and x_ < self.width and y_ >= 0 and y_ < self.height \
                and self.states.get(self.location_to_move([x_, y_]), None) is self.current_player):
                x_ += x_i; y_ += y_i; jumpsame[i] += 1
            while (x_ >= 0 and x_ < self.width and y_ >= 0 and y_ < self.height \
                and self.states.get(self.location_to_move([x_, y_]), None) is None):
                x_ += x_i; y_ += y_i; jumpempty[i] += 1
            while (x_ >= 0 and x_ < self.width and y_ >= 0 and y_ < self.height \
                and self.states.get(self.location_to_move([x_, y_]), None) is self.current_player):
                x_ += x_i; y_ += y_i; jumpjumpsame[i] += 1

        ### check balance breaker
        three_count, four_count = 0, 0
        for i in range(4):
            if adjsame[i] + adjsame[i+4] >= 5:
                return True # long-connection
            elif adjsame[i] + adjsame[i+4] == 3: # 4 stones connected
                isFour = False
                if adjsame[i] > 0:
                    isFour |= not self.check_keypoint_forbidden(loc_x, loc_y, i, adjsame[i])
                if adjsame[i+4] > 0:
                    isFour |= not self.check_keypoint_forbidden(loc_x, loc_y, i+4, adjsame[i])
                if isFour: four_count += 1
            elif adjsame[i] + adjsame[i+4] == 2: # 3 stones connected
                # advance four or active four checking
                if adjempty[i] == 1 and jumpsame[i] == 1:
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i, adjsame[i]):
                        four_count += 1
                if adjempty[i+4] == 1 and jumpsame[i+4] == 1:
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i+4, adjsame[i+4]):
                        four_count += 1
                # active three checking
                isThree = False
                if (adjempty[i] > 2 or adjempty[i] == 2 and jumpsame[i] == 0) \
                    and (adjempty[i+4] > 1 or adjempty[i+4] == 1 and jumpsame[i+4]):
                    isThree |= not self.check_keypoint_forbidden(loc_x, loc_y, i, adjsame[i])
                if (adjempty[i+4] > 2 or adjempty[i+4] == 2 and jumpsame[i+4] == 0) \
                    and (adjempty[i] > 1 or adjempty[i] == 1 and jumpsame[i]):
                    isThree |= not self.check_keypoint_forbidden(loc_x, loc_y, i+4, adjsame[i+4])
                if isThree: three_count += 1
            elif adjsame[i] + adjsame[i+4] == 1: # 2 stones connected
                # advance four or active four checking
                if adjempty[i] == 1 and jumpsame[i] == 2:
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i, adjsame[i]):
                        four_count += 1
                if adjempty[i+4] == 1 and jumpsame[i+4] == 2:
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i+4, adjsame[i+4]):
                        four_count += 1
                # active three checking
                if adjempty[i] == 1 and jumpsame[i] == 1 \
                    and (jumpempty[i] > 1 or jumpempty[i] == 1 and jumpjumpsame[i] == 0) \
                    and (adjempty[i+4] > 1 or adjempty[i+4] == 1 and jumpsame[i+4] == 0):
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i, adjsame[i]):
                        three_count += 1
                if adjempty[i+4] == 1 and jumpsame[i+4] == 1 \
                    and (jumpempty[i+4] > 1 or jumpempty[i+4] == 1 and jumpjumpsame[i+4] == 0) \
                    and (adjempty[i] > 1 or adjempty[i] == 1 and jumpsame[i] == 0):
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i+4, adjsame[i+4]):
                        three_count += 1
            elif adjsame[i] + adjsame[i+4] == 0: # 1 stone
                # advance four or active four checking
                if adjempty[i] == 1 and jumpsame[i] == 3:
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i, adjsame[i]):
                        four_count += 1
                if adjempty[i+4] == 1 and jumpsame[i+4] == 3:
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i+4, adjsame[i+4]):
                        four_count += 1
                # active three checking
                if adjempty[i] == 1 and jumpsame[i] == 2 \
                    and (jumpempty[i] > 1 or jumpempty[i] == 1 and jumpjumpsame[i] == 0) \
                    and (adjempty[i+4] > 1 or adjempty[i+4] == 1 and jumpsame[i+4] == 0):
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i, adjsame[i]):
                        three_count += 1
                if adjempty[i+4] == 1 and jumpsame[i+4] == 2 \
                    and (jumpempty[i+4] > 1 or jumpempty[i+4] == 1 and jumpjumpsame[i+4] == 0) \
                    and (adjempty[i] > 1 or adjempty[i] == 1 and jumpsame[i] == 0):
                    if not self.check_keypoint_forbidden(loc_x, loc_y, i+4, adjsame[i+4]):
                        three_count += 1

        ### counting the active three, advance four and active fours
        if four_count > 1 or three_count > 1:
            return True
        else:
            return False

    def show_board(self):
        return Game.graphic(self, self, *self.players)

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
