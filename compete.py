# -*- coding: utf-8 -*-
"""
AI model 1 VS AI model 2

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
import numpy as np
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
import torch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


def run(n_competition= 100, seed= 128):
    n = 5
    width, height = 15, 15
    model_file = 'best_policy_15_15_5_forbidden_pyt.model'
    try:
        board = Board(width=width, height=height, n_in_row=n, forbidden_check_level=-1)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        policy1 = PolicyValueNet(width, height, model_file = model_file)
        player1 = MCTSPlayer(policy1.policy_value_fn, c_puct=5, n_playout=400, save_tree_on_compete= False)

        policy2 = PolicyValueNet(width, height, model_file = model_file)
        player2 = MCTSPlayer(policy2.policy_value_fn, c_puct=5, n_playout=400, save_tree_on_compete= True)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        # try:
        #     policy_param = pickle.load(open(model_file, 'rb'))
        # except:
        #     policy_param = pickle.load(open(model_file, 'rb'),
        #                                encoding='bytes')  # To support python3
        # best_policy = PolicyValueNetNumpy(width, height, policy_param)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn,
        #                          c_puct=5,
        #                          n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # start multiple competes and count the winning rate
        player1_win, player2_win = 0, 0
        for competition_i in range(n_competition):
            np.random.seed(competition_i + seed)
            torch.manual_seed(competition_i + seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(competition_i + seed)
            winner = game.start_play(player1, player2, start_player=0, is_shown=0)
            if winner == 1: player1_win += 1
            elif winner == 2: player2_win += 1
            print("seed {}, Player {} wins".format(competition_i + seed, winner))
        print("Competition complete: player1 wins {:.0%}, player2 wins {:.0%}".format(player1_win/n_competition, player2_win/n_competition))
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
