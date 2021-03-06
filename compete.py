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
from policy_value_net_pytorch import PolicyValueNet, LinearNet  # Pytorch
import torch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras

dnn_file = 'best_policy_15_15_5_forbidden_pyt.model'
lin_file = 'linear_policy_15_15_forbidden_pytorch.model'
n = 5
width, height = 15, 15

def run(playerA, playerB, n_competition= 10, seed= 12138):
    try:
        board = Board(width=width, height=height, n_in_row=n, forbidden_check_level=-1)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # policyA = PolicyValueNet(width, height, model_file = dnn_file)
        # playerA = MCTSPlayer(policyA.policy_value_fn, c_puct=5, n_playout=400, save_tree_on_compete= False)

        # policyB = PolicyValueNet(width, height, PolicyValueNetCls= LinearNet, model_file = lin_file)
        # playerB = MCTSPlayer(policyB.policy_value_fn, c_puct=5, n_playout=400, save_tree_on_compete= False)

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
        print("player 1 start first")
        for competition_i in range(n_competition):
            np.random.seed(competition_i + seed)
            torch.manual_seed(competition_i + seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(competition_i + seed)
            winner = game.start_play(playerA, playerB, start_player=0, is_shown=0)
            if winner == 1: player1_win += 1
            elif winner == 2: player2_win += 1
            print("seed {}, Player {} wins".format(competition_i + seed, winner))
        print("player 2 start first")
        for competition_i in range(n_competition):
            np.random.seed(competition_i + seed)
            torch.manual_seed(competition_i + seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(competition_i + seed)
            winner = game.start_play(playerA, playerB, start_player=1, is_shown=0)
            if winner == 1: player1_win += 1
            elif winner == 2: player2_win += 1
            print("seed {}, Player {} wins".format(competition_i + seed, winner))
        print("Competition complete: player1 wins {:.0%}, player2 wins {:.0%}".format(player1_win/n_competition/2, player2_win/n_competition/2))
    except KeyboardInterrupt:
        print('\n\rquit')

def improvement_compete():

    policyA = PolicyValueNet(width, height, model_file = dnn_file)
    playerA = MCTSPlayer(policyA.policy_value_fn, c_puct=5, n_playout=400, save_tree_on_compete= True)

    policyB = PolicyValueNet(width, height, model_file = dnn_file)
    playerB = MCTSPlayer(policyB.policy_value_fn, c_puct=5, n_playout=400, save_tree_on_compete= False)

    run(playerA, playerB)

def linear_approximator_compete():
    policyA = PolicyValueNet(width, height, model_file = dnn_file)
    playerA = MCTSPlayer(policyA.policy_value_fn, c_puct=5, n_playout=400, save_tree_on_compete= False)

    policyB = PolicyValueNet(width, height, PolicyValueNetCls= LinearNet, model_file = lin_file)
    playerB = MCTSPlayer(policyB.policy_value_fn, c_puct=5, n_playout=400, save_tree_on_compete= False)

    run(playerA, playerB)


if __name__ == '__main__':
    
    improvement_compete()

    linear_approximator_compete()
