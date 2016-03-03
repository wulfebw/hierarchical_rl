
import matplotlib.pyplot as plt
import numpy as np
import os

import logger


def is_valid(key):
    return key.replace('-','').isalnum()

def load_key(filepath):
    assert os.path.exists(filepath), 'filepath: {} not found'.format(filepath)
    
    key = None
    with open(filepath, 'rb') as f:
        key = f.readline()
    if is_valid(key):
        return key
    else:
        raise ValueError('invalid key: {}'.format(key))


def graph_rewards_seq_len(filepaths):
    initrewards = []
    min_len = 10000000
    for f in filepaths:
        r = np.load(f)['values']
        mr = logger.moving_average(r, 5)
        if len(mr) < min_len:
            min_len = len(mr)
        initrewards.append(mr)

    rewards = []
    for r in initrewards:
        rewards.append(r[:min_len])

    r2 = plt.plot(rewards[0], label='length 2 sequence', color='orange')   
    r4 = plt.plot(rewards[1], label='length 4 sequence', color='crimson')  
    r8 = plt.plot(rewards[2], label='length 8 sequence', color='cyan')   
    r12 = plt.plot(rewards[3], label='length 12 sequence', color='brown')  
    r16 = plt.plot(rewards[4], label='length 16 sequence', color='blue')   
    r20 = plt.plot(rewards[5], label='length 20 sequence', color='black')  
    r24 = plt.plot(rewards[6], label='length 24 sequence', color='pink')   

    plt.legend(loc='lower right')
    plt.ylabel('Episode Rewards')
    plt.xlabel('Epochs')
    plt.savefig('/Users/wulfe/Dropbox/School/Stanford/winter_2016/cs239/project/hierarchical_rl/results/seqlen_rewards.png')

def graph_rewards(filepaths):
    rewards = []
    for f in filepaths:
        r = np.load(f)['values']
        mr = logger.moving_average(r, 10)
        rewards.append(mr)

    plt.plot(rewards[0], label='row/col + room', color='r')  
    plt.plot(rewards[1], label='row/col only', color='g') 
    plt.plot(rewards[2], label='tabular', color='b')    
    plt.plot(rewards[3], label='coordinates', color='magenta') 

    plt.legend(loc='upper left')
    plt.ylabel('Episode Rewards')
    plt.xlabel('Epochs')
    plt.savefig('/Users/wulfe/Dropbox/School/Stanford/winter_2016/cs239/project/hierarchical_rl/results/staterep_rewards.png')

if __name__ =='__main__':
    root = '/Users/wulfe/Desktop/logs2/promise_hrlstaterep'

    rowcolroom = os.path.join(root, 'QNetwork_2016-03-02T02.56.25.325166', 'rewards.npz')
    rowcol = os.path.join(root, 'QNetwork_2016-03-02T03.12.14.506093', 'rewards.npz')
    tabular = os.path.join(root, 'QNetwork_2016-03-02T03.35.00.107253', 'rewards.npz')
    coords = os.path.join(root, 'QNetwork_2016-03-02T04.01.10.893242', 'rewards.npz')
    filepaths = [rowcolroom, rowcol, tabular, coords]
    graph_rewards(filepaths)

    # r2 = os.path.join(root, 'single_layer_lstm_2016-02-29T12.52.41.641967', 'rewards.npz')
    # r4 = os.path.join(root + '_hrltimestep', 'single_layer_lstm_2016-03-01T15.31.29.414573', 'rewards.npz')
    # r8 = os.path.join(root, 'single_layer_lstm_2016-02-29T15.16.25.802919', 'rewards.npz')
    # r12 = os.path.join(root + '_hrltimestep', 'single_layer_lstm_2016-03-01T12.42.03.151324', 'rewards.npz')
    # r16 = os.path.join(root + '_hrltimestep', 'single_layer_lstm_2016-03-01T12.30.01.583816', 'rewards.npz')
    # r20 = os.path.join(root + '_hrltimestep', 'single_layer_lstm_2016-03-01T12.35.43.236976', 'rewards.npz')
    # r24 = os.path.join(root + '_hrltimestep', 'single_layer_lstm_2016-03-01T17.31.08.383906', 'rewards.npz')
    # filepaths = [r2, r4, r8, r12, r16, r20, r24]
    # graph_rewards(filepaths)