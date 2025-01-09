import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point


MAX_MEMORY = 100_000 #store max 100k items in this memory
BATCH_SIZE = 1000 
LR = 0.001 #learning rate

class Agent:
    def __init__(self):
        pass

    #get state of current environment
    #calculate next move based on state
    #get game to play step
    #get new state
    #store state into 
    #train the model

    def get_state(self,game): 
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self): #trains on only one step
        pass

    def get_action(self,state):
        pass

    def train():
        pass
    
    if __name__=="__main__":
        train()