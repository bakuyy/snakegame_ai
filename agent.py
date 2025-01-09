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
        self.n_games = 0
        self.epsilon = 0 #controls randomness
        self.gamma = 0 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if we exceed the memory, it'll popleft()

        self.model = None #must implement this
        self.trainer = None #must implement this
        pass

    #get state of current environment
    #calculate next move based on state
    #get game to play step
    #get new state
    #store state into 
    #train the model

    def get_state(self,game): 
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y) #20 is the block size
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # direction stored as a boolean. it any given state, only one direction is 1 (the rest are 0)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #popleft if MAX_MEMORY  is reached, append as a tuple
        

    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:

            mini_sample = random.sample(self.memory, BATCH_SIZE) #returns list of tuples
        else:
            mini_sample = self.memory

        # * is the unpacking operator, used to unpack iterable objects like lists/tuples into individual elements
        # zip maps the arguments in each tuple positionally and groups them

        '''
        example:
        mini_sample = [
        (state1, action1, reward1, next_state1, done1),
        (state2, action2, reward2, next_state2, done2),
        ]   

        states, actions, rewards, next_states, done = zip(*mini_sample)

        states will be (state1, state2, state3)
        actions will be (action1, action2, action3)
        rewards will be (reward1, reward2, reward3)
        '''
        states,actions,rewards,next_states,done = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, done)


    def train_short_memory(self,state, action, reward, next_state, done): #trains on only one step
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self,state):
        # start with random movies: tradeoff exploration/exploitation
        # we ensure we make random moves in the beginning- as we improve our model, we explore less and exploit the agent more
        self.epsilon = 80-self.n_games # more games played equals smaller epsilon
        final_move = [0,0,0]

        # the smaller the epsilon, the less frequent this statement will be true
        # if it reaches the negatives, this statement will not play out
        if random.randint(0,200) < self.epsilon: 
            move = random.randint(0,2) 
            final_move[move] = 1 #start by randomly moving around
        else:
            #converts state into a tensor, ensuring that data type is explicitly set to torch.float32
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model.predict(state0) #prediction might look like [5.0,2.7,1.0]
            move = torch.argmax(prediction).item() # from above, we take the max and represent it as the move [1,0,0]
            final_move[move] = 1 # see line above, we change it to one

        return final_move

    def train():
        plot_scores = []
        plot_mean_score = []

        total_score = 0
        record = 0

        agent = Agent()
        game = SnakeGameAI()

        while True:
            prev_state = agent.get_state(game) #previous state
            final_move = agent.get_action(prev_state) #get move
            reward,done,score = game.play_step(final_move)
            new_state = agent.get_state(game)

            #train short memory: based on one step
            agent.train_short_memory(prev_state,final_move,reward,new_state,done)
            agent.remember(prev_state,final_move,reward,new_state,done) #stores it into memory

            if done: #game is over
                #train the long memory, called replay memory or experience replay
                #plot the results
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    # agent.model.save()
                print('Game',agent.n_games,'Score', score, 'Record:', record)

                #TODO: plot the results

        pass
    
    if __name__=="__main__":
        train()