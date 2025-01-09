import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os #to save model

class Linear_QNet(nn.Module): #this will be a feedforward model with an input, hidden and ouput layer
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size) #input is input_size and output as hidden_size
        self.linear2 = nn.Linear(hidden_size,input_size) #input is hidden_size and output as input_size



    '''
    forward function defines the computation that our model will perform when given a tensor/input x (represents a batch of data)
    when you call the model, this function is executed automatically
    '''
    def forward(self, x): #needed when using torch, it gets the tensor x
        '''
        apply the linear layer and use actuation function-> actuation function introduces non-linearity into the model
        relu stands for Rectified Linear Unit (ReLU)
        ->ReLU(z)=max(0,z)
        '''
        x = F.relu(self.linear1(x)) #replaces any negative z values with 0
        x = self.linear2(x) #applies second linear layer
        return x #returns output tensor

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict)

    #optimizations
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion =nn.MSELoss()
    
    def train_step(self, state,action,reward,new_state,done): #can be a tuple, list or single value
        #convert into tensors
        state = torch.tensor(state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            #only one dimension, so we want to reshape
            # we want (1,x)
            state = torch.unsqueeze(state, 0) #appends 1 dim in the beginning
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

            #1 predicted Q values with the current state
            pred = self.model(state)
            target = pred.clone()
            for idx in range(len(done)):
                q_new = reward[idx]
                if not done[idx]:
                    q_new = reward[idx] + self.gamma*max(self.model(state[idx]))
                target[idx][torch.argmax(action).item()] = q_new

            self.optimizer.zero_grad() #empties the gradient
            loss = self.criterion(target, pred)
            loss.backward() #apply back propogation
            self.optimizer.step()


            #2 apply formula: r+y* max(next predicted q value) -> we ONLY do this if not done
