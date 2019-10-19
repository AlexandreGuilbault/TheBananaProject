import torch
import torch.nn as nn

class GRU_QNetwork(nn.Module):
    def __init__(self, n_frames, state_space, action_space, seed=0):
        super(GRU_QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.state_space = state_space
        self.action_space = action_space
        
        self.hidden_size = self.state_space
        self.num_layers = n_frames
        
        self.gru = nn.GRU(input_size=self.state_space, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.action_space)
        self.relu = nn.ReLU()
    
    def count_parameters(self):
        #https://arxiv.org/ftp/arxiv/papers/1701/1701.05923.pdf 
        m = self.state_space
        n = self.hidden_size
        return self.num_layers + 3*(n**2+n*m+n) + self.hidden_size * self.action_space
         
    def forward(self, state):
        x, h = self.gru(state)
        x = self.relu(x)
        return self.linear(x[:,-1,:])
    
class MLP_QNetwork(nn.Module):
    def __init__(self, n_frames, state_space, action_space, seed=0):
        super(MLP_QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.state_space = state_space
        self.action_space = action_space
        
        self.hidden_size = 64*self.state_space
        self.n_frames = n_frames

        self.network = nn.Sequential(
                nn.Linear(self.n_frames*self.state_space, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.action_space)                
                )

    def forward(self, state):
        return self.network(state.squeeze())