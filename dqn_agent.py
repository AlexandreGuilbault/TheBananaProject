import numpy as np
import random
from collections import namedtuple, deque

from models import GRU_QNetwork

import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# buffer_size		- replay buffer size
# batch_size		- minibatch size
# gamma			- discount factor
# tau 			- for soft update of target parameters
# lr 				- learning rate 
# update_every 		- how often to update the network

class Agent():
    def __init__(self, state_space, action_space, seed, buffer_size=100000, batch_size=64, gamma=0.99, tau=1000, lr=5e-4, repeat_frames=1, update_every=4):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.repeat_frames = repeat_frames
        self.update_every = update_every
        
        
        self.state_space = state_space
        self.action_space = action_space
        self.seed = random.seed(seed)

        self.qnetwork_local = GRU_QNetwork(n_frames=self.repeat_frames, state_space=self.state_space, action_space=self.action_space, seed=seed).to(device)
        self.qnetwork_target = GRU_QNetwork(n_frames=self.repeat_frames, state_space=self.state_space, action_space=self.action_space, seed=seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        self.replay_memory = ReplayBuffer(action_space, self.buffer_size, self.batch_size, seed)

        self.t_step = 0
    
    def step(self, state_buffer, action, reward, next_state, done):
        self.replay_memory.add(state_buffer, action, reward, next_state, done) # save into replay memory
        
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0: # If enough samples are available in replay_memory, get random subset and learn
            
            if len(self.replay_memory) > self.batch_size:
                experiences = self.replay_memory.sample()
                self.learn(experiences, self.gamma)


    def act(self, state_buffer, eps=0.):

        if random.random() > eps :  # Greedy
            state = torch.from_numpy(np.array(state_buffer)).float().unsqueeze(0).to(device)
            
            self.qnetwork_local.eval()
            with torch.no_grad(): action_values = self.qnetwork_local(state.reshape(-1,self.repeat_frames, self.state_space))
            self.qnetwork_local.train() 

            noise = 1.0#(torch.rand_like(action_values)*0.01+0.99) # To avoid being stuck in a loop
            
            return np.argmax((action_values*noise).cpu().data.numpy())
        else: # Exploration
            return random.choice(np.arange(self.action_space))


    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences
       
        with torch.no_grad():
            # DQN
            #q_target_reward = self.qnetwork_target(next_states.reshape(-1,self.repeat_frames,self.state_space)).max(1)[0].unsqueeze(1)
            # Double DQN
            hindsight_actions =  self.qnetwork_local(next_states.reshape(-1,self.repeat_frames,self.state_space)).max(1)[1].squeeze().unsqueeze(1)
            q_target_reward = self.qnetwork_target(next_states.reshape(-1,self.repeat_frames,self.state_space)).gather(1,hindsight_actions)
        
        target = rewards + gamma * q_target_reward * (1-dones)
        estimation = self.qnetwork_local(states.reshape(-1,self.repeat_frames,self.state_space)).gather(1,actions)

        loss = ((target - estimation)**2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau) # update target network

    def soft_update(self, local_model, target_model, tau):
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class ReplayBuffer:
    def __init__(self, action_space, buffer_size, batch_size, seed):

        self.action_space = action_space
        self.replay_memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.replay_memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.replay_memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([np.array(e.state) for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([np.array(e.next_state) for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.replay_memory)