import torch
import numpy as np
from collections import deque
import time

def DeepQNetwork(env, agent, score_for_success, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, repeat_frames=1, show_every=100, save_every=250):

    brain_name = env.brain_names[0]
    
    scores = []                               # list containing scores from each episode
    scores_window = deque(maxlen=show_every)  # last 100 scores
    eps = eps_start          

    start_time = time.time() 
    for i_episode in range(1, n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        
        state_buffer = []
        for _ in range(repeat_frames): 
            state_buffer.append(state)
        
        for _ in range(max_t):
            action = np.int(agent.act(state_buffer, eps))
            
            next_state_buffer = []
            total_reward = 0.
            for f in range(repeat_frames) : 
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                
                next_state_buffer.append(next_state)
                total_reward += reward

                if done : break
            
            while len(next_state_buffer) < repeat_frames: 
                next_state_buffer.append(next_state_buffer[-1])
                
            agent.step(state_buffer, action, total_reward, next_state_buffer, done)
            
            state_buffer = next_state_buffer             
            score += total_reward
            
            if done: 
                break 

        completion = (i_episode)/(n_episodes)
        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time * (1/completion - 1)
        
        em, es = divmod(elapsed_time, 60)
        eh, em = divmod(em, 60)
        
        rm, rs = divmod(remaining_time, 60)
        rh, rm = divmod(rm, 60)

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        print('\rEpisode {:4.0f}/{} | Average Score: {:5.2f} | Epsilon : {:.2f} | Elapsed time : {:.0f}h {:02.0f}min {:02.0f}sec | Remaining time : {:.0f}h {:02.0f}min {:02.0f}sec'.format(i_episode, n_episodes, np.mean(scores_window), eps, eh, em, es, rh, rm, rs), end="")

        if i_episode % show_every == 0: 
            print('\rEpisode {:4.0f}/{} | Average Score: {:5.2f} | Epsilon : {:.2f}'.format(i_episode, n_episodes, np.mean(scores_window), eps))
        if np.mean(scores_window) > score_for_success : 
            print('\rEpisode {:4.0f}/{} | Average Score: {:5.2f} | Epsilon : {:.2f}'.format(i_episode, n_episodes, np.mean(scores_window), eps))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_{}_e{}.pth'.format("Banana", i_episode))
            print()
            print("Task succesfully completed in {} episodes!".format(i_episode))
            return scores
        if i_episode % save_every == 0 : 
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_{}_e{}.pth'.format("Banana", i_episode))
  
    print()
    print("Task not completed after {} episodes...".format(i_episode))
    
    return scores