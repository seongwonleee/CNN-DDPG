import numpy as np
from ddpg import DDPG
from environment.vrep_env import Env
import os

#specify parameters here:
episodes=30000
is_batch_norm = False #batch normalization switch

def main():
    env = Env(20000)
    steps= 10000
    num_states = 59
    num_actions = 3

    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm)
    counter=0
    reward_per_episode = 0    
    total_reward=0
    reward_st = np.array([0])

    agent.actor_net.load_actor(os.getcwd() + '/weights/actor/model.ckpt')
    agent.critic_net.load_critic(os.getcwd() + '/weights/critic/model.ckpt')

    for i in range(episodes):
        # print "==== Starting episode no:",i,"====","\n"
        observation = env.reset()
        done =False
        reward_per_episode = 0
        for t in range(steps):
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            action = action[0]

            observation,reward,done = env.step(action)
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done or (t == steps-1)):
                print('Episode',i,'Steps: ',t,'Episode Reward:',reward_per_episode)
                reward_st = np.append(reward_st,reward_per_episode)
                # np.savetxt('episode_reward.txt', reward_st, newline="\n")
                break
    total_reward+=reward_per_episode            

if __name__ == '__main__':
    main()    