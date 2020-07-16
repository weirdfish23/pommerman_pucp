'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np
from tqdm import tqdm

def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # dqAgent = agents.BaselineAgent()
    # dqAgent.save('model')

    # Create a set of agents (exactly four)
    # agent_list = [
    #     agents.SimpleAgent(),
        # agents.BaselineAgent().load('model')
        #agents.SimpleAgent(),
        # agents.RandomAgent(),
        # agents.SimpleAgent(),
        # agents.PlayerAgent(agent_control="arrows"), # arrows to move, space to lay bomb
        # agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    # ]
    # Make the "Free-For-All" environment using the agent list
    # env = pommerman.make('OneVsOne-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    wins0 = 0
    wins1 = 0
    ties = 0
    batch_size = 15
    for i_episode in tqdm(range(100)):

        agent_list = [agents.RandomAgent()]
        adversarialAgent = agents.AdversarialSearchAgent()
        
        agent_list.append(adversarialAgent)
        env = pommerman.make('OneVsOne-v0', agent_list)
        env.set_init_game_state('jsons/000.json')
        env.seed(0)
        state = env.reset()
        
        done = False
        while not done:
            # env.render()

            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            if done:
                # print('Done:: ', reward)
                # print('info:: ', info)
                if reward[0] == 1 and reward[1] != 1:
                    wins0+=1    
                elif reward[1] == 1 and reward[0] != 1:
                    wins1+=1
                else: 
                    ties+=1
        # print('Episode {} finished'.format(i_episode))
        
    print('Results: {}/{}/{}'.format(wins0, wins1, ties))
        
    env.close()


if __name__ == '__main__':
    main()
