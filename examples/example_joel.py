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

    # dqAgent = agents.DeepQAgent()
    # dqAgent.save('model')

    # Create a set of agents (exactly four)
    # agent_list = [
    #     agents.SimpleAgent(),
        # agents.DeepQAgent().load('model')
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
    for i_episode in tqdm(range(10)):

        agent_list = [agents.SimpleAgent()]
        dqAgent = agents.DeepQAgent()
        dqAgent.initDQ(learn=False, epsilon=0.5, action_size=5)
        dqAgent.load('model_3x64_40_20_no_bombs')
        agent_list.append(dqAgent)
        env = pommerman.make('OneVsOne-v0', agent_list)
        env.set_init_game_state('jsons/000.json')
        env.seed(0)
        state = env.reset()
        
        done = False
        while not done:
            # env.render(record_json_dir='jsons/')
            env.render()
            actions = env.act(state)
            prevState = np.ravel(state[1]['board'])
            prevState= np.reshape(prevState, [1, 64])
            # print('prev State shape:: ', prevState.shape)

            # print('Actions:: ', actions)
            state, reward, done, info = env.step(actions)

            #print('state:: ', state[1]['board'])
            nextState = np.ravel(state[1]['board'])
            nextState= np.reshape(nextState, [1, 64])
            # print('nextState shape:: ', nextState.shape)

            dqAgent.remember(prevState, actions[1], reward[1], nextState, done )

            # print('Memory:: ', len(dqAgent.memory))

            if len(dqAgent.memory) > batch_size:  # si el agente tiene suficiente experiencias en su memoria -> ajusta su modelo neuronal 
                # print('done replay')
                dqAgent.replay(batch_size)
                # print("Done replay:: memory lem", len(dqAgent.memory))
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
        
        # dqAgent.save('model_3x64_40_20')
    print('Results: {}/{}/{}'.format(wins0, wins1, ties))
        
    env.close()


if __name__ == '__main__':
    main()
