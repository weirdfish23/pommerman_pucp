'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents


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
    batch_size = 500
    for i_episode in range(300):

        agent_list = [agents.SimpleAgent()]
        dqAgent = agents.BaselineAgent()
        dqAgent.initDQ()
        dqAgent.load('model')
        agent_list.append(dqAgent)
        env = pommerman.make('OneVsOne-v0', agent_list)
        state = env.reset()
        env.set_init_game_state('jsons/000.json')
        done = False
        while not done:
            # env.render(record_json_dir='jsons/')
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            if done:
                print('Done:: ', reward)
                # print('info:: ', info)
                if reward[0] == 1 and reward[1] != 1:
                    wins0+=1    
                elif reward[1] == 1 and reward[0] != 1:
                    wins1+=1
                else: 
                    ties+=1
        print('Episode {} finished'.format(i_episode))
        if len(dqAgent.memory) > batch_size:  # si el agente tiene suficiente experiencias en su memoria -> ajusta su modelo neuronal 
            dqAgent.replay(batch_size)
        dqAgent.save('model')
    print('Results: {}/{}/{}'.format(wins0, wins1, ties))
        
    env.close()


if __name__ == '__main__':
    main()
