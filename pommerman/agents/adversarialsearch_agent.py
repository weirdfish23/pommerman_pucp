from . import BaseAgent
from .. import forward_model
import random
import numpy as np

def game(obs,action):
    my_position = [obs['position'][0], obs['position'][1]] #tuple(obs['position'])
    board = np.array(obs['board'])
    # bombs = np.array(obs['bomb_blast_strength'])
    # enemies = [constants.Item(e) for e in obs['enemies']]
    # ammo = int(obs['ammo'])
    # blast_strength = int(obs['blast_strength'])
    
    if(action==0):
        pass
    elif(action==1):
        if my_position[1]-1 >= 0 and  (board[my_position[0],my_position[1]-1]==0):
            my_position[1]-=1
            board[my_position[0],my_position[1]]=10
            board[my_position[0],my_position[1]+1]=0
    elif(action==2):
        if my_position[1]+1 <= 7 and (board[my_position[0],my_position[1]+1]==0):
            my_position[1]+=1
            board[my_position[0],my_position[1]]=10
            board[my_position[0],my_position[1]-1]=0 
    elif(action==3):
        if my_position[0]-1 >= 0 and (board[my_position[0]-1,my_position[1]]==0):
            my_position[0]-=1
            board[my_position[0],my_position[1]]=10
            board[my_position[0]+1,my_position[1]]=0
    elif(action==4):
        if my_position[0]+1 <= 7 and (board[my_position[0]+1,my_position[1]]==0):
            my_position[0]+=1
            board[my_position[0],my_position[1]]=10
            board[my_position[0]-1,my_position[1]]=0 
    elif(action==5):
        board[my_position[0],my_position[1]]=3
        #debería el agente retroceder? o donde se pone al agente?
    obs['position'] = my_position 
    obs['board']=board
    return obs                
            
def terminal_test(obs):
    alive =  obs['alive']
    if (len(alive)<=1) :
        return True
    else:
        return False

def utility(obs):
    if '10' in obs['alive']:#se debe corregir el 10, ya que esa no siempre es el agente ?
        return 1
    else:
        return -1

class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""

    def __init__(self, parent=None, state=None, U=0, N=0):
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children = {}
        self.actions = None
        self.state = state
    

class AdversarialSearchAgent(BaseAgent):
    

    def ucb(self, n, C=1.4):
        if n.N == 0:
            return np.inf    
        else:
            return (n.U / n.N) + C * np.sqrt(np.log(n.parent.N) / n.N)       

    def select(self, n):
        """select a leaf node in the tree"""
        if n.children:
            return self.select(max(n.children.keys(), key=self.ucb))
        else:
            return n

    def expand(self, n):
        k = 3 # Maximo numero de nodos a colocar en el arbol en la expansion
        
        legal_actions = [0,1,2,3,4,5]
        
        n.children = {MCT_Node(state=game(n.state,action), parent=n): action
                    for action in random.sample(legal_actions, k = min(k,len(legal_actions)))}
        #"""expand the leaf node by adding all its children states"""
        #if not n.children and not game.terminal_test(n.state):
            # n.children = {MCT_Node(state=game.result(n.state, action), parent=n): action
            #                for action in game.actions(n.state)}
        return self.select(n)

    def simulate(self, state):
        """simulate the utility of current state by random picking a step"""
        i=0
        while not terminal_test(state) and i < 100:
            action = random.choice([0,1,2,3,4,5])
            state = game(state, action) #(obs, action)
            i+=1
        #como saber mi posicion actual (del agente)?
        v = utility(state)
        return -v

    def backprop(self, n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        # if utility == 0:
        #     n.U += 0.5
        n.N += 1
        if n.parent:
            self.backprop(n.parent, -utility) 

    def monte_carlo_tree_search(self, state, N=1000):
        root = MCT_Node(state=state)

        for _ in range(N):
            leaf = self.select(root)
            child = self.expand(leaf)
            result = self.simulate(child.state)  #(game, child.state)
            self.backprop(child, result)

        max_state = max(root.children, key=lambda p: p.N)

        return root.children.get(max_state)

        

        

    def act(self,obs,action_space):
        max_state=self.monte_carlo_tree_search(obs ) #, N=1000)
        #retornar la accion a realizar
        return max_state