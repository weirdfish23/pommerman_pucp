'''Baseline agent by Joel Cabrera Rios PUCP'''
from . import BaseAgent

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class BaselineAgent(BaseAgent):
    def initDQ(self, state_size=64, action_size=6, learn=True, epsilon=1.0):
        self.state_size = state_size     # tamaño de un estado (numero de atributos que representan un estado)
        self.action_size = action_size   # tamaño del vector de acciones 
        self.memory = deque(maxlen=3000)  # define la memoria del agente (2000 registros como maximo)
        self.gamma = 0.95                 # discount rate
        self.learning_rate = 0.001        # taza de aprendizaje 
        
        self.epsilon = epsilon       # factor de exploration inicial
        self.epsilon_min = 0.01     # factor de exploration minimo
        self.epsilon_decay = 0.995   # factor de decaimiento del factor de exploracion
        self.model = self._build_model()  # construye el modelo neuronal para estimar las utilidades
        self.learn = learn

    def _build_model(self):
        # Define y compila un modelo de red neuronal de 3 capas: state_size entradas X 20 neuronas X 20 neuronas x action_size neuronas de salida
        # model = Sequential()   # Informa que las capas que se van agregar son secuenciales
        # model.add(Dense(20, input_dim=self.state_size, activation='relu')) # 1ra capa de 20 neuronas, cada neurona recibe state_size entradas (4 para CartPole), activacion relu
        # model.add(Dense(20, activation='relu')) # 2da capa de 20 neuronas, funcion de activacion relu
        # model.add(Dense(self.action_size, activation='linear')) # 3ra capa (salida) de action_size neuronas (2 para CartPole)

        model = Sequential()   # Informa que las capas que se van agregar son secuenciales
        model.add(Dense(64, input_dim=self.state_size, activation='relu')) # 1ra capa de 64 neuronas, cada neurona recibe state_size entradas , activacion relu
        model.add(Dense(40, activation='relu')) # 2da capa de 40 neuronas, funcion de activacion relu
        model.add(Dense(20, activation='relu')) # 2ra capa de 20 neuronas, funcion de activacion relu
        model.add(Dense(self.action_size, activation='linear')) # 4ta capa (salida) 
       
        model.compile(loss='mse', optimizer = Adam(lr=self.learning_rate)) # la funcion de perdida es el error cuadratico medio (mse)
        return model

    # metodo para guardar una transicion del agente (experiencia): (estado, accion, reward resultante, nuevo estado, done)
    # done es un flag que indica que el entorno cayo en un estado terminal
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))

    # retorna una accion.  
    def act(self, obs, action_space):
        # x, y = obs['position']
        state = np.ravel(obs['board'])
        if self.learn and np.random.rand() <= self.epsilon:  # retorna una accion aleatoria con probabilidad self.epsilon
            return random.randrange(self.action_size)
        state= np.reshape(state, [1, 64])
        action_values = self.model.predict(state) # obtiene los q valores predichos por el modelo para cada accion
        return np.argmax(action_values[0])  # retorna la accion con el maximo q-valor predicho

    # def act(self, obs, action_space):
    #     print('Action space:: ', action_space)
    #     print('Obs:: ', obs)
    #     return action_space.sample()

    def replay(self, batch_size): # ajusta la red neuronal con una muestra de su memoria de tamaño batch_size
        # obtiene una muestra de su memoria de experiencias
        minibatch = random.sample(self.memory, batch_size) 
        
        # recorre cada experiencia del minibatch de experiencias
        for state, action, reward, next_state, done in minibatch:
            # print('State from memory:: ', state.shape)
            # print('nextState from memory:: ', next_state.shape)

            # target es el vector de Q values de las posibles acciones desde state (por defecto son los predichos por el modelo)
            target = self.model.predict(state)
            
            if done:  # si cayo en un estado terminal
                # Actualiza el Q valor del target correspondiente a action, colocando el valor Q = reward
                target[0][action] = reward   
            else:  # si  no es estado terminal 
                # Predice los valores Q del next_state usando el modelo
                Qvals_next_state = self.model.predict(next_state)[0]
                # Actualiza el Q value del target correspondiente a la accion action con el future discounted reward
                target[0][action] = reward + self.gamma * np.amax(Qvals_next_state)
 
            self.model.fit(state, target, epochs=1, verbose=0) # ajusta pesos de la red con el ejemplo: (state,target)

        # si no esta en el valor minimo del factor de exploracion -> hace un decaimiento del factor de exploracion
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    
