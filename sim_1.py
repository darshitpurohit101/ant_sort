from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers.merge import Add, Multiply
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import keras
import keras
from IPython.display import clear_output
import random
import time
import pickle
import numpy as np

import ant_sort

class Play:
    def __init__(self):
        self.epsilon = 0.02
        self.gamma=0.4
        self.epsilon=0.4
        self.alpha = 0.2
        self.batch_size = 64
        self.train_step = 2000
        self.update_step = 1000
    
        '''storing the enviroment at the end of each episode'''
        self.enviroment = []
        '''storing the state of the ants after each episode'''
        self.carrying_ant = []
        '''storing glabal ratios after each iteration, episode wise'''
        self.global_ratios = {}
        
        '''appending state-action pair for all the ants
        togenther for NN training'''
        self.replay_buffer = []
        
        ''' Q value info for each ants'''
        self.q_table = {}
        self.ant_id = 0
        
        '''track the states and number of time any action 
        is been chosen for that state'''
        self.state_tracker = []
        self.action_tracker = []
        
    
    def store(self, current_states, q_values, rewards, rewards_anti_act, actions):
        for i in range(len(current_states)):
            
            '''storing information for debugging'''
            if current_states[i].tolist() not in self.state_tracker:
                #state_tracker = [state1....staten]
                self.state_tracker.append(current_states[i].tolist())
                self.action_tracker.append([0,0])
            '''increment the selected action for particular state'''
            indx = self.state_tracker.index(current_states[i].tolist())
            self.action_tracker[indx][actions[i]] += 1
            
            if actions[i] == 0:
                reward_1 = rewards[i]
                reward_2 = rewards_anti_act[i]
            else:
                reward_1 = rewards_anti_act[i]
                reward_2 = rewards[i]        
            
            '''storing information to q table'''
            if self.q_table.get(self.ant_id) == None:
                #format >> q_table={ant_id:[[state, Qvalue_action_1, Qvalue_action_2]....[]]}
                self.q_table[self.ant_id] = []
            self.q_table[self.ant_id].append([current_states[i], q_values[i][0], q_values[i][1],
                                        self.action_tracker[indx][0], self.action_tracker[indx][1],
                                        reward_1, reward_2])
    
    def create_model(self):
        #DQN
        state_input = Input(shape=(9,))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(2, activation='relu')(h3)
        model = Model(inputs=state_input, outputs=output)
        adam  = Adam(lr=0.006)
        model.compile(loss="mse", optimizer=adam)
        
        return model
    
    def calculate_target(self, batch, target_model):
        states, actions, rewards, next_states, rewards_anti_act = [], [], [], [], []
        for i in batch:
            state, action, reward, next_state, reward_anti_act = i
            states.append(state.reshape((9,)))
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.reshape((9,)))
            rewards_anti_act.append(reward_anti_act)
        
        next_states = np.array(next_states)
        next_q_values = target_model.predict(next_states)
        #caculating target and updating q values
        for i in range(len(batch)):
            next_q_value = max(next_q_values[i])
            target = rewards[i] + self.gamma * next_q_value
            next_q_values[i][actions[i]] = target
        
        self.store(states, next_q_values, rewards, rewards_anti_act, actions)
        
        return next_states, next_q_values
    
    def training(self, x,y, model):
        X = np.array(x)
        Y = np.array(y)
        model.fit(X, Y, batch_size=self.batch_size, verbose=0)
        return model

if __name__ == "__main__" :
    play = Play()
    
    model = play.create_model() #should be trained with the updated q values
    target_model = play.create_model() #should be updated by copying weights of 'model'
    
    game = ant_sort.AntsGame()
    
    for episode in range(1):
        '''reset the game enivronment'''
        env = game.new_initial_state()
        ants = list(range(env.no_ants))
        play.global_ratios[episode] = []
        for timestep in range(1, 3000):
            global_ratio = env.global_ratio([0,0],2,0) #to get current global ratio give action=2 & ant_id can be any int value
            play.global_ratios[episode].append(global_ratio)
            random.shuffle(ants) #randomised order of turns
            initial_locations = env.start_locations #get the updated location of all the ants
            for ant in ants:
                a_flag = False #to check if legal action = 2
                play.ant_id = ant
                global_ratio_old = env.global_ratio([0,0],2,0)
                ant_states = env.ants_states
                ant_state = ant_states[play.ant_id]
                current_location = initial_locations[play.ant_id]
                current_state, _ = env.neighbours(current_location)
                current_state = np.array(current_state)
                legal_actions = env._legal_actions(current_state[4], play.ant_id)
                    
                if timestep >= play.train_step:
                    if timestep % play.update_step:
                            target_model.set_weights(model.get_weights())
                    batch = random.sample(play.replay_buffer, 5)
                    x, y = play.calculate_target(batch, target_model)
                    model = play.training(x,y,model)
                    
                    if len(legal_actions)==2:
                        q_value = target_model.predict(current_state.reshape(1,9))
                        
                        if random.random() < play.epsilon: 
                            '''Returns the index of maximum q value'''
                            action = np.argmax(q_value[0]) #q_value looks like [[q1, q2]] that is why using q_value[0]
                        else:
                            action = random.choice([0,1])
                        
                        a_flag = True
                    else:
                        action = 0
    
                else:
                    if len(legal_actions)==2:
                        #randomly choose actions untill enough examples are generated
                        action = random.choice([0,1])
                        '''flag is ture if there are 2 possible action, then q values are updated 
                        and sotred. Else only 1 action (PASS) is available so doesnt update of store the state '''
                        a_flag = True 
                    else:
                        action = 0
                        
                act = legal_actions[action]
                if a_flag == True:
                    next_state, map_to_food = env._apply_action(act,play.ant_id, current_location, current_state)
    
                    anti_act = legal_actions[(action - 1)]
                    next_state_fake, _ = env._apply_action(anti_act,play.ant_id, current_location, current_state, flag=False)
                    reward_anti_act = env._reward(current_state.tolist(), next_state_fake.tolist())
                    reward_anti_act = reward_anti_act*100
                    reward = env._reward(current_state.tolist(), next_state.tolist())
                    reward = reward*100
                    
                    if action == 0:
                        reward_1 = reward
                        reward_2 = reward_anti_act
                    else:
                        reward_1 = reward_anti_act
                        reward_2 = reward                
                    
                    play.replay_buffer.append(np.array([current_state, action, reward, next_state, reward_anti_act]))
                else:
                    next_state, map_to_food = env._apply_action(act, play.ant_id, current_location, current_state)
           
            play.enviroment.append(map_to_food)
            play.carrying_ant.append(ant_states)
            env.next_position()

ratio_file = r"A:\THESIS\trial_1\l2\trst_ratios"
with open(ratio_file, "wb") as f:
    pickle.dump(play.global_ratios,f)

q_file = r"A:\THESIS\trial_1\l2\test_q_table"
with open(q_file, "wb") as f:
    pickle.dump(play.q_table,f)
    
ant_file = r"A:\THESIS\trial_1\l2\test_state_of_ants"
with open(ant_file, "wb") as f:
    pickle.dump(play.carrying_ant,f)
    
board_file = r"A:\THESIS\trial_1\l2\test_board"
with open(board_file, "wb") as f:
    pickle.dump(play.enviroment,f)
    






