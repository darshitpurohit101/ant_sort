
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
import math
from array import array

import ant_sort
import tables_and_plot_sim_3 as tables_and_plot

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

print("RUNNING........................")

class Play:
    def __init__(self):
        self.epsilon = 0.02
        self.gamma=0.4
        self.epsilon=0.4
        self.alpha = 0.2
        self.batch_size = 128
        self.train_step = 2000
        self.update_step = 1000
        self.state_history = {}
        self.track_state = ''
        self.tp = tables_and_plot.Create()
    
        '''storing the enviroment at the end of each episode'''
        self.enviroment = []
        '''storing the state of the ants after each episode'''
        self.carrying_ant = []
        '''storing glabal ratios after each iteration, episode wise'''
        self.global_ratios = {}
        
        '''appending state-action pair for all the ants
        togenther for NN training'''
        self.replay_buffer = {}
        
        '''Q value info for each ants'''
        self.episodic_q_table = []
        self.predicted_episodic_q_table = {}
        self.ant_id = 0
        
        ''' track the states and number of time any action 
        is been chosen for that state '''
        self.state_tracker = []
        self.action_tracker = []
        
    def FindMaxLength(self, lst):
        maxList = max(lst, key = lambda i: len(i))
        maxLength = len(maxList)
         
        return maxLength    

    def store(self):
        q_table = self.replay_buffer
        for sample in q_table.keys():
            state_key,targets_key,q_values_key,reward_1_key,reward_2_key,_,chosen_action_key,action_track_key = list(q_table[sample])
            
            state = q_table[sample][state_key].tolist()
            hist_targets = q_table[sample][targets_key]
            targets = hist_targets[-1][0]
            hist_q_values = q_table[sample][q_values_key]
            q_values = hist_q_values[-1][0]
            reward_1 = q_table[sample][reward_1_key][-1]
            reward_2 = q_table[sample][reward_2_key][-1]
            hist_chosen_action = q_table[sample][chosen_action_key]
            chosen_action = hist_chosen_action[-1]
            action_track = q_table[sample][action_track_key]
            
            hist =[]
            for i in range(len(hist_targets)):
                act = hist_chosen_action[i]
                t = hist_targets[i][0]
                q = hist_q_values[i][0]
                hist.append([act,t,q])
            
            self.episodic_q_table.append([state,targets,q_values,chosen_action,action_track,reward_1,reward_2,hist])
            
        
    def create_model(self):
        #DQN
        state_input = Input(shape=(9,))
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48, activation='relu')(state_h1)
        action_input = Input((2,))
        action_h1 = Dense(24, activation='relu')(action_input)
        action_h2 = Dense(48, activation='relu')(action_h1)
        merged = keras.layers.Concatenate(axis=1)([state_h2, action_h2])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(2, activation='relu')(merged_h1)
        
        model = Model(inputs = [state_input,action_input], outputs = output)
        adam  = Adam(lr=0.006)
        model.compile(loss="mse", optimizer=adam)
        
        return model
    
    def one_hot_encode(self,action):
        actions = np.zeros(2)
        actions[action] = 1
        return actions
    
    def predict(self,model,state):
        actions = np.array([1,1])
        q_values = model.predict([state.reshape(1,9),actions.reshape((1,2))])
        return q_values
        
    def epsilon_greedy_approach(self,q_values):
        if random.random() < self.epsilon:
            action = random.choice([0,1])
        else:
            action = np.argmax(q_values)
        return action
    
    def calculate_target(self, q_values, next_q_values, reward, action_indx):
        next_q_value = max(next_q_values)
        target = q_values[0][action_indx] + self.alpha*(reward + (self.gamma * next_q_value) - q_values[0][action_indx])
        # target = reward + self.gamma * next_q_value
        # target = math.ceil(target*1000)/1000
        old_next_q_values = q_values.copy()
        q_values[0][action_indx] = target
        # anti_target = next_q_values[action_indx-1]
        # next_q_values[action_indx-1] = math.ceil(anti_target*1000)/1000
        return q_values, old_next_q_values
    
    def training_data(self, batch):
        states, actions, q_values, next_states = [], [], [], []
        for i in batch:
            sample = self.replay_buffer[str(i)]
            state,targets,_,_,_,action_indx,_,_ = list(sample)
            states.append(sample[state].reshape((9,)))
            q_values.append(sample[targets][-1].tolist()[0])
            indx = sample[action_indx][-1]
            action = self.one_hot_encode(indx)
            action = np.array(action)
            actions.append(action)

        states = np.array(states)
        actions = np.array(actions)
        q_values = np.array(q_values)
        
        return [states, actions], q_values
    
    def training(self, x,y, model):
        model.fit(x, y, batch_size=self.batch_size, verbose=2)
        return model

if __name__ == "__main__" :
    play = Play()
    
    model = play.create_model() #should be trained with the updated q values
    target_model = play.create_model() #should be updated by copying weights of 'model'
    
    game = ant_sort.AntsGame()
    t_flag = False
    print("STARTING NOW......................................")
    
    for episode in range(100):
        print("##################################################################################################################################################################")
        print("################################################################ EPISODE "+str(episode)+"  "+"#############################################################################################")
        '''reset the game enivronment'''
        env = game.new_initial_state()
        ants = list(range(env.no_ants))
        play.episodic_q_table = []
        play.updated_q_value_history = {}
        play.p_q_value_history = {}
        play.replay_buffer = {}
        play.global_ratios[episode] = []
        c = 1
        for timestep in range(10000):
            global_ratio = env.global_ratio() #to get current global ratio give action=2 & ant_id can be any int value
            play.global_ratios[episode].append(global_ratio)
            random.shuffle(ants) #randomised order of turns
            initial_locations = env.start_locations #get the updated location of all the ants
            for ant in ants:
                a_flag = False #to check if legal action = 2
                play.ant_id = ant
                ant_states = env.ants_states
                ant_state = ant_states[play.ant_id]
                current_location = initial_locations[play.ant_id]
                current_state, _ = env.neighbours(current_location)
                current_state = np.array(current_state)
                legal_actions = env._legal_actions(current_state[4], play.ant_id)
                    
                if len(legal_actions)==2:
                     q_values = play.predict(target_model, current_state)
                     p_q_values = q_values.copy()
                     a_flag = True
                     if timestep >= play.train_step:
                        if timestep % play.update_step:
                             target_model.set_weights(model.get_weights())
                        batch = random.sample(list(play.replay_buffer), play.batch_size)
                        x,y = play.training_data(batch)
                        model = play.training(x,y,model)
                       
                        action_indx = play.epsilon_greedy_approach(q_values)
                     else:
                        action_indx = random.choice([0,1])
    
                else:
                     action_indx = 0

                act = legal_actions[action_indx]
                if a_flag == True:
                    '''Tracking action taken at the particular state. Storing information for debugging'''
                    if current_state.tolist() not in play.state_tracker:
                         #state_tracker = [state1....staten]
                        play.state_tracker.append(current_state.tolist())
                        play.action_tracker.append([0,0])

                    '''increment the selected action for particular state'''
                    indx = play.state_tracker.index(current_state.tolist())
                    play.action_tracker[indx][action_indx] += 1

                    next_state, map_to_food = env._apply_action(act,play.ant_id, current_location, current_state)
                    next_q_values = play.predict(target_model, next_state)
  
                    anti_act = legal_actions[(action_indx - 1)]
                    next_state_fake, _ = env._apply_action(anti_act,play.ant_id, current_location, current_state, flag=False)
                    reward_anti_act = env._reward(current_state.tolist(), next_state_fake.tolist())
                    reward = env._reward(current_state.tolist(), next_state.tolist())
                    
                    if action_indx == 1:
                        chosen_action = "action 2"
                        if reward_anti_act < 0:
                            reward = abs(reward_anti_act)
                        reward_1 = reward_anti_act*100
                        reward_2 = reward*100
                    else:
                        chosen_action = "action 1"
                        reward_1 = reward*100
                        reward_2 = reward_anti_act*100
                        
                    reward_anti_act = reward_anti_act*100
                    reward = reward*100
                    
                    if play.replay_buffer.get(str(current_state.tolist())) == None:
                        target_q_values, old_vals = play.calculate_target(q_values, next_q_values[0], reward, action_indx)
                        play.replay_buffer[str(current_state.tolist())] = {'state':current_state,'targets':[],'q values':[],'reward action 1':[], 'reward action 2':[],
                                                                           'action indx':[],'chosen action':[],'action track':[]}
                    else:
                        update_values = play.replay_buffer[str(current_state.tolist())]['targets'][-1]
                        target_q_values, old_vals = play.calculate_target(update_values, next_q_values[0], reward, action_indx)
                        
                    buffer = play.replay_buffer[str(current_state.tolist())]
                    buffer['targets'].append(target_q_values)
                    buffer['q values'].append(p_q_values.tolist())
                    buffer['reward action 1'].append(reward_1)
                    buffer['reward action 2'].append(reward_2)
                    buffer['action indx'].append(action_indx)
                    buffer['chosen action'].append(chosen_action)
                    buffer['action track'] = play.action_tracker[indx]
                    
                else:
                    next_state, map_to_food = env._apply_action(act, play.ant_id, current_location, current_state)
           
            play.enviroment.append(map_to_food)
            play.carrying_ant.append(ant_states)
            env.next_position()
        
        play.store()
        play.tp.q_table(play.episodic_q_table,episode)
        model_name = "sim_9_model_ep_"+str(episode)+".h5"
        target_model.save(model_name)


ratio_file = "sim_9_4_clrs__ratio"
with open(ratio_file, "wb") as f:
    pickle.dump(play.global_ratios,f)

