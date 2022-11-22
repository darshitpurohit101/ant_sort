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

import ant_sort_06 as ant_sort
import tables_and_plot_sim_4 as tables_and_plot

import sys
import warnings

from tqdm import tqdm

if not sys.warnoptions:
    warnings.simplefilter("ignore")

print("RUNNING........................")

class Play:
    def __init__(self):
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.02
        self.decay = 0.01
        self.gamma=0.4
        self.alpha = 0.2
        self.batch_size = 64
        self.train_step = 1000
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
        self.replay_buffer = []
        self.state_hist = {}
        self.table = {}
        
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
        q_table = self.table
        for sample in q_table.keys():
            state_key,targets_key,q_values_key,reward_1_key,reward_2_key,_,chosen_action_key,action_track_key = list(q_table[sample])
            
            state = q_table[sample][state_key].tolist()
            hist_targets = q_table[sample][targets_key]
            targets = hist_targets[-1]
            hist_q_values = q_table[sample][q_values_key]
            q_values = hist_q_values[-1]
            reward_1 = q_table[sample][reward_1_key][-1]
            reward_2 = q_table[sample][reward_2_key][-1]
            hist_chosen_action = q_table[sample][chosen_action_key]
            chosen_action = hist_chosen_action[-1]
            action_track = q_table[sample][action_track_key]
            
            hist = []
            for i in range(len(hist_targets)):
                act = hist_chosen_action[i]
                t = hist_targets[i][0]
                q = hist_q_values[i][0]
                hist.append([act,t,q])
            
            self.episodic_q_table.append([state,targets,q_values,chosen_action,action_track,reward_1,reward_2,hist])
            
    
    def create_model(self):
        init = tf.keras.initializers.HeUniform()
        state_input = Input(shape=(1))
        state_h1 = Dense(48, activation='relu',  kernel_initializer=init)(state_input)
        state_h2 = Dense(24, activation='relu',  kernel_initializer=init)(state_h1)
        output = Dense(2, activation='relu',  kernel_initializer=init)(state_h2)
        
        model = Model(inputs = state_input, outputs = output)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        
        return model
    
    def predict(self,model,state):
        q_values = model.predict(state)
        return q_values
        
    def epsilon_greedy_approach(self,q_values):
        if random.random() < self.epsilon:
            action = random.choice([0,1])
        else:
            action = np.argmax(q_values)
        return action
    
    def calculate_target(self, batch, target_model, encoder):
        states,actions,rewards,next_states, reward_1s, reward_2s, action_trackers, chosen_actions, encoded_states, encoded_next_states = [],[],[],[],[],[],[],[],[],[]
        for sample in batch:
            state, action, reward, next_state, reward_1, reward_2, action_tracker, chosen_action = sample
            reward_1s.append(reward_1)
            reward_2s.append(reward_2)
            action_trackers.append(action_tracker)
            chosen_actions.append(chosen_action)
            #encode and get the append the states
            encoded_state = encoder.predict(state.reshape((1,9)))
            encoded_states.append(encoded_state)
            states.append(state.reshape((9,)))
            encoded_next_state = encoder.predict(next_state.reshape((1,9)))
            encoded_next_states.append(encoded_next_state)
            next_states.append(next_state.reshape((9,)))
            actions.append(action)
            rewards.append(reward)
        states = np.array(states)
        encoded_states = np.array(encoded_states)
        next_states = np.array(next_states)
        encoded_next_states = np.array(encoded_next_states)
        q_values = self.predict(target_model, encoded_states)
        p_q_values = q_values.copy()
        next_q_values = self.predict(target_model, encoded_next_states)
        
        for i in range(len(q_values)):
            next_q_value = max(next_q_values[i])
            target = rewards[i] + self.gamma * next_q_value
            q_values[i][actions[i]] = q_values[i][actions[i]] + self.alpha*(rewards[i] + (self.gamma * target) - q_values[i][actions[i]])
        
            if self.table.get(str(states[i].tolist())) == None:
                self.table[str(states[i].tolist())] = {'state':current_state,'targets':[],'q values':[],'reward action 1':[], 'reward action 2':[],
                                                                   'action indx':[],'chosen action':[],'action track':[]}
                
            buffer = self.table[str(states[i].tolist())]
            buffer['targets'].append(q_values[i].tolist())
            buffer['q values'].append(p_q_values[i].tolist())
            buffer['reward action 1'].append(reward_1s[i])
            buffer['reward action 2'].append(reward_2s[i])
            buffer['action indx'].append(actions[i])
            buffer['chosen action'].append(chosen_actions[i])
            buffer['action track'] = action_trackers[i]

            q_values = np.array(q_values)
        return encoded_states, q_values
    
    def training(self, x,y, model):
        model.fit(x, y, batch_size=self.batch_size, verbose=0)
        return model

if __name__ == "__main__" :
    play = Play()
    
    model = play.create_model() #should be trained with the updated q values
    target_model = play.create_model() #should be updated by copying weights of 'model'
    
    #loading encoder state
    encoder_path = r"A:/THESIS/trial_1/l3/encoder_9_1_1_9_epochs_10000.h5"
    encoder = keras.models.load_model(encoder_path)
    
    game = ant_sort.AntsGame()
    t_flag = False
    print("STARTING NOW......................................")
    
    for episode in tqdm(range(100)):
        print("##################################################################################################################################################################")
        print("################################################################ EPISODE "+str(episode)+"  "+"#############################################################################################")
        '''reset the game enivronment'''
        env = game.new_initial_state()
        ants = list(range(env.no_ants))
        play.episodic_q_table = []
        play.updated_q_value_history = {}
        play.p_q_value_history = {}
        #play.replay_buffer = {}
        play.global_ratios[episode] = []
        c = 1
        for timestep in tqdm(range(10000)):
            if timestep%50:
                if play.epsilon != play.min_epsilon:
                    play.epsilon = play.epsilon - play.decay
            global_ratio = env.global_ratio() #to get current global ratio give action=2 & ant_id can be any int value
            play.global_ratios[episode].append(global_ratio)
            random.shuffle(ants) #randomised order of turns
            initial_locations = env.start_locations #get the updated location of all the ants
            
            if timestep>=1000:
               if timestep % play.update_step:
                    target_model.set_weights(model.get_weights())
               batch = random.sample(play.replay_buffer, play.batch_size)
               x,y = play.calculate_target(batch,target_model,encoder)
               # x,y = play.training_data(batch)
               model = play.training(x,y,model)
               
            for ant in ants:
                a_flag = False #to check if legal action = 2
                play.ant_id = ant
                ant_states = env.ants_states
                temp_ant_states = ant_states.copy()
                ant_state = ant_states[play.ant_id]
                current_location = initial_locations[play.ant_id]
                current_state, _ = env.neighbours(current_location)
                current_state = np.array(current_state)
                legal_actions = env._legal_actions(current_state[4], play.ant_id)
                    
                if len(legal_actions)==2:
                     encoded_state = encoder.predict(current_state.reshape((1,9)))
                     q_values = play.predict(target_model, encoded_state)
                     a_flag = True
                     if timestep>=2000:
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
                    
                    play.replay_buffer.append([current_state,action_indx,reward,next_state,reward_1,reward_2,play.action_tracker[indx], chosen_action])

                else:
                    next_state, map_to_food = env._apply_action(act, play.ant_id, current_location, current_state)
           
            play.enviroment.append(map_to_food)
            play.carrying_ant.append(temp_ant_states)
            env.next_position()
            
            if timestep % 3000 == 0:
                model_name = r"A:\THESIS\trial_1\l3\test_from_prof\EN\2_0.65_6_gpu_EN_model_timestep_"+str(timestep)+".h5"
                target_model.save(model_name)
            
        play.store()
        play.tp.q_table(play.episodic_q_table,episode)
        play.tp.save_avg_global_ratio(play.global_ratios)
        model_name = r"A:\THESIS\trial_1\l3\test_from_prof\EN\2_0.65_6_gpu_EN_model_ep_"+str(episode)+".h5"
        target_model.save(model_name)
        replay_file = r"A:\THESIS\trial_1\l3\test_from_prof\EN\2_0.65_6_gpu_EN_replay_buffer"
        with open(replay_file,'wb') as f:
            pickle.dump(play.replay_buffer,f)
        carrying_ants_file = r"A:\THESIS\trial_1\l3\test_from_prof\EN\2_0.65_6_gpu_EN_ant_cayying_objects"
        with open(carrying_ants_file,'wb') as f:
           pickle.dump(play.carrying_ant,f)
 
ratio_file = r"A:\THESIS\trial_1\l3\test_from_prof\EN\2_0.65_6_gpu_EN_clrs_ratio"
with open(ratio_file, "wb") as f:
    pickle.dump(play.global_ratios,f)
