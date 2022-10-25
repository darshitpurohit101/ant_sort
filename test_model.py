import keras
import numpy as np
import random
import pickle
import sys
import ant_sort

class Play:
    def __init__(self):
        '''storing the enviroment at the end of each episode'''
        self.enviroment = []
        '''storing the state of the ants after each episode'''
        self.carrying_ant = []
        '''storing glabal ratios after each iteration, episode wise'''
        self.global_ratios = {}
    
    def predict(self,model,state):
        actions = np.array([1,1])
        q_values = model.predict([state.reshape(1,9),actions.reshape((1,2))])
        return q_values
    
    def load_model(self, model_path):
        model = keras.models.load_model(model_path)
        return model
        
if __name__ == "__main__" :
    play = Play()
    
    model_path = sys.argv
    model = play.load_model(model_path)
    print("model path: ",model)
    
    game = ant_sort.AntsGame()
    
    for episode in range(1):
        '''reset the game enivronment'''
        env = game.new_initial_state()
        ants = list(range(env.no_ants))
        play.global_ratios[episode] = []
        for timestep in range(10000):
            global_ratio = env.global_ratio([0,0],2,0) #to get current global ratio give action=2 & ant_id can be any int value
            play.global_ratios[episode].append(global_ratio)
            random.shuffle(ants) #randomised order of turns
            initial_locations = env.start_locations #get the updated location of all the ants
            for ant in ants:
                a_flag = False #to check if legal action = 2
                global_ratio_old = env.global_ratio([0,0],2,0)
                ant_states = env.ants_states
                current_location = initial_locations[play.ant_id]
                current_state, _ = env.neighbours(current_location)
                current_state = np.array(current_state)
                legal_actions = env._legal_actions(current_state[4], play.ant_id)
                
                if len(legal_actions)==2:
                    q_values = play.predict(model, current_state)
                    print("q values: ",q_values)
                    action_indx = np.argmax(q_values)
                    print("Chosen action: ",action_indx)
                    a_flag = True
                else:
                    action_indx = 0

                act = legal_actions[action_indx]
                if a_flag == True:
                    _, map_to_food = env._apply_action(act,play.ant_id, current_location, current_state)
                else:
                    _, map_to_food = env._apply_action(act, play.ant_id, current_location, current_state)
           
            play.enviroment.append(map_to_food)
            play.carrying_ant.append(ant_states)
            if timestep == 9999:
                print(ant_states)
                print("########################################################################")
                print(map_to_food)
            env.next_position()

ratio_file = "play_Global_ratios"
with open(ratio_file, "wb") as f:
    pickle.dump(play.global_ratios,f)

# q_file = "test_q_table"
# with open(q_file, "wb") as f:
#     pickle.dump(play.q_table,f)
    
board_file = "play_board"
with open(board_file, "wb") as f:
    pickle.dump(play.enviroment,f)

ant_state_file = "play_ant_states"
with open(ant_state_file, "wb") as f:
    pickle.dimp(play.carrying_ant,f)