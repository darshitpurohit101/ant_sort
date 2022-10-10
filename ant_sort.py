# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 22:56:52 2022

@author: 91820
"""

import enum

import numpy as np
import random
from math import *

class AntsGame():
    def __init__(self):
        self._rows = 20
        self._columns = 20
        self._no_items = 1
        self._items_density = (self._no_items * (self._rows * self._columns)) / (self._no_items+1)
        self._no_ants = 6
        self._colours = np.arange(2,self._no_items+2) #unique list of colours, '1' is reserved for empty cell
        self._game_legth = 10000
    
    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return AntsState()
    
    def rows(self):
        return self._rows
  
    def columns(self):
        return self._columns

    def no_items(self):
        return self._no_items

    def no_ants(self):
        return self._no_ants

    def colours(self): #checked
        return self._colours

    def game_length(self):
        return self._game_legth

    def board(self): #checked
        #creating array with shape nxm with values as cell number
        mat = np.arange(0,self._rows*self._columns)
        mat = np.reshape(mat,(self._rows,self._columns),order='C')
        return mat

    def initial_locations(self): #checked
        #all ants are placed at random locations initially
        locations = [random.sample(range(0,self._rows),2) for _ in range(self._no_ants)]
        return locations    

    def ants_states(self): #checked
        ant_state = [1 for _ in range(self._no_ants)]
        return ant_state

    def map_to_food(self): #checked
        #creatint array to track items in board
        #location of items through out the grid, items can not overlap
        mat = np.zeros((self._rows,self._columns))
        empty_cells = np.arange(0,self._rows*self._columns)
        items_per_colour =  floor((floor(self._items_density)/self._no_items))
        board = self.board()
        for item in self._colours:
            for i in range(1,items_per_colour): #colour starts from 2, 0 will be used to pad them
                loc = random.sample(list(empty_cells),1)[0]
                x, y = np.argwhere(board==loc)[0]
                mat[x][y] = item
                index = np.argwhere(empty_cells==loc)
                empty_cells = np.delete(empty_cells,index)
        for loc in empty_cells:
            x, y = np.argwhere(board==loc)[0]
            mat[x][y] = 1 #'1' signifies empty cell
        return mat


class AntsState():

    def __init__(self):
        """Constructor; should only be called by Game.new_initial_state."""
        game = AntsGame()
        self._rows = game.rows()
        self._columns = game.columns()
        self.no_items = game.no_items()
        self.no_ants = game.no_ants()
        self.board = game.board()
        self.map_to_food = game.map_to_food()
        self.colours = game.colours()
        self.start_locations = game.initial_locations()
        self.no_ants = game.no_ants()
        self.game_length = game.game_length()
        self.ants_states = game.ants_states()
        self.next_player = 0

    def neighbours(self, current_location): #checked
        '''given a current location, returns neighbours'''
        #surrounding 8 cells are considered as neighbour
        temp_loc = current_location
        neighbouring_items = []
        neighbours = []
        board_wise = []
        top = [[temp_loc[0]-1,temp_loc[1]-1], [temp_loc[0]-1,temp_loc[1]], [temp_loc[0]-1,temp_loc[1]+1]]
        middle = [[temp_loc[0],temp_loc[1]-1], [temp_loc[0],temp_loc[1]], [temp_loc[0],temp_loc[1]+1]]
        bottom = [[temp_loc[0]+1,temp_loc[1]-1], [temp_loc[0]+1,temp_loc[1]], [temp_loc[0]+1,temp_loc[1]+1]]
    
        #American donut style, all list will be of length 9
        for i in [top, middle, bottom]:
            temp = []
            temp_loc = []
            for j in i:
                if  0 > j[0] or j[0] >= self._rows:
                    j[0] = abs(j[0]-(self._rows - 2))
                if 0 > j[1] or j[1] >= self._columns:
                    j[1] = abs(j[1]-(self._columns - 2))
                neighbouring_items.append(self.map_to_food[j[0]][j[1]])
                board_wise.append(self.board[j[0]][j[1]])
                neighbours.append([j[0],j[1]])
#         #post padding neighbout_items to be of length 9 with 0
#         N = 9
#         neighbouring_items = (neighbouring_items + N * [0])[:N]
        return neighbouring_items, neighbours

    def _legal_actions(self,current_cell,ant_id): #checked
        '''returns legal action that can be prformed in the particular state, [pickup, drop, pass], 0=not_carrying/enpty
            if current cell is empty and ant carrying an item >> [DROP, PASS], else >> [PASS]
            if current cell is not empty and no carrying an item >> [PICKUP, PASS], else >> [PASS] 
        '''
        # print("Current cell: ",current_cell)
        if current_cell == 1:
            if self.ants_states[ant_id] == 1:
                return [2]
            else:
                return [1,2]
        else:
            if self.ants_states[ant_id] == 1:
                return [0,2]
            else:
                return [2]

    def next_position(self): #checked
        '''randomly move all ants to new position'''
        start_location = self.start_locations
        new_location = []
        for loc in start_location:
            _, neighbours = self.neighbours(loc)
            next_loc = random.sample(neighbours,1)[0]
            new_location.append(next_loc)
        self.start_locations = new_location

    def _apply_action(self, action, ant_id, ant_location, current_state, flag=True): #checked
        '''perfom action chosen by agent'''
        state = current_state.copy()
        x,y = ant_location
        temp = self.map_to_food[x][y].copy()
        item_pick = self.map_to_food[x][y]
        item_drop = self.ants_states[ant_id]
        if action == 0: #pick
            if flag == True:
                self.ants_states[ant_id] = item_pick
                self.map_to_food[x][y] = 1
            state[4] = 1
            return state, self.map_to_food
        elif action ==1: #drop
            if flag == True:
                self.map_to_food[x][y] = item_drop
                self.ants_states[ant_id] = 1
            state[4] = item_drop
            return state, self.map_to_food
        else: #pass
            return state, self.map_to_food

    def _reward(self, current_state, next_state): #checked
        '''ration = total number of item(droped/picked) / Total items in the state(9)
            reward = ration(current_State) - ration(next_state)'''
        #counting total valid cells, without considering '0'
        item = 2 #bcz we are considering only 1 item/color
        count = current_state.count(item) 
        empty = current_state.count(1)
        total = empty + count
        ratio_current_state = count/total
        #counting total valid cells, without considering '0'
        empty = next_state.count(1)
        count = next_state.count(item)
        total = empty + count
        ratio_next_state = count/total
        reward = ratio_next_state - ratio_current_state
        if reward < 0:
            return (-reward)
        else:
            return reward

    def global_ratio(self, current_location, act, ant_id): #checked
        '''(ratio of each cell)/total cells after every iteration'''
        #get global change for both all the available action
        env_ratio = []
        for row in range(len(self.map_to_food)):
            for col in range(len(self.map_to_food[0])):
                temp_loc = [row,col]
                items, _ = self.neighbours(temp_loc)
                if items[4] != 1:
                    if temp_loc == current_location: #append current loction change for both action
                        #call apply_action 
                        items = np.array(items)
                        flag = False
                        temp_items, _ = self._apply_action(act, ant_id, current_location, items, flag)
                        items = items.tolist()
                        ratio = items.count(temp_items[4])/len(temp_items)
                    else:
    #                     print("temp state: ",items)
                        ratio = items.count(items[4])/len(items)
                    env_ratio.append(ratio)

        global_ratio = sum(env_ratio)/len(env_ratio)
        return global_ratio

    def is_terminal(self,iteration):
        """Returns True if the game is over. Reached max iteration"""
        if iteration > self.game_length:
            return True
        else:
            return False

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""