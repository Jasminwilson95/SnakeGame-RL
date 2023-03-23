# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 10:21:10 2023

@author: jasmi
"""

# This will run the snake and record the scores simualteouly
# This uses snake_visual_test.py to render

#This takes a while to run and observe hence used Agent.py for testing
# It took 3 hours to finish only 500 episodes

# How to run -
# Select which learning you want to see the snake move and Run Agent_test.py and at the bottom 

import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from snake_visual_test import SNAKE


class Agent:
    def __init__(self):
        # define initial parameters
        self.discount_rate = 0.95
        self.learning_rate = 0.01
        self.eps = 0.5
        self.eps_discount = 0.9992
        self.min_eps = 0.001
        self.num_episodes = 10000
        self.lamda = 0.4
        self.Qtable = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        self.eligibility_trace = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        self.env = SNAKE()
        self.score = []
        self.survived = []
        self.plot_x =[]
        self.plot_y =[]
        self.plot_z =[]
        self.learnreward =[]


        
     # epsilon-greedy action choice
    def get_action(self, state):
         # select random action (exploration)
         if random.random() < self.eps:
             return random.choice([0, 1, 2, 3])
         
         # select best action (exploitation)
         return np.argmax(self.Qtable[state])
     
    def trainSarsa(self):
        for i in range(1, self.num_episodes + 1):
            if(i<=100):
                self.eps =0.01
                
            
            self.env  = SNAKE()
            steps_without_food = 0
            length = len(self.env.body)
            self.env.show_episode = True
            self.env.episode = i
            self.env.print_episode()
            
            
            if i % 25 == 0:
                print(f"Episodes: {i}, score: {np.mean(self.score)}, survived: {np.mean(self.survived)}, eps: {self.eps}, lr: {self.learning_rate}")
                
                self.plot_x.append(i)
                self.plot_z.append(np.mean(self.score))
                self.plot_y.append(np.mean(self.learnreward))

                # self.plot_e.append(self.eps)
                self.score = []
                self.survived = []
 
    
            current_state = self.env.get_state()
            self.eps = max(self.eps * self.eps_discount, self.min_eps)
            done = False
            # choose action and take it
            action = self.get_action(current_state)
            while not done:

                new_state, reward, done = self.env.game_loop(action)
                new_action = self.get_action(new_state)
                
                # Bellman Equation Update
                self.Qtable[current_state][action] = self.Qtable[current_state][action] + self.learning_rate\
                    * (reward + self.discount_rate * self.Qtable[new_state][new_action] - self.Qtable[current_state][action]) 
                current_state = new_state
                action = new_action
                
                steps_without_food += 1
                if length != len(self.env.body):
                    length = len(self.env.body)
                    steps_without_food = 0
                if steps_without_food == 1000:
                    # break out of loops
                    break
                
            self.score.append(len(self.env.body) - 2)
            self.learnreward.append(reward)
            self.survived.append(self.env.survived)
            
        # save it to excel for plotting later
        df = pd.DataFrame({
            'Episode':self.plot_x,
            'Reward':self.plot_y,
            'Score':self.plot_z,

            })
        df.to_excel("sarsa.xlsx")
     
        
     
           
    def trainQL(self):
        for i in range(1, self.num_episodes + 1):
            if(i<=100):
                self.eps =0.01
                
            
            self.env  = SNAKE()
            steps_without_food = 0
            length = len(self.env.body)
            self.env.show_episode = True
            self.env.episode = i
            self.env.print_episode()
            
            if i % 25 == 0:
                print(f"Episodes: {i}, score: {np.mean(self.score)}, survived: {np.mean(self.survived)}, eps: {self.eps}, lr: {self.learning_rate}")
                
                self.score = []
                self.survived = []

            current_state = self.env.get_state()
            self.eps = max(self.eps * self.eps_discount, self.min_eps)
            done = False
            while not done:
                # choose action and take it
                action = self.get_action(current_state)
                new_state, reward, done = self.env.game_loop(action)
                
                # Bellman Equation Update
                self.Qtable[current_state][action] = (1 - self.learning_rate)\
                    * self.Qtable[current_state][action] + self.learning_rate\
                    * (reward + self.discount_rate * max(self.Qtable[new_state])) 
                current_state = new_state
                
                steps_without_food += 1
                if length != len(self.env.body):
                    length = len(self.env.body)
                    steps_without_food = 0
                if steps_without_food == 1000:
                    # break out of loops
                    break
                
            self.score.append(len(self.env.body) - 2)
            self.survived.append(self.env.survived)
                        
        # save it to excel for plotting later
        df = pd.DataFrame({
            'Episode':self.plot_x,
            'Reward':self.plot_y,
            'Score':self.plot_z,

            })
        df.to_excel("sarsa.xlsx")
               
    def trainSarsaLamda(self):
        for i in range(1, self.num_episodes + 1):
            if(i<=100):
                self.eps =0.01
                
            
            self.env  = SNAKE()
            steps_without_food = 0
            length = len(self.env.body)
            self.env.show_episode = True
            self.env.episode = i
            self.env.print_episode()
            #reinitalize to 0
            self.eligibility_trace = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
            
            if i % 25 == 0:
                print(f"Episodes: {i}, score: {np.mean(self.score)}, survived: {np.mean(self.survived)}, eps: {self.eps}, lr: {self.learning_rate}")
                
                self.plot_x.append(i)
                self.plot_z.append(np.mean(self.score))
                self.plot_y.append(np.mean(self.learnreward))
                self.score = []
                self.survived = []
 
            
            current_state = self.env.get_state()
            # choose action and take it
            action = self.get_action(current_state)
            self.eps = max(self.eps * self.eps_discount, self.min_eps)
            done = False
            while not done:
 
                new_state, reward, done = self.env.game_loop(action)
                new_action = self.get_action(new_state)
                if steps_without_food == 1000:
                    # break out of loops
                    reward -=10
                delta = reward + self.learning_rate * self.Qtable[new_state][new_action] - self.Qtable[current_state][action]
                
                self.eligibility_trace[current_state][action] += 1
                #Update Qvalue
                self.Qtable += self.learning_rate * delta * self.eligibility_trace
                self.eligibility_trace *= self.discount_rate * self.lamda 
                
                
                current_state = new_state
                action = new_action
                
                steps_without_food += 1
                if length != len(self.env.body):
                    length = len(self.env.body)
                    steps_without_food = 0
                if steps_without_food == 1020:
                    # break out of loops
                    break
                
            self.score.append(len(self.env.body) - 2)
            self.learnreward.append(reward)
            self.survived.append(self.env.survived)
        # save it to excel for plotting later
        df = pd.DataFrame({
                'Episode':self.plot_x,
                'Reward':self.plot_y,
                 'Score':self.plot_z,
 
                })
        df.to_excel("sarsaLamda.xlsx")


def main():
    v= Agent() 
    v.trainSarsaLamda()
    # v.trainSarsa()
    # v.trainQL()


if __name__ == "__main__":
    main()                    
            