# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 21:08:30 2023

@author: jasmi
"""

# How to run -

# Run Agent.py for plot and score results for all models
# This saves the Q table in a pickle file that will be used later to display the UI of snake movement
# for UI run  snake-visual.py


import pygame,sys,random
from pygame.math import Vector2
from snake_novisual import LearnSNAKE
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

d = os.path.dirname(__file__) # directory of script
p = r'{}/pickle-sarsa'.format(d) # path to be created
q = r'{}/pickle-QL'.format(d) # path to be created
r = r'{}/pickle-sarsalamda'.format(d) # path to be created
if not os.path.exists(p):
    os.mkdir(p)
if not os.path.exists(q):
    os.mkdir(q)
if not os.path.exists(r):
    os.mkdir(r)

class Agent:
    def __init__(self):
        # define initial parameters
        self.discount_rate = 0.95
        self.learning_rate = 0.01
        self.eps = 0.4
        self.eps_discount = 0.9992
        self.min_eps = 0.001
        self.num_episodes = 5000
        self.lamda = 0.4
        self.Qtable = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        self.eligibility_trace = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        self.env = LearnSNAKE()
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
                
            
            self.env  = LearnSNAKE()
            steps_without_food = 0
            length = len(self.env.body)
            
            if i % 25 == 0:
                print(f"Episodes: {i}, score: {np.mean(self.score)}, survived: {np.mean(self.survived)}, eps: {self.eps}, lr: {self.learning_rate}")
                
                self.plot_x.append(i)
                self.plot_z.append(np.mean(self.score))
                self.plot_y.append(np.mean(self.learnreward))

                # self.plot_e.append(self.eps)
                self.score = []
                self.survived = []
 
                
            # #occasionally save latest model
            if (i < 500 and i % 10 == 0) or (i >= 500 and i < 1000 and i % 200 == 0) or (i >= 1000 and i % 500 == 0):
                with open(f'pickle-sarsa/{i}.pickle', 'wb') as file:
                    pickle.dump(self.Qtable, file)
            
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
                
            
            self.env  = LearnSNAKE()
            steps_without_food = 0
            length = len(self.env.body)
            
            if i % 25 == 0:
                print(f"Episodes: {i}, score: {np.mean(self.score)}, survived: {np.mean(self.survived)}, eps: {self.eps}, lr: {self.learning_rate}")
                
                self.plot_x.append(i)
                self.plot_z.append(np.mean(self.score))
                self.plot_y.append(np.mean(self.learnreward))
                # self.plot_e.append(self.eps)
                self.score = []
                self.survived = []
 
                
            # #occasionally save latest model
            if (i < 500 and i % 10 == 0) or (i >= 500 and i < 1000 and i % 200 == 0) or (i >= 1000 and i % 500 == 0):
                with open(f'pickle-QL/{i}.pickle', 'wb') as file:
                    pickle.dump(self.Qtable, file)
            
            current_state = self.env.get_state()
            self.eps = max(self.eps * self.eps_discount, self.min_eps) # step decay

            done = False
            while not done:
                # choose action and take it
                action = self.get_action(current_state)
                new_state, reward, done = self.env.game_loop(action)
                
                # Update Qvalue
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
            self.learnreward.append(reward)

        # save it to excel for plotting later
        df = pd.DataFrame({
                'Episode':self.plot_x,
                'Reward':self.plot_y,
                'Score':self.plot_z,

                })
        #to plot and average
        df.to_excel("q_learning.xlsx")  


    def trainSarsaLamda(self):
        for i in range(1, self.num_episodes + 1):
            if(i<=100):
                self.eps =0.01
                
            
            self.env  = LearnSNAKE()
            steps_without_food = 0
            length = len(self.env.body)
            #reinitalize to 0
            self.eligibility_trace = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
            
            if i % 25 == 0:
                print(f"Episodes: {i}, score: {np.mean(self.score)}, survived: {np.mean(self.survived)}, eps: {self.eps}, lr: {self.learning_rate}")
                
                self.plot_x.append(i)
                self.plot_z.append(np.mean(self.score))
                self.plot_y.append(np.mean(self.learnreward))
                self.score = []
                self.survived = []
 
                
            # #occasionally save latest model
            if (i < 500 and i % 10 == 0) or (i >= 500 and i < 1000 and i % 200 == 0) or (i >= 1000 and i % 500 == 0):
                with open(f'pickle-sarsalamda/{i}.pickle', 'wb') as file:
                    pickle.dump(self.Qtable, file)
            
            current_state = self.env.get_state()
            # choose action and take it
            action = self.get_action(current_state)
            done = False
            while not done:

                # new_state, reward, done = self.env.game_loop(steps_without_food,action) # n step penalty
                new_state, reward, done = self.env.game_loop(action)
                new_action = self.get_action(new_state)
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
        df.to_excel("sarsaLamda.xlsx")
        
    
            
def plotEpsilon():
    q = pd.read_excel("q-learning-static.xlsx")
    sar = pd.read_excel("q-learning-step.xlsx")
    sarlamda = pd.read_excel("q_learning.xlsx")
    qx=list(q['Episode'])
    qy=list(q['Score'])
    
    sarx=list(sar['Episode'])
    sary=list(sar['Score'])
    
    sarLamx=list(sarlamda['Episode'])
    sarLamy=list(sarlamda['Score'])

    plt.plot(qx,qy,linestyle='solid',label = "Static Epsilon")
    plt.plot(sarx,sary,linestyle='solid',label = "Step Epsilon")
    plt.plot(sarLamx,sarLamy,linestyle='solid',label = "Step + Static")

    plt.title("Training...")
    plt.xlabel("Episodes")
    plt.ylabel("Average Score")
    plt.legend()
    
def plotResults():
    q = pd.read_excel("q_learning.xlsx")
    sar = pd.read_excel("sarsa.xlsx")
    sarlamda = pd.read_excel("sarsaLamda.xlsx")
    
    qx=list(q['Episode'])
    qy=list(q['Reward'])
    qz=list(q['Score'])

    
    sarx=list(sar['Episode'])
    sary=list(sar['Reward'])
    sarz=list(sar['Score'])
    
    sarLamx=list(sarlamda['Episode'])
    sarLamy=list(sarlamda['Reward'])
    sarLamz=list(sarlamda['Score'])
    
    f1 = plt.figure(1)
    plt.plot(qx,qy,linestyle='solid',label = "Q-learning")
    plt.plot(sarx,sary,linestyle='solid',label = "Sarsa learning")
    plt.plot(sarLamx,sarLamy,linestyle='solid',label = "Sarsa λ Learning")
    plt.title("Training...")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    
    f1.show()
    
    f2= plt.figure(2)
    plt.figure(num=0,dpi=120)
    plt.plot(qx,qz,linestyle='solid',label = "Q-learning")
    plt.plot(sarx,sarz,linestyle='solid',label = "Sarsa learning")
    plt.plot(sarLamx,sarLamz,linestyle='solid',label = "Sarsa λ Learning")
    
    plt.title("Training...")
    plt.xlabel("Episodes")
    plt.ylabel("Average Scores")

    plt.legend()
    f2.show()
        



def main():
    v= Agent() 
    v1= Agent()
    v2 = Agent()
    v.trainSarsa()
    v1.trainQL()
    v2.trainSarsaLamda()

    plotResults()
    plotEpsilon() 

if __name__ == "__main__":
    main()                    
            