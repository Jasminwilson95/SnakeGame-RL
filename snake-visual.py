# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 01:02:36 2023

@author: jasmi
"""

import random,sys
import numpy as np
import pygame
import pickle
import time
from pygame.math import Vector2


class SNAKE:
    def __init__(self):
        # whether to show episode number at the top
        self.show_episode = False
        self.episode = None
        
        # scale adjusts size of whole board
        self.cell_size = 40
        self.cell_number = 20
        
        # starting location for the snake
        self.body = [Vector2(5,10),Vector2(4,10)]
        self.direction = Vector2(0,0)
        self.new_block = False
        
        self.game_close = False
        self.survived = 0
        self.snake_speed = 40

        
        pygame.init()
        
        self.screen = pygame.display.set_mode((self.cell_number * self.cell_size,self.cell_number * self.cell_size))
        self.clock = pygame.time.Clock()
        
        self.food_x, self.food_y= self.randomize()
        self.game_font = pygame.font.SysFont("Arial", 25)
        
        self.game_loop()
        
    
    def draw_score(self,score):
        # score_text = str(len(self.snake.body) - 2)
        value = self.game_font.render(f"Score: {score}", True, (56,74,12))
        self.screen.blit(value, [10 * self.cell_number, 10])
        
    def print_episode(self):
        if self.show_episode:
            value = self.game_font.render(f"Episode: {self.episode}", True, (56,74,12))
            self.screen.blit(value, [10, 10])
    
    def draw_snake(self):

        for block in self.body:
            x_pos = int(block.x * self.cell_size)
            y_pos = int(block.y * self.cell_size)
            block_rect = pygame.Rect(x_pos,y_pos,self.cell_size,self.cell_size)
            pygame.draw.rect(self.screen, (183,111,122), block_rect)

        
    def is_unsafe(self, x, y):
            if not 0 <= x < self.cell_number or not 0 <= y < self.cell_number:
                for block in self.body[1:]:
                    if block == (x,y):
                        return 1
                return 1
            return 0
         
    def randomize(self):
        self.x = random.randint(0,self.cell_number - 1)
        self.y = random.randint(0,self.cell_number - 1)
        self.pos = Vector2(self.x,self.y)
        
        return self.pos
    
    def get_state(self):
        head_x, head_y = self.body[0]
        state = []
        state.append(int(self.direction == Vector2(-1,0)))
        state.append(int(self.direction == Vector2(1,0)))
        state.append(int(self.direction == Vector2(0,-1)))
        state.append(int(self.direction == Vector2(0,1)))
        state.append(int(self.food_x < head_x))
        state.append(int(self.food_x > head_x))
        state.append(int(self.food_y < head_y))
        state.append(int(self.food_y > head_y))
        state.append(self.is_unsafe(head_x + 1, head_y))
        state.append(self.is_unsafe(head_x - 1, head_y))
        state.append(self.is_unsafe(head_x, head_y + 1))
        state.append(self.is_unsafe(head_x, head_y - 1))
        return tuple(state)
    
    def game_over(self):
        return self.game_close
    
    
    def check_fail(self):
        if not 0 <= self.body[0].x < self.cell_number or not 0 <= self.body[0].y < self.cell_number:
            # global reward
            # reward =-100
            self.game_close = True

        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                # reward =-100
                self.game_close = True
                
    def update_ui(self):
         self.screen.fill((175,215,70))

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x * self.cell_size),int(self.pos.y * self.cell_size),self.cell_size,self.cell_size)
        pygame.draw.rect(self.screen,(126,166,114),fruit_rect)
    
    def add_block(self):
        self.new_block = True
    
    def move_snake(self):
        if self.new_block == True:
            body_copy = self.body[:]
            body_copy.insert(0,body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0,body_copy[0] + self.direction)
            self.body = body_copy[:]
        
                
    
    def game_loop(self, action="None"):
        if action == "None":
            action = random.choice(["left", "right", "up", "down"])
        else:
            action = ["left", "right", "up", "down"][action]
        reward = 0    
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        if action == 'up':
            if self.direction.y != 1:
                self.direction = Vector2(0,-1)
        if action == 'right':
            if self.direction.x != -1:
                self.direction = Vector2(1,0)
        if action == 'down':
            if self.direction.y != -1:
                self.direction = Vector2(0,1)
        if action == 'left':
            if self.direction.x != 1:
                self.direction = Vector2(-1,0)
        
                
        if not 0 <= self.body[0].x < self.cell_number or not 0 <= self.body[0].y < self.cell_number:
            self.game_close = True
            
        self.move_snake()
        self.update_ui()
        
        self.draw_fruit()
        
        for block in self.body[1:]:
            if block == self.body[0]:
                self.game_close = True
        
        self.draw_snake()
        self.draw_score(len(self.body) - 2)
        self.print_episode()
        pygame.display.update()        
                
         #ate food       
        if self.pos == self.body[0]:
            self.food_x, self.food_y= self.randomize()
            self.add_block()
            
        for block in self.body[1:]:
            if block == self.pos:
                self.food_x, self.food_y= self.randomize()
        
        self.survived += 1
        
        
    def run_game(self,episode):
        self.show_episode = True
        self.episode = episode
        self.print_episode()
        pygame.display.update()
        
        filename = f"pickle/{episode}.pickle"
        with open(filename, 'rb') as file:
            table = pickle.load(file)
        time.sleep(5)
        
        current_length = 2
        steps_unchanged = 0
        while not self.game_over():
            if len(self.body) != current_length:
                steps_unchanged = 0
                current_length = len(self.body)
            else:
                steps_unchanged += 1        
            
            state = self.get_state()
            action = np.argmax(table[state])
            if steps_unchanged == 1000:
                # stop if snake hasn't eaten anything in 1000 episodes (stuck in a loop)
                break
            
            self.game_loop(action)
            self.clock.tick(self.snake_speed)
        if self.game_over() == True:
            self.update_ui()
            self.print_episode()
            self.draw_score((len(self.body)) - 2)
            pygame.display.update()
            time.sleep(5)
            pygame.quit()
        return len(self.body)
        

def main():
    v= SNAKE() 
    v.run_game(5000)


if __name__ == "__main__":
    main()                    
           

        
        
        
        
        

