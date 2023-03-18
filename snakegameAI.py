# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 10:16:09 2023

@author: jasmi
"""

import pygame,sys,random
from pygame.math import Vector2
import numpy as np

pygame.init()
cell_size = 40
cell_number = 20
game_font = pygame.font.SysFont("Arial", 25)
class SNAKE:
    def __init__(self):
        self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
        self.direction = Vector2(0,0)
        self.new_block = False
        

    def draw_snake(self,screen):

        for block in self.body:
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect = pygame.Rect(x_pos,y_pos,cell_size,cell_size)
            pygame.draw.rect(screen, (183,111,122), block_rect)

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

    def add_block(self):
        self.new_block = True

    def reset(self):
        self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
        self.direction = Vector2(0,0)


class FRUIT:
    def __init__(self):
        self.randomize()

    def draw_fruit(self,screen):
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size),int(self.pos.y * cell_size),cell_size,cell_size)
        pygame.draw.rect(screen,(126,166,114),fruit_rect)

    def randomize(self):
        self.x = random.randint(0,cell_number - 1)
        self.y = random.randint(0,cell_number - 1)
        self.pos = Vector2(self.x,self.y)

class MAIN:
    def __init__(self):
                  
        pygame.mixer.pre_init(44100,-16,2,512)
        self.snake = SNAKE()
        self.fruit = FRUIT()

        self.screen = pygame.display.set_mode((cell_number * cell_size,cell_number * cell_size))
        self.clock = pygame.time.Clock()
        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE,150)
        

    def update(self):
        self.snake.move_snake()
        self.check_collision()
        self.check_fail()

    def draw_elements(self,screen):
        self.fruit.draw_fruit(screen)
        self.snake.draw_snake(screen)
        self.draw_score(screen)

    def check_collision(self):
        if self.fruit.pos == self.snake.body[0]:
            self.fruit.randomize()
            self.snake.add_block()

        for block in self.snake.body[1:]:
            if block == self.fruit.pos:
                self.fruit.randomize()

    def check_fail(self):
        if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number:
            self.game_over()

        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                self.game_over()
        
    def game_over(self):
        self.snake.reset()
    
    def update_ui(self):
        self.screen.fill((175,215,70))
        self.draw_elements(self.screen)
        pygame.display.update()
        
    
    def game_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == self.SCREEN_UPDATE:
                    self.update()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        if self.snake.direction.y != 1:
                            self.snake.direction = Vector2(0,-1)
                    if event.key == pygame.K_RIGHT:
                        if self.snake.direction.x != -1:
                            self.snake.direction = Vector2(1,0)
                    if event.key == pygame.K_DOWN:
                        if self.snake.direction.y != -1:
                            self.snake.direction = Vector2(0,1)
                    if event.key == pygame.K_LEFT:
                        if self.snake.direction.x != 1:
                            self.snake.direction = Vector2(-1,0)
            self.update_ui()
            
    def draw_score(self,screen):
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render("Your Score: " + score_text,True,(56,74,12))
        score_x = int(cell_size * cell_number - 80)
        score_y = int(cell_size * cell_number - 40)
        score_rect = score_surface.get_rect(center = (score_x,score_y))
        

        #pygame.draw.rect(screen,(167,209,61),bg_rect)
        screen.blit(score_surface,score_rect)

                    
class QL_Agent:
    def __init__(self):
        # define initial parameters
        self.discount_rate = 0.95
        self.learning_rate = 0.01
        self.eps = 0.5
        self.eps_discount = 0.9992
        self.min_eps = 0.001
        self.num_episodes = 5000
        self.Qtable = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        self.env = MAIN()
        self.score = []
        self.survived = []
        
     # epsilon-greedy action choice
    def get_action(self, state):
         # select random action (exploration)
         if random.random() < self.eps:
             return random.choice([0, 1, 2, 3])
         
         # select best action (exploitation)
         return np.argmax(self.table[state])
     
    def get_state(self):
        head_r, head_c = self.snake.body[0]
        state = []
        state.append(int(self.dir == "left"))
        state.append(int(self.dir == "right"))
        state.append(int(self.dir == "up"))
        state.append(int(self.dir == "down"))
        state.append(int(self.food_r < head_r))
        state.append(int(self.food_r > head_r))
        state.append(int(self.food_c < head_c))
        state.append(int(self.food_c > head_c))
        state.append(self.is_unsafe(head_r + 1, head_c))
        state.append(self.is_unsafe(head_r - 1, head_c))
        state.append(self.is_unsafe(head_r, head_c + 1))
        state.append(self.is_unsafe(head_r, head_c - 1))
        return tuple(state)
        
        def is_unsafe(self, r, c):
            if self.valid_index(r, c):
              if self.board[r][c] == 1:
                  return 1
              return 0
            else:
              return 1
          
    def train(self):
        
        for i in range(1, self.num_episodes + 1):
            
            if(i<=100):
                self.eps =0.01

            self.env  = MAIN()
            steps_without_food = 0
            length = self.env.snake_length
            
            # print updates
            if i % 25 == 0:
                print(f"Episodes: {i}, score: {np.mean(self.score)}, survived: {np.mean(self.survived)}, eps: {self.eps}, lr: {self.learning_rate}")
                # plot_x.append(i)
                # plot_y.append(np.mean(self.score))
                # plot_e.append(self.eps)

                self.score = []
                self.survived = []
               
            # occasionally save latest model
            # if (i < 500 and i % 10 == 0) or (i >= 500 and i < 1000 and i % 200 == 0) or (i >= 1000 and i % 500 == 0):
            #     with open(f'pickle/{i}.pickle', 'wb') as file:
            #         pickle.dump(self.table, file)
                
            current_state = self.env.get_state()
            self.eps = max(self.eps * self.eps_discount, self.min_eps)
            done = False
            while not done:
                # choose action and take it
                action = self.get_action(current_state)
                new_state, reward, done = self.env.step(action)
                
                # Bellman Equation Update
                self.table[current_state][action] = (1 - self.learning_rate)\
                    * self.table[current_state][action] + self.learning_rate\
                    * (reward + self.discount_rate * max(self.table[new_state])) 
                current_state = new_state

                
                steps_without_food += 1
                if length != self.env.snake_length:
                    length = self.env.snake_length
                    steps_without_food = 0
                if steps_without_food == 1000:
                    # break out of loops
                    break
                
            # keep track of important metrics
            self.score.append(self.env.snake_length - 1)
            self.survived.append(self.env.survived)
            #print(plot_x,plot_y)
            #plot(plot_x,plot_y)
            # plot(plot_x,plot_e)
            
            # plot_scores.append(record)
            # total_score += record
            # mean_score = total_score / self.num_episodes
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
        return self.env.snake_length



def main():
    pygame.init()
    s = MAIN()
    s.game_loop()

if __name__ == "__main__":
    main() 
