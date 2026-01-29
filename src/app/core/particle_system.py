import random 
import cv2
import numpy as np

class Particle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def create_particle(self, effect_type, start_x=None):
        """ Create a New Weather particle """
        if start_x is None:
            x = random.randint(0, self.width)
        else:
            x = start_x
        
        if effect_type == "RAIN":
            return {
                'x': x,
                'y': 0,
                'speed': random.randint(25, 35),
                'type': 'rain'
            }
        elif effect_type == "SNOW":
            return {
                'x': x,
                'y': 0,
                'speed': random.randint(2, 5),
                'drift': random.randint(-1, 1),
                'type': 'snow'
            }
        elif effect_type == "LIGHTNING":
            segments = [(x, 0)]
            current_y = 0
            current_x = x
            while current_y < self.height:
                new_x = current_x + random.randint(-20, 20)
                current_y += random.randint(20, 50)
                segments.append((new_x, current_y))
                current_x = new_x
            
            return {
                'x': x,
                'y': 0,
                'lifetime': random.randint(5, 10),
                'segments': segments,
                'type': 'lightning',
                'flash_intensity': random.uniform(0.1, 0.3)
            }
    