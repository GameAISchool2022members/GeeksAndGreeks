import os
import pygame as pg
import numpy as np

def load_sprites_old(entities_path):
    images_path = [[]]*len(os.listdir(entities_path))
    for entity in os.listdir(entities_path):
        entity_n = int(entity[0])
        entity_path = os.path.join(entities_path, entity)
        entity_list = []
        for image in os.listdir(entity_path):
            image_path = os.path.join(entity_path, image)
            entity_list.append(image_path)
            images_path[entity_n] = entity_list
    return images_path

images = load_sprites('resources/images/entities')
print(images)


