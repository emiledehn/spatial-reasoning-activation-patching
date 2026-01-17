import itertools
import numpy as np

def generate_positions():
    coord_values = [-1, 0, 1]
    t = list(itertools.product(coord_values, repeat=3))
    t = [np.array(coord) for coord in t]
    transformations = []
    indices = list(itertools.product(range(0, len(t)), repeat=2))
    for i, j in indices:
        transformations.append((t[i], t[j], t[i] + t[j]))
    return transformations

def vector_to_direction(vector):
    dist = lambda d: f"{abs(d)} units" if abs(d) != 1 else f"{abs(d)} unit"
    direction = ""
    if vector[2] < 0:
        direction += f"{dist(vector[2])} to the bottom"
    elif vector[2] > 0:
        direction += f"{dist(vector[2])} to the top"
    if vector[1] < 0:
        if vector[2] != 0:
            direction += ", "
        direction += f"{dist(vector[1])} to the back"
    elif vector[1] > 0:
        if vector[2] != 0:
            direction += ", "
        direction += f"{dist(vector[1])} to the front"
    if vector[0] < 0:
        if vector[1] != 0 or vector[2] != 0:
            direction += ", "
        direction += f"{dist(vector[0])} to the left"
    elif vector[0] > 0:
        if vector[1] != 0 or vector[2] != 0:
            direction += ", "
        direction += f"{dist(vector[0])} to the right"
    if direction == "":
        direction = "at the position"
    return direction
