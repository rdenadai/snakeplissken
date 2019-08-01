import math
from configs import SNAKE_SIZE, SNAKE_SEPARATION, FPS, KEY, APPLE_SIZE, WALL_SIZE


class Segment:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.size = (SNAKE_SIZE, SNAKE_SIZE)
        self.direction = KEY["RIGHT"]


class Apple:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.position = (x, y, APPLE_SIZE, APPLE_SIZE)


class Wall:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.position = (x, y, WALL_SIZE, WALL_SIZE)


class Snake:
    """
    Class that represents the Snake...
    part of this code was taken from: https://gist.github.com/someoneigna/5022021
    """

    def __init__(self, x, y, head_color, body_color):
        seg = Segment(x + SNAKE_SEPARATION, y, head_color)
        self.stack = [seg]
        next_x = x
        for i in range(1, 3):
            next_x -= 10
            seg = Segment(next_x + SNAKE_SEPARATION, y, body_color)
            self.stack.append(seg)
        self.size_x = len(self.stack) * SNAKE_SIZE
        self.size_y = SNAKE_SIZE
        # Movement speed
        self.movement = SNAKE_SIZE + SNAKE_SEPARATION

    def head(self):
        return self.stack[0]

    def move(self):
        last_element = len(self.stack) - 1
        while last_element != 0:
            self.stack[last_element].direction = self.stack[last_element - 1].direction
            self.stack[last_element].x = self.stack[last_element - 1].x
            self.stack[last_element].y = self.stack[last_element - 1].y
            last_element -= 1
        if len(self.stack) < 2:
            last_segment = self
        else:
            last_segment = self.stack.pop(last_element)
        last_segment.direction = self.stack[0].direction
        if self.stack[0].direction == KEY["UP"]:
            last_segment.y = self.stack[0].y - self.movement
        elif self.stack[0].direction == KEY["DOWN"]:
            last_segment.y = self.stack[0].y + self.movement
        elif self.stack[0].direction == KEY["LEFT"]:
            last_segment.x = self.stack[0].x - self.movement
        elif self.stack[0].direction == KEY["RIGHT"]:
            last_segment.x = self.stack[0].x + self.movement
        self.stack.insert(0, last_segment)

    def grow(self):
        last_element = len(self.stack) - 1
        x = self.stack[last_element].x
        y = self.stack[last_element].y
        color = self.stack[last_element].color
        seg = Segment(x + SNAKE_SEPARATION, y, color)
        self.stack.append(seg)
