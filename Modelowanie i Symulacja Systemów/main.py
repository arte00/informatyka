import re
import numpy as np
import matplotlib.pyplot as plt
from pandas import wide_to_long

def calculate_resistance(age: int) -> int:
    if 15 > age >= 70:
        return np.random.randint(0, 4)
    elif 40 <= age < 70:
        return np.random.randint(3, 7)
    elif 15 <= age < 40:
        return np.random.randint(6, 11)  

DIRECTIONS_STR = {"top" : [1, 0], "top-right": [1, 1], "right":[0, 1], "bot-right" :[-1, 1], "bot":[-1, 0], 
                "bot-left": [-1, -1], "left": [0, -1], "top-left": [1, -1]}

DIRECTIONS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]] # 1, 4, 5, 6, 7

COLORS = ['darkred', 'yellow', 'orange', 'forestgreen']

class Entity:

    def __init__(self) -> None:
        self.x = np.random.randint(0, 100)
        self.y = np.random.randint(0, 100)
        self.velocity = np.random.randint(1, 4)
        self.direction = np.random.randint(0, 8)
        self.state = np.random.randint(0, 4)
        self.age = np.random.randint(0, 61)
        self.resistance = calculate_resistance(self.age)

    def move(self, board) -> None:

        # ruch poprawny

        height = board.height
        width = board.width

        # check for collision

        if self.x + DIRECTIONS[self.direction][1] * self.velocity < width  \
        and self.y + DIRECTIONS[self.direction][0] * self.velocity < height \
        and self.x + DIRECTIONS[self.direction][1] * self.velocity > 0 \
        and self.y + DIRECTIONS[self.direction][0] * self.velocity > 0:

            self.x += DIRECTIONS[self.direction][1] * self.velocity
            self.y += DIRECTIONS[self.direction][0] * self.velocity

        else:
            if self.direction == 0:
                self.direction = np.random.choice([2, 3, 4, 5, 6])
            elif self.direction == 1:
                self.direction = np.random.choice([4, 5, 6])
            elif self.direction == 2:
                self.direction = np.random.choice([1, 4, 5, 6, 7])
            elif self.direction == 3:
                self.direction = np.random.choice([1, 6, 7])
            elif self.direction == 4:
                self.direction = np.random.choice([1, 2, 3, 6, 7])
            elif self.direction == 5:
                self.direction = np.random.choice([1, 2, 3])
            elif self.direction == 6:
                self.direction = np.random.choice([1, 2, 3, 4, 5])
            elif self.direction == 7:
                self.direction = np.random.choice([2, 4, 5])

            self.x += DIRECTIONS[self.direction][1] * self.velocity
            self.y += DIRECTIONS[self.direction][0] * self.velocity




def test_entity(e: Entity) -> None:
    print("x: ", e.x, "y: ", e.y)
    print("direction: ", e.direction, ", velocity: ", e.velocity)
    print("age: ", e.age, ", resistance: ", e.resistance)
    print("state: ", e.state)

def test_move():
    e = Entity()
    m = Board(100, 100, 0)
    m.add_entity(e)

    e.x = 99
    e.y = 99
    e.direction = 7
    test_entity(e)
    e.move(m.height, m.width)
    test_entity(e)

def test_move2():
    m = Board(100, 100, 100)
    for _ in range(100):
        for e in m.entities:
            e.move(m.height, m.width)
            if e.x >= m.width or e.y >= m.height or e.x < 0 or e.y < 0:
                print(e.x, e.y, e.direction)

    

class Board:

    def __init__(self, x, y, n) -> None:
        self.height = x
        self.width = y
        self.entities = [Entity() for _ in range(n)]

    def add_entity(self, e: Entity) -> None:
        self.entities.append(e)

    def show(self):
        x = [e.x for e in self.entities]
        y = [e.y for e in self.entities]
        c = [COLORS[e.state] for e in self.entities]
        plt.scatter(x, y, c=c)
        plt.show()

def main():
    
    map = Board(100, 100, 100)
    map.show()


if __name__ == '__main__':
    test_move2()