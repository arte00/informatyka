import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class World():

    def __init__(self, width, height) -> None:
        self.particles = []
        self.width = width
        self.height = height

    def add_particle(self, particle):
        self.particles.append(particle)

    def remove_particle(self, particle):
        self.particles.remove(particle)

    def nano(self):
        for particle in self.particles:
            particle.move(self.width, self.height)

    def show(self):
        for particle in self.particles:
            plt.scatter(particle.x, particle.y, c='r')

class Particle:

    def __init__(self, world: World, x, y, direction, velocity) -> None:
        self.x = x
        self.y = y
        self.direction = direction
        self.velocity = velocity
        self.world = world

    def move(self, x, y):

        if self.x > 95 or self.y > 90:
            # self.world.remove_particle(self)
            self.direction = np.random.choice([3, 4, 5])
        if self.direction == 0:
            self.y += self.velocity
        elif self.direction == 1:
            self.x += self.velocity * 0.5
            self.y += self.velocity * 0.5
        elif self.direction == 2:
            self.y += self.velocity
        elif self.direction == 3:
            self.x += self.velocity * 0.5
            self.y -= self.velocity * 0.5
        elif self.direction == 4:
            self.y -= self.velocity
        elif self.direction == 5:
            self.x -= self.velocity * 0.5
            self.y -= self.velocity * 0.5
        elif self.direction == 6:
            self.x -= self.velocity
        elif self.direction == 7:
            self.x -= self.velocity * 0.5
            self.y += self.velocity * 0.5

class Cannon():

    def __init__(self, world: World, x, y) -> None:
        self.x = x
        self.y = y
        self.world = world

    def pass_frame(self, frame):
        if frame % 5 == 0 and frame != 0:
            particle = Particle(self.world, self.x, self.y, 0, 3)
            self.world.add_particle(particle)

def visualisation():

    fig, ax = plt.subplots(constrained_layout=True)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    # ax.set_xticks(np.arange(-10, 10, 2))
    # ax.set_yticks(np.arange(0, 100, 10))
    world = World(100, 100)
    cannon = Cannon(world, 0, 5)
    

    for i in range(100):
        
        ax.cla()
        ax.set_xlim([-10, 10])
        ax.set_ylim([0, 100])
        plt.scatter(cannon.x, cannon.y, c='b', marker='s')
        world.show()
        ax.set_title("frame {}".format(i))
        cannon.pass_frame(i)
        world.nano()
        # Note that using time.sleep does *not* work here!
        plt.pause(0.1)

if __name__ == "__main__":

    visualisation()