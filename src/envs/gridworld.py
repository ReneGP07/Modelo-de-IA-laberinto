class GridWorld:
    def __init__(self, width=5, height=5, start=(0,0), goal=(4,4), obstacles=None):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles or set()
        self.pos = start

    def actions(self):
        return [(-1,0),(1,0),(0,-1),(0,1)]

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, a):
        x, y = self.pos
        dx, dy = a
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height and (nx,ny) not in self.obstacles:
            self.pos = (nx, ny)
        r = 1.0 if self.pos == self.goal else -0.01
        terminado = self.pos == self.goal
        return self.pos, r, terminado
