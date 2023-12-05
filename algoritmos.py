import numpy as np

class Hillclimb:
    def __init__(self, f, minimize, x_range, y_range, size) -> None:
        self.f = f
        self.x_range = x_range
        self.y_range = y_range
        self.minimize = minimize
        self.max_it = 1000
        self.max_viz = 100
        self.e = size/self.max_viz 

    def perturb(self, x):
        return np.random.uniform(low=x-self.e, high=x+self.e)
        
    def run(self):
        x_opt = np.array([
            [np.random.uniform(low=self.x_range[0],high=self.x_range[1])],
            [np.random.uniform(low=self.y_range[0],high=self.y_range[1])]
        ])
        f_opt = self.f(x_opt[0,0], x_opt[1,0])

        i = 0
        melhoria = True
        while i < self.max_it and melhoria:
            melhoria = False
            i += 1
            for _ in range(self.max_viz):
                x_vizinho = self.perturb(x=x_opt)
                f_vizinho = self.f(x_vizinho[0,0], x_vizinho[1,0])
                if (self.minimize and f_vizinho < f_opt) or (not self.minimize and f_vizinho > f_opt):
                    x_opt = x_vizinho
                    f_opt = f_vizinho
                    melhoria = True
                    break
        print(x_opt)

class LRS:
    def __init__(self, f, minimize, x_range, y_range) -> None:
        self.f = f
        self.x_range = x_range
        self.y_range = y_range
        self.minimize = minimize
        self.max_it = 1000
        self.sigma = 0.2

    def run(self):
        x_opt = np.array([
            [np.random.uniform(low=self.x_range[0],high=self.x_range[1])],
            [np.random.uniform(low=self.y_range[0],high=self.y_range[1])]
        ])
        f_opt = self.f(x_opt[0,0], x_opt[1,0])

        for _ in range(self.max_it):
            x_candidato = np.array([
                [x_opt[0,0] + np.random.normal(0, self.sigma)],
                [x_opt[1,0] + np.random.normal(0, self.sigma)]
            ])

            if x_candidato[0,0] < self.x_range[0]:
                x_candidato[0,0] = self.x_range[0]
            if x_candidato[0,0] > self.x_range[1]:
                x_candidato[0,0] = self.x_range[1]
            if x_candidato[1,0] < self.y_range[0]:
                x_candidato[1,0] = self.y_range[0]
            if x_candidato[1,0] > self.y_range[1]:
                x_candidato[1,0] = self.y_range[1]

            f_candidato = self.f(x_candidato[0,0], x_candidato[1,0])
            if (self.minimize and f_candidato < f_opt) or (not self.minimize and f_candidato > f_opt):
                x_opt = x_candidato
                f_opt = f_candidato
        
        print(x_opt)

class GRS:
    def __init__(self, f, minimize, x_range, y_range) -> None:
        self.f = f
        self.x_range = x_range
        self.y_range = y_range
        self.minimize = minimize
        self.max_it = 1000

    def run(self):
        x_opt = np.array([
            [np.random.uniform(low=self.x_range[0],high=self.x_range[1])],
            [np.random.uniform(low=self.y_range[0],high=self.y_range[1])]
        ])
        f_opt = self.f(x_opt[0,0], x_opt[1,0])

        for _ in range(self.max_it):
            x_candidato = np.array([
                [np.random.uniform(low=self.x_range[0],high=self.x_range[1])],
                [np.random.uniform(low=self.y_range[0],high=self.y_range[1])]
            ])
            f_candidato = self.f(x_candidato[0,0], x_candidato[1,0])
            if (self.minimize and f_candidato < f_opt) or (not self.minimize and f_candidato > f_opt):
                x_opt = x_candidato
                f_opt = f_candidato
        
        print(x_opt)

class SA:
    def __init__(self, f, minimize, x_range, y_range) -> None:
        self.f = f
        self.x_range = x_range
        self.y_range = y_range
        self.minimize = minimize
        self.max_it = 1000
        self.sigma = 0.2
        self.T = 1000
    
    def run(self):
        x_opt = np.array([
            [np.random.uniform(low=self.x_range[0],high=self.x_range[1])],
            [np.random.uniform(low=self.y_range[0],high=self.y_range[1])]
        ])
        f_opt = self.f(x_opt[0,0], x_opt[1,0])
        
        for _ in range(self.max_it):
            x_candidato = np.array([
                [x_opt[0,0] + np.random.normal(0, self.sigma)],
                [x_opt[1,0] + np.random.normal(0, self.sigma)]
            ])

            if x_candidato[0,0] < self.x_range[0]:
                x_candidato[0,0] = self.x_range[0]
            if x_candidato[0,0] > self.x_range[1]:
                x_candidato[0,0] = self.x_range[1]
            if x_candidato[1,0] < self.y_range[0]:
                x_candidato[1,0] = self.y_range[0]
            if x_candidato[1,0] > self.y_range[1]:
                x_candidato[1,0] = self.y_range[1]

            f_candidato = self.f(x_candidato[0,0], x_candidato[1,0])

            P_ij = np.exp(-((f_candidato - f_opt)/self.T))
            if (self.minimize and f_candidato < f_opt) or (not self.minimize and f_candidato > f_opt) or P_ij >= np.random.uniform(0,1):
                x_opt = x_candidato
                f_opt = f_candidato
            self.T = 0.99 * self.T
        
        print(x_opt)