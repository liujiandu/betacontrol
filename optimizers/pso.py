#----------------------------------------
#Particle swarm algorithm
#Author : Liujiandu
#Date: 2018/1/4
#----------------------------------------
import numpy as np

class PSO(object):
    def __init__(self, func, init_position, init_velocity=None):
        self.func=func
        self.position = init_position
        self.particle_num, self.position_dim = self.position.shape

        if init_velocity==None:
            self.velocity = (np.random.random((self.particle_num,self.position_dim))-0.5)*20
        else:
            self.velocity = init_velocity

        self.omiga = 0.9
        self.lr1 = 0.5
        self.lr2 = 0.7

        self.indivdual_best_fitness = [self.func(self.position[i]) for i in range(self.particle_num)]
        self.indivdual_best_position = self.position.copy()
	self.group_best_fitness = min(self.indivdual_best_fitness)
	self.group_best_position = self.position[self.indivdual_best_fitness.index(self.group_best_fitness)]
    
    def update_position_velocity(self):
	c1 = self.omiga*self.velocity 
        c2 =self.lr1*np.multiply(np.random.random((self.particle_num,self.position_dim)),self.indivdual_best_position-self.position)
        c3 =self.lr2*np.multiply(np.random.random((self.particle_num,self.position_dim)),self.group_best_position-self.position)
        self.velocity = c1 + c2 + c3
        self.position = self.position+self.velocity
    
    def update_fitness(self):
	for i in range(self.particle_num):
	    fitness = self.func(self.position[i])
	    if fitness<self.indivdual_best_fitness[i]:
		self.indivdual_best_fitness[i] = fitness
		self.indivdual_best_position[i] = self.position[i]
		if fitness<self.group_best_fitness:
		    self.group_best_fitness = fitness
                    self.group_best_position = self.position[i]

    def optimization(self, max_iter):
        group_best_fitness = []
	group_best_position = []
	for _ in range(max_iter):
	    self.update_position_velocity()
            self.update_fitness()
	    group_best_fitness.append(self.group_best_fitness)	
	    group_best_position.append(self.group_best_position)
        return group_best_fitness, group_best_position

if __name__=="__main__":
    import sys
    sys.path.append('../')
    from function.holder_table import func
    init_position = 20.0*(np.random.random((5, 2))-0.5)
    pso = PSO(func, init_position)
    pso.optimization(10)
