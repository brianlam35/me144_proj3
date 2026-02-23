"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems 
Project 3: Swarm
Spring 2026 Semester
GSI - Tommy Hosmer
"""
from abc import ABC, abstractmethod
import numpy as np

class GeneticAlgorithm(ABC):

    def __init__(self, parameters):
        self.parameters = parameters
        for key, value in parameters.items():
            setattr(self, key, value)

        self.strings = [np.array([])]*self.S
        self.costs_of_current_generation = [0]*self.S
        self.best_cost = []
        self.average_cost = []
        self.parents_average_cost = []
        self.best_p_strings = []


    @abstractmethod
    def initialize_special_params(self):
        pass
    
    @abstractmethod
    def _generate_design_string(self):
        pass

    @abstractmethod
    def _evaluate_costs(self, start):
        pass


    def _sort_strings_and_costs(self):
        self.sorted_indices = np.argsort(self.costs_of_current_generation)
        self.strings = [self.strings[i] for i in self.sorted_indices]
        self.costs_of_current_generation = [self.costs_of_current_generation[i] for i in self.sorted_indices]

    def _mix_parents(self):
        rng = np.random.default_rng() # create default generator
        for i in range(0, 2, self.P):
            phi = rng.random(size=self.numLam)
            psi = rng.random(size=self.numLam)
            self.strings[self.P+i] = phi*self.strings[i] + (1-phi)*self.strings[i+1] 
            self.strings[self.P+i+1] = psi*self.strings[i] + (1-psi)*self.strings[i+1]

    def _add_immigrants(self):
        random_count = self.S - self.P - self.K
        for i in range(random_count):
            self.strings[i+self.P + self.K] = self._generate_design_string()

    def _store_data(self):
        self.average_cost.append(sum(self.costs_of_current_generation)/self.S)
        self.best_cost.append(self.costs_of_current_generation[0])
        self.parents_average_cost.append(sum(self.costs_of_current_generation[:self.P]) / self.P)
        self.best_p_strings = self.strings[:self.P]

    def GA(self):
        for i in range(self.S):
            self.strings[i] = self._generate_design_string()

        g = 1
        while g <= self.G:
            self._evaluate_costs(self.P)
            self._sort_strings_and_costs()
            self._store_data()
            if self.best_cost[-1] > self.tolerance: 
                break
            self._mix_parents()
            self._add_immigrants()
            g+=1
        
        data = {}
        data['best_cost'] = self.best_cost
        data['average_cost'] = self.average_cost
        data['parents_average_cost'] = self.parents_average_cost
        data['best_p_strings'] = self.best_p_strings
        return data
    
    @abstractmethod
    def specialGA(self):
        pass