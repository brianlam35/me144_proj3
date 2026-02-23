"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems 
Project 3: Swarm
Spring 2026 Semester
GSI - Tommy Hosmer
"""
from ga_class import GeneticAlgorithm
import numpy as np
from simulation import Simulation
from typing import Dict

class SwarmGA(GeneticAlgorithm):

    def initialize_special_params(self)-> None:
        self.Tstar = np.zeros(self.S)
        self.Mstar = np.zeros(self.S)
        self.Lstar = np.zeros(self.S)

        self.MstarMin = np.zeros(self.G)                                # Mstar value associated with best cost for each generation
        self.TstarMin = np.zeros(self.G)                                # Tstar value associated with best cost for each generation
        self.LstarMin = np.zeros(self.G)                                # Lstar value associated with best cost for each generation

        self.MstarPAve = np.zeros(self.G)                               # Average Mstar value for top parents for each generation
        self.TstarPAve = np.zeros(self.G)                               # Average Tstar value for top parents for each generation
        self.LstarPAve = np.zeros(self.G)                               # Average Lstar value for top parents for each generation

        self.MstarAve = np.zeros(self.G)                                # Average Mstar value for whole population for each generation
        self.TstarAve = np.zeros(self.G)                                # Average Tstar value for whole population for each generation
        self.LstarAve = np.zeros(self.G)                                # Average Lstar value for whole population for each generation

    def _generate_design_string(self)-> np.ndarray:
        # Assignment states that all design variables are assumed to lie in the closed set [0,2]
        return 2*np.random.random(self.numLam)


    def _evaluate_costs(self,start: int)-> None:
        for i in range(start, self.S):
            sim = Simulation(self.parameters,'')
            self.costs_of_current_generation[i], _, _, _, _, self.Mstar[i], self.Tstar[i], self.Lstar[i] = sim.run_simulation(read_LAM=True, LAM=self.strings[i])


    def _MLT_star_updates(self, g: int)-> None:
        # sort the M, L, and T star values
        # Store the averages and mins
        self.Tstar = self.Tstar[self.sorted_indices]
        self.Mstar = self.Mstar[self.sorted_indices]
        self.Lstar = self.Lstar[self.sorted_indices]

        self.MstarMin[g] = self.Mstar[0]
        self.TstarMin[g] = self.Tstar[0]
        self.LstarMin[g] = self.Lstar[0]

        self.MstarPAve[g] = np.mean(self.Mstar[0:self.P])
        self.TstarPAve[g] = np.mean(self.Tstar[0:self.P])
        self.LstarPAve[g] = np.mean(self.Lstar[0:self.P])

        self.MstarAve[g] = np.mean(self.Mstar)
        self.TstarAve[g] = np.mean(self.Tstar)
        self.LstarAve[g] = np.mean(self.Lstar)


    def specialGA(self,print_statements=False)-> Dict:
        # Have to define special GA function for the M*, L*, T* array calculations and updates
        for i in range(self.S):
            self.strings[i] = self._generate_design_string()

        g = 1
        start = 0 
        while g <= self.G:
            self._evaluate_costs(start)
            self._sort_strings_and_costs()
            self._store_data()
            self._MLT_star_updates(g=g-1)
            if self.best_cost[-1] < self.tolerance:
                break
            if print_statements:
                print(f'Generation: {g}')
                print(f'Best cost: {self.best_cost[-1]}')
                print(f'M*: {self.MstarMin}')
                print(f'T*: {self.TstarMin}')
                print(f'L*: {self.LstarMin}')
            self._mix_parents()
            self._add_immigrants()
            g+=1
            start = self.P
        
        data = {}
        data['best_cost'] = self.best_cost
        data['average_cost'] = self.average_cost
        data['parents_average_cost'] = self.parents_average_cost
        data['best_p_strings'] = self.best_p_strings
        data['MstarMin'] = self.MstarMin
        data['TstarMin'] = self.TstarMin
        data['LstarMin'] = self.LstarMin
        data['MstarPAve'] = self.MstarPAve
        data['TstarPAve'] = self.TstarPAve
        data['LstarPAve'] = self.LstarPAve
        data['MstarAve'] = self.MstarAve
        data['TstarAve'] = self.TstarAve
        data['LstarAve'] = self.LstarAve
        return data