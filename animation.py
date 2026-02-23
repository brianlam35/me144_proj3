"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems 
Project 3: Swarm
Spring 2026 Semester
GSI - Tommy Hosmer
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib import rc
from typing import List, Dict, Tuple
"""
This script features the functions to animate the simulation.

TODO:
- add comments
"""

class Animation:
    def __init__(self,counter: int, tarData: List[np.ndarray], posData: List[np.ndarray], obsData: np.ndarray, parameters: Dict):
        self.counter = counter
        self.posData = posData
        self.tarData = tarData
        self.obsData = obsData
        # environment parameters are fixed so we can set them once when we instantiate the class
        for key, value in parameters.items():
            setattr(self, key, value)

        # Read Obstacle Locations
        self.obs = self.obsData

        # Read Initial Target Locations
        self.tar0 = self.tarData[0]

        # Read Initial Drone Positions
        self.pos0 = self.posData[0]                         


    ################################## Plotting Initial System ####################################################
    def initial_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.obsData[:, 0], self.obsData[:, 1], self.obsData[:, 2], color='r')
        ax.scatter(self.tar0[:, 0], self.tar0[:, 1], self.tar0[:, 2], color='g')
        ax.scatter(self.pos0[:, 0], self.pos0[:, 1], self.pos0[:, 2], color='k')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.view_init(elev=70., azim=40)
        ax.legend(['Obstacles', 'Targets', 'Agents'])

    ################################# Plotting Best Solution ###############################################################

    def _drawframe(self, n: int)-> Tuple:  # Function to create plot
        self.dots1.set_xdata(self.obs[:, 0])
        self.dots1.set_ydata(self.obs[:, 1])
        self.dots1.set_3d_properties(self.obs[:, 2])

        self.dots2.set_xdata(self.tarData[n][:, 0])
        self.dots2.set_ydata(self.tarData[n][:, 1])
        self.dots2.set_3d_properties(self.tarData[n][:, 2])

        self.dots3.set_xdata(self.posData[n][:, 0])
        self.dots3.set_ydata(self.posData[n][:, 1])
        self.dots3.set_3d_properties(self.posData[n][:, 2])

        self.Title.set_text('Solution Animation: Time = {0:4f}'.format(n * self.dt))
        return (self.dots1, self.dots2, self.dots3)
    ###################################################################################################
    ######################################## Plotting #################################################
    ###################################################################################################

    def animation(self, storage_directory:str = '')-> None:
        # Set up plot for animation
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_xlim((-self.xmax, self.xmax))
        ax.set_ylim((-self.ymax, self.ymax))
        ax.set_zlim((-self.zmax, self.zmax))
        ax.view_init(elev=70., azim=40)

        self.Title = ax.set_title('')

        self.dots1, = ax.plot([], [], [], 'r.', ms=10)
        self.dots2, = ax.plot([], [], [], 'g.', ms=10)
        self.dots3, = ax.plot([], [], [], 'k.', ms=10)

        ax.legend(['Obstacles', 'Targets', 'Agents'])

        anim = animation.FuncAnimation(fig, self._drawframe, frames=self.counter, interval=50, blit=True)

        rc('animation', html='html5')

        anim
        writervideo = animation.FFMpegWriter(fps=60)
        if len(storage_directory):
            anim.save(f'{storage_directory}/{self.Nm}_agents_{self.No}_obs_{self.Nt}_tar.mp4', writer=writervideo)
        else:
            anim.save(f'{self.Nm}_agents_{self.No}_obs_{self.Nt}_tar.mp4', writer=writervideo)


    def plot_costs_over_generations(self, path_to_pickle: str, storage_directory:str ='')-> None:
        with open(path_to_pickle, 'rb') as f:
            ga_results = pickle.load(f)

        # Assuming ga_results is a dict with keys: 'avg_cost', 'best_cost', 'avg_parent_cost'
        avg_cost = ga_results['average_cost']
        best_cost = ga_results['best_cost']
        avg_parent_cost = ga_results['parents_average_cost']

        plt.figure(figsize=(10, 6))
        plt.plot(avg_cost, label='Average Cost')
        plt.plot(best_cost, label='Best Cost')
        plt.plot(avg_parent_cost, label='Average Parent Cost')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.title('GA Cost Metrics Over Generations')
        plt.legend()
        plt.grid(True)
        if len(storage_directory):
            plt.savefig(f'{storage_directory}/cost_plot.png')
        else:
            plt.show()


    def plot_star_vals(self, path_to_pickle: str, storage_directory:str ='')-> None:
        with open(path_to_pickle, 'rb') as f:
            ga_results = pickle.load(f)
        
        MstarMin = ga_results['MstarMin']
        TstarMin = ga_results['TstarMin']
        LstarMin = ga_results['LstarMin']
        plt.figure(figsize=(10, 6))
        plt.plot(MstarMin, label='MstarMin')
        plt.plot(TstarMin, label='TstarMin')
        plt.plot(LstarMin, label='LstarMin')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.title('Mins of Star Cost Metrics Over Generations')
        plt.legend()
        plt.grid(True)
        if len(storage_directory):
            plt.savefig(f'{storage_directory}/LMTmin_plot.png')
        else:
            plt.show()

        MstarPAve = ga_results['MstarPAve']
        TstarPAve = ga_results['TstarPAve']
        LstarPAve = ga_results['LstarPAve']
        plt.figure(figsize=(10, 6))
        plt.plot(MstarPAve, label='MstarPAve')
        plt.plot(TstarPAve, label='TstarPAve')
        plt.plot(LstarPAve, label='LstarPAve')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.title('Parent Averages of Star Cost Metrics Over Generations')
        plt.legend()
        plt.grid(True)
        if len(storage_directory):
            plt.savefig(f'{storage_directory}/LMTPAve_plot.png')
        else:
            plt.show()

        MstarAve = ga_results['MstarAve']
        TstarAve = ga_results['TstarAve']
        LstarAve = ga_results['LstarAve']
        plt.figure(figsize=(10, 6))
        plt.plot(MstarAve, label='MstarAve')
        plt.plot(TstarAve, label='TstarAve')
        plt.plot(LstarAve, label='LstarAve')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.title('Averages of Star Cost Metrics Over Generations')
        plt.legend()
        plt.grid(True)
        if len(storage_directory):
            plt.savefig(f'{storage_directory}/LMTAve_plot.png')
        else:
            plt.show()