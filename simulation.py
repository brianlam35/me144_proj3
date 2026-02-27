"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems 
Project 3: Swarm
Spring 2026 Semester
GSI - Tommy Hosmer
"""
import numpy as np
import pickle
from typing import Dict, List
import pdb
"""
########################################
####### STUDENT VERSION ################
########################################

This script features the function to run the physics simulation.
The function will accept two dictionaries:
    parameters - the system parameters that define the environment and are NOT to be adjusted
    drone_genes - the hyperparameters of the drones that control their flight behavior and that are optimized through the GA
"""
class Simulation:

    def __init__(self, parameters: Dict, path_to_genes: str):
        """
        Initialize values for the simulation class

        nT0 [int]: initial number of targets
        nA0 [int]: initial number of drones (aka agents or members)
        path_to_genes [str]: path to pkl of design variables that are functional
        No [int]: number of obstacles
        Nt [int]: number of targets
        Nm [int]: number of drones 
        obs [ndarray: No x (1x3)]: positions of the obstacles
        tar [ndarray: Nt x (1x3)]: positions of the targets
        pos [ndarray: Nm x (1x3)]: positions of the drones 
        vel [ndarray: Nm x (1x3)]: velocities of the drones
        pos0, vel0, tar0: initial values for the three variables above
        """
        np.random.seed(42)
        # Environment parameters are fixed so we can set them when we instantiate the class
        for key, value in parameters.items():
            setattr(self, key, value)
        
        # Save initial number of targets and agents for cost calculation
        self.nT0 = self.Nt
        self.nA0 = self.Nm
        self.path_to_genes = path_to_genes

        # Initialize Obstacle Locations
        self.obs = np.array([(self.locx - (-self.locx))*np.random.rand(self.No) + (-self.locx),
                        (self.locy - (-self.locy))*np.random.rand(self.No) + (-self.locy),
                        (self.locz - (-self.locz))*np.random.rand(self.No) + (-self.locz)])
        self.obs = self.obs.T

        # Initial Target Locations
        self.tar = np.array([(self.locx - (-self.locx))*np.random.rand(self.Nt) + (-self.locx),
                        (self.locy - (-self.locy))*np.random.rand(self.Nt) + (-self.locy),
                        (self.locz - (-self.locz))*np.random.rand(self.Nt) + (-self.locz)])
        self.tar = self.tar.T

        # Initialize drone positions and velocity
        pos = np.array([(self.xmax - 0.05*self.xmax)*np.ones(self.Nm),
                        np.linspace(-self.ymax + 0.05*self.ymax, self.ymax - 0.05*self.ymax, self.Nm),
                        np.zeros(self.Nm)])
        self.pos = pos.T
        self.vel = np.zeros([self.Nm, 3])

        self.pos0 = self.pos.copy()                              # Initial agent positions
        self.vel0 = self.vel.copy()                              # Initial agent velocities
        self.tar0 = self.tar.copy()                              # Initial target positions


    def _read_optimal_string(self)-> None:
        """
        Method to read an already functional design string. Used for simulations outside of GA when an optimal design has already been found.
        
        path_to_genes [str]: path to pkl of design variables that are functional
        """
        # Read the genes (hyperparameters) from pickle  
        with open(self.path_to_genes, 'rb') as file:
            drone_genes = pickle.load(file)
        # Add each hyperparameter value into the class
        for key, value in drone_genes.items():
            setattr(self, key, value)


    def _read_LAM_vector_from_GA(self, LAM: List[float])->None:
        """
        When in the GA, design strings are passed into the simulation and read into the class by this method.
        """
        ######## Step 1 ################
        self.Wmt = LAM[0]
        self.Wmo = LAM[1]
        self.Wmm = LAM[2]
        self.wt1 = LAM[3]
        self.wt2 = LAM[4]
        self.wo1 = LAM[5]
        self.wo2 = LAM[6]
        self.wm1 = LAM[7]
        self.wm2 = LAM[8]
        self.a1 = LAM[9]
        self.a2 = LAM[10]
        self.b1 = LAM[11]
        self.b2 = LAM[12]
        self.c1 = LAM[13]
        self.c2 = LAM[14]
        ################################ 


    def _compute_distances(self)-> None: 
        """
        Method to compute drone-to-target, drone-to-drone, and drone-to-obstacle distances

        mtDiff [ndarray]: matrix of the vector pointing from every drone to every target
        mmDiff [ndarray]: matrix of the vector pointing from every drone to every other drone
        mmDiff [ndarray]: matrix of the vector pointing from every drone to every obstacle
        mtDist [ndarray]: matrix of the distance from every drone to every target
        mmDiff [ndarray]: matrix of the distance pointing from every drone to every other drone
        mmDiff [ndarray]: matrix of the distance pointing from every drone to every obstacle
        """

        Nm = self.pos.shape[0]
        Nt = self.tar.shape[0]
        No = self.obs.shape[0]

        # Allocate arrays
        self.mtDiff = np.full((Nm, Nt, 3), np.nan)
        self.mmDiff = np.full((Nm, Nm, 3), np.nan)
        self.moDiff = np.full((Nm, No, 3), np.nan)

        self.mtDist = np.full((Nm, Nt), np.nan)
        self.mmDist = np.full((Nm, Nm), np.nan)
        self.moDist = np.full((Nm, No), np.nan)
        
        """
        The np.nan_to_num serves its purpose more clearly later, in remove_and_crash method
        """
        for j in range(Nm):
        # If drone j is dead, leave its row as NaN and skip
            if np.any(np.isnan(self.pos[j])):
                continue

            # --- Differences ---
            self.mtDiff[j, :, :] = self.tar - self.pos[j]     # (Nt,3)
            self.mmDiff[j, :, :] = self.pos - self.pos[j]     # (Nm,3)
            self.moDiff[j, :, :] = self.obs - self.pos[j]     # (No,3)

            # --- Distances ---
            self.mtDist[j, :] = np.linalg.norm(self.mtDiff[j, :, :], ord=2, axis=1)
            self.mmDist[j, :] = np.linalg.norm(self.mmDiff[j, :, :], ord=2, axis=1)
            self.moDist[j, :] = np.linalg.norm(self.moDiff[j, :, :], ord=2, axis=1)

            # No self-interactions for drone-drone
            self.mmDiff[j, j, :] = np.nan
            self.mmDist[j, j] = np.nan
    

    def _check_collisions(self)->None:
        """
        Knowing distances, check for targets that have been mapped, drones that have collided with obstacles, and drones that have collided with one another.

        mtHit [ndarray]: boolean array of all targets that have been mapped
        moHit [ndarray]: boolean array of drones that collided with obstacles
        mmHit [ndarray]: boolean array of drones that collided with other drones
        xLost [ndarray]: boolean array of drones that exited domain relative to the x-axis bounds 
        yLost [ndarray]: boolean array of drones that exited domain relative to the y-axis bounds
        zLost [ndarray]: boolean array of drones that exited domain relative to the z-axis bounds  
        mLost [ndarray]: boolean array of all unique drones that flew out of bounds
        tarMapped [ndarray]: boolean array of all unique targets that have been mapped
        mCrash [ndarray]: boolean array of all unique drones that have crashed or flown out of bounds

        Hint: review parameters agent_sight, crash_range, xmax, ymax, and zmax
        """
        ######## Step 2 ################
        #Collision
        #mapped target if within sight
        mtHit = np.where(self.mtDist <= self.agent_sight)
        #drone-obstacle collision if within crash range
        moHit = np.where(self.moDist <= self.crash_range)
        #drone-drone collision if within crash range
        mmHit = np.where(self.mmDist <= self.crash_range)
        ################################


        # Ignore hits coming from already-dead drones (pos is NaN)
        alive = ~np.isnan(self.pos[:, 0])

        # mtHit is (drone_indices, target_indices)
        mtHit = (mtHit[0][alive[mtHit[0]]], mtHit[1][alive[mtHit[0]]])

        # moHit is (drone_indices, obstacle_indices)
        moHit = (moHit[0][alive[moHit[0]]], moHit[1][alive[moHit[0]]])

        # mmHit is (drone_indices, other_drone_indices)
        mmHit = (mmHit[0][alive[mmHit[0]]], mmHit[1][alive[mmHit[0]]])



        ######## Step 3 ################
        # Out of bounds crashes
        # Lower bound?
        xLost = np.where(np.abs(self.pos[:, 0]) > self.xmax)
        yLost = np.where(np.abs(self.pos[:, 1]) > self.ymax)
        zLost = np.where(np.abs(self.pos[:, 2]) > self.zmax)
        ################################

        mLost = np.unique(np.hstack([xLost[0], yLost[0], zLost[0]]))
        self.tarMapped = np.unique(mtHit[1])
        self.mCrash = np.unique(np.hstack([mmHit[0], moHit[0], mLost]))


    def _label_crash_and_map(self)-> None:
        """
        Method to label drones that have crashed/exited domain and targets that have been mapped.
        Instead of deleting values, we replace with np.nan to avoid resizing arrays and constantly reshifting things around in memory.
        This is to improve time complexity.
        If curious why, then ask a GSI or google "berkeley cs267 spring 2025".
        """
        
        # Replace positions and distances of crashed agents with np.nan
        self.mtDist[self.mCrash, :] = np.nan
        self.mtDiff[self.mCrash, :, :] = np.nan

        self.mmDist[self.mCrash, :] = np.nan
        self.mmDist[:, self.mCrash] = np.nan
        self.mmDiff[self.mCrash, :, :] = np.nan
        self.mmDiff[:, self.mCrash, :] = np.nan

        self.moDist[self.mCrash, :] = np.nan
        self.moDiff[self.mCrash, :, :] = np.nan

        # Replace positions and distances of mapped targets with np.nan
        self.tar[self.tarMapped, :] = np.nan
        self.mtDist[:, self.tarMapped] = np.nan
        self.mtDiff[:, self.tarMapped, :] = np.nan


    def _compute_dynamics(self)-> None:
        """
        Method to compute forces acting on each drone. 

        nMT [ndarray]: normal vector pointing from each drone to all targets
        nMO [ndarray]: normal vector pointing from each drone to all obstacles
        nMM [ndarray]: normal vector pointing from each drone to all other drones
        nMThat [ndarray]: magnitude * normal vector pointing from each drone to all targets
        nMOhat [ndarray]: magnitude * normal vector pointing from each drone to all obstacles
        nMMhat [ndarray]: magnitude * normal vector pointing from each drone to all other drones
        Nmt [ndarray]: on each drone, the vector representing the sum of all vectors pointing from respective drone to targets
        Nmo [ndarray]: on each drone, the vector representing the sum of all vectors pointing from respective drone to obstacles
        Nmm [ndarray]: on each drone, the vector representing the sum of all vectors pointing from respective drone to drones
        Ntot [ndarray]: sum of all interaction vectors at each drone
        Ntot_norm [ndarray]: norm of the sum of all interaction vectors at each drone
        nProp [ndarray]: normal vector pointing in the direction of cumulative interactions
        fProp [ndarray]: force vector driven by propulsion; magnitude of propulsive force * normal vector of total interaction
        vNormDiff [ndarray]: normal direction of the drone's velocity relative to velocity of the air
        fDrag [ndarray]: drag force acting on each drone
        fTot [ndarray]: total force acting on the drone
        """
        
        nMT = np.nan_to_num(self.mtDiff / self.mtDist[:, :, np.newaxis])
        nMO = np.nan_to_num(self.moDiff / self.moDist[:, :, np.newaxis])
        nMM = np.nan_to_num(self.mmDiff / self.mmDist[:, :, np.newaxis])

        ######## Step 4 ################
        #Eq 13 - target interaction
        nMThat = (self.wt1 * np.exp(-self.a1 * self.mtDist) - self.wt2 * np.exp(-self.a2 * self.mtDist))
        nMThat = np.nan_to_num(nMThat[:, :, np.newaxis] * nMT)

        #Eq 17 - obstacle interaction
        nMOhat = (self.wo1 * np.exp(-self.b1 * self.moDist) - self.wo2 * np.exp(-self.b2 * self.moDist)) 
        nMOhat = np.nan_to_num(nMOhat[:, :, np.newaxis] * nMO)

        #Eq21 - drone interaction
        nMMhat = (self.wm1 * np.exp(-self.c1 * self.mmDist) - self.wm2 * np.exp(-self.c2 * self.mmDist))
        nMMhat = np.nan_to_num(nMMhat[:, :, np.newaxis] * nMM)
        ################################

        # Sum all vectors for each drone
        Nmt = np.nansum(nMThat, axis=1)
        Nmo = np.nansum(nMOhat, axis=1)
        Nmm = np.nansum(nMMhat, axis=1)

        ######## Step 5 ################
        # Sum the weighted attractive/repulsive forces at each drone
        Ntot = self.Wmt * Nmt + self.Wmo * Nmo + self.Wmm * Nmm
        Ntot_norm = np.linalg.norm(Ntot, 2, axis=1)[:, np.newaxis]
        Ntot_norm[Ntot_norm == 0] = np.nan
        
        # Normalize interactions vector and scale by propulsive force magnitude
        nProp = np.nan_to_num(Ntot / Ntot_norm)
        fProp = self.Fp * nProp
        
        velocity_diff = np.nan_to_num(self.va - self.vel)
        vNormDiff = np.linalg.norm(velocity_diff, 2, axis=1)[:, np.newaxis]
        # DOUBLE CHECK - Cdi & Ai?
        fDrag = 0.5 * self.ra * self.Cdi * self.Ai * vNormDiff * velocity_diff

        fTot = fProp + fDrag
        ################################

        ######## Step 6 ################
        # Semi-Implicit Euler update
        alive = ~np.isnan(self.pos[:, 0])

        acc = np.zeros_like(self.vel)
        acc[alive] = fTot[alive] / self.mi

        self.vel[alive] = self.vel[alive] + acc[alive] * self.dt
        self.pos[alive] = self.pos[alive] + self.vel[alive] * self.dt
        ################################


    def _compute_cost(self, counter: int)-> float:
        """
        Compute design cost for the string

        Mstar [float]: fraction of unmapped targets
        Tstar [float]: fraction of available time to reach end of simulation
        Lstar [float]: fraction of agents that crashed
        PI [float]: total cost
        """

        ######## Step 7 ################
        Mstar = np.sum(~np.isnan(self.tar[:, 0])) / self.nT0
        Tstar = (counter * self.dt) / self.tf
        Lstar = (np.sum(np.isnan(self.pos[:, 0]))) / self.nA0
        if Mstar < 0:
            print('Mstar ' + str(Mstar))
        if Lstar < 0:
            print('Lstar ' + str(Lstar))
        if Tstar < 0:
            print('Tstar ' + str(Tstar))
        PI = self.w1 * Mstar + self.w2 * Tstar + self.w3 * Lstar
        ################################

        return PI, Mstar, Tstar, Lstar
    

    def _remove_crash_and_map(self)-> None:
        """
        Method to remove crashed drones and mapped targets 
        """
        # Replace positions and velocities of crashed agents with np.nan
        self.pos[self.mCrash, :] = np.nan
        self.vel[self.mCrash, :] = np.nan

        # Replace positions of mapped targets with np.nan
        self.tar[self.tarMapped, :] = np.nan

        ######## Step 8 ################
        # Update the number of targets and agents
        self.nT = int(np.sum(~np.isnan(self.tar[:, 0])))
        self.nA = int(np.sum(~np.isnan(self.pos[:, 0])))
        ################################


    def run_simulation(self, read_LAM=False, LAM=[]):
        """
        Method to bring it all togeter
        """
        if read_LAM:
            if len(LAM) == 0:
                raise ValueError("LAM is empty but read_LAM is true.")
            self._read_LAM_vector_from_GA(LAM)
        else:
            self._read_optimal_string()
        
        tStep = int(np.ceil(self.tf / self.dt))  # total time steps
        counter = 0  # counter for actual number of time steps
        posData = []  # store mPosition data
        tarData = []  # store mTarget data
        posData.append(self.pos0.copy())
        tarData.append(self.tar0.copy())
        obsData = self.obs.copy()
        self.pos = self.pos0.copy()
        self.tar = self.tar0.copy()
        
        # Time Loop
        for i in range(tStep):
            self.timestep = i
            ######## Step 9 ################
            # Determine order of the methods from above 
            self._compute_distances()

            self._check_collisions()

            self._label_crash_and_map()
            
            self._compute_dynamics()

            self._remove_crash_and_map() # put this after break?

            # if all agents are lost, crashed, or eliminated, stop the simulation
            if np.all(np.isnan(self.tar)) or np.all(np.isnan(self.pos)):
                break

            ################################
            
            posData.append(self.pos.copy())
            tarData.append(self.tar.copy())
            
            counter += 1

        PI, Mstar, Tstar, Lstar = self._compute_cost(counter)

        return PI, posData, tarData, obsData, counter, Mstar, Tstar, Lstar


# Example
if __name__ == "__main__":
    # Load parameters from a pickle file
    with open('parameters.pkl', 'rb') as file:
        parameters = pickle.load(file)
    
    sim = Simulation(parameters, path_to_genes='')
    
    # Design string from a previous run of the GA
    LAM=[0.3988442888626489, 1.949567329046, 0.5777614822547401, 1.064381542531382, 0.2389800770766377, 1.8583543294267468, 0.08971249931120329, 0.18965425881131015, 1.3688472010259771, 0.13871317151794504, 0.9680810718049682, 1.8708028880195542, 1.3845315310998025, 0.9138950782301458, 0.5957025792584691]
    
    PI, posData, tarData, obsData, counter, Mstar, Tstar, Lstar = sim.run_simulation(read_LAM=True,LAM=LAM)
    print(PI)
    print("M*:", Mstar, "T*:", Tstar, "L*:", Lstar)
    print("timesteps:", counter)
    print("Remaining targets:", sim.nT)
    print("Remaining agents:", sim.nA)
    print(parameters)
    

