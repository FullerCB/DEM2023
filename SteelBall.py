"""
2-d implementation With Rotation of DEM particles
for Dr.  Wilkerson
@author: Fuller Collins-Bilyeu
"""


'''
Mark 3, with the dynamic interactions working on a 3d scale (on a 2d plane),
as well as the tangential displacemets being calculated - optimization has begun
working on removing all n^2 complexities as well as simplfying and speeding up
wall interactions will also be expanded upon
'''
#importing libraries

import numpy as np
import scipy.spatial
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#import tester
from line_profiler import LineProfiler
import random
#-----------------------------------------------------------------------------------------#

#variables and array intialization


n = 300 #Number of Particles
r = 0.5 #radius of particle
m = .01 #mass

DOF = 4 #Dimensional Degrees of Freedom - for example, 4 if theres position, velocity, angular position and angular velcoity
DOFP = 3 #Positional Degrees of freedom  - For example, 2 if theres x-pos, y-pos
DOFA = 3 #Angular Degrees of freedom - for example, 1 if theres angularspeed/velocity about the z axis

# Define the data types for the structured array
dtype = [
    ('position', float, (DOFP,)), 
    ('velocity', float, (DOFP,)),
    ('angular_position', float, (DOFA,)),
    ('angular_velocity', float, (DOFA,))
]

#Create an empty structured numpy array with 'n' elements
DOFarray = np.zeros(n, dtype=dtype)

#Creating a sparse array to hold all accumulated vectors
#accumulated_vector = scipy.sparse.lil_matrix((n * n, DOFP), dtype=np.float64)

#creating default dict where dicts that dont exist return zeros array
ref_vector = defaultdict(lambda: defaultdict(lambda: np.zeros(DOFP)))
accumulated_vector = defaultdict(lambda: defaultdict(lambda: np.zeros(DOFP)))

time = 1 #how long to run
dt = 0.00005 #step size

WallLpos, WallRpos, WallBpos, WallTpos = 0, 200, 0, 150 #bounds for wall
Wall = np.array([WallLpos, WallBpos, WallRpos, WallTpos])

#Spring Constants
cn = 20 #normal damping coefficient
ct = 4 #tangential damping coefficient

v = 0.3 #poissons ratio
E = 10**6 #youngs mod

us = 0.4 #static friction coeff
uk = 0.3 #kinetic friction coeff
cor = 0.85 #coeff of restitution
crr = 0.005 #rolling coeff of friction

I = 2 / 5 * m * r**2

E_var = E / (2 * (1 - v ** 2))
R_var = r / 2

#steel ball info
steel_radius = 5
steel_mass = 0.008 * 4/3 * np.pi * 5**3
dtype_steel = [
    ('position', float, (DOFP,)), 
    ('velocity', float, (DOFP,)),
    ('acceleration', float, (DOFP,))
]
steel_array = np.zeros(1,dtype = dtype_steel)
steel_array['position'] = [WallRpos / 2, 60, 0]

#-----------------------------------------------------------------------------------------#

#functions

def normmag(arraymag): #much more efficient than numpy for this project
    mag = math.sqrt(arraymag[0]**2+arraymag[1]**2+arraymag[2]**2)
    return mag

def vector_cross(left, right): #numpy cross function is super slow due to broad use case
    x = ((left[1] * right[2]) - (left[2] * right[1]))
    y = ((left[2] * right[0]) - (left[0] * right[2]))
    z = ((left[0] * right[1]) - (left[1] * right[0]))
    return np.array([x,y,z])

def distance_vector(i, j):
    
    distance_vec = DOFarray[j]['position'] - DOFarray[i]['position']
    
    return distance_vec

def relative_velocity(i,j):
        
    relative_velocity = np.subtract(DOFarray[j]['velocity'], DOFarray[i]['velocity'])
    
    dishold = distance_vector(i,j)
    r_vector = (dishold / normmag(dishold)) * r
    
    #the cross products here factor in the tangential speeds of the edge relative to the center of mass
    relative_velocity += vector_cross(-DOFarray[j]['angular_velocity'], r_vector)
    relative_velocity -= vector_cross(-DOFarray[i]['angular_velocity'], -r_vector)
    return relative_velocity

#-----------------------------------------------------------------------------------------#

#normal forces
def normal_contact_force(i, j, x):
    
    E_var = E / (2*(1-v**2)) #given formula from research pdf
    R_var = r / 2 #given formula
    dishold = distance_vector(i,j) #holding variable for distance
    norm = dishold / normmag(dishold) #unit vector in direction of contact
    NCF = -(4/3) * E_var * np.sqrt(R_var) * (x**(3/2)) * norm
    
    return NCF

def normal_damping_force(i, j):
    dishold = distance_vector(i,j) #holding variable for distance
    norm = dishold / normmag(dishold) #unit vector in direction of contact
    NDF = cn * np.dot(relative_velocity(i, j), norm) * norm #formula
    return NDF

#tangential forces
def t_contact_force(i, j, NCF, t_displacement, x):
    #used found formula for sand 'scale1' and 'scale2' stuff  is just for readability
    max_t_displacement = us * x * (2-v) / (2 * (1-v)) 
    scale1 = (-us * normmag(NCF) / normmag(t_displacement))
    scale2 = (1 - (1 - min(normmag(t_displacement), max_t_displacement))**(3/2))
    TCF = scale1 * scale2 * np.array(t_displacement)
    return TCF

def t_damping_force(i, j):
    dishold = distance_vector(i,j) #holding variable for distance
    norm = dishold / normmag(dishold) #unit vector in direction of contact
    rel_vel = relative_velocity(i, j)
    TDF = -ct * vector_cross(vector_cross(rel_vel, norm), norm)
    return TDF

#force totals
def normal_forces(i, j, overlap): #reducancy for readability
    NCF = normal_contact_force(i, j, overlap)
    NDF = normal_damping_force(i, j)
    return NCF, NDF

def tang_forces(i, j, t_displacement, NCF, overlap): #reducancy for readability
    TCF = t_contact_force(i, j, NCF, t_displacement, overlap)
    TDF = t_damping_force(i, j)
    return TCF, TDF

#-----------------------------------------------------------------------------------------#

#search algo
'''
K-D Tree method w/ SciPy:
kdtree = scipy.spatial.KDTree(X)
pairs = kdtree.query_pairs(Radius)
for (i,j) in pairs:
	Collision calc with X[ i ] and X[ j ]
…
For x in X
	update x position

'''


    
def master_function(DOFarray):
    
    #initializations
    global ref_vector, accumulated_vector, accel
    
    accel = np.zeros(shape=(n, DOFP))
    angaccel = np.zeros(shape=(n, DOFA))
    
    kdtree = scipy.spatial.KDTree(DOFarray['position'], compact_nodes=True) #kdtree method should be most efficient collision method - no need to change for 3d
    pairs = kdtree.query_pairs(2.01*r) #input is search distance will be slightly greater than 2r because tangential displacement reset

    accel[:,1] += -9810

    #particle-wall
    
    DOFarray['velocity'][:, 0] = np.where(DOFarray['position'][:,0] <= r, np.abs(DOFarray['velocity'][:, 0]), DOFarray['velocity'][:, 0])
    DOFarray['velocity'][:, 0] = np.where(DOFarray['position'][:,0] >= (WallRpos - r), -np.abs(DOFarray['velocity'][:, 0]), DOFarray['velocity'][:, 0])
    
    wcontact = (4 / 3) * E_var * np.sqrt(R_var) * (abs(r - DOFarray['position'][:, 1]) ** (3 / 2))
    wdamp = -25 * DOFarray['velocity'][:, 1] #0.7 is defaut pw damp coeff
    
    accel[:, 1] += np.where(DOFarray['position'][:, 1] <= r, np.add(wdamp, wcontact)[:] / m, 0)
    
    #particle-particle
    for (i,j) in pairs:
        
        distance_array = distance_vector(i, j)
        distance = normmag(distance_array)
        
        overlap = 2 * r - distance

        in_contact = overlap > 0
        
        if in_contact:    
            
            #determining tangential and normal displacements
            if np.all(accumulated_vector[i][j] == 0) or np.all(accumulated_vector[i][j] == np.zeros(DOFP, dtype=np.float64)):
                ref_vector[i][j] = distance_vector(i, j) #creates vector to project to to determine normal/tangential displacement
            rel_center_velocity = np.subtract(DOFarray[i]['velocity'], DOFarray[j]['velocity'])
            displacement = rel_center_velocity * dt
            accumulated_vector[i][j] += displacement
            #vector projection of accumulated vector onto normal vector
            normal_vector = ref_vector[i][j] * (np.dot(accumulated_vector[i][j], ref_vector[i][j]) / np.dot(ref_vector[i][j], ref_vector[i][j]))
            t_displacement = np.subtract(accumulated_vector[i][j], normal_vector)
            
            #normal forces
            
            NCF, NDF = normal_forces(i, j, overlap)
            
            #tangential forces
            if np.any(t_displacement != 0):
                TCF, TDF = tang_forces(i, j, t_displacement, NCF, overlap)

                #dynamic angular calcs
                Torque = vector_cross((distance_array / distance) * r, np.add(TCF, TDF))
                angaccel[i] += Torque / I
                angaccel[j] += Torque / I
                
                #friction torque
                
                if np.any(DOFarray[i]['angular_velocity'] != 0):
                    angaccel[i] += (-crr * normmag(NCF) * DOFarray[i]['angular_velocity'] / normmag(DOFarray[i]['angular_velocity'])) / I
                if np.any(DOFarray[j]['angular_velocity'] != 0):
                    angaccel[j] += (-crr * normmag(NCF) * DOFarray[j]['angular_velocity'] / normmag(DOFarray[j]['angular_velocity'])) / I
                
            
            total_force = np.add(NCF, NDF)
            
            total_acceleration = np.array(total_force) / m
            
            accel[i] += total_acceleration
            accel[j] -= total_acceleration
            
            
        else:
            # Reset the accumulated vector to zero when particles are no longer in contact
            accumulated_vector[i][j] = np.zeros(DOFP)
            #based on my understanding setting dict entries back to 0 will ultimatley be more efficient than deleting them
    
    #External objects interactions
    
    #steel ball
    
    sand_steel_impacts = kdtree.query_ball_point(steel_array['position'], r + steel_radius)
    steel_array['acceleration'] = np.zeros(3)
    steel_array['acceleration'][0][1] += -9810
    for i in sand_steel_impacts[0]:
        sand_steel_distance = DOFarray['position'][i] - steel_array['position'][0]
        sand_steel_overlap = r + steel_radius - normmag(sand_steel_distance)
        if sand_steel_overlap > 0:
            
            sand_steel_norm = sand_steel_distance / normmag(sand_steel_distance) #unit vector in direction of contact
            NCF = -2 * (4/3) * E_var * np.sqrt(R_var) * (sand_steel_overlap**(3/2)) * sand_steel_norm  #acting on steel
            NDF = 0.7 * np.dot(DOFarray[i]['velocity'] - steel_array['velocity'][0], sand_steel_norm) * sand_steel_norm #acting on steel
            steel_array['acceleration'][0] += (NDF + NCF) / steel_mass
            accel[i] -= (NDF + NCF) / m
    
    #particle-wall-friction
    
    #when particle rolls
    rolling_vel = -(DOFarray['velocity'][:] * [1, 0, 1]) / np.linalg.norm(DOFarray['velocity'][:] * [1, 0, 1], axis=1)[:, np.newaxis]
    floor_roll_friction = np.zeros(shape=(n,3))
    touching_ground = DOFarray['position'][:, 1] <= r
    
    floor_roll_friction[touching_ground] = crr * np.multiply(wcontact[:, np.newaxis], rolling_vel)[touching_ground]
    
    #when particle is slipping/sliding
    
    tang_speed = np.cross(DOFarray['angular_velocity'][:], [0,-r,0])
    total_slide_speed = -np.add(tang_speed, DOFarray['velocity'] * [1,0,1])[:] / np.linalg.norm(np.add(tang_speed, DOFarray['velocity'][:] * [1,0,1]), axis=1)[:, np.newaxis]
    kinetic_slip_friction = np.zeros(shape=(n,3))
    kinetic_slip_friction[touching_ground] = uk * np.multiply(wcontact[:, np.newaxis], total_slide_speed[:])[touching_ground]
    
    accel += np.add(kinetic_slip_friction, floor_roll_friction) / m
    angaccel += np.add(np.cross([0,-r,0], kinetic_slip_friction[:]), np.cross([0,-r,0], floor_roll_friction[:])) / I 
    
    return accel, angaccel, steel_array['acceleration']


#---------------------------#
#defining particle conditions
#---------------------------#

for i in range(n):
    DOFarray[i][0][0] = random.uniform(90, 110)
    DOFarray[i][0][1] = random.uniform(Wall[1] + r, 70)
    for j in range(2):
        #DOFarray[i][0][j] = random.uniform(Wall[j] + r, Wall[j + 2] - r)
        DOFarray[i][1][j] = random.uniform(-2, 2)


'''
DOFarray['position'][0] = [2.2, 6.75, 0]
#DOFarray['position'][0] = [5, 0.9, 0]
DOFarray['velocity'][0] = [0, 20, 0]
#DOFarray['velocity'][0] = [0, -5, 0]
DOFarray['angular_velocity'][0] = [0, 0, 4]
DOFarray['position'][1] = [2, 8, 0]
DOFarray['velocity'][1] = [0, -3, 0]
'''
'''
DOFarray['position'][0] = [5, 0.5, 0]
DOFarray['angular_position'][0] = [0, 0, 3/2 * np.pi]
DOFarray['angular_velocity'][0] = [0, 0, 50]
'''
#----------------------------------------------------------------------------#
#numerical method
#----------------------------------------------------------------------------#

def verlet(DOFarray, steel_array, dt):

    accel, angaccel, steelaccel= master_function(DOFarray)

    # Initial update using the velocity Verlet method
    
    DOFarray['position'] += np.add(DOFarray['velocity'] * dt, 0.5 * (dt ** 2) * accel)
    DOFarray['angular_position'] += np.add(DOFarray['angular_velocity'] * dt, 0.5 * (dt ** 2) * angaccel)
    
    #non-particle updates
    steel_array['position'] += np.add(steel_array['velocity'] * dt, 0.5 * (dt ** 2) * steelaccel)
    
    #secondary
    accelnew, angaccelnew, steelaccelnew = master_function(DOFarray)

    DOFarray['velocity'] += 0.5 * dt * np.add(accelnew, accel)
    DOFarray['angular_velocity'] += 0.5 * dt * np.add(angaccelnew, angaccel)
    
    #non-particle updates
    steel_array['velocity'] += 0.5 * dt * np.add(steelaccel, steelaccelnew)
    

#----------------------------------------------------------------------------#
#energy calcs below

def calculate_kinetic_energy(i):
    total_kenergy = 0
    total_kenergy += 1/2 * m * (abs(DOFarray[i][1][0])**2 + abs(DOFarray[i][1][1])**2)
    total_kenergy += 1/2 * I * (DOFarray[i]['angular_velocity'][2])**2
    #total_kenergy += m * 9.81 * DOFarray[i]['position'][1] #also potential (ik it says kinetic)
    return total_kenergy

def calculate_interactive_energy(overlap): #guestimate specific potential energy interactions
    total_potential = 0
    total_potential += 1/2 * 6000 * overlap
    return total_potential

#----------------------------------------------------------------------------#

# Visualization
import matplotlib.patches as patches
fig, ax = plt.subplots()
particles = [patches.Circle((x, y, z), radius=r) for x, y, z in DOFarray['position']]
for particle in particles:
    ax.add_patch(particle)
steelballs = [patches.Circle((x, y, z), radius=steel_radius) for x, y, z in steel_array['position']]
for steel in steelballs:
    ax.add_patch(steel)

lines = []  # List to store line objects for each particle
ini_total_energy = 0
for i in range(n): #assumes no intital particle contact
    ini_total_energy += calculate_kinetic_energy(i)

for i in range(len(DOFarray)):
    line, = ax.plot([], [], 'k-', lw=1)  # Line to visualize rotation
    lines.append(line)

def update(frame, energy_label, deltaE_label):
    global DOFarray, steel_array
    verlet(DOFarray, steel_array, dt)

    for particle, position in zip(particles, DOFarray['position']):
        particle.center = position[:2]  # Update the position of the circle
        
    for steel, position in zip(steelballs, steel_array['position']):
        steel.center = position[:2]  # Update the center of the circle
    for i in range(len(DOFarray)):
        x_center = DOFarray['position'][i, 0]
        y_center = DOFarray['position'][i, 1]
        theta = DOFarray['angular_position'][i, 2]
        lines[i].set_data([x_center, x_center + r * np.cos(theta)], [y_center, y_center + r * np.sin(theta)])

    return [*lines, *particles, *steelballs, energy_label, deltaE_label]

energy_label = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top')
deltaE_label = ax.text(0.55, 0.95, '', transform=ax.transAxes, ha='left', va='top')

ani = animation.FuncAnimation(fig, update, frames=2500 , interval=15, blit=True,
                              fargs=(energy_label, deltaE_label))

ax.set_xlim(Wall[0] - r, Wall[2] + r)
ax.set_ylim(Wall[1] - r, Wall[3] + r)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Particle Simulation')
ax.set_aspect('equal')

#ani.save('2dSandPile2.mp4')
#----------------------------------------------------------------------------------------#

#testing

#Number of times to run the function
num_iterations = 1
#Create a loop to run the profiler multiple times
def verlet_tester():
    DOFarray = np.zeros(n, dtype=dtype)
    for i in range(n):
        DOFarray[i][0][0] = random.uniform(90, 110)
        DOFarray[i][0][1] = random.uniform(Wall[1] + r, Wall[1 + 2] - r)
        for j in range(2):
            #DOFarray[i][0][j] = random.uniform(Wall[j] + r, Wall[j + 2] - r)
            DOFarray[i][1][j] = random.uniform(-2, 2)
    for result in range(num_iterations):
        DOFarray = verlet(DOFarray, steel_array, dt)

Lp = LineProfiler()
'''
Lp.add_function(t_damping_force)
Lp.add_function(tang_forces)
Lp.add_function(normal_contact_force)
Lp.add_function(normal_damping_force)
Lp.add_function(normal_forces)
'''
Lp.add_function(master_function)
Lp.add_function(verlet)
Lp_wrapper = Lp(verlet_tester)
Lp_wrapper()
Lp.print_stats()
