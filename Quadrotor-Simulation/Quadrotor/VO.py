

import numpy as np
from scipy import spatial



# hyperparameters
sphere_radius = 0.09    # this value is read by the drone specifications in Quadrotor/Params.py
safe_radius = sphere_radius * 100
dangerous_radius = sphere_radius*10

def ComputeVelocityObstacle(state, des_state, rev_state, dt):
    """
    1) Evaluates whether close enough dynamic obstacles are in a collision course (according to a radius R).
    2) Reads as input the positions of two dynamic obstacles: one in 'STATE' and one 'REV_STATE' variables.
    3) predicts according to its static velocity if there will be a collision course
    4) assumes obstacles are spheres of radius R

    OUTPUT:
    5) if there is a collision, assigns a new collision free velocity, to the Desired State
    else, the desired state remains unchanged.
    """

    if DyanmicObstaclesInCloseRange(dangerous_radius,state, rev_state):
        if DroneIsInACollisonCourseWithTheDynamicObstacle(des_state, rev_state):
            collisionFreeVelocity = Orth_RetrieveCollisionFreeVelocity(state, des_state, rev_state, dt)
            EstimatedCollisionFreeDesiredPosition = EstimateDesiredPosition(state, collisionFreeVelocity, dt)
            return EstimatedCollisionFreeDesiredPosition
        else:
            return des_state.pos
    else:
        return des_state.pos

def Orth_RetrieveCollisionFreeVelocity(state, des_state, rev_state, dt):
    vector = np.cross(des_state.vel, rev_state.vel)
    norm = np.linalg.norm(vector)
    normal_array = vector / norm
    return normal_array * np.linalg.norm(des_state.vel)


def DyanmicObstaclesInCloseRange(dangerous_radius,state, rev_state):
    dist = np.linalg.norm(state.pos - rev_state.pos)
    if dist < dangerous_radius:
        return True
    else:
        return False

def DroneIsInACollisonCourseWithTheDynamicObstacle(des_state, rev_state):
    dist = np.linalg.norm(des_state.pos - rev_state.pos)
    if dist < dangerous_radius:
        return True
    else:
        return False


def RandomNumberUniform():
    val = np.random.uniform(low=0.0, high=1.0)
    return val/np.cos(val)

def SampleVelUniform(state, des_state, max_vel):
    x = RandomNumberUniform()
    y = RandomNumberUniform()
    z = RandomNumberUniform()
    vector = np.array((x,y,z))
    norm = np.linalg.norm(vector)
    normal_array = vector / norm
    return  max_vel * normal_array

def SampleVelTriangular(state, des_state, max_vel):
    # https://www.omnicalculator.com/math/angle-between-two-vectors
    # angle= np.random.triangular(left, mode, right, size=None)
    # angle = (2*np.pi - 0) * np.random.random_sample() + 0
    # # module = (max_vel - 0.9 *max_vel)  * np.random.random_sample() + 0.5 *max_vel
    # module = max_vel
    # vx = np.cos(angle) * module
    # vy = np.sin(angle) * module
    # V = np.array([vx, vy, vz])
    pass

def CollisionFree(state, rev_state, sampled_V, dt):
    tau = dt
    n_steps = 10
    # check about no collision with other moving obstacles
    vj = rev_state.vel # velocity of non collaborating moving obstacle
    vi = sampled_V # velocity of ourself
    pj =  rev_state.pos # pos of obstacle
    pi =  state.pos # pos of us
    rj = sphere_radius  # radius of obstacle
    ri = sphere_radius  # radius of obstacle
    times = np.linspace(start=0, stop=tau, num=n_steps)
    truth = np.zeros(n_steps, dtype=bool)
    for i in range(0, n_steps):
        a = (pi + vi*times[i])
        b = (pj + vj*times[i])
        if np.linalg.norm(a-b) > (rj+ri) :
            truth[i] = True
        else:
            truth[i] = False

    if np.all(truth==True):
        return True
    else:
        # print("false")
        return False

def RetrieveCollisionFreeVelocity(state, des_state, rev_state, dt):
    # desired_velocity = ComputeDesiredVelocity(state, des_state)
    desired_velocity = des_state.vel

    # fill an array of N collision free velocities
    num_cf_V = 3
    cf_V = []
    count_cv_V = 0
    iterations = 0
    while iterations < 10:
        # sample velocity
        sampled_V = SampleVelUniform(state, des_state, max_vel=np.linalg.norm(des_state.vel))
        # check if velocity leads to collision
        if CollisionFree(state, rev_state, sampled_V, dt):
            print("collision Free")
            cf_V.append(sampled_V)
            count_cv_V += 1
        iterations += 1

    # get best collision free velocity from the array (according to Cosine Similarity metric)
    cosine_similarity = np.zeros(num_cf_V)
    for i in range(0, num_cf_V):
        cosine_similarity[i] = spatial.distance.cosine(desired_velocity, cf_V[i], w=None)
    minIdx = np.argmin(cosine_similarity)
    VO_velocity = cf_V[minIdx]
    return VO_velocity


def EstimateDesiredPosition(state, collisionFreeVelocity, dt):
    ending_position = state.pos + collisionFreeVelocity
    return ending_position


def ComputeDesiredVelocity(state, des_state):
    p1 = state.pos
    p2 = des_state.pos
    velocity =  p2 - p1
    return velocity