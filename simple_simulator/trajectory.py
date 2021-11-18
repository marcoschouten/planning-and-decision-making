import numpy as np

# start_pos 3 by 1 vector
# pos 3 by 1 vector

def tj_from_line(start_pos, end_pos, time_ttl, t_c):
    v_max = (end_pos - start_pos) * 2 / time_ttl
    if (t_c >= 0 and t_c < time_ttl/2):
        vel = v_max*t_c/(time_ttl/2)
        pos = start_pos + t_c*vel/2
        acc = v_max/(time_ttl/2)
    elif (t_c >= time_ttl/2 and t_c <= time_ttl):
        vel = v_max * (time_ttl - t_c) / (time_ttl / 2)
        pos = end_pos - (time_ttl - t_c) * vel / 2
        acc = - v_max/(time_ttl / 2)
    else:
        if (type(start_pos) == int) or (type(start_pos) == float):
            pos, vel, acc = 0, 0, 0
        else:
            pos, vel, acc = np.zeros(start_pos.shape), np.zeros(
                start_pos.shape), np.zeros(start_pos.shape)
    return pos, vel, acc

# hover jrajectory
def hover(t):
    pos, vel, acc, final_pos = np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1))
    return pos, vel, acc, final_pos

# diamond jrajectory
def diamond(t):
    T1, T2, T3, T4 = 3, 3, 3, 3
    points = []
    points.append(np.zeros((3, 1)))
    points.append(np.array([[0], [np.sqrt(2)], [np.sqrt(2)]]))
    points.append(np.array([[1], [0], [2*np.sqrt(2)]]))
    points.append(np.array([[1], [-np.sqrt(2)], [np.sqrt(2)]]))
    points.append(np.array([[1], [0], [0]]))

    if (0 < t) and (t <= T1):
        pos, vel, acc = tj_from_line(points[0], points[1], T1, t)
    elif (T1 < t) and (t <= (T1+T2)):
        pos, vel, acc = tj_from_line(points[1], points[2], T2, t-T1)
    elif ((T1 + T2) < t) and (t <= (T1 + T2 + T3)):
        pos, vel, acc = tj_from_line(points[2], points[3], T3, t - (T1 + T2))
    elif ((T1 + T2 + T3) < t) and (t <= (T1 + T2 + T3 + T4)):
        pos, vel, acc = tj_from_line(
            points[3], points[4], T4, t - (T1 + T2 + T3))
    elif (t > (T1 + T2 + T3 + T4)):
        pos, vel, acc = points[4], np.zeros((3, 1)), np.zeros((3, 1))
    else:
        pos, vel, acc = np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1))

    final_pos = points[-1]
    return pos, vel, acc, final_pos

# circle jrajectory
def circle(t):
    T = 30
    radius = 5
    angle, _, _ = tj_from_line(0, 2*np.pi, T, t)
    pos = np.array([[radius*np.cos(angle)], [radius*np.sin(angle)],
                   [2.5*angle/(2*np.pi)]]) + np.array([[-radius], [0], [0]])
    angle2, _, _ = tj_from_line(0, 2*np.pi, T, t+0.01)
    pos2 = np.array([[radius*np.cos(angle2)], [radius*np.sin(angle2)],
                    [2.5*angle2/(2*np.pi)]]) + np.array([[-radius], [0], [0]])
    vel = (pos2 - pos)/0.01
    acc = np.zeros((3, 1))
    final_point = np.array([[radius], [0], [2.5]]) + \
        np.array([[-radius], [0], [0]])
    return pos, vel, acc, final_point

# TUD jrajectory
def pos_from_angle(a, ori_point=np.zeros((3, 1)), radius=2, displacement=np.array([[0], [2], [0]])):
    pos = np.array([[0], [radius*np.cos(a)], [radius*np.sin(a)]]
                   ) + ori_point + displacement
    return pos

def get_vel(t, T, dt, ori_point=np.zeros((3, 1)), s_ang=np.pi, e_ang=2*np.pi):
    angle1, _, _ = tj_from_line(s_ang, e_ang, T, t)
    pos1 = pos_from_angle(angle1, ori_point)
    angle2, _, _ = tj_from_line(s_ang, e_ang, T, t+dt)
    vel = (pos_from_angle(angle2, ori_point) - pos1)/dt
    return vel

def TUD(t):
    points = []
    # Start of T
    T1, T2, T3 = 3, 3, 3   # T
    points.append(np.zeros((3, 1)))
    points.append(np.array([[0], [0], [7]]))
    points.append(np.array([[0], [-2], [7]]))
    points.append(np.array([[0], [2], [7]]))
    # T to U
    T4 = 3
    # Start of U
    T5, T6, T7 = 3, 4, 3
    points.append(np.array([[0], [3], [7]]))
    points.append(np.array([[0], [3], [2]]))
    points.append(np.array([[0], [7], [2]]))
    points.append(np.array([[0], [7], [7]]))
    # U to D
    T8 = 3
    # Start of D
    T9, T10 = 4, 5
    points.append(np.array([[0], [8], [7]]))
    points.append(np.array([[0], [8], [0]]))
    points.append(np.array([[0], [8], [7.1]]))

    # start of T
    if (0 < t) and (t <= T1):
        pos, vel, acc = tj_from_line(points[0], points[1], T1, t)
    elif (T1 < t) and (t <= (T1+T2)):
        pos, vel, acc = tj_from_line(points[1], points[2], T2, t-T1)
    elif ((T1 + T2) < t) and (t <= (T1 + T2 + T3)):
        pos, vel, acc = tj_from_line(points[2], points[3], T3, t - (T1 + T2))
    # from T to U
    elif ((T1 + T2 + T3) < t) and (t <= (T1 + T2 + T3 + T4)):
        pos, vel, acc = tj_from_line(
            points[3], points[4], T4, t - (T1 + T2 + T3))
    # start of U
    elif ((T1 + T2 + T3 + T4) < t) and (t <= (T1 + T2 + T3 + T4 + T5)):
        pos, vel, acc = tj_from_line(
            points[4], points[5], T5, t - (T1 + T2 + T3 + T4))
    elif ((T1 + T2 + T3 + T4 + T5) < t) and (t <= (T1 + T2 + T3 + T4 + T5 + T6)):
        # point[5] to point[6]
        angle, _, _ = tj_from_line(
            np.pi, 2*np.pi, T6, t - (T1 + T2 + T3 + T4 + T5))
        pos = pos_from_angle(angle, points[5])
        vel = get_vel(t - (T1 + T2 + T3 + T4 + T5), T6, 0.01, points[5])
        acc = np.zeros((3, 1))
    elif ((T1 + T2 + T3 + T4 + T5 + T6) < t) and (t <= (T1 + T2 + T3 + T4 + T5 + T6 + T7)):
        pos, vel, acc = tj_from_line(
            points[6], points[7], T7, t - (T1 + T2 + T3 + T4 + T5 + T6))
    # from U to D
    elif ((T1 + T2 + T3 + T4 + T5 + T6 + T7) < t) and (t <= (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)):
        pos, vel, acc = tj_from_line(
            points[7], points[8], T8, t - (T1 + T2 + T3 + T4 + T5 + T6 + T7))
    # start of D
    elif ((T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8) < t) and (t <= (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)):
        pos, vel, acc = tj_from_line(
            points[8], points[9], T9, t - (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8))
    elif ((T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9) < t) and (t <= (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)):
        angle, _, _ = tj_from_line(-0.5*np.pi, 0.5*np.pi, T10,
                                   t - (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9))
        radius = 3.5
        pos = np.array([[0], [radius*np.cos(angle)], [radius*np.sin(angle)]]
                       ) + points[9] + np.array([[0], [0], [radius]])

        angle1, _, _ = tj_from_line(-0.5*np.pi, 0.5*np.pi, T10,
                                    t - (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9))
        pos1 = np.array([[0], [radius*np.cos(angle1)], [radius*np.sin(angle1)]]
                        ) + points[9] + np.array([[0], [0], [radius]])
        angle2, _, _ = tj_from_line(-0.5*np.pi, 0.5*np.pi, T10,
                                    t - (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9) + 0.01)
        pos2 = np.array([[0], [radius*np.cos(angle2)], [radius*np.sin(angle2)]]
                        ) + points[9] + np.array([[0], [0], [radius]])

        vel = (pos2 - pos1)/0.01
        acc = np.zeros((3, 1))
    elif (t > (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)):
        pos, vel, acc = points[-1], np.zeros((3, 1)), np.zeros((3, 1))
    else:
        pos, vel, acc = np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1))

    final_pos = points[-1]
    return pos, vel, acc, final_pos


# flat_output
# x, x_dot, x_ddot, yaw, yaw_dot
def generate_trajec(t, trajec=TUD):
    x, x_dot, x_ddt, final_x = trajec(t)
    yaw = 0
    yaw_dot = 0
    flat_output = {'x': x, 'x_dot': x_dot,
                   'x_ddt': x_ddt, 'yaw': yaw, 'yaw_dot': yaw_dot}
    return flat_output, final_x
