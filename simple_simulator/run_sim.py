import quadrotor
import controller
import trajectory
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

# start simulation
if __name__=="__main__":
    quad_model = quadrotor.Quadrotor()
    quad_model.reset()
    quad_controller = controller.PDcontroller()
    time_step = 1e-2
    simu_time = 60
    num_iter = int(simu_time / time_step)
    cur_time = 0

    # init performance values
    accu_error_pos = np.zeros((3, 1))
    error_pos = np.zeros((3, 1))
    total_time = 0
    square_ang_vel = np.zeros((4, ))
    real_trajectory = {'x': [], 'y': [], 'z': []}
    cmd_trajectory = {'x': [], 'y': [], 'z': []}
    for i in range(num_iter):
        # Please change the trajectory [trajectory.TUD, trajectory.circle, trajectory.diamond, trajectory.hover]
        flat_output, final_x = trajectory.generate_trajec(cur_time, trajectory.diamond)    
        control_input = quad_controller.control(flat_output, quad_model.state)
        cmd_rotor_speeds = control_input["cmd_motor_speeds"]
        obs, _, _, _ = quad_model.step(cmd_rotor_speeds)
        cur_time += time_step
        if np.all(abs(obs['x'] - final_x.reshape(obs['x'].shape)) < np.full((3, 1), 1e-2)):
            total_time = cur_time
            break
        else:
            total_time = simu_time
        # record performance values
        error_pos = control_input['error_pos'] * time_step
        accu_error_pos += error_pos 
        square_ang_vel += cmd_rotor_speeds ** 2 * time_step
        real_trajectory['x'].append(obs['x'][0])
        real_trajectory['y'].append(obs['x'][1])
        real_trajectory['z'].append(obs['x'][2])
        cmd_trajectory['x'].append(flat_output['x'][0][0])
        cmd_trajectory['y'].append(flat_output['x'][1][0])
        cmd_trajectory['z'].append(flat_output['x'][2][0])
        current_state = obs
    
    # Print three required criterions
    print("Tracking performance: ", np.sum(accu_error_pos**2))
    print("Total time needed: ", total_time)
    print("Sum of square of angular velocities: ", np.sum(square_ang_vel))

    # Visualization
    start_ani = 1
    if (start_ani):
        fig = plt.figure()
        ax1 = p3.Axes3D(fig)  # 3D place for drawing
        real_trajectory['x'] = np.array(real_trajectory['x'])
        real_trajectory['y'] = np.array(real_trajectory['y'])
        real_trajectory['z'] = np.array(real_trajectory['z'])
        cmd_trajectory['x'] = np.array(cmd_trajectory['x'])
        cmd_trajectory['y'] = np.array(cmd_trajectory['y'])
        cmd_trajectory['z'] = np.array(cmd_trajectory['z'])
        point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'ro',
                        label='Quadrotor')
        line1, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]],
                        label='Real_Trajectory')
        line2, = ax1.plot(cmd_trajectory['x'][0], cmd_trajectory['y'][0], cmd_trajectory['z'][0],
                        label='CMD_Trajectory')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_title('3D animate')
        ax1.view_init(35, 35)
        ax1.legend(loc='lower right')

        def animate(i):
            line1.set_xdata(real_trajectory['x'][:i + 5])
            line1.set_ydata(real_trajectory['y'][:i + 5])
            line1.set_3d_properties(real_trajectory['z'][:i + 5])
            line2.set_xdata(cmd_trajectory['x'][:i + 5])
            line2.set_ydata(cmd_trajectory['y'][:i + 5])
            line2.set_3d_properties(cmd_trajectory['z'][:i + 5])
            point.set_xdata(real_trajectory['x'][i])
            point.set_ydata(real_trajectory['y'][i])
            point.set_3d_properties(real_trajectory['z'][i])

        ani = animation.FuncAnimation(fig=fig,
                                    func=animate,
                                    frames=len(real_trajectory['x']),
                                    interval=1,
                                    repeat=False,
                                    blit=False)
        plt.show()
