# NN_sim.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from heading import heading_sim
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor


# Read in dataset created by the PID for training and testing, of the structure
# ['x', 'y', 'psi', 'u', 'v', 'r', 'e_psi', 'cmd_rpm','act_rpm','cmd_rudder','act_rudder']
S = pd.read_csv("heading_data.csv")

# Separate data to be used for training and testing
# X (input)
rudder_data_list = S.iloc[0:-1, [4, 5, 6, 7]]  # u,v,r, e_psi in m/s, m/s, rad, rad
# Y (output)
rudder_label_list = S.iloc[1:, 10]  # rudder command in degrees

# Shuffle and split dataset for training and testing
X2, y2 = shuffle(rudder_data_list.values, rudder_label_list, random_state=0)
test_size = 0.2
x2_train, x2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=test_size)

# Create NN and normalize data
mlp_psi = make_pipeline(
    StandardScaler(),
    MLPRegressor(solver='adam', hidden_layer_sizes=(32, 32), activation='relu', max_iter=1000),
)

# Train NN on the training data
mlp_psi.fit(x2_train, y2_train.ravel())

# Test trained NN on the test data and report
print('mlp_psi.score: ', mlp_psi.score(x2_test, y2_test))

# Test control performance from new initial conditions

# Assign initial values for testing NN control
e_thresh = .001  # rad, acceptable margin of error in final heading, used to terminate simulation
psi_d = 45  # degrees, Defines target heading which will be passed to heading_sim
psi_d_rad = np.radians(psi_d)  # rad
init_x = 0  # m, initial x position
init_y = 0  # m, initial y position
test_num = 100  # Number of randomized iterations to perform
success_num = 0  # Initialize a counter for # of iterations completed successfully (superfluous)
init_rpm = 80  # rpm, constant shaft speed
psi = (np.random.random(test_num)-0.5)*(2*np.pi)  # Vector of randomized initial headings for each training scenario

# Create a state vector of position, velocity and control inputs:
# ['x', 'y', 'psi', 'u', 'v', 'r','e_x', 'e_y', 'e_psi','cmd_rpm','act_rpm','cmd_rudder','act_rudder']
df_empty = np.zeros((0, 11))

for i in range(test_num):
    x_d = np.array([0, 0])  # Defines target position which will be passed to heading_sim
    init_psi = psi[i]  # rad, initial randomized heading
    # Rotation matrix for converting position and linear velocity between body frame and world frame
    R = np.array([[np.cos(init_psi), -np.sin(init_psi)],
                  [np.sin(init_psi), np.cos(init_psi)]])
    # Error signals
    e_pos = np.linalg.inv(R).dot(np.array([[init_x], [init_y]])-x_d.reshape(-1, 1))  # m, 2x1 array, positional error
    e_psi = init_psi - psi_d_rad  # rad, heading error

    # Map the heading error signal to the interval [0, 2pi] to avoid extra rotations:
    if e_psi > np.pi:
        e_psi -= 2 * np.pi
    elif e_psi <= -np.pi:
        e_psi += 2 * np.pi

    # Vector of initial conditions for passing to heading_sim object
    x0 = [init_x, init_y, init_psi, 0, 0, 0, e_psi, init_rpm, init_rpm, 0, 0]

    # Instantiate and initialize docking_PID object for calculating trajectory under PID control from
    # initial to final condition
    dp = heading_sim(X0=np.array(x0),  # Initial conditions
                 x_d=x_d,  # Target position as a (x,y) array in m
                 psi_d=psi_d_rad,  # Target heading in rad
                 h=1,  # s, Timestep
                 t_f=2000,  # s, Maximum simulation length
                 pid_rudder=[.2, 0.001, 5],  # 3x1 array of PID parameters kp, ki and kd for cmd_rudder
                 e_thresh=.001)  # rad, acceptable margin of error in final heading, used to terminate simulation

    # Run heading_sim simulation
    success, NN_data_b = dp.run_NN_control(mlp_psi)
    # print('restemp.shape ', restemp.shape)
    # When complete, concatenate the simulated time series of position, velocity, error signals, and commands
    # to the local variable "df_empty"
    if success:
        success_num += 1
        df_empty = np.concatenate((df_empty, NN_data_b))

# Report number of successfully completed simulations at command line (superflous)
print('total success NN: ', success_num)

# Write simulation results to an external .csv data file
data = pd.DataFrame(df_empty, columns=['x', 'y', 'psi', 'u', 'v', 'r', 'e_psi', 'cmd_rpm',
                                         'act_rpm', 'cmd_rudder', 'act_rudder'])
result_file = "heading_dataNN.csv"
data.to_csv(result_file)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

idx_start = 0
idx_end = 0
for i in range(1, len(data)):
    # Since the initial condition for each simulation is u=0 and v=0, each new simulation run can be identified by
    # a 0 value in these columns of the dataframe; for the end of the last run, use the length of the data
    if np.linalg.norm(data.iloc[i, 3:5]) == 0 \
            or i==len(data)-1:
        idx_end = i
        ax1.plot(data.iloc[idx_start:idx_end, 0], data.iloc[idx_start:idx_end, 1])  # trajectory
        ax2.plot(np.arange(idx_end - idx_start), np.rad2deg(data.iloc[idx_start:idx_end, 2]))  # heading
        ax3.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 3], label='Surge speed')  # u
        ax4.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 4], label='Sway speed')  # v
        ax5.plot(np.arange(idx_end - idx_start), np.rad2deg(data.iloc[idx_start:idx_end, 6]))  # e_psi
        ax6.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 10])  # act rudder

        idx_start = idx_end

ax1.set_title('Trajectory')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.grid()

ax2.set_title('Heading')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('Heading (deg)')
ax2.grid()

ax3.set_title('Surge speed, u')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('(m/s)')
ax3.grid()

ax4.set_title('Sway speed, v')
ax4.set_xlabel('time (s)')
ax4.set_ylabel('(m/s)')
ax4.grid()

ax5.set_title('Heading error')
ax5.set_xlabel('time (s)')
ax5.set_ylabel('e_psi (deg)')
ax5.grid()

ax6.set_title('Actual rudder')
ax6.set_xlabel('time (s)')
ax6.set_ylabel('Angle (deg)')
ax6.grid()

plt.show()