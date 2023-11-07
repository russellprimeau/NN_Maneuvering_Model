import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../data file/docking_data.csv')

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
    # a 0 value in these columns of the dataframe
    if np.linalg.norm(data.iloc[i, 4:6]) == 0 \
            or i==len(data)-1:
        # print(i, 'th pos=',data.iloc[i, :6])
        idx_end = i
        ax1.plot(data.iloc[idx_start:idx_end, 1], data.iloc[idx_start:idx_end, 2])  # trajectory
        ax2.plot(np.arange(idx_end - idx_start), np.rad2deg(data.iloc[idx_start:idx_end, 3]))  # heading
        ax3.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 4], label='Surge speed')  # u
        ax4.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 5], label='Sway speed')  # v
        ax5.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 11])  # act rpm
        ax6.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 13])  # act rudder

        idx_start = idx_end
        #break

ax1.set_title('Trajectory')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.grid()

ax2.set_title('Heading')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('Heading (deg)')
ax2.grid()

ax3.set_title('Surge speed')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('u (m/s)')
ax3.grid()

ax4.set_title('Sway speed')
ax4.set_xlabel('time (s)')
ax4.set_ylabel('v (m/s)')
ax4.grid()

ax5.set_title('Actual RPM')
ax5.set_xlabel('time (s)')
ax5.set_ylabel('RPM')
ax5.grid()

ax6.set_title('Actual rudder')
ax6.set_xlabel('time (s)')
ax6.set_ylabel('Angle (deg)')
ax6.grid()
# plt.savefig('C:/data_visual', bbox_inches='tight')


fig2 = plt.figure(2)
ax7 = fig2.add_subplot(131)
ax7.grid()
ax8 = fig2.add_subplot(132)
ax8.grid()
ax9 = fig2.add_subplot(133)
ax9.grid()

idx_start = 0
idx_end = 0
for i in range(1, len(data)):
    # since u=0 and v=0 for initial condition, we can use it to find a new start of test
    if np.linalg.norm(data.iloc[i, 4:6]) == 0 \
            or i==len(data)-1:
        idx_end = i
        ax7.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 7])  # e_pos_x
        ax8.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 8])  # e_pos_y
        ax9.plot(np.arange(idx_end - idx_start), np.rad2deg(data.iloc[idx_start:idx_end, 9]))  # e_psi
        idx_start = idx_end
        #break

ax7.set_title('e_pos_x')
ax7.set_xlabel('time (s)')
ax7.set_ylabel('e_x (m)')

ax8.set_title('e_pos_y')
ax8.set_xlabel('time (s)')
ax8.set_ylabel('e_y (m)')

ax9.set_title('e_psi')
ax9.set_xlabel('time (s)')
ax9.set_ylabel('e_psi (deg)')

plt.show()
