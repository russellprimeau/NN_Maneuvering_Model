# heading.py
import numpy as np
import pandas as pd

# Object for simulating control of a ship toward a desired position and orientation
class heading_sim:
    def __init__(self, X0, x_d, psi_d, h, t_f, pid_rudder, e_thresh):
        """
        Initialization function
        X0 = 11 by 1 array, initial state for ship model [x, y, psi, u, v, r, e_psi, cmd_rpm, act_rpm, cmd_rudder_angle,
        act_rudder_angle]'
        psi in rad
        psi_d = desired heading in rad
        t_f = final simulation time
        h = timestep size in s
        pid_rudder = 3x1 array of PID parameters kp, ki and kd for controlling rudder angle
        e_thresh = acceptable margin of error in final heading, used to terminate simulation
        """

        self.pid_rudder = pid_rudder

        self.tf = t_f
        self.h = h
        self.x_d = x_d
        self.psi_d = psi_d
        self.e_thresh = e_thresh

        # Create time range for simulation
        self.T = np.arange(0, self.tf, self.h)

        # Number of timesteps
        self.N = len(self.T)

        # Array of state vector X at each timestep: rows 0:3->eta, 3:6->nv, 6:9 e_pos_x, e_pos_y, e_psi in body frame,
        # 9->cmd_rpm, 10->act_rpm, 11->cmd_rudder, 12->act_rudder
        self.X = np.zeros((11, self.N))

        # Set initial condition of state vector
        self.X[:, 0] = X0

        # Mass matrix from the problem statement
        self.M = np.array([[1.0e7, 0, 0],
                          [0, 1.0e7, 8.4e6],
                          [0, 8.4e6, 5.8e9]])

        # Invert mass matrix for calculations
        self.M_inv = np.linalg.inv(self.M)

        # Damping matrix from the problem statement
        self.D = np.array([[3.0e5, 0, 0],
                          [0, 5.5e5, 6.0e5],
                          [0, 6.0e5, 1.3e8]])

        # Nominal force vs. RPM data from the problem statement
        Dt = pd.read_csv('rpm.txt', names=['rpm', 'f_coef'])
        # Generate a 5th order polynomial function from the data to approximate the relationship
        self.rpm_poly = np.polyfit(Dt.rpm, Dt.f_coef, deg=5)

        # Propulsion coefficient data from the problem statement
        Dr = pd.read_csv('rudder.txt',
                         names=['angle_lift', 'lift_coef', 'angle_prop', 'prop_coef'])
        # Generate 5th order polynomial functions from the data to approximate the relationships
        self.l_poly = np.polyfit(Dr.angle_lift, Dr.lift_coef, deg=5)
        self.p_poly = np.polyfit(Dr.angle_prop, Dr.prop_coef, deg=5)

    # Run simulation using PID rudder control
    def run(self):
        """
        run one simulation from initial state to final state (desired heading, etc.)
        :return: flag for successful completion and the final state vector X
        """
        success = 0  # flag to indicate successful docking to x_d

        # Initialize variables for I and D error in shaft speed and rudder angle
        e_rpm_previous = 0
        e_rpm_int = 0

        e_rudder_previous = 0
        e_rudder_int = 0

        for i in range(self.N-1):
            # Command constant shaft speed
            cmd_rpm = 80

            # Read in previously calculated state for current timestep
            eta = self.X[:3, i]  # x, y, psi (world frame position and heading)
            nu = self.X[3:6, i]  # u, v, r (ship frame velocity and yaw rate)
            psi = eta[-1]  # psi is the yaw angle in rad

            # Update to error signal for current timestep
            e_psi = self.X[6, i]

            # Calculate the integral of the heading error
            e_rudder_int = e_rudder_int + e_psi
            # Calculate the rudder command using PID control on the heading error signal
            cmd_rudder = np.rad2deg(-self.pid_rudder[0] * e_psi - self.pid_rudder[1] * e_rudder_int * self.h
                                    - self.pid_rudder[2] * (e_psi - e_rudder_previous) / self.h)  # degrees
            # Save the current heading error for calculating the next derivative error term
            e_rudder_previous = e_psi  # rad

            # Calculate the actual rpm and rudder angle that will apply to the ship model during this timestep
            # after accounting for the limits to adjustment rates
            act = self.get_actual(
                cmd_rpm=cmd_rpm, cmd_rudder=cmd_rudder,
                act_rpm=self.X[8, i], act_rudder_angle=self.X[10, i],
                delt_t=self.h
            )

            # Calculate ship position and velocity at timestep i+1 from the position, velocity, forces and torques
            # at timestep i,  using the 4th order RK method
            k1 = self.ship(self.T[i], self.X[:6, i], act)
            k2 = self.ship(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k1, act)
            k3 = self.ship(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k2, act)
            k4 = self.ship(self.T[i] + self.h, self.X[:6, i] + self.h * k3, act)
            self.X[:6, i + 1] = self.X[:6, i] + self.h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)

            # Map heading to the interval [-pi, pi] to avoid correcting for excess rotations
            if self.X[2, i + 1] > np.pi:
               self.X[2, i + 1] -= 2*np.pi
            elif self.X[2, i + 1] <= -np.pi:
               self.X[2, i + 1] += 2*np.pi

            # Calculate heading error signal and map to the interval [-pi, pi] to avoid correcting for excess rotations
            e_psi_future = self.X[2, i+1] - self.psi_d
            if e_psi_future > np.pi:
                e_psi_future -= 2*np.pi
            elif e_psi_future <= -np.pi:
                e_psi_future += 2*np.pi

            # Write the calculated error, cmd and act shaft speeds and rudder angles into the state vector
            self.X[6, i + 1] = e_psi_future
            self.X[7, i + 1] = cmd_rpm
            self.X[8, i + 1] = act[0]
            self.X[9, i + 1] = cmd_rudder
            self.X[10, i + 1] = act[1]

            # Terminate simulation when desired heading achieved
            if np.abs(e_psi) < self.e_thresh:
                success = 1
                break
        res = self.X[:, :i+2].reshape(11, -1)  # Vector of final state, since we update values to X[i+1]
        return success, res.T

    def get_force(self, rpm, rudder):
        """
        :param rpm: current shaft speed in rpm
        :param rudder: current rudder angle in degrees
        :return: vector of force in surge, sway directions in N and torque in yaw direction in Nm
        """

        # Flag shaft speed and rudder angles which are outside the ship's capabilities
        if np.abs(rpm) > 132 or np.abs(rudder) > 45:
            print('input thruster rpm or rudder angle out of range')

        arm = -41.5  # m, Lever arm of the lift and drag force on the ship's center of gravity, based on ship geometry
        nominalf = np.polyval(self.rpm_poly, rpm)
        propulsionf = np.polyval(self.p_poly, rudder) * nominalf
        liftf = np.polyval(self.l_poly, rudder) * nominalf

        # Calculate a vector of [surge force, sway force, yaw torque] for the two propeller-rudder assemblies
        # for all possible shaft speeds
        if rpm > 0:
            ft = np.array([2 * propulsionf, 2 * liftf, 2 * arm * liftf]).reshape(-1, 1)
        elif rpm == 0:
            ft = np.array([0, 0, 0]).reshape(-1, 1)
        else:
            ft = np.array([2 * nominalf, 0, 0]).reshape(-1, 1)
        return ft

    def get_actual(self, cmd_rudder, cmd_rpm, act_rudder_angle, act_rpm, delt_t):
        """
        :param cmd_rudder: command rudder angle in degrees
        :param cmd_rpm: command shaft speed in RPM
        :param act_rudder_angle: current rudder angle in degrees
        :param act_rpm: current shaft speed in RPM
        :param delt_t: time step in s
        :return: actual rudder anger in degree and actual RPM
        """

        # Implement physical limits given in the problem statement
        rudder_max = 45  # degrees, Maximum rudder angle
        rudder_dot_max = 3.7  # degrees/s, Maximum rudder adjustment rate

        rpm_max = 132  # RPM, Shaft speed
        rpm_dot_max = 13  # RPM/s, Maximum shaft speed adjustment rate

        # Limit the rudder response to the commanded based on the acceleration limits
        if act_rudder_angle < cmd_rudder:
            if act_rudder_angle + rudder_dot_max * delt_t > cmd_rudder:
                act_rudder_angle_ = cmd_rudder
            else:
                act_rudder_angle_ = act_rudder_angle + rudder_dot_max * delt_t
        else:
            if act_rudder_angle - rudder_dot_max * delt_t < cmd_rudder:
                act_rudder_angle_ = cmd_rudder
            else:
                act_rudder_angle_ = act_rudder_angle - rudder_dot_max * delt_t

        # Map the rudder angle to the range of physically feasible angles
        if act_rudder_angle_ > rudder_max:
            act_rudder_angle_ = rudder_max
        elif act_rudder_angle_ < -rudder_max:
            act_rudder_angle_ = -rudder_max

        # Limit the shaft speed response to the command based on the limits to shaft acceleration
        if act_rpm < cmd_rpm:
            if act_rpm + rpm_dot_max * delt_t > cmd_rpm:
                act_rpm_ = cmd_rpm
            else:
                act_rpm_ = act_rpm + rpm_dot_max * delt_t
        else:
            if act_rpm - rpm_dot_max * delt_t < cmd_rpm:
                act_rpm_ = cmd_rpm
            else:
                act_rpm_ = act_rpm - rpm_dot_max * delt_t

        # Map the shaft speed to the range of physically feasible speeds
        if act_rpm_ > rpm_max:
            act_rpm_ = rpm_max
        elif act_rpm_ < -rpm_max:
            act_rpm_ = -rpm_max

        return np.array([float(act_rpm_), float(act_rudder_angle_)])

    # Ship dynamic model: calculate velocities and accelerations based on incident forces and torques
    def ship(self, t, xship, act):
        """
        :param t: current time
        :param xship: x,y,psi, u,v,r, i.e., [eta nv]'
        :param act: array of current shaft speed and rudder angle in rpm and degrees
        :return: time derivative of state vector of ship (position, yaw, speed, yaw rate)
        """

        # Reshape the position and velocity vector for linear algebra
        xship = np.array(xship).reshape(-1, 1)

        # Calculate forces on the ship in the current timestep
        F = self.get_force(act[0], act[1])  # return 3 by 1 array

        # Compute rotation matrix for converting between world and body frame at the current heading
        tmp_psi = xship[2, 0]
        R = np.array([[np.cos(tmp_psi), -np.sin(tmp_psi), 0],
                      [np.sin(tmp_psi), np.cos(tmp_psi), 0],
                      [0, 0, 1]], dtype=float)

        # Rearrange the arrays of the ships state and characteristics to simplify linear algebra functions
        r1 = np.hstack((np.zeros((3, 3)), R))
        r2 = np.hstack((np.zeros((3, 3)), -np.dot(self.M_inv, self.D)))
        A = np.vstack((r1, r2))
        B = np.vstack((np.zeros((3, 1)), np.dot(self.M_inv, F)))

        # Calculate the ship's state at the next timestep based on the current forces
        xplus = (np.dot(A, xship) + B).T

        return xplus

    # Run simulation using NN for control instead of PID
    def run_NN_control(self, mlp):
        """
        run for one instance of NN controller for docking operation
        :param mlp, multilayer perceptron neural network model for rudder angle prediction in sklearn format
        """

        # The NN uses u,v,r, and e_psi to predict rudder angle
        success = 0  # flag for indicating successful docking to x_d
        for i in range(self.N-1):
            # Current state
            cmd_rpm = 80  # Constant shaft speed command
            rudder_input = np.array([self.X[3, i], self.X[4, i], self.X[5, i], self.X[6, i]])
            # Predict rudder command
            cmd_rudder = mlp.predict(rudder_input.reshape(1, -1))

            # Calculate the actual rpm and rudder angle that will apply to the ship model during this timestep
            # from the commands after accounting for the limits to adjustment rates
            act = self.get_actual(
                cmd_rpm=cmd_rpm, cmd_rudder=cmd_rudder,
                act_rudder_angle=self.X[10, i], act_rpm=self.X[8, i],
                delt_t=self.h
            )

            # Calculate ship position and velocity at timestep i+1 from the position, velocity, forces and torques
            # at timestep i,  using the 4th order RK method
            k1 = self.ship(self.T[i], self.X[:6, i], act)
            k2 = self.ship(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k1, act)
            k3 = self.ship(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k2, act)
            k4 = self.ship(self.T[i] + self.h, self.X[:6, i] + self.h * k3, act)
            self.X[:6, i + 1] = self.X[:6, i] + self.h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)

            # Map heading to the interval [-pi, pi] to avoid correcting for excess rotations
            if self.X[2, i + 1] > np.pi:
                self.X[2, i + 1] -= 2 * np.pi
            elif self.X[2, i + 1] <= -np.pi:
                self.X[2, i + 1] += 2 * np.pi

            # Calculate heading error signal and map to the interval [-pi, pi] to avoid correcting for excess rotations
            e_psi_future = self.X[2, i + 1] - self.psi_d
            if e_psi_future > np.pi:
                e_psi_future -= 2*np.pi
            elif e_psi_future <= -np.pi:
                e_psi_future += 2*np.pi
            self.X[6, i + 1] = e_psi_future

            # Write the calculated cmd_ and act_ rpm and rudder angle to the state vector X for timestep i+1
            self.X[7, i + 1] = cmd_rpm
            self.X[8, i + 1] = act[0]
            self.X[9, i + 1] = cmd_rudder
            self.X[10, i + 1] = act[1]

            # Terminate simulation when desired heading achieved
            if np.abs(e_psi_future) < self.e_thresh:
                success = 1
                break
        res = self.X[:, :i+2].reshape(11, -1)  # since we update values to X[i+1]
        return success, res.T
