import numpy as np
from gym import spaces
import math
from sumo_utils import true_acceleration
import random


class TrafficSignal:
    def __init__(
            self,
            ts_id: str,
            yellow_time: int,
            simulation_time: float,
            delta_rs_update_time: int,
            reward_fn: str,
            sumo
    ):
        self.ts_id = ts_id
        self.yellow_time = yellow_time
        self.simulation_time = simulation_time
        self.delta_rs_update_time = delta_rs_update_time
        # reward_state_update_time
        self.rs_update_time = 0
        self.reward_fn = reward_fn
        self.sumo = sumo
        self.green_phase = None
        self.yellow_phase = None
        self.end_min_time = 0
        self.end_max_time = 0
        self.accel = 0.73
        self.decel = 1.67
        self.all_phases = self.sumo.trafficlight.getAllProgramLogics(ts_id)[0].phases
        self.all_green_phases = [phase for phase in self.all_phases if 'y' not in phase.state]
        self.num_green_phases = len(self.all_green_phases)
        self.lanes_id = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.ts_id)))
        self.lanes_length = {lane_id: self.sumo.lane.getLength(lane_id) for lane_id in self.lanes_id}
        # self.attribute_to_method = {
        #     'maxspeed': 'getMaxSpeed',
        #     'tau': 'getTau',
        #     'accel': 'getAccel',
        #     'mingap': 'getMinGap'
        # }

        # IDM Parameters
        self.delta = 4  # Default value for IDM delta
        self.iterations = 10  # Number of substeps for iterative updates
        self.time_step = 1.0  # Simulation time step (s)
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.lanes_id), dtype=np.float32),
            high=np.ones(len(self.lanes_id), dtype=np.float32))
        self.action_space = spaces.Discrete(self.num_green_phases)
        self.last_measure = 0
        self.dict_lane_veh = None


    def compute_density(self, den, tau, mingap, maxspeed):
        return np.array(
            [i1 * (1 + (40 - i2) / 400 + i3 / 30 + i4 / 100) / 1.2 for i1, i2, i3, i4 in zip(den, maxspeed, tau, mingap)],
            dtype=np.float32
        )

    def compute_state(self):
        """
        Compute the current state (density) using IDM parameters.
        """
        # 获取车道密度
        den = self.get_lanes_density()

        # 从 IDM 参数中获取 tau, mingap, maxspeed
        tau_list, mingap_list, maxspeed_list, _ = self.get_idm_parameters()

        # 计算 transformed density
        return self.compute_density(den, tau_list, mingap_list, maxspeed_list)

    def compute_next_state(self):
        """
        Compute the next state (density) if the simulation time has reached the update time.
        """
        current_time = self.sumo.simulation.getTime()
        if current_time >= self.rs_update_time:
            # 获取车道密度
            den = self.get_lanes_density()

            # 从 IDM 参数中获取 tau, mingap, maxspeed
            tau_list, mingap_list, maxspeed_list, _ = self.get_idm_parameters()

            # 计算 transformed density
            return self.compute_density(den, tau_list, mingap_list, maxspeed_list)
        else:
            return None


    def change_phase(self, new_green_phase):
        """
        :param new_green_phase:
        :return: do_action -> the real action operated; if is None, means the new_green_phase is not appropriate,
        need to choose another green_phase and operate again
        """
        # yellow_phase has not finished yet
        # yellow_phase only has duration, no minDur or maxDur

        # do_action mapping (int -> Phase)
        new_green_phase = self.all_green_phases[new_green_phase]
        do_action = new_green_phase
        current_time = self.sumo.simulation.getTime()
        if self.yellow_phase is not None:
            if current_time >= self.end_max_time:
                self.yellow_phase = None
                self.update_end_time()
                self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.green_phase.state)
                do_action = self.green_phase
            else:
                do_action = self.yellow_phase
        else:
            # if old_green_phase has finished
            if current_time >= self.end_min_time:
                if new_green_phase.state == self.green_phase.state:
                    if current_time < self.end_max_time:
                        do_action = self.green_phase
                    else:
                        # current phase has reached the max operation time, have to find another green_phase instead
                        other_phases = [phase for phase in self.all_green_phases if
                                        phase.state != new_green_phase.state]
                        if other_phases:
                            new_green_phase = random.choice(other_phases)
                            yellow_state = ''
                            for s in range(len(new_green_phase.state)):
                                if self.green_phase.state[s] == 'G' and new_green_phase.state[s] == 'r':
                                    yellow_state += 'y'
                                else:
                                    yellow_state += self.green_phase.state[s]
                            self.yellow_phase = self.sumo.trafficlight.Phase(self.yellow_time, yellow_state)
                            self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.yellow_phase.state)
                            self.green_phase = new_green_phase
                            self.rs_update_time = current_time + self.yellow_time + self.delta_rs_update_time  # 更新奖励时间
                            self.update_end_time()
                            do_action = self.yellow_phase
                        else:
                            do_action = None
                else:
                    # need to set a new plan(yellow + new_green)
                    yellow_state = ''
                    for s in range(len(new_green_phase.state)):
                        if self.green_phase.state[s] == 'G' and new_green_phase.state[s] == 'r':
                            yellow_state += 'y'
                        else:
                            yellow_state += self.green_phase.state[s]
                    self.yellow_phase = self.sumo.trafficlight.Phase(self.yellow_time, yellow_state)
                    self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.yellow_phase.state)
                    self.green_phase = new_green_phase
                    self.rs_update_time = current_time + self.yellow_time + self.delta_rs_update_time  # update reward after 10 seconds of the operated action
                    self.update_end_time()
                    do_action = self.yellow_phase
            else:
                do_action = self.green_phase

        if do_action is None:
            return None

        # do_action mapping (Phase -> int)
        if 'y' in do_action.state:
            do_action = -1
        else:
            for i, green_phase in enumerate(self.all_green_phases):
                if do_action.state == green_phase.state:
                    do_action = i
                    break

        return do_action

    def update_end_time(self):
        current_time = self.sumo.simulation.getTime()
        if self.yellow_phase is None:
            self.end_min_time = current_time + 15
            self.end_max_time = current_time + 60
            # self.end_min_time = current_time + self.green_phase.minDur
            # self.end_max_time = current_time + self.green_phase.maxDur
        else:
            self.end_min_time = current_time + self.yellow_time
            self.end_max_time = current_time + self.yellow_time

    def compute_reward(self, start, do_action):
        update_reward = False
        current_time = self.sumo.simulation.getTime()
        if current_time >= self.rs_update_time:
            self.rs_update_time = self.simulation_time + self.delta_rs_update_time
            update_reward = True
        if self.reward_fn == 'choose-min-waiting-time':
            return self._choose_min_waiting_time(start, update_reward, do_action)
        else:
            return None

    def _choose_min_waiting_time(self, start, update_reward, do_action):
        if start:
            self.dict_lane_veh = {}
            for lane_id in self.lanes_id:
                veh_list = self.sumo.lane.getLastStepVehicleIDs(lane_id)
                wait_veh_list = [veh_id for veh_id in veh_list if
                                 self.sumo.vehicle.getAccumulatedWaitingTime(veh_id) > 0]
                self.dict_lane_veh[lane_id] = len(wait_veh_list)
            # merge wait_time by actions
            dict_action_wait_time = [self.dict_lane_veh['n_t_0'] + self.dict_lane_veh['s_t_0'],
                                     self.dict_lane_veh['n_t_1'] + self.dict_lane_veh['s_t_1'],
                                     self.dict_lane_veh['e_t_0'] + self.dict_lane_veh['w_t_0'],
                                     self.dict_lane_veh['e_t_1'] + self.dict_lane_veh['w_t_1']]
            best_action = np.argmax(dict_action_wait_time)
            if best_action == do_action:
                self.last_measure = 1
            else:
                self.last_measure = -1

        if update_reward:
            return self.last_measure
        else:
            return None

    def compute_queue(self):
        total_queue = 0
        for lane_id in self.lanes_id:
            total_queue += self.sumo.lane.getLastStepHaltingNumber(lane_id)
        return total_queue


    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane_id) / (
                    self.lanes_length[lane_id] / vehicle_size_min_gap))
                for lane_id in self.lanes_id]

    def _rmse_cluster(self, data, v0, d_min, a, b, T):
        """
        Compute RMSE for a single trajectory data point using IDM parameters.
        :param data: A single trajectory data point (list)
                     [ego_speed, pred_speed, gap_to_pred, ego_acceleration, leader_flag]
        :param v0: Desired maximum speed (m/s)
        :param d_min: Minimum gap (m)
        :param a: Maximum acceleration (m/s^2)
        :param b: Comfortable deceleration (m/s^2)
        :param T: Desired time headway (s)
        :return: RMSE value for the data point
        """
        # Extract data
        ego_speed = data[0]  # Ego vehicle speed
        pred_speed = data[1]  # Preceding vehicle speed
        gap_to_pred = data[2]  # Gap to preceding vehicle
        ego_acc = data[3]  # Ego vehicle's actual acceleration (from SUMO/TRACI)
        leader_flag = data[4]  # Whether there is a preceding vehicle (1 for yes, 0 for no)

        # Set IDM parameters
        self.max_speed = v0
        self.min_gap = d_min
        self.headway_time = T

        acc = self.idm(ego_speed, pred_speed, gap_to_pred, self.max_speed, self.min_gap
                           , self.accel, self.decel, self.headway_time, leader_flag)

        # Compute RMSE for the acceleration
        return (acc - ego_acc) ** 2

    def idm(self, v, v_l, d, v0, d_min, a, b, T, leader_flag):
        """
        Intelligent Driver Model (IDM) function.
        Parameters:
        v : float
            Current speed of the vehicle.
        v0 : float
            Desired speed of the vehicle.
        vl : float
            speed of the leader vehicle.
        d : float
            Gap to the vehicle in front.
        a : float
            Maximum acceleration.
        b : float
            Comfortable deceleration.
        T : float
            Desired time headway.
        delta : float
            Acceleration exponent.

        Returns:
        float
            Calculated acceleration based on IDM.
        """

        if leader_flag == 1:
            d = max(d, 0.1)
            d_star = d_min + max(0, v * T - (v * (v_l - v)) / (2 * np.sqrt(a * b)))
            acc = max(a * (1 - (v / v0) ** 4 - (d_star / d) ** 2), -4)
        else:
            acc = max(a * (1 - (v / v0) ** 4), -4)
        return acc


    def get_idm_parameters(self):
        """
        Dynamically compute IDM parameters (v0, d_min, a, b, T) using Method 3 and SUMO trajectory data.
        """
        tau_list, mingap_list, maxspeed_list, acc_list = [], [], [], []

        for lane_id in self.lanes_id:
            vehicles = self.sumo.lane.getLastStepVehicleIDs(lane_id)
            if len(vehicles) == 0:
                # If there are no vehicles on the lane, use default values
                tau_list.append(1.0)
                mingap_list.append(2.5)
                maxspeed_list.append(30.7)
                acc_list.append(2.6)
                continue

            # Store parameters for all vehicles on the lane
            lane_tau, lane_mingap, lane_maxspeed, lane_acc = [], [], [], []

            for vehicle_id in vehicles:
                # Get vehicle information
                # self.max_speed = self.sumo.vehicle.getMaxSpeed(vehicle_id)
                # self.min_gap = self.sumo.vehicle.getMinGap(vehicle_id)
                # self.headway_time = self.sumo.vehicle.getTau(vehicle_id)
                ego_v = self.sumo.vehicle.getSpeed(vehicle_id)  # Ego vehicle speed
                leader_info = self.sumo.vehicle.getLeader(vehicle_id)  # Preceding vehicle info (ID and distance)
                ego_a = self.sumo.vehicle.getAcceleration(vehicle_id)
                if leader_info is not None and leader_info[0] != "":  # If there is a preceding vehicle
                    leader_id = leader_info[0]  # Preceding vehicle ID
                    leader_v = self.sumo.vehicle.getSpeed(leader_id)  # Preceding vehicle speed
                    gap = leader_info[1]  # Distance to preceding vehicle (including minGap)
                    ###true value from
                    # ego_a = true_acceleration(ego_v, gap, leader_v, self.accel, self.decel, self.max_speed, self.min_gap, self.headway_time, self.delta, self.iterations, self.time_step)
                    # Trajectory data includes preceding vehicle info
                    trajectory_data = [ego_v, leader_v, gap, ego_a, 1]
                else:  # If there is no preceding vehicle
                    # ego_a = true_acceleration(ego_v, None, None, self.accel, self.decel, self.max_speed, self.min_gap,
                    #                           self.headway_time, self.delta, self.iterations, self.time_step)
                    # Trajectory data does not include preceding vehicle info
                    trajectory_data = [ego_v, None, None, ego_a, 0]

                ########### use 5fast.py here to identify driving clusters
                # Prototype IDM parameters
                params_prototype = [
                    [34.1, 0.18, 0.73, 1.67, 0.97],
                    [40.0, 2.66, 0.73, 1.67, 0.70],
                    [40.0, 0.90, 0.73, 1.67, 1.45],
                    [10.6, 4.26, 0.73, 1.67, 0.30]
                ]

                # Find the best IDM parameters for the vehicle
                min_rmse = float('inf')
                best_params = None
                for params in params_prototype:
                    v0, d_min, a, b, T = params
                    current_rmse = self._rmse_cluster(trajectory_data, v0, d_min, a, b, T)
                    if current_rmse < min_rmse:
                        min_rmse = current_rmse
                        best_params = params

                # Extract the best parameters
                v0, d_min, a, b, T = best_params
                lane_tau.append(T)  # T is reaction time (tau)
                lane_mingap.append(d_min)  # d_min is minimum gap
                lane_maxspeed.append(v0)  # v0 is maximum speed
                lane_acc.append(a)  # a is maximum acceleration

            # Take the average of all vehicle parameters on the lane
            tau_list.append(np.mean(lane_tau))
            mingap_list.append(np.mean(lane_mingap))
            maxspeed_list.append(np.mean(lane_maxspeed))
            acc_list.append(np.mean(lane_acc))

        return tau_list, mingap_list, maxspeed_list, acc_list