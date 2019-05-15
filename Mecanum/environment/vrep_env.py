from .env_modules import vrep
import numpy as np
import time

class Env():
    def __init__(self, port):
        self.clientID = vrep.simxStart('127.0.0.1', port, True, True, 5000, 5)
        self.lidar_handles = {}
        self.lidar_list = ['fastHokuyo_sensor1', 'fastHokuyo_sensor2', 'fastHokuyo_sensor3']
        self.motor_handles = {}
        self.motor_list = ['rollingJoint_fl', 'rollingJoint_fr', 'rollingJoint_rl', 'rollingJoint_rr']
        
        self.robot_handle = vrep.simxGetObjectHandle(self.clientID, 'dualarm_mobile', vrep.simx_opmode_blocking)[1]
        self.target_handle = vrep.simxGetObjectHandle(self.clientID, 'target', vrep.simx_opmode_blocking)[1]
        for name in self.motor_list:
            self.motor_handles[name] = vrep.simxGetObjectHandle(self.clientID, name, vrep.simx_opmode_blocking)[1]
        for name in self.lidar_list:
            self.lidar_handles[name] = vrep.simxGetObjectHandle(self.clientID, name, vrep.simx_opmode_blocking)[1]

        self.goal_set = [[-4.0, -2.0], [-4.0, -1.0], \
                         [-3.0, -4.0], [-3.0, 1.0], [-3.0, 3.0],[-3.0, 4.0], \
                         [-2.0, -3.0], [-2.0, 0.0], [-2.0, 1.0], [-2.0, 2.0], [-2.0, 3.0], \
                         [-1.0, -3.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0], [-1.0, 3.0], \
                         [0.0, -1.0], [0.0, 1.0], [0.0, 3.0], [0.0, 4.0], \
                         [1.0, -1.0], [1.0, 1.0], [1.0, 3.0], [1.0, 4.0], \
                         [2.0, -3.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0], [2.0, 3.0], [2.0, 4.0], \
                         [3.0, -4.0], [3.0, -3.0], [3.0, -2.0], [3.0, -1.0], [3.0, 0.0], [3.0, 1.0], [3.0, 4.0], \
                         [4.0, -3.0], [4.0, -2.0], [4.0, -1.0], [4.0, 1.0], [4.0, 2.0]]

        vrep.simxSynchronous(self.clientID, True)

    def reset(self):
        # Choose random goal position
        goal_position = self.goal_set[np.random.randint(len(self.goal_set))] # Modify to randomize!

        vrep.simxSynchronous(self.clientID, True)
        time.sleep(0.5)
        vrep.simxPauseCommunication(self.clientID, 1)
        vrep.simxSetObjectPosition(self.clientID, self.robot_handle, -1, (0.0, 0.0, 0.15), vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(self.clientID, self.robot_handle, -1, (0.0, np.pi/2.0, np.pi), vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(self.clientID, self.target_handle, -1, goal_position, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(self.clientID, self.target_handle, -1, (0.0, np.pi/2.0, np.pi), vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.clientID, 0)
        
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

        vrep.simxPauseCommunication(self.clientID, 1)
        self.set_action([-1.0, -1.0, -1.0]) # Stop when reset
        vrep.simxPauseCommunication(self.clientID, 0)
        vrep.simxSynchronousTrigger(self.clientID)

        while vrep.simxGetStringSignal(self.clientID, "lidar", vrep.simx_opmode_blocking)[0] != 0:
            vrep.simxSynchronousTrigger(self.clientID)
        data = vrep.simxGetStringSignal(self.clientID, "lidar", vrep.simx_opmode_blocking)[1]

        raw_lidar = vrep.simxUnpackFloats(data)
        minpooled_lidar = []
        for i in range(54):
            minpooled_lidar.append(min(raw_lidar[19*i:19*(i+1)]))
        normalized_lidar_state = [item/10.0 for item in minpooled_lidar] # Normalized

        target_position = vrep.simxGetObjectPosition(self.clientID, self.target_handle ,self.robot_handle, vrep.simx_opmode_blocking)[1] # Target position w.r.t. robot position
        target_x = target_position[1]
        target_y = target_position[2]
        dist_target = (target_x**2 + target_y**2)**0.5
        normalized_target_state = [target_x/5.0, target_y/5.0] # Normalized

        state = normalized_lidar_state + [-1.0, -1.0, -1.0] + normalized_target_state # lidar, prev_action, target
        self.prev_distance = dist_target
        
        return state

    def step(self, action):
        done = False

        vrep.simxPauseCommunication(self.clientID, 1)
        self.set_action(action)
        vrep.simxPauseCommunication(self.clientID, 0)
        vrep.simxSynchronousTrigger(self.clientID)

        _, data = vrep.simxGetStringSignal(self.clientID, "lidar", vrep.simx_opmode_blocking)
        while _ != 0:
            _, data = vrep.simxGetStringSignal(self.clientID, "lidar", vrep.simx_opmode_blocking)

        raw_lidar = vrep.simxUnpackFloats(data)
        minpooled_lidar = []
        for i in range(54):
            minpooled_lidar.append(min(raw_lidar[19*i:19*(i+1)]))
        normalized_lidar_state = [item/10.0 for item in minpooled_lidar] # Normalized
        
        target_position = vrep.simxGetObjectPosition(self.clientID, self.target_handle ,self.robot_handle, vrep.simx_opmode_blocking)[1]
        target_x = target_position[1]
        target_y = target_position[2]
        dist_target = (target_x**2 + target_y**2)**0.5
        normalized_target_state = [target_x/5.0, target_y/5.0] # Normalized
        state1 = normalized_lidar_state + action.tolist() + normalized_target_state
        
        robot_position = vrep.simxGetObjectPosition(self.clientID, self.robot_handle, -1, vrep.simx_opmode_blocking)[1]
        if robot_position[0] > 4.5 or robot_position[0] < -4.5 or robot_position[1] > 4.5 or robot_position[1] < -4.5:
            done = True
        if min(minpooled_lidar) < 0.55:
            done = True
        if dist_target < 0.3:
            done = True
        
        target_reward = (self.prev_distance - dist_target) * 100
        costmap_reward = -2 * np.exp(-(min(minpooled_lidar)-0.3)**2/(2*0.2**2))
        reward = target_reward + costmap_reward

        if done == True:
            vrep.simxPauseCommunication(self.clientID, 1)
            self.set_action([-1.0, -1.0, -1.0]) # Stop before reset
            vrep.simxPauseCommunication(self.clientID, 0)
            vrep.simxSynchronousTrigger(self.clientID)
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)

        self.prev_distance = dist_target # Update prev_distance
        return state1, reward, done

    def set_action(self, action): # Send action signal to simulator
        # 1. Inverse Method
        a = 0.2355 # lateral wheel separation from midline
        b = 0.281 # longitudinal wheel separation from midline
        l = a + b
        r = 0.152 / 2.0 # wheel radius
        
        u_x, u_y, u_p = action
        u_x = u_x * 0.5
        u_y = u_y * 0.5
        fl = (u_x - u_y - l * u_p) / r
        fr = (u_x + u_y + l * u_p) / r
        rl = (u_x + u_y - l * u_p) / r
        rr = (u_x - u_y + l * u_p) / r
        
        # # 2. Forward Method
        # fl, fr, rl, rr = action
        # fl =  fl * 500.0 * np.pi / 180  # Need calibration
        # fr =  fr * 500.0 * np.pi / 180
        # rl =  rl * 500.0 * np.pi / 180
        # rr =  rr * 500.0 * np.pi / 180

        vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles['rollingJoint_fl'], fl, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles['rollingJoint_fr'], fr, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles['rollingJoint_rl'], rl, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles['rollingJoint_rr'], rr, vrep.simx_opmode_oneshot)
        
    def close(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(0.5)