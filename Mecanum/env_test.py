import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPG
from ou_noise import OUNoise
# from environment.vrep_env import Env
from environment.env_modules import vrep
import time
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
lidar_list = ['fastHokuyo_sensor1', 'fastHokuyo_sensor2', 'fastHokuyo_sensor3']
lidar_handles = {}
for name in lidar_list:
    lidar_handles[name] = vrep.simxGetObjectHandle(clientID, name, vrep.simx_opmode_blocking)[1]
# env = Env(19997)
vrep.simxSynchronous(clientID, True)
time.sleep(0.5)
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)


while vrep.simxGetStringSignal(clientID, "lidar", vrep.simx_opmode_blocking)[0] != 0:
    vrep.simxSynchronousTrigger(clientID)

data = vrep.simxGetStringSignal(clientID, "lidar", vrep.simx_opmode_blocking)[1]

measuredData=vrep.simxUnpackFloats(data)
lidar = []
for i in range(18):
    lidar.append(min(measuredData[57*i:57*(i+1)]))
r = np.asarray(measuredData)
r1 = np.asarray(lidar)
theta = np.linspace(0.0, np.pi*2, 1026)
theta1 = np.linspace(0.0, np.pi*2, 18)
ax = plt.subplot(111, projection='polar')
ax.plot(theta1, r1)
ax.set_rmax(11)
plt.show()