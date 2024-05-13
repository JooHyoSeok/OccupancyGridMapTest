import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython.display import HTML

# Calculates the inverse measurement model for a laser scanner.
# It identifies three regions. The first where no information is available occurs
# outside of the scanning arc. The second where objects are likely to exist, at the
# end of the range measurement within the arc. The third are where objects are unlikely
# to exist, within the arc but with less distance than the range measurement.

def inverse_scanner(num_rows, num_cols, robot_x, robot_y, robot_theta, measurement_angles, measurement_ranges, max_range, alpha, beta):
    
    # *이 함수는 센서 측정값과 로봇의 위치 및 방향을 바탕으로 맵 추정을 생성합.*
    ''' p(z|map)
        num_rows, num_cols: 맵의 차원.
        robot_x, robot_y: 로봇의 위치를 나타내는 행과 열 인덱스.
        robot_theta: 로봇의 방향(라디안 단위).
        measurement_angles: 로봇의 방향을 기준으로 측정된 각도 배열.
        measurement_ranges: measurement_angles에 해당하는 거리 측정 배열.
        max_range: 센서의 최대 범위.
        alpha: 장애물의 너비(거리 불확실성).
        beta: 센서 빔의 각도 너비(각도 불확실성).

    '''
    # print(f'robot x : {robot_x } robot_ y : {robot_y}  robot_theta : {robot_theta}')
    # print(f"meas_angle : {measurement_angles} meas_ranges : {measurement_ranges}")
    #* robot position input as pixel coordinates
    map_estimation = np.zeros((num_rows, num_cols))
    
    for row in range(num_rows):
        for col in range(num_cols):
            # Find range and bearing relative to the input state (x, y, theta).
            distance = math.sqrt((row - robot_x)**2 + (col - robot_y)**2)
            angle = (math.atan2(col - robot_y, row - robot_x) - robot_theta + math.pi) % (2 * math.pi) - math.pi
            # Find the range measurement associated with the relative bearing.
            closest_measurement_index = np.argmin(np.abs(angle - measurement_angles))
            # 로봇의 현재 위치와 특정 픽셀 사이의 방향과 가장 유사한 각도로 측정된 레이저 스캔 값 중에서 인덱스
            # If the range is greater than the maximum sensor range, or behind our range
            # measurement, or is outside of the field of view of the sensor, then no

            # new information is available. 정보 부재: 셀이 센서의 최대 범위를 초과하거나 가장 가까운 측정의 각도 범위를 벗어난 경우, 중립 확률을 할당.
            if distance > min(max_range, measurement_ranges[closest_measurement_index] + alpha / 2.0) or abs(angle - measurement_angles[closest_measurement_index]) > beta / 2.0:
                map_estimation[row, col] = 0.5

            # If the range measurement lied within this cell, it is likely to be an object. 객체 감지: 측정값이 객체가 셀 안에 있음을 나타내는 경우, 높은 점유 확률을 할당.
            elif measurement_ranges[closest_measurement_index] < max_range and abs(distance - measurement_ranges[closest_measurement_index]) < alpha / 2.0:
                map_estimation[row, col] = 0.7 # l_occ
            # If the cell is in front of the range measurement, it is likely to be empty. 빈 공간 표시: 셀이 감지된 범위 앞에 있는 경우, 낮은 점유 확률을 할당.
            elif distance <= measurement_ranges[closest_measurement_index]:
                map_estimation[row, col] = 0.3 # l_free

    return map_estimation

# Generates range measurements for a laser scanner based on a map, vehicle position,
# and sensor parameters.
# Uses the ray tracing algorithm.

def get_ranges(true_map, position, measurement_angles, max_range):

    '''
        Ray Tracing
    
    @   Param

        true_map: 알려진 환경을 나타내는 2D 배열로, 1은 Obstacle, 0은 FreeSpace.
        position: 로봇의 x, y 좌표와 방향을 포함하는 튜플.
        measurement_angles: 센서가 거리를 측정하는 각도 배열.
        max_range: 센서가 측정할 수 있는 최대 거리.
    
    '''
    
    (num_rows, num_cols) = np.shape(true_map)
    x, y, theta = position
    ranges = max_range * np.ones(measurement_angles.shape)

    for idx, meas_angle in enumerate(measurement_angles):
        # Iterate over each unit step up to and including rmax.
        for r in range(1, max_range + 1):

            # Determine the coordinates of the cell.
            target_x = int(round(x + r * math.cos(theta + meas_angle)))
            target_y = int(round(y + r * math.sin(theta + meas_angle)))

            # If not in the map, set measurement there and stop going further.

            if target_x <= 0 or target_x >= (num_rows - 1) or target_y <= 0 or target_y >= (num_cols - 1): # 경계 밖 
                ranges[idx] = r
                break

            # If in the map, but hitting an obstacle, set the measurement range
            # and stop ray tracing.

            elif true_map[target_x, target_y] == 1: # 장애물이 있다면 
                ranges[idx] = r
                break
    # print(f"ranges : {ranges}")
    return ranges


# Simulation time initialization.
T_MAX = 200
time_steps = np.arange(T_MAX)
print("Time Step : ", time_steps)
# Initializing the robot's location.
x_0 = [30, 30, 0]

# The sequence of robot motions.
u = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
u_i = 1

# Robot sensor rotation command
w =  np.ones(len(time_steps)) * 0.3

 
# True map (note, columns of map correspond to y axis and rows to x axis, so 
# robot position x = x(1) and y = x(2) are reversed when plotted to match

M = 100
N = 100

occupancy_map = np.zeros((M, N))
occupancy_map[0:20, 0:20] = 1
occupancy_map[30:35, 40:45] = 1
occupancy_map[5:20,40:60] = 1
occupancy_map[20:30,25:29] = 1
occupancy_map[40:50,5:25] = 1

# plt.figure(figsize=(8, 8))
# cmap = plt.cm.jet  # 컬러맵 설정
# norm = plt.Normalize(vmin=-1, vmax=1)
# plt.imshow(occupancy_map, cmap=cmap, norm=norm)
# plt.colorbar()  # 컬러바 표시
# plt.title('2D Occupancy Grid Map')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.grid(False)  
# plt.show()

# Initialize the belief map.
# We are assuming a uniform prior.
map_prob = np.multiply(0.5, np.ones((M, N)))

# Initialize the log odds ratio.
L0 = np.log(np.divide(map_prob, 1 -  map_prob))
L = L0

# Parameters for the sensor model.
meas_phi = np.arange(0, np.deg2rad(360), 0.05)
print(f"meas_phi  : {meas_phi}")
rmax = 50 # Max beam range.
alpha = 1 # Width of an obstacle (distance about measurement to fill in).
beta = 0.05 # Angular width of a beam.

# Initialize the vector of states for our simulation.
x = np.zeros((3, len(time_steps)))# 3 x 150 matrix
x[:, 0] = x_0



# Intitialize figures.
map_fig = plt.figure()
map_ax = map_fig.add_subplot(111)
map_ax.set_xlim(0, N)
map_ax.set_ylim(0, M)

invmod_fig = plt.figure()
invmod_ax = invmod_fig.add_subplot(111)
invmod_ax.set_xlim(0, N)
invmod_ax.set_ylim(0, M)

belief_fig = plt.figure()
belief_ax = belief_fig.add_subplot(111)
belief_ax.set_xlim(0, N)
belief_ax.set_ylim(0, M)

# INITIAL SETTING

meas_rs = []
meas_r = get_ranges(occupancy_map, x[:, 0], meas_phi, rmax)
print('meas_r : ',meas_r)
meas_rs.append(meas_r)
invmods = []
invmod = inverse_scanner(M, N, x[0, 0], x[1, 0], x[2, 0], meas_phi, meas_r,rmax, alpha, beta)
invmods.append(invmod)
map_list = []
map_list.append(map_prob)

def logit(x):
    '''
        확률 p를 입력으로 받아 로그 오즈 값 𝐿 로 변환 
    '''
    return np.log(x/(1-x))

# Main simulation loop.
for t in range(1, len(time_steps)):
    # Perform robot motion.
    move = np.add(x[0:2, t-1], u[:, u_i]) 
    # If we hit the map boundaries, or a collision would occur, remain still.
    if (move[0] >= M - 1) or (move[1] >= N - 1) or (move[0] <= 0) or (move[1] <= 0) \
        or occupancy_map[int(round(move[0])), int(round(move[1]))] == 1:
        x[:, t] = x[:, t-1]
        u_i = (u_i + 1) % 4
    else:
        x[0:2, t] = move
    x[2, t] = (x[2, t-1] + w[t]) % (2 * math.pi)
    
    # Gather the measurement range data, which will be converted to occupancy probabilities
    # using the simple inverse measurement model.
    meas_r = get_ranges(occupancy_map, x[:,t], meas_phi, rmax)
    # for simulation environment
    meas_rs.append(meas_r)
    
    # Given the range measurements and the robot location, apply the inverse scanner model
    # to measure the probabilities of occupancy.
    robot_x , robot_y , robot_theta = x[0, t], x[1, t], x[2, t]
    invmod = inverse_scanner(M, N, robot_x , robot_y , robot_theta, meas_phi, meas_r, rmax, alpha, beta)
    # print(f"invmod : {invmod}")
    invmods.append(invmod)

    # Calculate and update the log odds of the occupancy grid, given the measured
    # occupancy probabilities from the inverse model.
    L = logit(invmod) + L - L0

    # Calculate a grid of probabilities from the log odds.
    map_prob = (np.exp(L)) / (1 + np.exp(L))
    # log_odd to prob

    # print(map_prob[20:25,20:25].round(3))
    # print(len(map_prob) , len(map_prob[0]))
    map_list.append(map_prob)
    # print(len(map_list))
    
def map_update(i):
    map_ax.clear()
    map_ax.set_xlim(0, N)
    map_ax.set_ylim(0, M)
    map_ax.imshow(np.subtract(1, occupancy_map), cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    x_plot = x[1, :i+1]
    y_plot = x[0, :i+1]
    map_ax.plot(x_plot, y_plot, "bx-")

def invmod_update(i):
    invmod_ax.clear()
    invmod_ax.set_xlim(0, N)
    invmod_ax.set_ylim(0, M)
    invmod_ax.imshow(invmods[i], cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    for j in range(len(meas_rs[i])):
        invmod_ax.plot(x[1, i] + meas_rs[i][j] * math.sin(meas_phi[j] + x[2, i]), \
                       x[0, i] + meas_rs[i][j] * math.cos(meas_phi[j] + x[2, i]), "ko")
    invmod_ax.plot(x[1, i], x[0, i], 'bx')
    
def belief_update(i):
    belief_ax.clear()
    belief_ax.set_xlim(0, N)
    belief_ax.set_ylim(0, M)
    belief_ax.imshow(map_list[i], cmap='gray_r', origin='lower', vmin=0.0, vmax=1.0)
    belief_ax.plot(x[1, max(0, i-10):i], x[0, max(0, i-10):i], 'bx-')
    
map_anim = anim.FuncAnimation(map_fig, map_update, frames=len(x[0, :]), repeat=False)
invmod_anim = anim.FuncAnimation(invmod_fig, invmod_update, frames=len(x[0, :]), repeat=False)
belief_anim = anim.FuncAnimation(belief_fig, belief_update, frames=len(x[0, :]), repeat=False)
belief_anim.save('occmap.mp4', writer='ffmpeg', fps=30)
plt.show()
