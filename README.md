# Legible_MPPI 
MPPI Implementation for various legible algorithms for social robot navigation. Controller parameters are turned for the AgileX Ranger Mini robot. 

The implmentation is based on the [Pytorch-MPPI](https://github.com/UM-ARM-Lab/pytorch_mppi/tree/master) package. The algorithims included are:

- [Social Momentum](https://github.com/fluentrobotics/Legible_MPPI/blob/main/src/sm_mppi.py)
-  ~~DS Legibility~~
-  ~~DS Legibility (passing side goal)~~
-  ~~DS Legibility (passing side dynamic goals)~~
-  ~~Vanilla MPPI + Constant Velocity predictions~~

# Installation
To install ROS2 packages and launch robot base for Ranger Mini, follow the instructions shown in this [repository](https://github.com/agilexrobotics/ranger_ros2.git). 


```shell
pip install pytorch-mppi
```

An example on how to use the MPPIController for an algorithim with ROS2 can be found in [ros2_wrapper.py](https://github.com/fluentrobotics/Legible_MPPI/blob/main/src/ros2_wrapper.py). Note that this example is based on a AgileX Ranger Mini robot with the human poses obtained from motion capture and published on the ROS2 tf.

# Runinng the Controller
Initiate Ranger Mini base
```shell
cd ~/ranger_ws/
source install/setup.bash
sudo bash src/ranger_ros2/ranger_bringup/scripts/bringup_can2usb.bash
ros2 launch ranger_bringup ranger_mini_v3.launch.xml
```

Begin publishing MOCAP transformations 
```shell
cd ~/ranger_ws/
source install/setup.bash
ros2 launch mocap_optitrack_client launch_y_up_launch.py
```
run the ROS2 wrapper 
```shell
cd ~/SM-mppi-ranger_mini/src
python3 ros2_wrapper.py
```
# Parameter Tuning
|parameter |value | note |
------- | -----| -------|
| DT  | 0.4 | time increment between consecutive points in the trajectory duruing planning
| HORIZON_LENGTH | 12| the number of time steps the controller simulates into the future
| HZ | 0.025| time interval between velocity commands sent to the robot 
| NUM_SAMPLES | 350| how many random trajectories (rollouts) are generated at each control cycle
| cov | 0.01| random perturbation added control inputs during trajectory rollouts to explore different possible futures
|kernel sigma | 1.5| kernel in the trajectory time space (1 dimensional)
|num_support_pts |2| number of control points to sample 
|goal_weight | 100| weight of goal cost 
|action_weight | 8000| weight of action cost (cost is high if action changes rapidly)
|heading_weight | 1000| weight of action cost (cost is high if robot heading changes rapidly)
|dynamic_obstacle_weight | 100| weight of dynamic obstacle avoidance cost
|sm_weight | 10| weight of social moment cost
