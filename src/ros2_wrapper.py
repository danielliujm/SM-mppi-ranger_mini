import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_wrapper import TF2Wrapper
import torch
from pytorch_mppi import MPPI
from config import *
from utils import dynamics, normalize_angle, save_data
import numpy as np
import math
import time
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.vectorized import contains
from sm_mppi import SMMPPIController
from datetime import datetime
from vis_utils import *
from visualization_msgs.msg import Marker, MarkerArray

EPSILON = 1e-12

class MPPLocalPlannerMPPI(Node):
    def __init__(self):
        super().__init__('mpc_local_planner_mppi')

        # Initialize parameters

        self.tf2_wrapper = TF2Wrapper(self)
        self.rollouts = torch.zeros((7, NUM_SAMPLES, 2))
        self.costs = torch.zeros((7, NUM_SAMPLES, 2))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ROS2 setup
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.timer = self.create_timer(HZ, self.plan_and_publish)
        self.time_steps = []
        self.linear_velocities = []
        self.angular_velocities = []
        self.start_time = time.time()
        self.controller = SMMPPIController(STATIC_OBSTACLES, self.device)
        self.counter = 0
        self.filtered_action = torch.tensor([0.0, 0.0,0.0], dtype=torch.float32).to(self.device)
        self.alpha = 1.0
        

        self.current_state = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(self.device)  # [x, y, yaw]
        self.robot_velocity = torch.tensor([0.0, 0.0], dtype=torch.float32).to(self.device)
        self.previous_robot_state = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(self.device)

        self.agent_states = {i: torch.tensor([10.0, 10.0, 0.0], dtype=torch.float32).to(self.device) for i in range(ACTIVE_AGENTS)}
        self.agent_velocities = {i: torch.tensor([0.0, 0.0], dtype=torch.float32).to(self.device) for i in range(ACTIVE_AGENTS)}
        self.previous_agent_states = {i: torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(self.device) for i in range(ACTIVE_AGENTS)}
        self.viz_tool = VisualizationUtils(self)
        
        self.rob_marker_pub = self.create_publisher (Marker, "/debug_robot_location",1)

        self.prev_control = 0.0
        self.control_variation = 0.0
    
    
    def create_rollout_video(self, output_file="rollout_animation.mp4"):
        if self.min_rollouts is None:
            print("Min rollout data not loaded. Please call `read_data()` first.")
            return
        min_rollouts_xy = self.min_rollouts[:, :5, :2]
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Min Rollout Trajectories (5 Time Horizons)")
        ax.grid(True)
        min_lines = [ax.plot([], [], color="red", linestyle=":", linewidth=1)[0]
                    for _ in range(min_rollouts_xy.shape[1])]  # 5 rollouts in red
        # Set axis limits based on the data
        x_min = np.min(min_rollouts_xy[:, :, 0])
        x_max = np.max(min_rollouts_xy[:, :, 0])
        y_min = np.min(min_rollouts_xy[:, :, 1])
        y_max = np.max(min_rollouts_xy[:, :, 1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # Add a legend
        ax.legend(handles=[
            plt.Line2D([], [], color="red", linestyle=":", label="Min Rollouts")
        ])
        # Function to update the plot for each frame
        def update(frame):
            for i in range(len(min_lines)):
                # Update min rollout lines up to the current frame
                min_lines[i].set_data(min_rollouts_xy[:frame, i, 0], min_rollouts_xy[:frame, i, 1])
            return min_lines
        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=min_rollouts_xy.shape[0], interval=50, blit=True
        )

    def plan_and_publish(self):

        timer_callback_start_time = time.time()
        self.counter += 1
        
        T_map_baselink = self.tf2_wrapper.get_latest_pose("map", "base_link")
        
        if T_map_baselink is None:
            print("Can't find robot pose")
            return
            # Convert pose to MPPI state representation
        yaw = 2.0 * math.atan2(
            T_map_baselink.rotation.z, T_map_baselink.rotation.w
        )  # NOTE: assuming roll and pitch are negligible
        
        
        self.current_state = torch.tensor(
                [
                    T_map_baselink.translation.x,
                    T_map_baselink.translation.y,
                    yaw,
                ],
                dtype=torch.float32,
            ).to(self.device)
        
        #############################################################################
        min_marker = Marker()
        min_marker.header.frame_id = "map"
        min_marker.id = 0
        min_marker.type = Marker.SPHERE_LIST
        min_marker.scale.x = 0.05
        min_marker.scale.y = 0.05
        min_marker.scale.z = 0.05
        min_marker.lifetime = Duration(seconds= 0.07).to_msg()  
        min_marker.color.g = min_marker.color.a = 1.0
        min_marker.points.append (Point(x=self.current_state[0].item(), y=self.current_state[1].item()))
        self.rob_marker_pub.publish (min_marker)
        #############################################################################

        # print ("r location is ", self.current_state)
        
        if self.counter == 3:
            self.previous_robot_state = self.current_state
            
        if torch.any(self.previous_robot_state):
            self.robot_velocity = (self.current_state[:2] - self.previous_robot_state[:2])/HZ
            self.previous_robot_state = self.current_state
        for i in range(ACTIVE_AGENTS):
            tt = time.time()
            T_map_agent = self.tf2_wrapper.get_latest_pose("map", HUMAN_FRAME + "_" + str(i+1))
            #print("tt ", time.time() - tt)
            if T_map_agent is None:
                print("Can't find human pose") 
                agent_state = torch.tensor([100.0, 100.0, 0.0], dtype=torch.float32).to(self.device)    
                # self.agent_states[i] = agent_state
            else:
                #print("found human ", i)
                yaw = 2.0 * math.atan2(
                    T_map_agent.rotation.z, T_map_agent.rotation.w
                )  # NOTE: assuming roll and pitch are negligible
                agent_state = torch.tensor([T_map_agent.translation.x, T_map_agent.translation.y, yaw], dtype=torch.float32).to(self.device)
                agent_state[2] = torch.arctan2(torch.sin(agent_state[2]), torch.cos(agent_state[2]))
                self.agent_states[i] = agent_state
                # Calculate velocity if previous state exists
            if i in self.previous_agent_states:
                prev_x, prev_y, prev_yaw = self.previous_agent_states[i]
                velocity = torch.tensor([
                            (agent_state[0] - prev_x) / HZ,
                            (agent_state[1] - prev_y) / HZ
                        ], dtype=torch.float32).to(self.device)

                self.agent_velocities[i] = velocity
            self.previous_agent_states[i] = agent_state

    

      

        # b4_time = datetime.now()
        # Compute optimal control using MPPI
        action, self.rollouts, self.costs, termination = self.controller.compute_control(
            self.current_state, self.previous_robot_state, self.robot_velocity, self.agent_states, self.previous_agent_states, self.agent_velocities
        )
        

        
        self.rollouts = self.rollouts.unsqueeze(0)
        
        # print ("action calculation took ", (datetime.now().microsecond-b4_time.microsecond)/1000, "ms")
        
        
        # print ("cost is ", self.costs)

        if action is not None and not termination:
            # print ("shape of rollouts ", self.rollouts.size())
            # b4_time = datetime.now()
            self.viz_tool.visualize_rollouts (self.rollouts, self.costs)
            # print ("visualization took ", (datetime.now().microsecond-b4_time.microsecond)/1000, "ms")
            
            twist_stamped = Twist()
            # twist_stamped.header.stamp = self.get_clock().now().to_msg()
            
            self.filtered_action = self.alpha * action + (1 - self.alpha) * self.filtered_action
            action = self.filtered_action


            
            
            x_effort= action[0].item() if abs(action[0].item()) < VMAX else np.sign(action[0].item())*VMAX 
            y_effort = action[1].item() if abs(action[1].item()) < VMAX else np.sign(action[1].item())*VMAX 

            if abs(x_effort) < 0.03:
                x_effort = 0.0
            if abs (y_effort) < 0.03:
                y_effort = EPSILON
            if abs (action[2].item()) < 0.03:
                action[2] = 0.0


            print (x_effort, y_effort, action[2].item())
            
            twist_stamped.linear.x = x_effort 
            twist_stamped.linear.y = y_effort
            twist_stamped.angular.z = action[2].item()

            self.control_variation += np.sqrt((self.prev_control - y_effort)**2)
            self.prev_control = y_effort

            
            # twist_stamped.linear.x = 0.0
            # twist_stamped.linear.y = 0.0
            # twist_stamped.angular.z = 0.0

            # twist_stamped.angular.z = action[1].item() #min(action[1].item(), VMAX)
            self.cmd_vel_pub.publish(twist_stamped)
           
            
            self.linear_velocities.append(x_effort)
            self.angular_velocities.append(y_effort)
            self.time_steps.append(time.time() - self.start_time)

            ### print control variation ###
            print ("control variation is ", self.control_variation)
        elif termination:
            print("Reached the goal!!!!!")
            self.controller.move_to_next_goal()
            # self.save_plot()
        else:
            self.get_logger().warn("Failed to compute optimal controls")
        timer_callback_end_time = time.time()
        # print(timer_callback_end_time - timer_callback_start_time)


        
def main(args=None):
    rclpy.init(args=args)
    node = MPPLocalPlannerMPPI()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
