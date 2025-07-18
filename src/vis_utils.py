import math

import torch

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class VisualizationUtils:
    def __init__(self, node: Node) -> None:
        self._node = node

        self._rollouts_pub = self._node.create_publisher(
            MarkerArray, f"/{self._node.get_name()}/vis/rollouts", 1
        )

        self._path_pub = self._node.create_publisher(
            Path, f"/{self._node.get_name()}/vis/path", 1
        )

    def visualize_rollouts(self, rollouts: torch.Tensor, costs: torch.Tensor) -> None:
        """
        Input:
        rollouts: (shape: 1 x NUM_SAMPLES x HORIZON x 3)
        costs: (shape: NUM_SAMPLES)
        """

        assert rollouts.ndim == 4 and rollouts.shape[0] == 1 and rollouts.shape[-1] == 3

        min_cost = torch.min(costs).item()
        max_cost = torch.max(costs).item()

        marker_array = MarkerArray()
        
        
        min_cost_id = torch.argmin (costs).item()
        max_cost_id = torch.argmax (costs).item()
        
        min_marker = Marker()
        min_marker.header.frame_id = "map"
        min_marker.id = min_cost_id
        min_marker.type = Marker.LINE_STRIP
        min_marker.scale.x = 0.005
        min_marker.scale.y = 0.005
        min_marker.scale.z = 0.005
        min_marker.lifetime = Duration(seconds= 0.07).to_msg()  
        min_marker.color.g = min_marker.color.a = 1.0
        
        for state_idx in range (rollouts.shape[2]):
            state = rollouts [0,min_cost_id, state_idx,:]
            min_marker.points.append (Point(x=state[0].item(), y=state[1].item()))
        
        max_marker = Marker()
        max_marker.header.frame_id = "map"
        max_marker.id = max_cost_id
        max_marker.type = Marker.LINE_STRIP
        max_marker.scale.x = 0.005
        max_marker.scale.y = 0.005
        max_marker.scale.z = 0.005
        max_marker.lifetime = Duration(seconds=0.1).to_msg()  
        max_marker.color.r = max_marker.color.a = 1.0
        
        for state_idx in range (rollouts.shape[2]):
            state = rollouts [0,max_cost_id, state_idx,:]
            max_marker.points.append (Point(x=state[0].item(), y=state[1].item()))

        
        marker_array.markers.append (min_marker)
        # marker_array.markers.append (max_marker)
    
        
        # marker = Marker()
        # for sample_idx in range(rollouts.shape[1]):
        #     marker = Marker()
        #     marker.header.frame_id = "map"
        #     marker.id = sample_idx
        #     marker.type = Marker.SPHERE_LIST
        #     marker.scale.x = 0.005
        #     marker.scale.y = 0.005
        #     marker.scale.z = 0.005
        #     marker.lifetime = Duration(seconds=1.0).to_msg()  # type: ignore
            

        #     cost = costs[sample_idx].item()
        #     if cost == min_cost:
        #         marker.color.r = marker.color.g = marker.color.a = 1.0
        #     else:
        #         cost_prop = (cost - min_cost) / (max_cost - min_cost)
        #         # Smooth transition from red -> yellow -> green
        #         # prop: 1.0 -> 0.5 -> 0.0
        #         # r   : 1.0 -> 1.0 -> 0.0
        #         # g   : 0.0 -> 1.0 -> 1.0
        #         marker.color.r = 1.0 #min(1.0, 2 * cost_prop)
        #         marker.color.g = 0.0 #max(0.0, 1 - 2 * cost_prop)
        #         marker.color.a = 1.0
                
                
        #     min_cost_id = torch.argmin (costs)
        #     print ("min cost id is ", min_cost_id)

        #     for state_idx in range(rollouts.shape[2]):
        #         state = rollouts[0, min_cost_id, state_idx,:]
        #         marker.points.append(Point(x=state[0].item(), y=state[1].item()))
                
            
        #     # max_cost_id = torch.argmax (costs)
        #     # for state_idx in range(rollouts.shape[2]):
        #     #     state = rollouts [0, max_cost_id, state_idx, :]
        #     #     marker.points.append(Point(x=state[0].item(), y=state[1].item()))
            
        #     marker_array.markers.append (marker)
            


        self._rollouts_pub.publish(marker_array)

    def visualize_path(self, path: list[tuple[float, float, float]]) -> None:
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self._node.get_clock().now().to_msg()

        for state in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self._node.get_clock().now().to_msg()

            pose.pose.position.x = state[0]
            pose.pose.position.y = state[1]
            pose.pose.orientation.w = math.cos(state[2] / 2)
            pose.pose.orientation.z = math.sin(state[2] / 2)

            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)
