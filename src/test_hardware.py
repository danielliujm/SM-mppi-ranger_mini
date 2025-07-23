import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ranger_msgs.msg import ActuatorState,ActuatorStateArray,DriverState, MotionState,MotorState 
from std_msgs.msg import Header
import csv

class TestHardware(Node):
    def __init__ (self):
        super().__init__('test_hardware')
        # self.create_timer (1.0, self.malformed_cmd_callback)
        self.create_timer (0.02, self.normal_cmd_callback)

        self.cmd_publisher = self.create_publisher (Twist, '/cmd_vel',10)

        self.create_subscription (ActuatorStateArray, '/actuator_state', self.actuator_state_callback, 10)
        self.create_subscription (MotionState, '/motion_state', self.motion_state_callback, 10)

        self.republish_actuator = self.create_publisher(ActuatorStateArray, '/republish_actuator_states', 10)
        self.republish_motion = self.create_publisher(MotionState, '/republish_motion_state', 10)



    def actuator_state_callback(self, msg):
        array_msg = ActuatorStateArray()
        for i in range(7):
            if msg.states[i].id == 0 or msg.states[i].id == 1 or msg.states[i].id == 2 or msg.states[i].id == 3:
                array_msg.states.append (msg.states[i])

        # self.get_logger().info(f'Received actuator states: {len(array_msg.states)} states')
        self.republish_actuator.publish(array_msg)

    def motion_state_callback(self, msg):
        if msg.motion_mode == 1:
            self.republish_motion.publish(msg)
            self.get_logger().info(f'Received motion state: {msg.motion_mode}')
    
   

    def malformed_cmd_callback (self):
        msg = Twist()
        msg.linear.x = -0.3
        msg.linear.y = 0.1
        msg.angular.z = 0.0
        # for i in range(10):
        self.cmd_publisher.publish(msg)
        self.get_logger().info('Published malformed command')
    
    def normal_cmd_callback (self):
        msg = Twist()
        msg.linear.x = 0.1
        msg.linear.y = 1e-9
        msg.angular.z = 0.0
        # msg.angular.z = 0.33

        self.cmd_publisher.publish(msg)
        self.get_logger().info('Published normal command')
    
def main ():
    rclpy.init()
    node = TestHardware()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()