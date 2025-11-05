from sensor_msgs.msg import JointState

class PiDogSimDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        self.__motor_0  = self.__robot.getDevice('body_to_front_left_leg_b')
        self.__motor_1  = self.__robot.getDevice('front_left_leg_b_to_a')
        
        self.__motor_2  = self.__robot.getDevice('body_to_front_right_leg_b')
        self.__motor_3  = self.__robot.getDevice('front_right_leg_b_to_a')
        
        self.__motor_4  = self.__robot.getDevice('body_to_back_left_leg_b')
        self.__motor_5  = self.__robot.getDevice('back_left_leg_b_to_a')
        
        self.__motor_6  = self.__robot.getDevice('body_to_back_right_leg_b')
        self.__motor_7  = self.__robot.getDevice('back_right_leg_b_to_a')
    
        self.__motor_8  = self.__robot.getDevice('motor_8_to_tail')

        self.__motor_9  = self.__robot.getDevice('neck1_to_motor_9')
        self.__motor_10 = self.__robot.getDevice('neck2_to_motor_10')
        self.__motor_11 = self.__robot.getDevice('neck3_to_motor_11')

        self.motor_list = [self.__motor_0, self.__motor_1, self.__motor_2, self.__motor_3,
                  self.__motor_4, self.__motor_5, self.__motor_6, self.__motor_7,
                  self.__motor_8, self.__motor_9, self.__motor_10, self.__motor_11]

        # Safety: set a non-zero speed for position control
        for m in self.motor_list:
            if m is None:
                raise RuntimeError("Missing motor device — check names in the PROTO/WBT.")
            # m.setVelocity(1.0)
            m.setPosition(0.0)

        self.joint_states = [0.0] * 12

        # Store webots_node - it should have a 'robot' and potentially other attributes
        self.node = webots_node

        # Debug: print available attributes to find the right API
        print(f"WebotsNode attributes: {[attr for attr in dir(webots_node) if not attr.startswith('_')]}")

        # Try to access the underlying ROS node if it exists
        try:
            # Many webots plugins use webots_node as a ROS node directly
            self.node.create_subscription(JointState, 'motor_pos', self.__cmd_pos_callback, 1)
            self.joint_pub = self.node.create_publisher(JointState, 'joint_states', 1)
            print("PiDogSimDriver initialized with subscriptions.")
        except AttributeError as e:
            print(f"Subscription creation failed: {e}")
            print("Running without subscriptions - will use zero positions.")

    def __cmd_pos_callback(self, msg):
        """Callback to receive motor position commands from gait generator."""
        self.joint_states = msg.position

    def step(self):
        # Webots framework handles spinning - just apply motor commands
        for i, m in enumerate(self.motor_list):
            m.setPosition(self.joint_states[i])
