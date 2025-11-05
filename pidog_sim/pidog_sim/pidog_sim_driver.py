class PiDogSimDriver:
    """Minimal Webots driver for PiDog robot - receives commands via ROS topics."""

    def init(self, webots_node, properties):
        """Initialize the driver with webots_node."""
        print("=" * 60)
        print("PiDogSimDriver.init() called!")
        print(f"webots_node type: {type(webots_node)}")
        print(f"webots_node attributes: {[attr for attr in dir(webots_node) if not attr.startswith('_')]}")
        print("=" * 60)

        self.__robot = webots_node.robot

        # Get all motor devices
        motor_names = [
            'body_to_front_left_leg_b', 'front_left_leg_b_to_a',
            'body_to_front_right_leg_b', 'front_right_leg_b_to_a',
            'body_to_back_left_leg_b', 'back_left_leg_b_to_a',
            'body_to_back_right_leg_b', 'back_right_leg_b_to_a',
            'motor_8_to_tail',
            'neck1_to_motor_9', 'neck2_to_motor_10', 'neck3_to_motor_11'
        ]

        self.motor_list = []
        for name in motor_names:
            motor = self.__robot.getDevice(name)
            if motor is None:
                raise RuntimeError(f"Missing motor device: {name}")
            self.motor_list.append(motor)
            motor.setPosition(0.0)  # Initialize to zero

        # Stand position (degrees to radians)
        import math
        stand_angles_deg = [0, 45, 0, 45, 0, 45, 0, 45, 0, 0, 0, 0]
        self.joint_states = [angle * math.pi / 180.0 for angle in stand_angles_deg]

        # Store webots_node for ROS operations
        self.node = webots_node

        # Try to create ROS subscription
        try:
            from sensor_msgs.msg import JointState
            self.node.create_subscription(JointState, 'motor_pos', self.__cmd_pos_callback, 1)
            print("✓ Successfully created subscription to /motor_pos")
        except Exception as e:
            print(f"✗ Could not create subscription: {e}")
            print("  Driver will use stand position only")

        print("PiDogSimDriver initialized successfully!")

    def __cmd_pos_callback(self, msg):
        """Callback to receive motor position commands from gait generator."""
        self.joint_states = list(msg.position)

    def step(self):
        """Called every simulation step - apply motor commands."""
        for i, motor in enumerate(self.motor_list):
            motor.setPosition(self.joint_states[i])
