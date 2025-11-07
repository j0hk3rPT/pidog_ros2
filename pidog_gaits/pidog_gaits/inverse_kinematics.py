"""
Inverse Kinematics for PiDog

Converts [y, z] coordinates to joint angles using 2-link leg geometry.
Includes head/tail control for balancing during locomotion.
Based on SunFounder PiDog implementation.
"""

from math import sqrt, acos, atan2, pi, sin, cos


class LegIK:
    """
    Inverse kinematics for a 2-DOF leg.

    Leg structure:
        - Upper leg (LEG): 42mm
        - Lower leg (FOOT): 76mm
        - 2 joints: hip (alpha) and knee (beta)
    """

    # Physical constants (in mm, same as SunFounder)
    LEG = 42.0   # Upper leg length
    FOOT = 76.0  # Lower leg length

    @classmethod
    def coord2angles(cls, y, z):
        """
        Convert [y, z] coordinate to [leg_angle, foot_angle].

        Args:
            y (float): Horizontal distance from hip (mm)
            z (float): Vertical distance from hip (mm)

        Returns:
            tuple: (leg_angle, foot_angle) in radians
        """
        # Calculate distance from hip to foot
        u = sqrt(y**2 + z**2)

        # Law of cosines for knee angle (beta)
        cos_angle1 = (cls.FOOT**2 + cls.LEG**2 - u**2) / (2 * cls.FOOT * cls.LEG)
        cos_angle1 = max(min(cos_angle1, 1.0), -1.0)  # Clamp to [-1, 1]
        beta = acos(cos_angle1)

        # Calculate hip angle (alpha)
        angle1 = atan2(y, z)
        cos_angle2 = (cls.LEG**2 + u**2 - cls.FOOT**2) / (2 * cls.LEG * u)
        cos_angle2 = max(min(cos_angle2, 1.0), -1.0)  # Clamp to [-1, 1]
        angle2 = acos(cos_angle2)
        alpha = angle2 + angle1

        # Return in radians (SI units for ros2_control)
        return alpha, beta

    @classmethod
    def legs_coords_to_angles(cls, leg_coords):
        """
        Convert 4-leg coordinates to 8 joint angles.

        Args:
            leg_coords (list): List of 4 [y, z] coordinates, one per leg
                              [[y1, z1], [y2, z2], [y3, z3], [y4, z4]]
                              Order: leg1 (FL), leg2 (FR), leg3 (BL), leg4 (BR)

        Returns:
            list: 8 angles in controller order [BR, FR, BL, FL] in radians
                  Format: [BR_shoulder, BR_knee, FR_shoulder, FR_knee,
                           BL_shoulder, BL_knee, FL_shoulder, FL_knee]
        """
        # Generate angles in gait order: FL, FR, BL, BR
        gait_order_angles = []

        for i, coord in enumerate(leg_coords):
            y, z = coord
            # NO y-coordinate inversion (SunFounder uses coordinates directly)
            leg_angle, foot_angle = cls.coord2angles(y, z)

            # Match SunFounder implementation exactly:
            # 1. NO shoulder offset (alpha = angle2 + angle1, no subtraction)
            # 2. foot_angle -= 90° (not π/2 - foot_angle)
            # 3. Negate RIGHT legs (odd indices), not left legs
            foot_angle = foot_angle - (pi / 2)

            # SunFounder negates RIGHT side legs (odd indices: FR=1, BR=3)
            # But our leg order is [FL=0, FR=1, BL=2, BR=3]
            # AND our URDF has left legs with flipped axes (rpy="0 1.57 3.1415")
            # So we need to negate RIGHT legs like SunFounder
            if i % 2 != 0:  # Right legs (FR=1, BR=3)
                leg_angle = -leg_angle
                foot_angle = -foot_angle

            gait_order_angles.extend([leg_angle, foot_angle])

        # Reorder from gait order [FL, FR, BL, BR] to controller order [BR, FR, BL, FL]
        # gait_order_angles[0:2] = FL → controller[6:8]
        # gait_order_angles[2:4] = FR → controller[2:4]
        # gait_order_angles[4:6] = BL → controller[4:6]
        # gait_order_angles[6:8] = BR → controller[0:2]
        controller_order_angles = [
            gait_order_angles[6], gait_order_angles[7],  # BR (from index 3)
            gait_order_angles[2], gait_order_angles[3],  # FR (from index 1)
            gait_order_angles[4], gait_order_angles[5],  # BL (from index 2)
            gait_order_angles[0], gait_order_angles[1],  # FL (from index 0)
        ]

        return controller_order_angles


class HeadTailController:
    """
    Head and tail control for balancing and natural movement.

    Based on SunFounder PiDog head_rpy_to_angle() implementation:
    - 3 neck servos for yaw, roll, pitch control
    - 1 tail servo for counterbalance
    - Coupled pitch/roll control based on yaw angle
    """

    @classmethod
    def neutral_pose(cls):
        """Return neutral head/tail position: [tail, yaw, roll, pitch]"""
        return [0.0, 0.0, 0.0, 0.0]

    @classmethod
    def head_rpy_to_angles(cls, yaw, roll, pitch, roll_comp=0.0, pitch_comp=0.0):
        """
        Convert head orientation (yaw, roll, pitch) to servo angles.

        Implements SunFounder's coupled servo control where pitch and roll
        servos blend based on yaw angle ratio for diagonal head orientation.

        Args:
            yaw (float): Head yaw angle in radians (-pi/2 to pi/2)
            roll (float): Head roll angle in radians (-pi/2 to pi/2)
            pitch (float): Head pitch angle in radians (-pi/4 to pi/6)
            roll_comp (float): Roll compensation for balancing (radians)
            pitch_comp (float): Pitch compensation for balancing (radians)

        Returns:
            list: [yaw_servo, roll_servo, pitch_servo] in radians
        """
        # Calculate yaw ratio (0 to 1 based on absolute yaw)
        # SunFounder uses 90° max yaw, ratio = abs(yaw) / 90
        yaw_rad_max = pi / 2  # 90 degrees
        signed = -1.0 if yaw < 0 else 1.0
        ratio = min(abs(yaw) / yaw_rad_max, 1.0)

        # Coupled pitch/roll control (SunFounder algorithm):
        # - Pitch servo blends: roll*ratio + pitch*(1-ratio) + pitch_comp
        # - Roll servo blends: -(signed * (roll*(1-ratio) + pitch*ratio) + roll_comp)
        pitch_servo = roll * ratio + pitch * (1.0 - ratio) + pitch_comp
        roll_servo = -(signed * (roll * (1.0 - ratio) + pitch * ratio) + roll_comp)
        yaw_servo = yaw

        # Clamp to safe servo ranges (matching URDF limits: -1.57 to 1.57 rad)
        yaw_servo = max(min(yaw_servo, 1.57), -1.57)
        roll_servo = max(min(roll_servo, 1.57), -1.57)
        pitch_servo = max(min(pitch_servo, 1.57), -1.57)

        return [yaw_servo, roll_servo, pitch_servo]

    @classmethod
    def tail_angle_for_balance(cls, gait_phase, gait_type='walk'):
        """
        Calculate tail angle for counterbalancing during movement.

        The tail moves opposite to the body's center of mass to maintain balance.

        Args:
            gait_phase (float): Current gait cycle phase (0.0 to 1.0)
            gait_type (str): Gait type ('walk', 'trot', 'static')

        Returns:
            float: Tail servo angle in radians
        """
        if gait_type == 'static':
            # Neutral position for static poses
            return 0.0
        elif gait_type == 'walk':
            # Gentle side-to-side wag during walking (slower than leg movement)
            # Phase shifted by 180° to counterbalance leg movement
            tail_angle = 0.3 * sin(2.0 * pi * gait_phase + pi)
        elif gait_type == 'trot':
            # Faster wag for trotting (matches trot frequency)
            tail_angle = 0.4 * sin(4.0 * pi * gait_phase + pi)
        else:
            tail_angle = 0.0

        # Clamp to safe range
        return max(min(tail_angle, 1.57), -1.57)

    @classmethod
    def head_compensation_for_walking(cls, gait_phase, gait_type='walk', direction='forward'):
        """
        Calculate head pitch/roll compensation to keep head level during walking.

        When the body pitches forward/backward or rolls side-to-side during
        walking, the head compensates to maintain level gaze.

        Args:
            gait_phase (float): Current gait cycle phase (0.0 to 1.0)
            gait_type (str): Gait type ('walk', 'trot', 'static')
            direction (str): Movement direction ('forward', 'backward', 'left', 'right')

        Returns:
            tuple: (yaw, roll, pitch, roll_comp, pitch_comp) in radians
        """
        if gait_type == 'static':
            # Neutral head position for static poses
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        # Base head orientation
        yaw = 0.0  # Forward
        roll = 0.0
        pitch = 0.0

        if gait_type == 'walk':
            # Walking creates subtle pitch/roll oscillations
            # Compensate to keep head level (SunFounder uses pitch_comp for sitting: -30°)

            if direction in ['forward', 'backward']:
                # Pitch compensation: body pitches down when front legs lift
                # Compensate upward to keep head level
                pitch_comp = 0.15 * sin(2.0 * pi * gait_phase)  # ±8.6°
                roll_comp = 0.0
            elif direction in ['left', 'right']:
                # Roll compensation: body rolls when side legs lift
                roll_comp = 0.15 * sin(2.0 * pi * gait_phase)
                pitch_comp = 0.0
            else:
                pitch_comp = 0.0
                roll_comp = 0.0

        elif gait_type == 'trot':
            # Trotting creates more pronounced oscillations
            if direction in ['forward', 'backward']:
                pitch_comp = 0.2 * sin(4.0 * pi * gait_phase)  # ±11.5°
                roll_comp = 0.1 * sin(4.0 * pi * gait_phase + pi/2)  # Phase shifted
            elif direction in ['left', 'right']:
                roll_comp = 0.2 * sin(4.0 * pi * gait_phase)
                pitch_comp = 0.1 * sin(4.0 * pi * gait_phase + pi/2)
            else:
                pitch_comp = 0.0
                roll_comp = 0.0
        else:
            pitch_comp = 0.0
            roll_comp = 0.0

        return (yaw, roll, pitch, roll_comp, pitch_comp)

    @classmethod
    def get_head_tail_angles(cls, gait_phase, gait_type='walk', direction='forward'):
        """
        Get complete head and tail angles for a given gait phase.

        Args:
            gait_phase (float): Current gait cycle phase (0.0 to 1.0)
            gait_type (str): Gait type ('walk', 'trot', 'static')
            direction (str): Movement direction

        Returns:
            list: [tail_angle, yaw_servo, roll_servo, pitch_servo] in radians
        """
        # Get head compensation
        yaw, roll, pitch, roll_comp, pitch_comp = cls.head_compensation_for_walking(
            gait_phase, gait_type, direction
        )

        # Convert to servo angles
        head_angles = cls.head_rpy_to_angles(yaw, roll, pitch, roll_comp, pitch_comp)

        # Get tail angle
        tail_angle = cls.tail_angle_for_balance(gait_phase, gait_type)

        # Return in motor order: [tail, yaw, roll, pitch]
        return [tail_angle] + head_angles
