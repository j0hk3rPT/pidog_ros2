"""
Inverse Kinematics for PiDog

Converts [y, z] coordinates to joint angles using 2-link leg geometry.
Based on SunFounder PiDog implementation.
"""

from math import sqrt, acos, atan2, pi


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
