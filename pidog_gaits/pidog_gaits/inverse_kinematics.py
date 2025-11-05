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
            tuple: (leg_angle, foot_angle) in degrees
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

        # Convert to degrees
        alpha_deg = alpha * 180.0 / pi
        beta_deg = beta * 180.0 / pi

        return alpha_deg, beta_deg

    @classmethod
    def legs_coords_to_angles(cls, leg_coords):
        """
        Convert 4-leg coordinates to 8 joint angles.

        Args:
            leg_coords (list): List of 4 [y, z] coordinates, one per leg
                              [[y1, z1], [y2, z2], [y3, z3], [y4, z4]]
                              Order: leg1 (left front), leg2 (right front),
                                     leg3 (left back), leg4 (right back)

        Returns:
            list: 8 angles [a1, a2, a3, a4, a5, a6, a7, a8]
                  where (a1,a2) = leg1, (a3,a4) = leg2, etc.
                  Format matches PiDog's motor order
        """
        angles = []

        for i, coord in enumerate(leg_coords):
            y, z = coord
            # Invert Y for URDF coordinate system (forward is negative Y)
            y = -y
            leg_angle, foot_angle = cls.coord2angles(y, z)

            # Adjust for leg mounting orientation
            # SunFounder convention: foot_angle -= 90
            foot_angle = foot_angle - 90

            # Right side legs (odd indices) are mirrored
            if i % 2 != 0:
                leg_angle = -leg_angle
                foot_angle = -foot_angle

            angles.extend([leg_angle, foot_angle])

        return angles
