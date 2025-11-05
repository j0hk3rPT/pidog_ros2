"""
Walk Gait Generator

Sequential leg movement pattern for stable walking.
Adapted from SunFounder PiDog implementation.
"""

from math import cos, pi


class Walk:
    """
    Walking gait generator.

    Generates smooth walking motion using sequential leg lifting pattern.
    One leg lifts at a time in the order: 1 → 4 → 2 → 3

    A full walk cycle is divided into 8 sections, each with 6 steps.
    """

    # Direction constants
    FORWARD = 1
    BACKWARD = -1
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

    # Gait timing parameters
    SECTION_COUNT = 8  # Number of sections in one complete cycle
    STEP_COUNT = 6     # Number of steps per section
    LEG_ORDER = [1, 0, 4, 0, 2, 0, 3, 0]  # Sequence: leg1, pause, leg4, pause, leg2, pause, leg3, pause

    # Movement parameters (in mm)
    LEG_STEP_HEIGHT = 20     # How high the leg lifts
    LEG_STEP_WIDTH = 80      # How far forward/back the leg moves
    CENTER_OF_GRAVITY = -15  # Body center of gravity offset
    LEG_POSITION_OFFSETS = [-10, -10, 20, 20]  # Per-leg offsets for balance
    Z_ORIGIN = 80            # Standing height

    # Turning parameters
    TURNING_RATE = 0.3
    LEG_STEP_SCALES_LEFT = [TURNING_RATE, 1, TURNING_RATE, 1]  # Scale factors for left turn
    LEG_STEP_SCALES_MIDDLE = [1, 1, 1, 1]                      # No turning
    LEG_STEP_SCALES_RIGHT = [1, TURNING_RATE, 1, TURNING_RATE] # Scale factors for right turn
    LEG_ORIGINAL_Y_TABLE = [0, 2, 3, 1]  # Initial leg positions in cycle
    LEG_STEP_SCALES = [LEG_STEP_SCALES_LEFT, LEG_STEP_SCALES_MIDDLE, LEG_STEP_SCALES_RIGHT]

    def __init__(self, fb=FORWARD, lr=STRAIGHT):
        """
        Initialize walk gait.

        Args:
            fb: Direction - FORWARD (1) or BACKWARD (-1)
            lr: Turn - LEFT (-1), STRAIGHT (0), or RIGHT (1)
        """
        self.fb = fb
        self.lr = lr

        # Calculate center of gravity offset
        self.y_offset = 0 + self.CENTER_OF_GRAVITY

        # Calculate step parameters for each leg
        self.leg_step_width = [
            self.LEG_STEP_WIDTH * self.LEG_STEP_SCALES[self.lr + 1][i] for i in range(4)
        ]

        self.section_length = [
            self.leg_step_width[i] / (self.SECTION_COUNT - 1) for i in range(4)
        ]

        self.step_down_length = [
            self.section_length[i] / self.STEP_COUNT for i in range(4)
        ]

        self.leg_origin = [
            self.leg_step_width[i] / 2 + self.y_offset +
            (self.LEG_POSITION_OFFSETS[i] * self.LEG_STEP_SCALES[self.lr + 1][i])
            for i in range(4)
        ]

    def step_y_func(self, leg, step):
        """
        Calculate Y coordinate for a stepping leg (uses cosine for smooth motion).

        Args:
            leg (int): Leg index (0-3)
            step (int): Current step number (0 to STEP_COUNT-1)

        Returns:
            float: Y coordinate in mm
        """
        theta = step * pi / (self.STEP_COUNT - 1)
        temp = (self.leg_step_width[leg] * (cos(theta) - self.fb) / 2 * self.fb)
        y = self.leg_origin[leg] + temp  # Original: +temp (reverted from -temp)
        return y

    def step_z_func(self, step):
        """
        Calculate Z coordinate for a stepping leg (linear descent).

        Args:
            step (int): Current step number (0 to STEP_COUNT-1)

        Returns:
            float: Z coordinate in mm
        """
        return self.Z_ORIGIN - (self.LEG_STEP_HEIGHT * step / (self.STEP_COUNT - 1))

    def get_coords(self):
        """
        Generate complete walk cycle coordinates.

        Returns:
            list: List of 4-leg coordinate sets for each timestep
                  [[[y1,z1], [y2,z2], [y3,z3], [y4,z4]], ...]
                  Total frames: SECTION_COUNT * STEP_COUNT + 1
        """
        # Starting position for all legs
        origin_leg_coord = [
            [
                self.leg_origin[i] - self.LEG_ORIGINAL_Y_TABLE[i] * 2 * self.section_length[i],
                self.Z_ORIGIN
            ]
            for i in range(4)
        ]

        leg_coord = list(origin_leg_coord)
        leg_coords = []

        # Generate coordinates for each section and step
        for section in range(self.SECTION_COUNT):
            for step in range(self.STEP_COUNT):
                # Determine which leg is lifting this section
                if self.fb == 1:  # Forward
                    raise_leg = self.LEG_ORDER[section]
                else:  # Backward
                    raise_leg = self.LEG_ORDER[self.SECTION_COUNT - section - 1]

                # Update all 4 legs
                for i in range(4):
                    if raise_leg != 0 and i == raise_leg - 1:
                        # This leg is lifting - use step motion
                        y = self.step_y_func(i, step)
                        z = self.step_z_func(step)
                    else:
                        # Other legs slide on ground
                        y = leg_coord[i][0] + self.step_down_length[i] * self.fb  # Original: + (reverted from -)
                        z = self.Z_ORIGIN

                    leg_coord[i] = [y, z]

                leg_coords.append([list(coord) for coord in leg_coord])

        # Add final position to loop back
        leg_coords.append(origin_leg_coord)

        return leg_coords
