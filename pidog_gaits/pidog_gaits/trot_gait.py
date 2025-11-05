"""
Trot Gait Generator

Diagonal leg pairing pattern for faster movement.
Adapted from SunFounder PiDog implementation.
"""

from math import cos, pi


class Trot:
    """
    Trotting gait generator.

    Generates trotting motion using diagonal leg pairing.
    Legs move in pairs: (1,4) together, then (2,3) together.
    Faster than walking, more dynamic.
    """

    # Direction constants
    FORWARD = 1
    BACKWARD = -1
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

    # Gait timing parameters
    SECTION_COUNT = 2  # Only 2 sections (two diagonal pairs)
    STEP_COUNT = 3     # Steps per section
    LEG_RAISE_ORDER = [[1, 4], [2, 3]]  # Diagonal pairs

    # Movement parameters (in mm)
    LEG_STEP_HEIGHT = 20      # How high the leg lifts
    LEG_STEP_WIDTH = 100      # How far forward/back (wider than walk)
    CENTER_OF_GRAVITY = -17   # Body center of gravity offset
    LEG_STAND_OFFSET = 5      # Leg stance offset
    Z_ORIGIN = 80             # Standing height

    # Turning parameters
    TURNING_RATE = 0.5
    LEG_STAND_OFFSET_DIRS = [-1, -1, 1, 1]  # Offset directions per leg
    LEG_STEP_SCALES_LEFT = [TURNING_RATE, 1, TURNING_RATE, 1]
    LEG_STEP_SCALES_MIDDLE = [1, 1, 1, 1]
    LEG_STEP_SCALES_RIGHT = [1, TURNING_RATE, 1, TURNING_RATE]
    LEG_ORIGINAL_Y_TABLE = [0, 1, 1, 0]  # Initial positions
    LEG_STEP_SCALES = [LEG_STEP_SCALES_LEFT, LEG_STEP_SCALES_MIDDLE, LEG_STEP_SCALES_RIGHT]

    def __init__(self, fb=FORWARD, lr=STRAIGHT):
        """
        Initialize trot gait.

        Args:
            fb: Direction - FORWARD (1) or BACKWARD (-1)
            lr: Turn - LEFT (-1), STRAIGHT (0), or RIGHT (1)
        """
        self.fb = fb
        self.lr = lr

        # Calculate center of gravity offset based on direction
        if self.fb == self.FORWARD:
            if self.lr == self.STRAIGHT:
                self.y_offset = 0 + self.CENTER_OF_GRAVITY
            else:
                self.y_offset = -2 + self.CENTER_OF_GRAVITY
        elif self.fb == self.BACKWARD:
            if self.lr == self.STRAIGHT:
                self.y_offset = 8 + self.CENTER_OF_GRAVITY
            else:
                self.y_offset = 1 + self.CENTER_OF_GRAVITY
        else:
            self.y_offset = self.CENTER_OF_GRAVITY

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

        self.leg_offset = [
            self.LEG_STAND_OFFSET * self.LEG_STAND_OFFSET_DIRS[i] for i in range(4)
        ]

        self.leg_origin = [
            self.leg_step_width[i] / 2 + self.y_offset +
            (self.leg_offset[i] * self.LEG_STEP_SCALES[self.lr + 1][i])
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
        Generate complete trot cycle coordinates.

        Returns:
            list: List of 4-leg coordinate sets for each timestep
                  [[[y1,z1], [y2,z2], [y3,z3], [y4,z4]], ...]
                  Total frames: SECTION_COUNT * STEP_COUNT
        """
        # Starting position for all legs
        origin_leg_coord = [
            [
                self.leg_origin[i] - self.LEG_ORIGINAL_Y_TABLE[i] * self.section_length[i],
                self.Z_ORIGIN
            ]
            for i in range(4)
        ]

        leg_coords = []

        # Generate coordinates for each section and step
        for section in range(self.SECTION_COUNT):
            for step in range(self.STEP_COUNT):
                # Determine which legs are lifting this section (diagonal pair)
                if self.fb == 1:  # Forward
                    raise_legs = self.LEG_RAISE_ORDER[section]
                else:  # Backward
                    raise_legs = self.LEG_RAISE_ORDER[self.SECTION_COUNT - section - 1]

                leg_coord = []

                # Update all 4 legs
                for i in range(4):
                    if (i + 1) in raise_legs:
                        # This leg is lifting - use step motion
                        y = self.step_y_func(i, step)
                        z = self.step_z_func(step)
                    else:
                        # Other legs slide on ground
                        y = origin_leg_coord[i][0] + self.step_down_length[i] * self.fb  # Original: + (reverted from -)
                        z = self.Z_ORIGIN

                    leg_coord.append([y, z])

                origin_leg_coord = leg_coord
                leg_coords.append([list(coord) for coord in leg_coord])

        return leg_coords
