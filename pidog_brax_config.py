"""
PiDog Brax System Configuration
Converted from pidog.urdf for GPU-accelerated training

Based on URDF structure:
- Body: main link (0.3 kg)
- 4 legs × 2 DOF = 8 actuated joints
- 4 feet (acrylic pads with sphere collision)
- Simplified geometry (no meshes - use primitives)
"""

from brax import base
from brax.base import Motion, Transform
from brax.io import mjcf
import jax.numpy as jnp

def create_pidog_config():
    """
    Create Brax system config for PiDog quadruped

    Coordinate system (Brax uses standard physics convention):
    - X: forward
    - Y: left
    - Z: up

    Joint naming convention:
    - Motor 0-1: Back Right (BR) - shoulder, knee
    - Motor 2-3: Front Right (FR) - shoulder, knee
    - Motor 4-5: Back Left (BL) - shoulder, knee
    - Motor 6-7: Front Left (FL) - shoulder, knee
    """

    # System parameters (from URDF)
    LEG_UPPER_LENGTH = 0.042  # meters (42mm)
    LEG_LOWER_LENGTH = 0.076  # meters (76mm)

    # Joint positions on body (from URDF origins)
    # Back right: xyz="0.0405 -0.06685 -0.009"
    # Front right: xyz="0.0405 0.05315 -0.009"
    # Back left: xyz="-0.0405 -0.06685 -0.009" (mirrored)
    # Front left: xyz="-0.0405 0.05315 -0.009" (mirrored)

    config = {
        'dt': 0.02,  # 50Hz timestep (same as ros2_control)
        'substeps': 10,  # 10 physics substeps per control step
        'gravity': (0, 0, -9.81),

        'bodies': [
            # Main body
            {
                'name': 'body',
                'colliders': [{
                    'box': {'halfsize': (0.05, 0.05, 0.05)},  # Simplified box collision
                }],
                'inertia': {
                    'mass': 0.300,  # 300g total (RPi, battery, structure)
                    'i': (3.85e-4, 1.60e-4, 3.45e-4),  # From URDF inertia diagonal
                },
                'frozen': {'position': (False, False, False),
                          'rotation': (False, False, False)}
            },

            # ============ BACK RIGHT LEG ============
            # Upper leg (shoulder to knee)
            {
                'name': 'br_upper',
                'colliders': [{
                    'capsule': {
                        'radius': 0.008,
                        'length': LEG_UPPER_LENGTH,
                        'end': (LEG_UPPER_LENGTH, 0, 0)
                    }
                }],
                'inertia': {'mass': 0.010, 'i': (2.58e-7, 2.78e-6, 3.03e-6)}
            },
            # Lower leg (knee to foot)
            {
                'name': 'br_lower',
                'colliders': [{
                    'capsule': {
                        'radius': 0.008,
                        'length': LEG_LOWER_LENGTH,
                        'end': (LEG_LOWER_LENGTH, 0, 0)
                    }
                }],
                'inertia': {'mass': 0.010, 'i': (2.58e-7, 2.78e-6, 3.03e-6)}
            },
            # Foot (acrylic pad with sphere collision)
            {
                'name': 'br_foot',
                'colliders': [{
                    'sphere': {'radius': 0.008}
                }],
                'inertia': {'mass': 0.003, 'i': (1.64e-8, 6.52e-8, 7.99e-8)},
                'frozen': {'position': (False, False, False),
                          'rotation': (True, True, True)}  # Foot doesn't rotate
            },

            # ============ FRONT RIGHT LEG ============
            {
                'name': 'fr_upper',
                'colliders': [{
                    'capsule': {
                        'radius': 0.008,
                        'length': LEG_UPPER_LENGTH,
                        'end': (LEG_UPPER_LENGTH, 0, 0)
                    }
                }],
                'inertia': {'mass': 0.010, 'i': (2.58e-7, 2.78e-6, 3.03e-6)}
            },
            {
                'name': 'fr_lower',
                'colliders': [{
                    'capsule': {
                        'radius': 0.008,
                        'length': LEG_LOWER_LENGTH,
                        'end': (LEG_LOWER_LENGTH, 0, 0)
                    }
                }],
                'inertia': {'mass': 0.010, 'i': (2.58e-7, 2.78e-6, 3.03e-6)}
            },
            {
                'name': 'fr_foot',
                'colliders': [{
                    'sphere': {'radius': 0.008}
                }],
                'inertia': {'mass': 0.003, 'i': (1.64e-8, 6.52e-8, 7.99e-8)},
                'frozen': {'position': (False, False, False),
                          'rotation': (True, True, True)}
            },

            # ============ BACK LEFT LEG ============
            {
                'name': 'bl_upper',
                'colliders': [{
                    'capsule': {
                        'radius': 0.008,
                        'length': LEG_UPPER_LENGTH,
                        'end': (LEG_UPPER_LENGTH, 0, 0)
                    }
                }],
                'inertia': {'mass': 0.010, 'i': (2.58e-7, 2.78e-6, 3.03e-6)}
            },
            {
                'name': 'bl_lower',
                'colliders': [{
                    'capsule': {
                        'radius': 0.008,
                        'length': LEG_LOWER_LENGTH,
                        'end': (LEG_LOWER_LENGTH, 0, 0)
                    }
                }],
                'inertia': {'mass': 0.010, 'i': (2.58e-7, 2.78e-6, 3.03e-6)}
            },
            {
                'name': 'bl_foot',
                'colliders': [{
                    'sphere': {'radius': 0.008}
                }],
                'inertia': {'mass': 0.003, 'i': (1.64e-8, 6.52e-8, 7.99e-8)},
                'frozen': {'position': (False, False, False),
                          'rotation': (True, True, True)}
            },

            # ============ FRONT LEFT LEG ============
            {
                'name': 'fl_upper',
                'colliders': [{
                    'capsule': {
                        'radius': 0.008,
                        'length': LEG_UPPER_LENGTH,
                        'end': (LEG_UPPER_LENGTH, 0, 0)
                    }
                }],
                'inertia': {'mass': 0.010, 'i': (2.58e-7, 2.78e-6, 3.03e-6)}
            },
            {
                'name': 'fl_lower',
                'colliders': [{
                    'capsule': {
                        'radius': 0.008,
                        'length': LEG_LOWER_LENGTH,
                        'end': (LEG_LOWER_LENGTH, 0, 0)
                    }
                }],
                'inertia': {'mass': 0.010, 'i': (2.58e-7, 2.78e-6, 3.03e-6)}
            },
            {
                'name': 'fl_foot',
                'colliders': [{
                    'sphere': {'radius': 0.008}
                }],
                'inertia': {'mass': 0.003, 'i': (1.64e-8, 6.52e-8, 7.99e-8)},
                'frozen': {'position': (False, False, False),
                          'rotation': (True, True, True)}
            },
        ],

        'joints': [
            # ============ BACK RIGHT LEG ============
            # Shoulder (body to upper leg) - revolute around Y axis
            {
                'name': 'br_shoulder',
                'parent': 'body',
                'child': 'br_upper',
                'parent_offset': (0.0405, -0.06685, -0.009),  # From URDF
                'rotation': (0, 90, 0),  # rpy="0 1.57 0" from URDF
                'stiffness': 5000,
                'angular_damping': 20,
                'limit_strength': 500,
                'angle_limit': (-1.57, 1.57),  # ±90° from URDF limits
                'reference_rotation': (0, 0, 0),
            },
            # Knee (upper leg to lower leg) - revolute around Z axis
            {
                'name': 'br_knee',
                'parent': 'br_upper',
                'child': 'br_lower',
                'parent_offset': (LEG_UPPER_LENGTH, 0, 0),  # End of upper leg
                'rotation': (0, 0, 90),  # rpy="0 0 1.57" from URDF
                'stiffness': 5000,
                'angular_damping': 20,
                'limit_strength': 500,
                'angle_limit': (-1.57, 1.57),
            },
            # Ankle (lower leg to foot) - fixed
            {
                'name': 'br_ankle',
                'parent': 'br_lower',
                'child': 'br_foot',
                'parent_offset': (LEG_LOWER_LENGTH, 0, 0),
                'stiffness': 10000,  # Very stiff = effectively fixed
                'angular_damping': 100,
            },

            # ============ FRONT RIGHT LEG ============
            {
                'name': 'fr_shoulder',
                'parent': 'body',
                'child': 'fr_upper',
                'parent_offset': (0.0405, 0.05315, -0.009),
                'rotation': (0, 90, 0),
                'stiffness': 5000,
                'angular_damping': 20,
                'limit_strength': 500,
                'angle_limit': (-1.57, 1.57),
            },
            {
                'name': 'fr_knee',
                'parent': 'fr_upper',
                'child': 'fr_lower',
                'parent_offset': (LEG_UPPER_LENGTH, 0, 0),
                'rotation': (0, 0, 90),
                'stiffness': 5000,
                'angular_damping': 20,
                'limit_strength': 500,
                'angle_limit': (-1.57, 1.57),
            },
            {
                'name': 'fr_ankle',
                'parent': 'fr_lower',
                'child': 'fr_foot',
                'parent_offset': (LEG_LOWER_LENGTH, 0, 0),
                'stiffness': 10000,
                'angular_damping': 100,
            },

            # ============ BACK LEFT LEG ============
            {
                'name': 'bl_shoulder',
                'parent': 'body',
                'child': 'bl_upper',
                'parent_offset': (-0.0405, -0.06685, -0.009),  # Mirrored X
                'rotation': (0, 90, 180),  # Left side mirror
                'stiffness': 5000,
                'angular_damping': 20,
                'limit_strength': 500,
                'angle_limit': (-1.57, 1.57),
            },
            {
                'name': 'bl_knee',
                'parent': 'bl_upper',
                'child': 'bl_lower',
                'parent_offset': (LEG_UPPER_LENGTH, 0, 0),
                'rotation': (0, 0, 90),
                'stiffness': 5000,
                'angular_damping': 20,
                'limit_strength': 500,
                'angle_limit': (-1.57, 1.57),
            },
            {
                'name': 'bl_ankle',
                'parent': 'bl_lower',
                'child': 'bl_foot',
                'parent_offset': (LEG_LOWER_LENGTH, 0, 0),
                'stiffness': 10000,
                'angular_damping': 100,
            },

            # ============ FRONT LEFT LEG ============
            {
                'name': 'fl_shoulder',
                'parent': 'body',
                'child': 'fl_upper',
                'parent_offset': (-0.0405, 0.05315, -0.009),
                'rotation': (0, 90, 180),
                'stiffness': 5000,
                'angular_damping': 20,
                'limit_strength': 500,
                'angle_limit': (-1.57, 1.57),
            },
            {
                'name': 'fl_knee',
                'parent': 'fl_upper',
                'child': 'fl_lower',
                'parent_offset': (LEG_UPPER_LENGTH, 0, 0),
                'rotation': (0, 0, 90),
                'stiffness': 5000,
                'angular_damping': 20,
                'limit_strength': 500,
                'angle_limit': (-1.57, 1.57),
            },
            {
                'name': 'fl_ankle',
                'parent': 'fl_lower',
                'child': 'fl_foot',
                'parent_offset': (LEG_LOWER_LENGTH, 0, 0),
                'stiffness': 10000,
                'angular_damping': 100,
            },
        ],

        'actuators': [
            # Position control actuators (8 total - 2 per leg)
            # Torque limits match real servo: 0.15 Nm (from URDF effort limits)
            {'name': 'br_shoulder_act', 'joint': 'br_shoulder', 'strength': 150.0, 'ctrl_range': (-1.57, 1.57)},
            {'name': 'br_knee_act', 'joint': 'br_knee', 'strength': 150.0, 'ctrl_range': (-1.57, 1.57)},
            {'name': 'fr_shoulder_act', 'joint': 'fr_shoulder', 'strength': 150.0, 'ctrl_range': (-1.57, 1.57)},
            {'name': 'fr_knee_act', 'joint': 'fr_knee', 'strength': 150.0, 'ctrl_range': (-1.57, 1.57)},
            {'name': 'bl_shoulder_act', 'joint': 'bl_shoulder', 'strength': 150.0, 'ctrl_range': (-1.57, 1.57)},
            {'name': 'bl_knee_act', 'joint': 'bl_knee', 'strength': 150.0, 'ctrl_range': (-1.57, 1.57)},
            {'name': 'fl_shoulder_act', 'joint': 'fl_shoulder', 'strength': 150.0, 'ctrl_range': (-1.57, 1.57)},
            {'name': 'fl_knee_act', 'joint': 'fl_knee', 'strength': 150.0, 'ctrl_range': (-1.57, 1.57)},
        ],

        'collisions': [
            # Ground plane contact
            ('br_foot', 'ground'),
            ('fr_foot', 'ground'),
            ('bl_foot', 'ground'),
            ('fl_foot', 'ground'),
            ('body', 'ground'),  # In case robot falls
        ],

        'defaults': {
            'geom_friction': 0.8,  # From URDF mu1
            'geom_restitution': 0.0,  # No bounce
            'joint_damping': 0.5,  # From URDF dynamics
            'joint_stiffness': 5000,
        }
    }

    return config


if __name__ == '__main__':
    """Test configuration validity"""
    config = create_pidog_config()
    print("PiDog Brax Configuration:")
    print(f"  Bodies: {len(config['bodies'])}")
    print(f"  Joints: {len(config['joints'])}")
    print(f"  Actuators: {len(config['actuators'])}")
    print(f"  Timestep: {config['dt']}s ({1/config['dt']:.0f} Hz)")
    print("\nActuator mapping (matches neural network output order):")
    for i, act in enumerate(config['actuators']):
        print(f"  Motor {i}: {act['name']}")
