"""
Optimized Vision Environment for Speed-Focused + Safe Training

Goals:
1. Balanced speed+stability (leaning towards speed)
2. CRITICAL obstacle avoidance (hardware safety)
3. Sim-to-real robustness

Key Changes from Original:
- Increased speed rewards (8.0 → 10.0)
- More aggressive obstacle penalties
- Added speed milestones for progressive improvement
- Better sim-to-real transfer rewards
"""

# Import base environment
from .pidog_rl_env_vision import PiDogVisionEnv
import numpy as np
import math
from scipy.spatial.transform import Rotation


class PiDogVisionEnvOptimized(PiDogVisionEnv):
    """
    Optimized vision environment with tuned rewards for:
    - Speed-focused locomotion
    - Critical obstacle avoidance
    - Sim-to-real transfer
    """

    def __init__(self, node_name='pidog_vision_env_opt', headless=False):
        super().__init__(node_name, headless)

        # Track performance for curriculum learning
        self.best_speed = 0.0
        self.episode_speed_samples = []

    def _calculate_reward(self, action):
        """
        OPTIMIZED reward function for speed + safety + sim-to-real.

        Priority:
        1. Don't fall (critical for hardware)
        2. Go fast (primary objective)
        3. CRITICAL obstacle avoidance (hardware safety)
        4. Smooth movements (sim-to-real transfer)
        """
        reward = 0.0
        done = False
        info = {}

        # Parse gait command
        gait_vec = self.gait_params.get(self.target_gait, [0.0, 0.0, 0.0])
        desired_direction = gait_vec[1] if len(gait_vec) > 1 else 0.0
        desired_turn = gait_vec[2] if len(gait_vec) > 2 else 0.0

        # Get orientation
        qx, qy, qz, qw = self.imu_orientation
        try:
            r = Rotation.from_quat([qx, qy, qz, qw])
            euler = r.as_euler('xyz')
            roll, pitch, yaw = euler
        except:
            roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
            pitch = math.asin(np.clip(2*(qw*qy - qz*qx), -1, 1))
            yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

        # Get velocities
        vel_x, vel_y, vel_z = self.body_velocity
        forward_vel = vel_x
        lateral_vel = vel_y
        vertical_vel = vel_z
        speed = np.linalg.norm(self.body_velocity[:2])

        # Track speed for curriculum
        self.episode_speed_samples.append(speed)

        # ========================================
        # 1. STABILITY (Critical for Hardware)
        # ========================================

        # Height reward - SLIGHTLY more tolerant for speed
        target_height = 0.10
        height_error = abs(self.body_position[2] - target_height)

        if self.body_position[2] > 0.07:  # Relaxed from 0.08 for speed
            reward += 1.5 - height_error * 6.0  # Reduced penalty (was 8.0)
        else:
            reward -= 3.0  # Still penalize crouching

        # Orientation - BALANCED for speed+stability
        roll_penalty = abs(roll) * 1.5  # Reduced from 2.0
        pitch_penalty = abs(pitch) * 1.5  # Reduced from 2.0

        if abs(roll) < 0.3 and abs(pitch) < 0.3:  # More tolerant (was 0.2)
            reward += 1.5
        else:
            reward -= roll_penalty + pitch_penalty

        # Head clearance - CRITICAL
        head_z = self.body_position[2] + 0.05
        if head_z < 0.02:
            reward -= 10.0  # Increased from 8.0
            done = True
            info['head_contact'] = True

        # Fall detection - CRITICAL
        if abs(roll) > 1.2 or abs(pitch) > 1.2 or self.body_position[2] < 0.05:
            reward -= 20.0  # Increased from 15.0 for hardware safety
            done = True
            info['fallen'] = True

        # ========================================
        # 2. OBSTACLE AVOIDANCE (CRITICAL!)
        # ========================================

        # MORE AGGRESSIVE obstacle avoidance for hardware safety
        critical_distance = 0.15  # 15cm = CRITICAL
        danger_distance = 0.30     # 30cm = DANGER (increased from 0.20)
        safe_distance = 0.60       # 60cm = WARNING (increased from 0.50)

        if self.ultrasonic_range < critical_distance:
            # CRITICAL: Immediate danger
            reward -= 15.0  # Massive penalty (was 5.0)
            info['obstacle_critical'] = True
            # HARD penalty for approaching
            if forward_vel > 0:
                reward -= forward_vel * 10.0  # Increased from 3.0
        elif self.ultrasonic_range < danger_distance:
            # DANGER: Very close
            reward -= 8.0  # Increased from 5.0
            info['obstacle_danger'] = True
            if forward_vel > 0:
                reward -= forward_vel * 5.0
        elif self.ultrasonic_range < safe_distance:
            # WARNING: Approaching
            proximity_penalty = (safe_distance - self.ultrasonic_range) / safe_distance
            reward -= proximity_penalty * 4.0  # Increased from 2.0
            info['obstacle_warning'] = True
        else:
            # SAFE: Bonus for good distance
            reward += 0.5  # Increased from 0.3

        # ========================================
        # 3. SPEED REWARDS (Primary Objective)
        # ========================================

        if self.target_gait in ['trot_forward', 'walk_forward']:
            # INCREASED speed multiplier for speed-focused training
            reward += forward_vel * 10.0  # Increased from 8.0

            # Progressive speed milestones
            if forward_vel > 0.10:  # 10 cm/s
                reward += 1.0
            if forward_vel > 0.15:  # 15 cm/s
                reward += 2.0
            if forward_vel > 0.20:  # 20 cm/s (ambitious!)
                reward += 5.0  # BIG bonus
            if forward_vel > 0.25:  # 25 cm/s (very fast!)
                reward += 10.0  # HUGE bonus

            # Penalize backward movement
            if forward_vel < 0:
                reward -= abs(forward_vel) * 5.0  # Increased from 3.0

            # Penalize lateral drift (stay straight)
            reward -= abs(lateral_vel) * 2.0  # Increased from 1.5

        elif self.target_gait in ['trot_backward', 'walk_backward']:
            # Backward motion
            if desired_direction < 0:
                reward += abs(forward_vel) * 8.0 if forward_vel < 0 else -forward_vel * 5.0
            reward -= abs(lateral_vel) * 2.0

        elif 'left' in self.target_gait or 'right' in self.target_gait:
            # Turning
            if desired_turn != 0:
                reward += abs(lateral_vel) * 6.0  # Increased from 5.0
            reward -= abs(forward_vel) * 1.5  # Penalize drift

        # Penalize bouncing (important for sim-to-real)
        reward -= abs(vertical_vel) * 3.0  # Increased from 2.0

        # ========================================
        # 4. AGILITY & SIM-TO-REAL TRANSFER
        # ========================================

        # Smooth movements (CRITICAL for hardware)
        joint_accel = self.joint_positions - self.last_joint_positions
        jerk_penalty = np.sum(np.square(joint_accel)) * 0.02  # Increased from 0.015
        reward -= jerk_penalty

        # Energy efficiency (battery life on hardware)
        if speed < 0.05:
            energy_waste = np.sum(np.square(self.joint_velocities)) * 0.03  # Increased from 0.02
            reward -= energy_waste

        # Penalize extreme joint positions (hardware limits)
        extreme_joints = np.sum(np.abs(self.joint_positions) > 1.4)  # > 80 degrees
        if extreme_joints > 0:
            reward -= extreme_joints * 0.5

        # Angular velocity control
        yaw_rate = self.imu_angular_vel[2]
        if desired_turn != 0:
            reward += abs(yaw_rate) * desired_turn * 2.0
        else:
            reward -= abs(yaw_rate) * 1.5  # Increased penalty for drift

        # ========================================
        # 5. TASK COMPLETION & CURRICULUM
        # ========================================

        # Episode completion bonus
        if self.episode_step >= self.max_episode_steps:
            done = True
            info['timeout'] = True

            # Calculate average speed this episode
            avg_speed = np.mean(self.episode_speed_samples) if self.episode_speed_samples else 0.0

            # Completion bonus scaled by performance
            if not info.get('fallen', False):
                base_bonus = 5.0
                speed_bonus = avg_speed * 20.0  # Reward average speed
                reward += base_bonus + speed_bonus

                # Track best performance
                if avg_speed > self.best_speed:
                    self.best_speed = avg_speed
                    reward += 10.0  # Personal best bonus!
                    info['new_best'] = True

        # ========================================
        # Info logging
        # ========================================
        info['body_z'] = self.body_position[2]
        info['forward_vel'] = forward_vel
        info['speed'] = speed
        info['ultrasonic_range'] = self.ultrasonic_range
        info['roll'] = roll
        info['pitch'] = pitch
        info['jerk_penalty'] = jerk_penalty

        return reward, done, info

    def reset(self, seed=None, options=None):
        """Reset with speed tracking."""
        # Reset speed tracking
        self.episode_speed_samples = []

        # Call parent reset
        return super().reset(seed=seed, options=options)
