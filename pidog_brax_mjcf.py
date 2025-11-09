"""
Generate MuJoCo XML (MJCF) from PiDog URDF for Brax

Brax v2+ uses MuJoCo's XML format (MJCF) as the standard system description.
This script converts PiDog URDF to MJCF for Brax training.
"""


def generate_pidog_mjcf():
    """
    Generate MJCF XML for PiDog quadruped

    Based on pidog.urdf structure with simplified geometry for Brax.

    Returns:
        str: MJCF XML string
    """

    mjcf_xml = """
<mujoco model="pidog">
  <compiler angle="radian" coordinate="local" inertiafromgeom="false"/>

  <option timestep="0.02" iterations="50" solver="Newton" jacobian="dense">
    <flag warmstart="enable"/>
  </option>

  <default>
    <joint axis="0 0 1" limited="true" damping="0.5" armature="0.01"/>
    <geom contype="1" conaffinity="1" condim="3" friction="0.8 0.005 0.0001"
          rgba="0.8 0.6 0.4 1"/>
    <motor ctrllimited="true" ctrlrange="-1.57 1.57"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.3 0.3"
             rgb2="0.4 0.4 0.4" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>

    <!-- Main body -->
    <body name="torso" pos="0 0 0.1">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.3"
                diaginertia="3.85e-4 1.60e-4 3.45e-4"/>
      <geom name="torso_geom" type="box" size="0.05 0.05 0.05"
            rgba="0.4 0.4 0.8 1"/>

      <!-- ========== BACK RIGHT LEG ========== -->
      <body name="br_upper" pos="0.0405 -0.06685 -0.009">
        <joint name="br_shoulder" type="hinge" axis="0 1 0"
               range="-1.57 1.57" damping="0.5"/>
        <inertial pos="0.021 0 0" mass="0.010"
                  diaginertia="2.58e-7 2.78e-6 3.03e-6"/>
        <geom name="br_upper_geom" type="capsule" size="0.008"
              fromto="0 0 0 0.042 0 0" rgba="0.8 0.3 0.3 1"/>

        <body name="br_lower" pos="0.042 0 0">
          <joint name="br_knee" type="hinge" axis="0 0 1"
                 range="-1.57 1.57" damping="0.5"/>
          <inertial pos="0.038 0 0" mass="0.010"
                    diaginertia="2.58e-7 2.78e-6 3.03e-6"/>
          <geom name="br_lower_geom" type="capsule" size="0.008"
                fromto="0 0 0 0.076 0 0" rgba="0.8 0.3 0.3 1"/>

          <body name="br_foot" pos="0.076 0 0">
            <inertial pos="0 0 0" mass="0.003"
                      diaginertia="1.64e-8 6.52e-8 7.99e-8"/>
            <geom name="br_foot_geom" type="sphere" size="0.008"
                  rgba="0.2 0.2 0.2 1" friction="1.0 0.005 0.0001"/>
          </body>
        </body>
      </body>

      <!-- ========== FRONT RIGHT LEG ========== -->
      <body name="fr_upper" pos="0.0405 0.05315 -0.009">
        <joint name="fr_shoulder" type="hinge" axis="0 1 0"
               range="-1.57 1.57" damping="0.5"/>
        <inertial pos="0.021 0 0" mass="0.010"
                  diaginertia="2.58e-7 2.78e-6 3.03e-6"/>
        <geom name="fr_upper_geom" type="capsule" size="0.008"
              fromto="0 0 0 0.042 0 0" rgba="0.8 0.3 0.3 1"/>

        <body name="fr_lower" pos="0.042 0 0">
          <joint name="fr_knee" type="hinge" axis="0 0 1"
                 range="-1.57 1.57" damping="0.5"/>
          <inertial pos="0.038 0 0" mass="0.010"
                    diaginertia="2.58e-7 2.78e-6 3.03e-6"/>
          <geom name="fr_lower_geom" type="capsule" size="0.008"
                fromto="0 0 0 0.076 0 0" rgba="0.8 0.3 0.3 1"/>

          <body name="fr_foot" pos="0.076 0 0">
            <inertial pos="0 0 0" mass="0.003"
                      diaginertia="1.64e-8 6.52e-8 7.99e-8"/>
            <geom name="fr_foot_geom" type="sphere" size="0.008"
                  rgba="0.2 0.2 0.2 1" friction="1.0 0.005 0.0001"/>
          </body>
        </body>
      </body>

      <!-- ========== BACK LEFT LEG ========== -->
      <body name="bl_upper" pos="-0.0405 -0.06685 -0.009">
        <joint name="bl_shoulder" type="hinge" axis="0 1 0"
               range="-1.57 1.57" damping="0.5"/>
        <inertial pos="-0.021 0 0" mass="0.010"
                  diaginertia="2.58e-7 2.78e-6 3.03e-6"/>
        <geom name="bl_upper_geom" type="capsule" size="0.008"
              fromto="0 0 0 -0.042 0 0" rgba="0.3 0.8 0.3 1"/>

        <body name="bl_lower" pos="-0.042 0 0">
          <joint name="bl_knee" type="hinge" axis="0 0 1"
                 range="-1.57 1.57" damping="0.5"/>
          <inertial pos="-0.038 0 0" mass="0.010"
                    diaginertia="2.58e-7 2.78e-6 3.03e-6"/>
          <geom name="bl_lower_geom" type="capsule" size="0.008"
                fromto="0 0 0 -0.076 0 0" rgba="0.3 0.8 0.3 1"/>

          <body name="bl_foot" pos="-0.076 0 0">
            <inertial pos="0 0 0" mass="0.003"
                      diaginertia="1.64e-8 6.52e-8 7.99e-8"/>
            <geom name="bl_foot_geom" type="sphere" size="0.008"
                  rgba="0.2 0.2 0.2 1" friction="1.0 0.005 0.0001"/>
          </body>
        </body>
      </body>

      <!-- ========== FRONT LEFT LEG ========== -->
      <body name="fl_upper" pos="-0.0405 0.05315 -0.009">
        <joint name="fl_shoulder" type="hinge" axis="0 1 0"
               range="-1.57 1.57" damping="0.5"/>
        <inertial pos="-0.021 0 0" mass="0.010"
                  diaginertia="2.58e-7 2.78e-6 3.03e-6"/>
        <geom name="fl_upper_geom" type="capsule" size="0.008"
              fromto="0 0 0 -0.042 0 0" rgba="0.3 0.8 0.3 1"/>

        <body name="fl_lower" pos="-0.042 0 0">
          <joint name="fl_knee" type="hinge" axis="0 0 1"
                 range="-1.57 1.57" damping="0.5"/>
          <inertial pos="-0.038 0 0" mass="0.010"
                    diaginertia="2.58e-7 2.78e-6 3.03e-6"/>
          <geom name="fl_lower_geom" type="capsule" size="0.008"
                fromto="0 0 0 -0.076 0 0" rgba="0.3 0.8 0.3 1"/>

          <body name="fl_foot" pos="-0.076 0 0">
            <inertial pos="0 0 0" mass="0.003"
                      diaginertia="1.64e-8 6.52e-8 7.99e-8"/>
            <geom name="fl_foot_geom" type="sphere" size="0.008"
                  rgba="0.2 0.2 0.2 1" friction="1.0 0.005 0.0001"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- Position servos with torque limits matching real hardware (0.15 Nm) -->
    <position name="br_shoulder_act" joint="br_shoulder"
              ctrlrange="-1.57 1.57" kp="150" kv="10"/>
    <position name="br_knee_act" joint="br_knee"
              ctrlrange="-1.57 1.57" kp="150" kv="10"/>
    <position name="fr_shoulder_act" joint="fr_shoulder"
              ctrlrange="-1.57 1.57" kp="150" kv="10"/>
    <position name="fr_knee_act" joint="fr_knee"
              ctrlrange="-1.57 1.57" kp="150" kv="10"/>
    <position name="bl_shoulder_act" joint="bl_shoulder"
              ctrlrange="-1.57 1.57" kp="150" kv="10"/>
    <position name="bl_knee_act" joint="bl_knee"
              ctrlrange="-1.57 1.57" kp="150" kv="10"/>
    <position name="fl_shoulder_act" joint="fl_shoulder"
              ctrlrange="-1.57 1.57" kp="150" kv="10"/>
    <position name="fl_knee_act" joint="fl_knee"
              ctrlrange="-1.57 1.57" kp="150" kv="10"/>
  </actuator>

  <sensor>
    <!-- IMU sensors for observation -->
    <framequat name="torso_quat" objtype="body" objname="torso"/>
    <framepos name="torso_pos" objtype="body" objname="torso"/>
    <framelinvel name="torso_vel" objtype="body" objname="torso"/>
    <frameangvel name="torso_angvel" objtype="body" objname="torso"/>
  </sensor>
</mujoco>
"""

    return mjcf_xml.strip()


if __name__ == '__main__':
    """Generate and save MJCF XML"""
    xml = generate_pidog_mjcf()

    # Save to file
    output_path = 'pidog.xml'
    with open(output_path, 'w') as f:
        f.write(xml)

    print(f"✅ Generated MJCF XML: {output_path}")
    print(f"   Size: {len(xml)} bytes")
    print("\nYou can now use this with Brax:")
    print("  from brax.io import mjcf")
    print("  sys = mjcf.load('pidog.xml')")
