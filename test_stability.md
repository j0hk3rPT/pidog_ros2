# PiDog Stability Diagnostic

## What I've Changed:

### 1. Ultra-Minimal Control Gains
- **P gain**: 10 → **1.0** (barely any position control)
- **D gain**: 15 → **2.0** (minimal damping)
- **I gain**: 0.01 → **0.0** (completely disabled)
- **Plugin gain**: 5 → **1.0** (minimal force)

### 2. Maximum Joint Damping
- **Damping**: 5.0 → **10.0** (heavy resistance to motion)
- **Friction**: 0.5 → **1.0** (doubled friction)

### 3. Extended Settling Time
- **Delay**: 5s → **15s** (triple the wait time)
- **Controller**: Starts **INACTIVE** (no control at startup)

### 4. Soft Contact Physics
- **kp**: 10,000 (soft contact stiffness)
- **kd**: 1,000 (high contact damping)

## Testing Steps:

1. **Launch Gazebo:**
   ```bash
   ros2 launch pidog_description gazebo.launch.py
   ```

2. **Observe for 15 seconds:**
   - Robot should spawn in standing pose
   - Controller is INACTIVE - no commands sent
   - Watch if bouncing still occurs
   - This tells us if physics or controller is the problem

3. **Check what's happening:**
   - If bouncing STOPS after a few seconds → Physics is settling, this is normal
   - If bouncing NEVER stops → Physics simulation is unstable
   - If bouncing starts AFTER 15s → Controller is causing it

4. **Manual controller activation (optional):**
   ```bash
   ros2 control set_controller_state position_controller active
   ```

## What to Report:

1. **Does bouncing occur immediately at spawn?** (Y/N)
2. **Does bouncing continue past 5 seconds?** (Y/N)
3. **Does bouncing continue past 15 seconds?** (Y/N)
4. **Does the robot eventually settle and stay still?** (Y/N)
5. **When you activate the controller, does bouncing start again?** (Y/N)

This will tell us exactly where the problem is!
