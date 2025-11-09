#!/bin/bash
#
# PiDog Production Training Pipeline
#
# Complete end-to-end training from data collection to deployable model.
# This script is self-contained and creates a timestamped experiment folder.
#
# Usage:
#   ./train_production_pipeline.sh <experiment_name>
#
# Example:
#   ./train_production_pipeline.sh "pidog_fast_runner_v1"
#
# Output:
#   experiments/<experiment_name>_YYYYMMDD_HHMMSS/
#     ├── config.txt                  # Training configuration
#     ├── data/                       # Raw training data
#     ├── imitation_model/            # Imitation learning results
#     ├── rl_model/                   # RL fine-tuning results
#     ├── logs/                       # All logs
#     └── final_model/                # Deployable model + metadata
#

set -e  # Exit on error

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

# CRITICAL: OpenBLAS thread limit for Gazebo performance
# Prevents thread management overhead that reduces CPU usage
# See: https://discourse.openrobotics.org/t/massive-gazebo-performance-improvement-ymmv/39181
export OPENBLAS_NUM_THREADS=4

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment name from argument
EXPERIMENT_NAME=${1:-"pidog_production_$(date +%Y%m%d_%H%M%S)"}

# Create timestamped experiment folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}_${TIMESTAMP}"

# Training parameters
DATA_COLLECTION_DURATION=120  # seconds per gait (2 minutes)
IMITATION_EPOCHS=300           # epochs for imitation learning
IMITATION_BATCH_SIZE=1024      # batch size for imitation
RL_TIMESTEPS=200000            # total RL training steps
RL_PARALLEL_ENVS=4             # parallel environments for RL
DEVICE="cuda"                  # cuda or cpu

# Physics quality: production, medium, or fast
PHYSICS_QUALITY="medium"       # Balanced quality + speed

# =============================================================================
# FUNCTIONS
# =============================================================================

log_header() {
    echo ""
    echo -e "${BLUE}${BOLD}========================================${NC}"
    echo -e "${BLUE}${BOLD}$1${NC}"
    echo -e "${BLUE}${BOLD}========================================${NC}"
    echo ""
}

log_step() {
    echo -e "${GREEN}[$(date +%H:%M:%S)] $1${NC}"
}

log_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

check_prerequisites() {
    log_step "Checking prerequisites..."

    # Check if in ROS2 environment
    if [ -z "$ROS_DISTRO" ]; then
        log_warn "ROS2 not sourced, sourcing now..."
        source /opt/ros/rolling/setup.bash 2>/dev/null || true
        source install/setup.bash 2>/dev/null || true
    fi

    # Check Python packages
    python3 -c "import torch, stable_baselines3, gymnasium" 2>/dev/null || {
        log_error "Required Python packages not installed!"
        exit 1
    }

    # Check NumPy version
    NUMPY_VER=$(python3 -c "import numpy; print(numpy.__version__)")
    if [[ "$NUMPY_VER" == 2.* ]]; then
        log_error "NumPy 2.x detected! Run ./fix_numpy.sh first"
        exit 1
    fi

    log_success "All prerequisites met"
}

setup_experiment() {
    log_step "Setting up experiment directory..."

    mkdir -p "$EXPERIMENT_DIR"/{data,imitation_model,rl_model,logs,final_model}

    # Save configuration
    cat > "$EXPERIMENT_DIR/config.txt" <<EOF
Experiment: $EXPERIMENT_NAME
Started: $(date)
Hostname: $(hostname)

=== Configuration ===
Data Collection Duration: ${DATA_COLLECTION_DURATION}s per gait
Imitation Learning Epochs: $IMITATION_EPOCHS
Imitation Batch Size: $IMITATION_BATCH_SIZE
RL Total Timesteps: $RL_TIMESTEPS
RL Parallel Envs: $RL_PARALLEL_ENVS
Physics Quality: $PHYSICS_QUALITY
Device: $DEVICE

=== System Info ===
CPU Cores: $(nproc)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || rocm-smi --showproductname 2>/dev/null | grep "Card series" || echo "No GPU detected")
Python: $(python3 --version)
PyTorch: $(python3 -c "import torch; print(torch.__version__)")
NumPy: $(python3 -c "import numpy; print(numpy.__version__)")
EOF

    log_success "Experiment directory created: $EXPERIMENT_DIR"
    log_info "Configuration saved to: $EXPERIMENT_DIR/config.txt"
}

# =============================================================================
# PIPELINE STAGES
# =============================================================================

stage_1_data_collection() {
    log_header "STAGE 1/4: DATA COLLECTION"

    log_step "Collecting enhanced training data with noise augmentation..."
    log_info "Duration: ${DATA_COLLECTION_DURATION}s per gait"
    log_info "Gaits: walk_forward, walk_backward, trot_forward, stand, sit"

    # Launch data collection (includes Gazebo + gait generator + data collector)
    log_step "Starting Gazebo + gait generator + data collector..."
    ros2 launch pidog_gaits collect_data_enhanced.launch.py \
        > "$EXPERIMENT_DIR/logs/data_collection.log" 2>&1 &
    COLLECTOR_PID=$!

    # Wait for Gazebo to initialize
    log_info "Waiting for Gazebo to stabilize..."
    sleep 15

    if ! kill -0 $COLLECTOR_PID 2>/dev/null; then
        log_error "Data collection launch failed!"
        cat "$EXPERIMENT_DIR/logs/data_collection.log"
        exit 1
    fi

    log_success "Data collection running (PID: $COLLECTOR_PID)"

    # Define gaits to collect
    GAITS=("walk_forward" "walk_backward" "trot_forward" "stand" "sit")
    TOTAL_GAITS=${#GAITS[@]}
    TOTAL_TIME=$((DATA_COLLECTION_DURATION * TOTAL_GAITS))

    log_info "Will collect $TOTAL_GAITS gaits × ${DATA_COLLECTION_DURATION}s = $TOTAL_TIME seconds (~$((TOTAL_TIME / 60)) minutes)"

    # Cycle through each gait
    for i in "${!GAITS[@]}"; do
        GAIT="${GAITS[$i]}"
        NUM=$((i + 1))

        log_step "[$NUM/$TOTAL_GAITS] Recording gait: $GAIT"

        # Set stand pose first
        ros2 topic pub /gait_command std_msgs/msg/String "data: 'stand'" --once >/dev/null 2>&1
        sleep 1

        # Reset robot position in Gazebo
        gz service -s /world/pidog_world/set_pose \
            --reqtype gz.msgs.Pose \
            --reptype gz.msgs.Boolean \
            --timeout 2000 \
            --req "name: 'Robot.urdf', position: {x: 0.0, y: 0.0, z: 0.12}, orientation: {x: 0, y: 0, z: 0, w: 1}" \
            >/dev/null 2>&1 || log_warn "Gazebo reset failed, continuing..."
        sleep 2

        # Start gait
        log_info "  Starting gait: $GAIT (${DATA_COLLECTION_DURATION}s)"
        ros2 topic pub /gait_command std_msgs/msg/String "data: '$GAIT'" --once >/dev/null 2>&1
        sleep 1

        # Wait for collection with progress
        for ((sec=1; sec<=DATA_COLLECTION_DURATION; sec++)); do
            if [ $((sec % 10)) -eq 0 ]; then
                log_info "  Progress: $sec/${DATA_COLLECTION_DURATION}s"
            fi
            sleep 1
        done

        log_success "  Completed: $GAIT"
    done

    # Stop data collection (this triggers save in data_collector_enhanced.py)
    log_step "Stopping data collection..."
    kill -INT $COLLECTOR_PID 2>/dev/null || true
    sleep 3  # Give time to save
    kill -9 $COLLECTOR_PID 2>/dev/null || true

    # Move data to experiment folder
    DATA_FILE=$(ls -t training_data/gait_data_enhanced_*.npz 2>/dev/null | head -1)
    if [ -z "$DATA_FILE" ]; then
        log_error "No data file generated!"
        log_error "Check logs: $EXPERIMENT_DIR/logs/data_collection.log"
        cat "$EXPERIMENT_DIR/logs/data_collection.log" | tail -50
        exit 1
    fi

    cp "$DATA_FILE" "$EXPERIMENT_DIR/data/"
    TRAINING_DATA="$EXPERIMENT_DIR/data/$(basename $DATA_FILE)"

    # Verify data
    SAMPLES=$(python3 -c "import numpy as np; d=np.load('$TRAINING_DATA'); print(d['inputs'].shape[0])")
    log_success "Data collection complete: $SAMPLES samples"
    log_info "Data saved to: $TRAINING_DATA"

    echo "$TRAINING_DATA" > "$EXPERIMENT_DIR/.data_path"
}

stage_2_imitation_learning() {
    log_header "STAGE 2/4: IMITATION LEARNING"

    TRAINING_DATA=$(cat "$EXPERIMENT_DIR/.data_path")

    log_step "Training imitation model (SimpleLSTM)..."
    log_info "Data: $TRAINING_DATA"
    log_info "Epochs: $IMITATION_EPOCHS"
    log_info "Batch size: $IMITATION_BATCH_SIZE"
    log_info "Model: GaitNetSimpleLSTM (best for sim-to-real)"

    python3 -m pidog_gaits.train \
        --data "$TRAINING_DATA" \
        --model simple_lstm \
        --epochs $IMITATION_EPOCHS \
        --batch_size $IMITATION_BATCH_SIZE \
        --device $DEVICE \
        --save_dir "$EXPERIMENT_DIR/imitation_model" \
        2>&1 | tee "$EXPERIMENT_DIR/logs/imitation_training.log"

    if [ ! -f "$EXPERIMENT_DIR/imitation_model/best_model.pth" ]; then
        log_error "Imitation model training failed!"
        exit 1
    fi

    log_success "Imitation learning complete"
    log_info "Model: $EXPERIMENT_DIR/imitation_model/best_model.pth"
    log_info "Training plot: $EXPERIMENT_DIR/imitation_model/training_history.png"

    echo "$EXPERIMENT_DIR/imitation_model/best_model.pth" > "$EXPERIMENT_DIR/.imitation_model_path"
}

stage_3_rl_training() {
    log_header "STAGE 3/4: REINFORCEMENT LEARNING"

    IMITATION_MODEL=$(cat "$EXPERIMENT_DIR/.imitation_model_path")

    log_step "Fine-tuning with RL (vision-based)..."
    log_info "Pretrained model: $IMITATION_MODEL"
    log_info "Total timesteps: $RL_TIMESTEPS"
    log_info "Parallel envs: $RL_PARALLEL_ENVS"
    log_info "Physics: $PHYSICS_QUALITY quality"

    # Select world file based on physics quality
    case $PHYSICS_QUALITY in
        "production")
            WORLD_FILE="pidog.sdf"
            ;;
        "medium")
            WORLD_FILE="pidog_rl_medium.sdf"
            ;;
        "fast")
            WORLD_FILE="pidog_rl_fast.sdf"
            ;;
        *)
            log_warn "Unknown physics quality, using medium"
            WORLD_FILE="pidog_rl_medium.sdf"
            ;;
    esac

    log_info "World file: $WORLD_FILE"

    # Launch Gazebo headless
    log_step "Starting Gazebo ($PHYSICS_QUALITY quality, headless)..."

    # Modify launch to use selected world
    ros2 launch pidog_description gazebo_rl_fast.launch.py \
        > "$EXPERIMENT_DIR/logs/rl_training_gazebo.log" 2>&1 &
    GAZEBO_PID=$!

    sleep 15

    if ! kill -0 $GAZEBO_PID 2>/dev/null; then
        log_error "Gazebo failed to start!"
        cat "$EXPERIMENT_DIR/logs/rl_training_gazebo.log"
        exit 1
    fi

    log_success "Gazebo running (PID: $GAZEBO_PID)"

    # Cleanup function for RL training
    cleanup_rl() {
        log_step "Stopping RL training..."
        kill $GAZEBO_PID 2>/dev/null || true
        sleep 2
        kill -9 $GAZEBO_PID 2>/dev/null || true
    }
    trap cleanup_rl EXIT INT TERM

    # Run RL training
    log_step "Starting RL training (this will take a while)..."
    log_info "Monitor with: tensorboard --logdir $EXPERIMENT_DIR/rl_model/tensorboard"

    python3 -m pidog_gaits.train_rl_vision \
        --pretrained "$IMITATION_MODEL" \
        --output "$EXPERIMENT_DIR/rl_model" \
        --timesteps $RL_TIMESTEPS \
        --envs $RL_PARALLEL_ENVS \
        --device $DEVICE \
        2>&1 | tee "$EXPERIMENT_DIR/logs/rl_training.log"

    # Stop Gazebo
    cleanup_rl
    trap - EXIT INT TERM

    if [ ! -f "$EXPERIMENT_DIR/rl_model/final_model.zip" ]; then
        log_error "RL training failed!"
        exit 1
    fi

    log_success "RL training complete"
    log_info "Model: $EXPERIMENT_DIR/rl_model/final_model.zip"
    log_info "PyTorch model: $EXPERIMENT_DIR/rl_model/final_model.pth"
}

stage_4_finalization() {
    log_header "STAGE 4/4: FINALIZATION"

    log_step "Preparing final deployable model..."

    # Copy final model
    cp "$EXPERIMENT_DIR/rl_model/final_model.pth" "$EXPERIMENT_DIR/final_model/"
    cp "$EXPERIMENT_DIR/rl_model/final_model.zip" "$EXPERIMENT_DIR/final_model/"

    # Create deployment metadata
    cat > "$EXPERIMENT_DIR/final_model/README.md" <<EOF
# PiDog Model: $EXPERIMENT_NAME

Trained: $(date)

## Model Files

- \`final_model.pth\` - PyTorch model for deployment
- \`final_model.zip\` - Stable-Baselines3 model for testing/evaluation

## Training Details

- **Data Collection**: ${DATA_COLLECTION_DURATION}s per gait
- **Imitation Learning**: $IMITATION_EPOCHS epochs, SimpleLSTM
- **RL Training**: $RL_TIMESTEPS timesteps, $RL_PARALLEL_ENVS parallel envs
- **Physics Quality**: $PHYSICS_QUALITY
- **Device**: $DEVICE

## Sensors Used

- Camera: 84x84 RGB vision
- IMU: Orientation + angular velocity
- Ultrasonic: Distance measurement (HC-SR04)
- Joint Encoders: 12 joint positions + velocities

## Testing

\`\`\`bash
# Test in Gazebo
./test_model_in_gazebo.sh $EXPERIMENT_DIR/final_model/final_model.zip 10

# Or manually
python3 test_rl_model.py --model $EXPERIMENT_DIR/final_model/final_model.zip --episodes 10
\`\`\`

## Deployment to Hardware

1. Copy \`final_model.pth\` to robot
2. Use with \`nn_controller\` node
3. Launch with vision-based configuration

## Performance Metrics

See \`../logs/rl_training.log\` for training curves and TensorBoard data.
EOF

    # Create summary
    cat > "$EXPERIMENT_DIR/SUMMARY.txt" <<EOF
========================================
EXPERIMENT SUMMARY: $EXPERIMENT_NAME
========================================

Completed: $(date)
Duration: $(( ($(date +%s) - $(stat -c %Y "$EXPERIMENT_DIR/config.txt")) / 60 )) minutes

OUTPUTS:
--------
✓ Training data: $(basename $(cat "$EXPERIMENT_DIR/.data_path"))
✓ Imitation model: imitation_model/best_model.pth
✓ RL model: rl_model/final_model.{zip,pth}
✓ Final deployable: final_model/final_model.pth

NEXT STEPS:
-----------
1. Test the model:
   ./test_model_in_gazebo.sh $EXPERIMENT_DIR/final_model/final_model.zip

2. Review training:
   tensorboard --logdir $EXPERIMENT_DIR/rl_model/tensorboard

3. Deploy to hardware:
   Copy $EXPERIMENT_DIR/final_model/final_model.pth to robot

For more details, see:
- Configuration: config.txt
- Training logs: logs/
- Model metadata: final_model/README.md

========================================
EOF

    log_success "Finalization complete"
    log_info "Deployable model: $EXPERIMENT_DIR/final_model/final_model.pth"
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

main() {
    log_header "PiDog Production Training Pipeline"

    echo -e "${BOLD}Experiment:${NC} $EXPERIMENT_NAME"
    echo -e "${BOLD}Directory:${NC} $EXPERIMENT_DIR"
    echo ""

    log_info "This will run a complete training pipeline:"
    echo "  1. Data Collection (~10 minutes)"
    echo "  2. Imitation Learning (~20-40 minutes)"
    echo "  3. RL Training (~40-60 minutes)"
    echo "  4. Finalization (~1 minute)"
    echo ""
    log_info "Total estimated time: ~1.5-2 hours"
    echo ""

    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Aborted by user"
        exit 0
    fi

    START_TIME=$(date +%s)

    # Run pipeline
    check_prerequisites
    setup_experiment

    stage_1_data_collection
    stage_2_imitation_learning
    stage_3_rl_training
    stage_4_finalization

    END_TIME=$(date +%s)
    DURATION=$(( (END_TIME - START_TIME) / 60 ))

    # Update summary with actual duration
    sed -i "s/Duration: .*/Duration: $DURATION minutes/" "$EXPERIMENT_DIR/SUMMARY.txt"

    # Display summary
    log_header "PIPELINE COMPLETE!"
    cat "$EXPERIMENT_DIR/SUMMARY.txt"

    log_success "All done! Your model is ready to test and deploy."
}

# Run main
main
