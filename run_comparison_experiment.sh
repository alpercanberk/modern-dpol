export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

configs=(
    "exp_unet_ddim_vpred_lowdim_workspace.yaml"
    "exp_unet_ddim_lowdim_workspace.yaml"
    "exp_unet_ddim_ztsnr_lowdim_workspace.yaml"
    "exp_unet_rf_lognorm_lowdim_workspace.yaml"
    "exp_unet_rf_lowdim_workspace.yaml"
)

tasks=(
    "pusht_lowdim"
    "lift_lowdim"
    "tool_hang_lowdim"
    "square_lowdim"
    "transport_lowdim"
)

#takes in a single parameter called partition, which is between 0 and 1
partition=$1    

# Convert inference_steps array to proper bash syntax
# inference_steps=(8 16) if partition is 0
# inference_steps=(32 64) if partition is 1

if [ $partition -eq 0 ]; then
    inference_steps=(8 16)
else
    inference_steps=(32 64)
fi

# Define seeds array (42, 43, 44, 45)
seeds=($(seq 42 45))

N_GPUS=8
N_PROCESSES_AT_A_TIME=16

# Add run counter at the start
run_counter=0

# Move inference_steps loop to be outermost
for steps in "${inference_steps[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Starting experiments for task: $task with inference steps: $steps"
        for ((i=0; i<${#configs[@]}; i+=8)); do
            echo "------- Starting new batch of parallel runs -------"
            for j in {0..7}; do
                config_idx=$((i + j))
                if [ $config_idx -lt ${#configs[@]} ]; then
                    config_name=${configs[$config_idx]}
                    
                    for current_seed in "${seeds[@]}"; do
                        # Use run_counter to determine GPU
                        gpu_idx=$(((run_counter % N_GPUS)))
                        
                        command="CUDA_VISIBLE_DEVICES=$gpu_idx python train.py \
                            --config-dir=diffusion_policy/config \
                            --config-name=$config_name \
                            training.seed=$current_seed \
                            training.device=cuda:0 \
                            num_inference_steps=$steps \
                            hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}_steps${steps}_seed${current_seed}' \
                            task=${task}"
                        
                        echo "Launching on GPU $gpu_idx: Task=$task, Config=${config_name}, Seed=$current_seed, Inference steps=$steps"
                        echo "Command: $command"
                        
                        eval $command &
                        
                        # Increment run counter
                        run_counter=$((run_counter + 1))

                        #wait for 3 seconds
                        sleep 10

                        # Wait after every N_GPUS launches
                        if [ $((run_counter % N_PROCESSES_AT_A_TIME)) -eq 0 ]; then
                            echo "Reached $N_PROCESSES_AT_A_TIME parallel processes, waiting for completion..."
                            wait
                            echo "Processes complete, continuing..."
                        fi
                    done
                fi
            done
            wait
            echo "------- Batch complete -------"
        done
        echo "Completed all configs for task: $task"
        echo "================================="
    done
done