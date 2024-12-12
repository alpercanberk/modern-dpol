export CUDA_VISIBLE_DEVICES=0,1,2,3

configs=(
    "exp_dec11_unet_ddim_vpred_lowdim_workspace.yaml"
    "exp_dec11_unet_ddim_lowdim_workspace.yaml"
    "exp_dec11_unet_ddim_ztsnr_lowdim_workspace.yaml"
)

tasks=(
    "square_lowdim"
    "lift_lowdim"
    "transport_lowdim"
)

seeds=($(seq 42 44))

N_GPUS=4
N_PROCESSES_AT_A_TIME=12

run_counter=0

for task in "${tasks[@]}"; do
    for config_name in "${configs[@]}"; do
        for current_seed in "${seeds[@]}"; do
            gpu_idx=$((run_counter % N_GPUS))

            echo "Launching on GPU $gpu_idx: Task=$task, Config=${config_name}, Seed=$current_seed"
            
            command="CUDA_VISIBLE_DEVICES=$gpu_idx python train.py \
                        --config-dir=diffusion_policy/config \
                        --config-name=$config_name \
                        training.seed=$current_seed \
                        training.device=cuda:0 \
                        hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}_seed${current_seed}' \
                        task=${task}"

            echo "Launching on GPU $gpu_idx: Task=$task, Config=${config_name}, Seed=$current_seed, Inference steps=$steps"
            echo "Command: $command"
                    
            eval $command &
            
            run_counter=$((run_counter + 1))
            sleep 12
            
            if [ $((run_counter % N_PROCESSES_AT_A_TIME)) -eq 0 ]; then
                wait
            fi
        done
    done
done

if [ $((run_counter % N_PROCESSES_AT_A_TIME)) -ne 0 ]; then
    wait
fi
