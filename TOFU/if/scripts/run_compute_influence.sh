#!/bin/bash

# Model configuration
model_family="llama2-7b"  # Options: llama2-7b, phi, stablelm, pythia-1.4, zephyr-7b-beta
checkpoint_dir="path/to/your/checkpoint"  # Path to the trained model checkpoint

# Factor configuration
factors_path="path/to/your/factors"  # Path to where factors are stored
factors_name="ekfac_factors"         # Name of the factors directory
factor_strategy="ekfac"              # Strategy used for factors

# Dataset configuration
data_path="locuslab/TOFU"  # HuggingFace dataset path or local path
forget_split="forget10"     # Options: forget01, forget05, forget10
retain_split="retain90"     # Options: retain99, retain95, retain90
max_length=512

# Computation configuration
query_batch_size=8
train_batch_size=8
use_half_precision="--use_half_precision"  # Remove or comment out to disable
# use_compile="--use_compile"  # Uncomment to enable torch.compile

# Output configuration
save_dir="./influence_results"
save_id=""  # Optional ID to append to output names

# Get script directory and navigate to parent
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

# Build command
cmd="python compute_influence.py \
    --model_name ${checkpoint_dir} \
    --model_family ${model_family} \
    --data_path ${data_path} \
    --forget_split ${forget_split} \
    --retain_split ${retain_split} \
    --max_length ${max_length} \
    --factors_path ${factors_path} \
    --factors_name ${factors_name} \
    --factor_strategy ${factor_strategy} \
    --query_batch_size ${query_batch_size} \
    --train_batch_size ${train_batch_size} \
    ${use_half_precision} \
    --save_dir ${save_dir}"

# Add optional save_id if specified
if [ ! -z "$save_id" ]; then
    cmd="${cmd} --save_id ${save_id}"
fi

# Add compile flag if enabled
if [ ! -z "$use_compile" ]; then
    cmd="${cmd} ${use_compile}"
fi

echo "Running command:"
echo "$cmd"
echo ""

# Execute
eval $cmd
