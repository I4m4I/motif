#!/usr/bin/env bash

set -euo pipefail

# Ensure we run from the script's directory so python finds main.py and params.ini
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

datasets=("ant")
run_ids=(1 2 3 4 5 6 7 8 9 10)

# GPUs assigned per batch; edit as needed.
cuda_ids=(0 1 2 3 4 5 6 7)

# prefixes=(
# 	"MOP_E"
# 	"MOP"
# 	"ORBI"
# 	"ORBI_E"
# )

# fre_lists=(
# 	"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.130366 0.449035 -1"
# 	"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.130366 0.249035 -1"
# 	"0.091003 0.287659 0.178217 0.107868 -1 -1 -1 -1 -1 -1 -1 -1 -1"
# 	"0.19003 0.287659 0.278217 0.107868 -1 -1 -1 -1 -1 -1 -1 -1 -1"
# )

prefixes=(
	# "2"
	# "2E"
	# "12"
	# "12E"
	"Vanilla"
)

fre_lists=(
	# "-1 0.25 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"
	# "-1 0.4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"
	# "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.25 -1"
	# "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.4 -1"
	"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"
)

num_cudas=${#cuda_ids[@]}
if (( num_cudas == 0 )); then
	echo "No CUDA ids configured. Please set at least one entry in cuda_ids." >&2
	exit 1
fi

# 1. Collect all tasks
tasks=()
for dataset in "${datasets[@]}"; do
	for run_id in "${run_ids[@]}"; do
		for idx in "${!prefixes[@]}"; do
			tasks+=("${dataset}|${run_id}|${idx}")
		done
	done
done

echo "Total tasks: ${#tasks[@]}"
echo "Distributing tasks across ${num_cudas} GPUs..."

# 2. Define worker function
run_worker() {
	local worker_idx=$1
	local cuda_id=$2
	
	echo "[Worker $worker_idx] Started on CUDA $cuda_id"
	
	# Stride loop: 0, 4, 8... or 1, 5, 9...
	for (( j=worker_idx; j<${#tasks[@]}; j+=num_cudas )); do
		task="${tasks[$j]}"
		IFS='|' read -r dataset run_id prefix_idx <<< "$task"
		prefix="${prefixes[$prefix_idx]}"
		fre="${fre_lists[$prefix_idx]}"
		
		echo "[Worker $worker_idx] Running task $j: env=${dataset}, prefix=${prefix}, run_id=${run_id}"
		
		python main.py \
			--env "${dataset}" \
			--seed "${run_id}" \
			--prefix "${prefix}" \
			--cuda "${cuda_id}" \
			--fre ${fre}
	done
	
	echo "[Worker $worker_idx] Finished all assigned tasks."
}

# 3. Launch N independent workers
pids=()

for (( i=0; i<num_cudas; i++ )); do
	cuda_id=${cuda_ids[$i]}
	# Launch function in background
	run_worker "$i" "$cuda_id" &
	pids+=($!)
done

# 4. Wait for all workers to finish
for pid in "${pids[@]}"; do
	wait "$pid"
done

echo "All tasks completed."

