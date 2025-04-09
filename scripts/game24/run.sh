# Game 24 BFS
python3 run.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 915 \
    --method_search bfs \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_evaluate_sample 3 \
    --n_select_sample 3 \
    --backend gpt-3.5-turbo \
    ${@}

# Game 24 DFS
python3 run.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 915 \
    --method_search dfs \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_evaluate_sample 3 \
    --n_select_sample 3 \
    --backend gpt-3.5-turbo \
    ${@}
