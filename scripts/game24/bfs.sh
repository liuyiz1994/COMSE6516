python3 run.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 901 \
    --method_generate sample \
    --method_evaluate value \
    --method_select greedy \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    --backend gpt-4o-mini \
    ${@}
