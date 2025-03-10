import itertools
import time

import numpy as np
from functools import partial
from tot.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    print('four digits for game 24 are:')
    print(x)
    print('-' * 80)

    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}


def dfs_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # initial input
    print('four digits for game 24 are:')
    print(x)
    print('-' * 80)


    # parameters
    T = task.steps  # step limit
    vthres = 0.5  # threshold for pruning; adjust as needed
    max_states_per_depth = 3

    # storage for recorded outputs at maximum depth
    results = []
    infos = []

    def dfs(s, t):
        # If we have reached the step limit, record output
        if t >= T:
            # According to the pseudo-code, we record G(pθ, s, 1)
            # i.e. we consider s as a final solution
            if to_print:
                print(f"Reached step limit T={T}, recording solution:\n{s}")
            results.append(s)
            return

        # Generate candidates from current state `s`
        if args.method_generate == 'sample':
            candidates = get_samples(task, x, s, args.n_generate_sample, args.prompt_sample, stop=task.stops[t])
        elif args.method_generate == 'propose':
            candidates = get_proposals(task, x, s)
        else:
            raise ValueError(f"Unknown generation method: {args.method_generate}")

        # Evaluate candidates
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, candidates, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, candidates, args.n_evaluate_sample, cache_value=False)
        else:
            raise ValueError(f"Unknown evaluation method: {args.method_evaluate}")

        # Sort candidates by value descending
        sorted_candidates_values = sorted(zip(candidates, values), key=lambda x: x[1], reverse=True)

        # Logging current step if needed
        if to_print:
            candidate_strs = [f"{c} (val={v})" for c, v in sorted_candidates_values]
            print(f"Step {t}, Expanding state:\n{s}\nCandidates:\n{candidate_strs}\n")

        top_candidates = sorted_candidates_values[:max_states_per_depth]

        # Prune and recurse
        for cand, val in top_candidates:
            if val > vthres:
                # Recurse deeper with candidate
                dfs(cand, t + 1)

    # Initiate DFS from empty solution
    dfs('', 0)

    # Return all recorded solutions and any info collected
    if to_print:
        print("DFS Completed. Final solutions:", results)
    return results, {'steps': infos}


def dfs_solve_new(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # initial input
    print('Sudoku puzzle:')
    print(x)
    print('-' * 80)

    # parameters
    T = task.steps[idx]  # step limit
    vthres = 0.5  # threshold for pruning; adjust as needed
    max_states_per_depth = 3

    # storage for recorded outputs at maximum depth
    results = []
    infos = []
    found_solution = False  # Flag to indicate that a solution has been found
    generate_times = []
    evaluate_times = []
    num_of_generations = 0
    num_of_evaluations = 0
    total_start_time = time.time()

    def dfs(s, t):
        nonlocal found_solution
        nonlocal num_of_generations
        nonlocal num_of_evaluations
        nonlocal generate_times
        nonlocal evaluate_times
        # If a solution was already found, stop exploring further
        if found_solution:
            return

        # If we have reached the step limit, record output as a solution
        if t >= T:
            if to_print:
                print(f"Reached step limit T={T}, recording solution:\n{s}")
            results.append(s)
            if task.test_output(idx, s)["r"] == 1:
                found_solution = True
            return

        # Generate candidates from current state `s`
        start_time = time.time()
        if args.method_generate == 'sample':
            candidates = get_samples(task, x, s, args.n_generate_sample, args.prompt_sample, stop=task.stops[t])
        elif args.method_generate == 'propose':
            candidates = get_proposals(task, x, s)
        else:
            raise ValueError(f"Unknown generation method: {args.method_generate}")
        generate_times.append(time.time() - start_time)
        num_of_generations += 1

        # Evaluate candidates
        start_time = time.time()
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, candidates, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, candidates, args.n_evaluate_sample, cache_value=False)
        else:
            raise ValueError(f"Unknown evaluation method: {args.method_evaluate}")
        evaluate_times.append(time.time() - start_time)
        num_of_evaluations += 1

        # Sort candidates by value descending
        sorted_candidates_values = sorted(zip(candidates, values), key=lambda x: x[1], reverse=True)

        top_candidates = sorted_candidates_values[:max_states_per_depth]
        
        # Logging current step if needed
        if to_print:
            candidate_strs = [f"{c} (val={v})" for c, v in top_candidates]
            print(f"Step {t}, Expanding state:\n{s}\nCandidates:\n{candidate_strs}\n")

        # Prune and recurse
        for cand, val in top_candidates:
            if found_solution:
                break
            if val > vthres:
                # Recurse deeper with candidate
                dfs(cand, t + 1)

    # Initiate DFS from empty solution
    dfs('', 0)

    # Return all recorded solutions and any info collected
    if to_print:
        print("DFS Completed. Final solutions:", results)
    
    logs = {
        'steps': infos, 
        'total_time': time.time() - total_start_time, 
        'generate_time': sum(generate_times), 
        'evaluate_time': sum(evaluate_times), 
        'num_of_generations': num_of_generations, 
        'num_of_evaluations': num_of_evaluations, 
        'num_of_steps': T,
    }
    return results, logs


def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}


import heapq
from functools import partial


def astar_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # initial input
    print('Sudoku Puzzle:')
    print(x)
    print('-' * 80)

    # Parameters
    T = task.steps[idx]  # maximum depth (similar to BFS/DFS)
    max_expansions = 5  # limit expansions if desired

    # Priority queue holds tuples of (-f_score, depth, state_string)
    # We use a max-heap by negating the f_score, since heapq in Python is a min-heap.
    # f_score = h(s) - g(s), where g(s) = depth and h(s) is the heuristic from value model.
    frontier = []

    # Initial state is empty solution
    initial_state = ''
    # Compute heuristic for initial state
    initial_val = get_values(task, x, [initial_state], args.n_evaluate_sample, cache_value=False)[0]
    initial_f = initial_val - 0  # f = h - g = initial_val - 0
    heapq.heappush(frontier, (-initial_f, 0, initial_state))

    results = []
    infos = []
    expansions = 0
    generate_times = []
    evaluate_times = []
    total_start_timne = time.time()
    num_of_generations = 0
    num_of_evaluations = 0

    while frontier and expansions < max_expansions:
        # Pop state with highest f-score
        neg_f, depth, s = heapq.heappop(frontier)
        f = -neg_f

        # If we reached the step limit, record solution
        if depth >= T:
            if to_print:
                print(f"Reached step limit T={T}, recording solution:\n{s}")
            results.append(s)
            return

        # Generate candidates from current state `s`
        start_time = time.time()
        if args.method_generate == 'sample':
            candidates = get_samples(task, x, s, args.n_generate_sample, args.prompt_sample, stop=task.stops[depth])
        elif args.method_generate == 'propose':
            candidates = get_proposals(task, x, s)
        else:
            raise ValueError(f"Unknown generation method: {args.method_generate}")
        generate_times.append(time.time() - start_time)
        num_of_generations += 1

        # Evaluate candidates
        start_time = time.time()
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, candidates, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, candidates, args.n_evaluate_sample, cache_value=False)
        else:
            raise ValueError(f"Unknown evaluation method: {args.method_evaluate}")
        evaluate_times.append(time.time() - start_time)
        num_of_evaluations += 1

        # Sort candidates by their value (heuristic), descending
        sorted_candidates_values = sorted(zip(candidates, values), key=lambda cv: cv[1], reverse=True)

        # Log current step if needed
        if to_print:
            candidate_strs = [f"{c} (val={v})" for c, v in sorted_candidates_values]
            print(f"Depth {depth}, Expanding state:\n{s}\nCandidates:\n{candidate_strs}\n")

        # Push top candidates into the frontier
        for cand, val in sorted_candidates_values:
            # Compute f-score for candidate: f = h(cand) - g(cand)
            # g(cand) = depth + 1 since we go one step deeper
            cand_f = val - (depth + 1)
            heapq.heappush(frontier, (-cand_f, depth + 1, cand))

        expansions += 1

    if to_print:
        print("A* Search Completed. Final solutions:", results)
    logs = {
        'steps': infos, 
        'total_time': time.time() - total_start_timne, 
        'generate_time': sum(generate_times), 
        'evaluate_time': sum(evaluate_times), 
        'num_of_generations': num_of_generations, 
        'num_of_evaluations': num_of_evaluations, 
        'num_of_steps': T,
    }
    return results, logs
