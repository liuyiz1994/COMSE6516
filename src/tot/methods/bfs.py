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
    # print(f"proposals: {proposals}")
    # valid_proposals = task.check_proposals(x, y, proposals)
    # print(f"valid proposals: {valid_proposals}")
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
    print(f"x: \n{x}")
    print("-"*20)
    ys = ['']  # current output candidates
    infos = []
    num_steps = task.steps[idx]
    total_times = []
    generate_times = []
    evaluate_times = []
    num_of_generations = [] # number of times LLM is called for generations
    num_of_evaluations = [] # number of times LLM is called for evaluations
    end_time = time.time()
    for step in range(num_steps):
        # generation
        start_time = time.time()
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        generate_times.append(time.time() - start_time)
        num_of_generations.append(len(new_ys))
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))

        # evaluation
        start_time = time.time()
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        evaluate_times.append(time.time() - start_time)
        num_of_evaluations.append(len(values))

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
        total_times.append(time.time() - end_time)
        end_time = time.time()

        # log
        if to_print:
            try:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                print(f"Step: {step}")
                print("--- new_ys and values ---")
                for p_y, p_v in zip(sorted_new_ys, sorted_values):
                    print(f"-- new_y: \n{p_y}")
                    print(f"-- new_v: {p_v}\n--")
                print(
                    f"""-- selected choices --: {select_new_ys}\n"""
                )
            except Exception as e:
                print(f"Error: {e}")
                print(f"new_ys: {new_ys}")
                print(f"values: {values}")
    
    if to_print: 
        print(ys)
    results = {
        'steps': infos, 
        'total_time': sum(total_times), 
        'generate_time': sum(generate_times), 
        'evaluate_time': sum(evaluate_times), 
        'num_of_generations': sum(num_of_generations), 
        'num_of_evaluations': sum(num_of_evaluations), 
        'num_of_steps': num_steps,
    }
    return ys, results

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}