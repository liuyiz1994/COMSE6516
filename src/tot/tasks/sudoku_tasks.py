import re
import os
import json
import ast

import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.sudoku_prompts import * 


def verify_3x3_sudoku(puzzle):
    """
    Verifies if a 3x3 Sudoku puzzle (provided as a string) is complete and valid.

    Args:
        grid_string (str): String representation of a 3x3 Sudoku grid, e.g., "[[3, 1, 2],[2, 3, 1],[1, 2, 3]]".

    Returns:
        bool: True if the puzzle is complete and valid, False otherwise.
    """
    try:
        grid = preprocess(puzzle)

        # Ensure it's a 3x3 grid
        if len(grid) != 3 or any(len(row) != 3 for row in grid):
            return False
        
        # Define the valid set of numbers for the puzzle
        valid_set = {'1', '2', '3'}
        
        # Check rows
        for row in grid:
            if set(row) != valid_set:
                return False

        # Check columns
        for col in range(3):
            column = {grid[row][col] for row in range(3)}
            if column != valid_set:
                return False

        return True
    except (ValueError, SyntaxError):
        # Handle invalid input strings
        return False


def verify_3x3_sudoku_completeness(puzzle):
    try:
        grid = preprocess(puzzle)

        # Ensure it's a 3x3 grid
        if len(grid) != 3 or any(len(row) != 3 for row in grid):
            return False
        
        # Check rows
        for row in grid:
            if "*" in set(row):
                return False
        return True
    except (ValueError, SyntaxError):
        # Handle invalid input strings
        return False


def preprocess(puzzle: str) -> list:
    """Convert a string puzzle into a list of list of string"""
    preprocessed_puzzle = re.sub(r'(\d+|\*)', r"'\1'", puzzle)
    grid = ast.literal_eval(preprocessed_puzzle)
    return grid


class Sudoku(Task):
    """
    Input (x): a string of 81 numbers
    Output (y): a trajectory of x number of steps to solve the sudoku puzzle with x being number of blanks/zeros
    Reward (r): 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        "[[1, *, *], [*, 1, *], [*, 2, *]]"
    Output Example:
        "[[1, *, *], [*, 1, *], [*, 2, *]]"
        ... (number of steps = number of asterisks)
        "[[1, 3, 2], [2, 1, 3], [3, 2, 1]]"
    """
    def __init__(self, file="sudoku_3x3.json"):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, "sudoku", file)
        with open(path, "r") as file:
            self.data = json.load(file)
        self.value_cache = {}
        self.steps = []
        self.preprocessed_data = []
        count = 0
        for puzzle in self.data:
            grid = preprocess(puzzle)
            self.preprocessed_data.append(grid)
            for r in range(3):
                for c in range(3):
                    if grid[r][c] == '*':
                        count += 1
            self.steps.append(count)
            count = 0
        
    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def test_output(self, idx: int, output: str):
        """check if the generated output is valid"""
        solved_puzzle = output.strip().split("\n")[-1].split("next: ")[-1].split(")")[0]
        #print(f"test_output: \n{solved_puzzle}")
        try:
            return {'r': int(verify_3x3_sudoku(solved_puzzle))}
        except Exception as e:
            # print(e)
            return {'r': 0}

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        """Makes propose prompt
        """
        if not y:
            current_puzzle = x
        else:
            current_puzzle = y.strip().split("\n")[-1]#.split('next: ')[-1].split(')')[0]
        #print(f"propose_prompt_wrap: \n{current_puzzle}")
        prompt = propose_prompt.format(input=current_puzzle)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        current_puzzle = y.strip().split("\n")[-1]#.split('next: ')[-1].split(')')[0]
        if verify_3x3_sudoku_completeness(current_puzzle):  # last step
            #print(f"value_prompt_wrap (last line): \n{current_puzzle}")
            return value_last_step_prompt.format(answer=current_puzzle)
        #print(f"value_prompt_wrap: \n{current_puzzle}")
        return value_prompt.format(input=current_puzzle)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        #print(f"value_names: {value_names}")
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value

    # @staticmethod
    # def check_proposals(x: str, y: str, proposals: list) -> list:
    #     if not y:
    #         current_puzzle = "".join(x.split('\n')).strip()
    #     else:
    #         #current_puzzle = "".join([p.strip() for p in y.strip().split('\n')[-10:-1]]).strip()
    #         current_puzzle = "".join([p.strip() for p in y.strip().split(".\n")[-1].split('\n')])[:-1]
    #     print(f"check_proposals (current_puzzle): \n{current_puzzle}")
    #     valid_proposals = []
    #     for proposal in proposals:
    #         if not proposal:
    #             continue
    #         #print(proposal)
    #         #row = int(proposal.split("row ")[-1].split(",")[0])
    #         #col = int(proposal.split("column ")[-1].split(").")[0])
    #         #next_puzzle = "".join([p.strip() for p in proposal.strip().split('\n')[-10:-1]]).strip()
    #         next_puzzle = "".join([p.strip() for p in proposal.strip().split('\n')])
    #         #print(row, col)
    #         print(f"check_proposals (next_puzzle): \n{next_puzzle}")
    #         #row_col_i = (row - 1)*9 + (col - 1)
    #         i = 0
    #         valid = False
    #         count = 0
    #         while i < len(current_puzzle):
    #             if current_puzzle[i] != next_puzzle[i] and current_puzzle[i] == "0" and next_puzzle[i] != "0": # and i == row_col_i:
    #                 valid = True
    #                 count += 1
    #             elif current_puzzle[i] != next_puzzle[i] and current_puzzle[i] != "0":
    #                 valid = False
    #             i += 1
    #         if valid and count == 1:
    #             valid_proposals.append(proposal)
    #     return valid_proposals
