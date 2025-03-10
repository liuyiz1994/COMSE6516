
# 1-shot
propose_prompt = """
Input is a 3x3 Sudoku puzzle in row-major format. The asterisk elements are blanks to be filled.
Please output possible next steps following the below example with one example per line without outputing any explanations or texts.
Input:
[[1, *, 3], [3, *, 2], [*, 3, *]]
Possible next steps: 
[[1, 2, 3], [3, *, 2], [*, 3, *]]
[[1, *, 3], [3, 1, 2], [*, 3, *]]
[[1, *, 3], [3, *, 2], [2, 3, *]]
[[1, *, 3], [3, *, 2], [*, 3, 1]]
Input:
{input}
Possible next steps:
"""


value_prompt = """
Evaluate if the given state of the Sudoku puzzle could be solved. Please only output one of the following 3 values for evaluation (sure/likely/impossible).
Input:
[[3, 1, 2], [2, *, 1], [1, 2, 3]]
Analysis:
[[3, 1, 2], [2, 3, 1], [1, 2, 3]]
Value:
sure
Input:
[[1, *, 3], [*, 2, *], [*, *, 1]]
Analysis:
[[1, 2, 3], [*, 2, *], [*, *, 1]]
Contradictions at column 2 with the number 2 repeated twice.
Value:
impossible
Input:
[[3, *, 2], [*, *, 1], [*, *, 3]]
Analysis:
[[3, 1, 2], [*, *, 1], [*, *, 3]]
[[3, 1, 2], [*, *, 1], [1, *, 3]]
[[3, 1, 2], [2, *, 1], [1, *, 3]]
[[3, 1, 2], [2, 3, 1], [1, *, 3]]
[[3, 1, 2], [2, 3, 1], [1, 2, 3]]
Value:
sure
Input:
[[*, 1, *], [*, *, *], [*, *, *]]
Analysis:
[[2, 1, *], [*, *, *], [*, *, *]]
[[2, 1, 3], [*, *, *], [*, *, *]]
I cannot solve the Sudoku puzzle now, but there are no contradictions so far. 
Value:
likely
Input:
[[1, *, *], [*, 2, *], [*, 1, *]]
Analysis:
[[1, 3, *], [*, 2, *], [*, 1, *]]
[[1, 3, 2], [*, 2, *], [*, 1, *]]
I cannot solve the Sudoku puzzle now, but there are no contradictions so far.
Value:
likely
Input:
[[*, 2, *], [3, *, 1], [*, *, 2]]
Analysis:
[[*, 2, 3], [3, *, 1], [*, *, 2]]
[[*, 2, 3], [3, 2, 1], [*, *, 2]]
Contradictions at column 2 with the number 2 repeated twice.
Value:
impossible
Input:
[[3, *, 2], [2, *, 1], [1, *, 3]]
Analysis:
[[3, 1, 2], [2, *, 1], [1, *, 3]]
[[3, 1, 2], [2, 3, 1], [1, *, 3]]
[[3, 1, 2], [2, 3, 1], [1, 2, 3]]
Value:
sure
Input:
[[1, 3, *], [3, 1, *], [3, *, 1]]
Analysis:
Contradictions at column 1 with the number 3 repeated twice.
Value:
impossible
Input:
[[1, *, 2], [2, 1, 3], [3, *, 1]]
Analysis:
[[1, *, 2], [2, 1, 3], [3, 2, 1]]
[[1, 3, 2], [2, 1, 3], [3, 2, 1]]
Value:
sure
Input:
{input}
Analysis:
"""


value_last_step_prompt = """
Given the Sudoku puzzle, verify (sure/impossible) its completeness and validity.
Answer:
[[3, 1, 2], [2, 3, 1], [1, 2, 3]]
Judge:
sure
Answer:
[[3, 1, 1], [2, 3, 1], [1, 2, 3]]
Judge:
impossible
Answer:
[[3, 2, 1], [1, 3, 2], [2, 1, 3]]
Judge:
sure
Answer:
[[1, 2, 3], [3, 1, 2], [2, 3, 1]]
Judge:
sure
Answer:
[[1, 2, 3], [3, 2, 1], [2, 3, 2]]
Judge:
impossble
Answer:
[[2, 3, 2], [1, 2, 3], [3, 2, 1]]
Judge:
impossible
Answer:
[[3, 1, 2], [2, 3, 1], [1, 1, 3]]
Judge:
impossible
Answer:
{answer}
Judge:
"""
