# Evaluation results (2026-11-09 16:39, 16 tasks)

## Agent vs. baselines (same in-scope + out-of-scope questions)

| metric | plain LLM | search once | full agent |
|---|---|---|---|
| in-scope questions answered | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| answers with valid citations | 0/8 (0%) | 8/8 (100%) | 8/8 (100%) |
| answers citing a correct page | 0/8 (0%) | 5/8 (62%) | 7/8 (88%) |
| keyword recall (completeness) | 44% | 66% | 87% |
| out-of-scope honestly declined | 0/3 (0%) | 2/3 (67%) | 2/3 (67%) |
| avg seconds per question | 3.7 | 3.3 | 20.8 |
| avg LLM calls per question | 1 | 1 | 5.6 |
| avg tokens per question | 316.0 | 1494.0 | 8129.0 |
| total cost | set prices in run_eval.py | set prices in run_eval.py | set prices in run_eval.py |

## Agent only (all tasks)

| metric | full agent |
|---|---|
| routing accuracy (all tasks) | 16/16 (100%) |
| quiz in a valid format | 1/1 (100%) |
| plan paused for human approval | 1/1 (100%) |
| chat/off-topic kept out of the book | 3/3 (100%) |
| avg retries (extra searches) | 0.9 |
| answers rejected by the critic | 2 |
| answers that failed the checks | 1 |
| avg seconds per task (all tasks) | 17.3 |
| total cost (all tasks) | set prices in run_eval.py |

## Agent, task by task

| task | route OK | answered | valid citations | correct page | keyword recall | retries | critic rejections | seconds | LLM calls |
|---|---|---|---|---|---|---|---|---|---|
| definition | ✅ | yes | ✅ | ✅ | 100% | 0 | 0 | 10.82 | 3 |
| design_steps | ✅ | yes | ✅ | ✅ | 78% | 2 | 0 | 32.77 | 7 |
| design_steps_typos | ✅ | yes | ✅ | ✅ | 67% | 2 | 0 | 24.61 | 7 |
| correctness | ✅ | yes | ✅ | ✅ | 100% | 0 | 1 | 17.0 | 5 |
| efficiency | ✅ | yes | ✅ | - | 50% | 2 | 2 | 77.2 | 10 |
| problem_types | ✅ | yes | ✅ | ✅ | 100% | 0 | 0 | 6.25 | 3 |
| design_technique | ✅ | yes | ✅ | ✅ | 100% | 1 | 0 | 11.57 | 5 |
| follow_up | ✅ | yes | ✅ | ✅ | 100% | 0 | 0 | 8.7 | 3 |
| quiz | ✅ | yes | ✅ | ✅ | - | 0 | 0 | 10.67 | 3 |
| study_plan | ✅ | yes | ✅ | - | - | 2 | 0 | 30.39 | 6 |
| photosynthesis | ✅ | no | - | - | - | 2 | 0 | 10.74 | 6 |
| french_revolution | ✅ | no | - | - | - | 2 | 0 | 11.02 | 6 |
| capital_of_japan | ✅ | yes | ✅ | - | - | 2 | 0 | 18.23 | 7 |
| greeting | ✅ | no | - | - | - | 0 | 0 | 2.36 | 2 |
| recipe | ✅ | no | - | - | - | 0 | 0 | 2.87 | 2 |
| thanks | ✅ | no | - | - | - | 0 | 0 | 2.09 | 2 |

## Problems to look at

- efficiency: did not cite any of the expected pages [42]
- capital_of_japan: answered a question the book does not cover
