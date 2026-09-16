from dotenv import load_dotenv
from phoenix.evals import ClassificationEvaluator
from phoenix.evals.llm import LLM

load_dotenv()

PORTFOLIO_REQUIREMENT_PROMPT_TEMPLATE = """
You will be given a portfolio-construction request and an answer. Decide whether
the answer follows the user's requirements.

An answer is "incorrect" if it:
- does not provide exactly five stocks or ETFs;
- ignores requested themes, risk tolerance, income needs, or asset-type constraints;
- omits requested current evidence or material risks;
- presents irrelevant or contradictory holdings; or
- presents the result as personalized financial advice.

    [BEGIN DATA]
    ************
    [Request]: {{input}}
    ************
    [Answer]: {{output}}
    ************
    [END DATA]

Return one final LABEL:
- "correct" if the answer follows the request accurately and fully.
- "incorrect" if it violates or omits any material requirement.

Your final output must be only one word: "correct" or "incorrect".
"""

ALLOCATION_CONSISTENCY_PROMPT_TEMPLATE = """
You will be given a portfolio-construction request and an answer. Determine
whether the proposed allocations are complete and mathematically consistent.

An answer is "incorrect" if:
- the allocations do not sum to exactly 100%;
- any allocation is missing, duplicated, negative, or ambiguous;
- the number of allocations does not match the number of holdings;
- the answer violates an allocation limit stated in the request; or
- allocation totals stated in the prose contradict the listed percentages.

    [BEGIN DATA]
    ************
    [Request]: {{input}}
    ************
    [Answer]: {{output}}
    ************
    [END DATA]

Return one final LABEL:
- "correct" if all allocations are complete and mathematically consistent.
- "incorrect" if any allocation requirement is violated.

Your final output must be only one word: "correct" or "incorrect".
"""

evaluator_llm = LLM(provider="openai", model="gpt-4o")

requirement_evaluator = ClassificationEvaluator(
    name="PORTFOLIO REQUIREMENT ADHERENCE",
    llm=evaluator_llm,
    prompt_template=PORTFOLIO_REQUIREMENT_PROMPT_TEMPLATE,
    choices={"correct": 1.0, "incorrect": 0.0},
)

allocation_evaluator = ClassificationEvaluator(
    name="ALLOCATION CONSISTENCY",
    llm=evaluator_llm,
    prompt_template=ALLOCATION_CONSISTENCY_PROMPT_TEMPLATE,
    choices={"correct": 1.0, "incorrect": 0.0},
)
