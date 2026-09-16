from dotenv import load_dotenv
from phoenix.otel import register

load_dotenv()

tracer_provider = register(
    project_name="openai-agents",
    auto_instrument=True,
)

from textwrap import dedent
from typing import Literal

from agents import Agent, Runner, TResponseInputItem
from pydantic import BaseModel, Field

CATALYSTS = """topline revenue growth, margin expansion, moat expansion, free cash flow generation, usage, pricing, distribution, share buyback, dividend, new products, regulation, competition, management team, mergers, acquisitions, analyst ratings, trading volume, technical indicators, price momentum"""


class EvaluationFeedback(BaseModel):
    feedback: str = Field(
        description=f"What is missing from the research report on positive and negative catalysts for a particular stock ticker. Catalysts include changes in {CATALYSTS}."
    )
    score: Literal["pass", "needs_improvement", "fail"] = Field(
        description="A score on the research report. Pass if the report is complete and contains at least 3 positive and 3 negative catalysts for the right stock ticker, needs_improvement if the report is missing some information, and fail if the report is completely wrong."
    )


report_agent = Agent(
    name="Catalyst Report Agent",
    instructions=dedent(
        """You are a research assistant specializing in stock research. Given a stock ticker, generate a report of 3 positive and 3 negative catalysts that could move the stock price in the future in 50 words or less."""
    ),
    model="gpt-4.1",
)

evaluation_agent = Agent(
    name="Evaluation Agent",
    instructions=dedent(
        """You are a senior financial analyst. You will be provided with a stock research report with positive and negative catalysts. Your task is to evaluate the report and provide feedback on what to improve."""
    ),
    model="gpt-4.1",
    output_type=EvaluationFeedback,
)

report_feedback = "fail"
input_items: list[TResponseInputItem] = [{"content": "AAPL", "role": "user"}]
iteration = 0
max_iterations = 3

while report_feedback != "pass" and iteration < max_iterations:
    iteration += 1
    report = Runner.run_sync(report_agent, input_items)
    print("### REPORT ###")
    print(report.final_output)
    input_items = report.to_input_list()

    evaluation = Runner.run_sync(evaluation_agent, str(report.final_output))
    evaluation_feedback = evaluation.final_output_as(EvaluationFeedback)
    print("### EVALUATION ###")
    print(str(evaluation_feedback))
    report_feedback = evaluation_feedback.score

    if report_feedback != "pass":
        print("Re-running with feedback")
        input_items.append({"content": f"Feedback: {evaluation_feedback.feedback}", "role": "user"})

if report_feedback != "pass":
    print(f"Stopped after {max_iterations} iterations without a passing evaluation.")
