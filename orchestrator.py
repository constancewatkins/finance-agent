from dotenv import load_dotenv
from phoenix.otel import register

load_dotenv()

tracer_provider = register(
    project_name="openai-agents",
    auto_instrument=True,
)

from pprint import pprint
from textwrap import dedent
from typing import Literal

from agents import Agent, Runner, TResponseInputItem, WebSearchTool
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field


class PortfolioItem(BaseModel):
    ticker: str = Field(description="The ticker of the stock or ETF.")
    allocation: float = Field(
        description="The percentage allocation of the ticker in the portfolio. The sum of all allocations should be 100."
    )
    reason: str = Field(description="The reason why this ticker is included in the portfolio.")


class Portfolio(BaseModel):
    tickers: list[PortfolioItem] = Field(
        description="A list of tickers that could support the user's stated investment strategy."
    )


class EvaluationFeedback(BaseModel):
    feedback: str = Field(
        description="What data is missing in order to create a portfolio of stocks and ETFs based on the user's investment strategy."
    )
    score: Literal["pass", "needs_improvement", "fail"] = Field(
        description="A score on the research report. Pass if you have at least 5 tickers with data that supports the user's investment strategy to create a portfolio, needs_improvement if you do not have enough supporting data, and fail if you have no tickers."
    )


evaluation_agent = Agent(
    name="Evaluation Agent",
    instructions=dedent(
        """You are a senior financial analyst. You will be provided with a stock research report with positive and negative catalysts. Your task is to evaluate the report and provide feedback on what to improve."""
    ),
    model="gpt-4.1",
    output_type=EvaluationFeedback,
)

portfolio_agent = Agent(
    name="Portfolio Agent",
    instructions=dedent(
        """You are a senior financial analyst. You will be provided with a stock research report. Your task is to create a portfolio of stocks and ETFs that could support the user's stated investment strategy. Include facts and data from the research report in the stated reasons for the portfolio allocation."""
    ),
    model="o4-mini",
    output_type=Portfolio,
)

research_agent = Agent(
    name="FinancialSearchAgent",
    instructions=dedent(
        """You are a research assistant specializing in financial topics. Given a stock ticker, use web search to retrieve up‑to‑date context and produce a short summary of at most 50 words. Focus on key numbers, events, or quotes that will be useful to a financial analyst."""
    ),
    model="gpt-4.1",
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required", parallel_tool_calls=True),
)

orchestrator_agent = Agent(
    name="Routing Agent",
    instructions=dedent("""You are a senior financial analyst. You are trying to create a portfolio based on my stated investment strategy. Your task is to handoff to the appropriate agent or tool.

    First, handoff to the research_agent to give you a report on stocks and ETFs that could support the user's stated investment strategy.
    Then, handoff to the evaluation_agent to give you a score on the research report. If the evaluation_agent returns a needs_improvement or fail, continue using the research_agent to gather more information.
    Once the evaluation_agent returns a pass, handoff to the portfolio_agent to create a portfolio."""),
    model="gpt-4.1",
    handoffs=[
        research_agent,
        evaluation_agent,
        portfolio_agent,
    ],
)

user_input = input("Enter your investment strategy: ")
input_items: list[TResponseInputItem] = [{"content": user_input, "role": "user"}]
orchestrator_output = None
max_runs = 6

for _ in range(max_runs):
    orchestrator = Runner.run_sync(orchestrator_agent, input_items)
    orchestrator_output = orchestrator.final_output
    pprint(orchestrator_output)

    input_items = orchestrator.to_input_list()
    if isinstance(orchestrator_output, Portfolio):
        break
    print("Going back to orchestrator")
    # input_items.append({"content": f"Keep going", "role": "user"})

if isinstance(orchestrator_output, Portfolio):
    print("AGENT COMPLETE")
else:
    print(f"Stopped after {max_runs} orchestrator runs without producing a portfolio.")
