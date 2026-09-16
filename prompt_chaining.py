from dotenv import load_dotenv
from phoenix.otel import register

load_dotenv()

tracer_provider = register(
    project_name="openai-agents",
    auto_instrument=True,
)

from pprint import pprint
from textwrap import dedent

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
        """You are a research assistant specializing in financial topics. Given an investment strategy, use web search to retrieve up‑to‑date context and produce a short summary of stocks that support the investment strategy at most 50 words. Focus on key numbers, events, or quotes that will be useful to a financial analyst."""
    ),
    model="gpt-4.1",
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required", parallel_tool_calls=True),
)

user_input = input("Enter your investment strategy: ")
input_items: list[TResponseInputItem] = [
    {"content": f"My investment strategy: {user_input}", "role": "user"}
]


research_output = Runner.run_sync(research_agent, input_items)
pprint(research_output.final_output)

input_items = research_output.to_input_list()
portfolio_output = Runner.run_sync(portfolio_agent, input_items)
pprint(portfolio_output.final_output)
