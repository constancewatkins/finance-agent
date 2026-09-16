from dotenv import load_dotenv
from phoenix.otel import register

load_dotenv()

tracer_provider = register(
    project_name="openai-agents",
    auto_instrument=True,
)

from pprint import pprint

import yfinance as yf


def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    data = stock.info
    return {
        "symbol": data.get("symbol"),
        "current_price": data.get("currentPrice"),
        "market_cap": data.get("marketCap"),
        "sector": data.get("sector"),
        "industry": data.get("industry"),
        "description": data.get("longBusinessSummary"),
        "trailing_pe": data.get("trailingPE"),
        "forward_pe": data.get("forwardPE"),
        "dividend_yield": data.get("dividendYield"),
        "beta": data.get("beta"),
        "fifty_two_week_high": data.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": data.get("fiftyTwoWeekLow"),
        "fifty_day_moving_average": data.get("fiftyDayAverage"),
        "two_hundred_day_moving_average": data.get("twoHundredDayAverage"),
        "recommendation_key": data.get("recommendationKey"),
        "revenue_growth": data.get("revenueGrowth"),
        "earnings_growth": data.get("earningsGrowth"),
        "profit_margins": data.get("profitMargins"),
    }


pprint(get_stock_data("AAPL"))

from pprint import pprint
from textwrap import dedent

from agents import Agent, Runner, TResponseInputItem, WebSearchTool, function_tool
from agents.model_settings import ModelSettings


@function_tool
def get_stock_data_tool(ticker_symbol: str) -> dict:
    """
    Get stock data for a given ticker symbol.
    Args:
        ticker_symbol: The ticker symbol of the stock to get data for.
    Returns:
        A dictionary containing stock data such as price, market cap, and more.
    """
    return get_stock_data(ticker_symbol)


research_agent = Agent(
    name="FinancialSearchAgent",
    instructions=dedent(
        """You are a research assistant specializing in financial topics. Given a stock ticker, use web search to retrieve up‑to‑date context and produce a short summary of at most 50 words. Focus on key numbers, events, or quotes that will be useful to a financial analyst."""
    ),
    model="gpt-4.1",
    tools=[WebSearchTool(), get_stock_data_tool],
    model_settings=ModelSettings(tool_choice="required", parallel_tool_calls=True),
)

user_input = input("Enter the stock tickers you want to research: ")
input_items: list[TResponseInputItem] = [{"content": user_input, "role": "user"}]

orchestrator = Runner.run_sync(research_agent, input_items)
orchestrator_output = orchestrator.final_output
pprint(orchestrator_output)
