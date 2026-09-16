import yfinance as yf
from agents import Agent, Runner, WebSearchTool, function_tool
from agents.model_settings import ModelSettings
from dotenv import load_dotenv
from opentelemetry import trace
from phoenix.otel import register

load_dotenv()

tracer_provider = register(
    project_name="finance-experiments",
    auto_instrument=True,
)
tracer = trace.get_tracer(__name__)


@tracer.chain(name="finance-data-api")
def _get_stock_data(ticker_symbol: str) -> dict:
    """Retrieve selected market and fundamental data from Yahoo Finance."""
    data = yf.Ticker(ticker_symbol).info
    return {
        "symbol": data.get("symbol"),
        "current_price": data.get("currentPrice"),
        "market_cap": data.get("marketCap"),
        "sector": data.get("sector"),
        "industry": data.get("industry"),
        "trailing_pe": data.get("trailingPE"),
        "forward_pe": data.get("forwardPE"),
        "dividend_yield": data.get("dividendYield"),
        "beta": data.get("beta"),
        "revenue_growth": data.get("revenueGrowth"),
        "earnings_growth": data.get("earningsGrowth"),
        "profit_margins": data.get("profitMargins"),
        "recommendation_key": data.get("recommendationKey"),
    }


@tracer.chain(name="price-history-api")
def _get_risk_metrics(ticker_symbol: str) -> dict:
    """Calculate one-year return and risk metrics from Yahoo Finance prices."""
    history = yf.Ticker(ticker_symbol).history(period="1y", auto_adjust=True)
    close = history["Close"].dropna()

    if close.empty:
        return {"symbol": ticker_symbol, "error": "No price history available."}

    daily_returns = close.pct_change().dropna()
    rolling_peak = close.cummax()
    drawdowns = close / rolling_peak - 1

    return {
        "symbol": ticker_symbol,
        "one_year_return_percent": round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
        "annualized_volatility_percent": round(daily_returns.std() * (252**0.5) * 100, 2),
        "maximum_drawdown_percent": round(drawdowns.min() * 100, 2),
    }


@function_tool
def get_stock_data(ticker_symbols: list[str]) -> list[dict]:
    """
    Get current market and fundamental data for several stocks or ETFs.

    Args:
        ticker_symbols: The ticker symbols to research in one batch.
    """
    return [_get_stock_data(ticker_symbol) for ticker_symbol in ticker_symbols]


@function_tool
def get_risk_metrics(ticker_symbols: list[str]) -> list[dict]:
    """
    Get one-year return, annualized volatility, and maximum drawdown for several tickers.

    Args:
        ticker_symbols: The ticker symbols to analyze in one batch.
    """
    return [_get_risk_metrics(ticker_symbol) for ticker_symbol in ticker_symbols]


@function_tool
def check_portfolio_allocations(
    ticker_symbols: list[str], allocation_percentages: list[float]
) -> dict:
    """
    Check whether proposed portfolio allocations are valid and sum to 100%.

    Args:
        ticker_symbols: The five proposed ticker symbols.
        allocation_percentages: Percentage allocation for each ticker in the same order.
    """
    total = sum(allocation_percentages)
    return {
        "holding_count": len(ticker_symbols),
        "total_allocation_percent": round(total, 2),
        "has_five_holdings": len(ticker_symbols) == 5,
        "ticker_and_allocation_counts_match": len(ticker_symbols)
        == len(allocation_percentages),
        "has_no_negative_allocations": all(
            value >= 0 for value in allocation_percentages
        ),
        "sums_to_100_percent": abs(total - 100) < 0.01,
    }


portfolio_agent = Agent(
    name="Finance Portfolio Agent",
    instructions=(
        "You are an educational financial research assistant. Given an investment "
        "strategy, propose a diversified portfolio of exactly five stocks or ETFs. "
        "Give each ticker a percentage allocation that sums to exactly 100%. "
        "First choose five candidate tickers. Call each finance data tool only once, "
        "passing all five tickers together in a single batch. Use at most one web "
        "search unless a tool reports an error. "
        "Use tools to gather current fundamentals and risk evidence, explain how each "
        "holding supports the request, and identify material risks. Check the proposed "
        "allocations with the allocation tool before answering. Treat the requested "
        "risk tolerance as a binding constraint: compare candidate volatility and "
        "drawdown, replace holdings whose risk conflicts with it, and keep any "
        "necessary higher-risk thematic holding to a small, explicitly justified "
        "allocation. State how the completed portfolio matches the requested risk "
        "profile, and never describe it as more aggressive than requested. Clearly "
        "distinguish sourced facts from your analysis. Do not present the result as "
        "personalized financial advice."
    ),
    model="gpt-4.1-mini",
    tools=[
        WebSearchTool(),
        get_stock_data,
        get_risk_metrics,
        check_portfolio_allocations,
    ],
    model_settings=ModelSettings(parallel_tool_calls=True),
)


def run_finance_agent(query: str) -> str:
    """Run the finance agent for one experiment input."""
    result = Runner.run_sync(portfolio_agent, query)
    return result.final_output


if __name__ == "__main__":
    user_input = input("Enter an investment strategy: ")
    print(run_finance_agent(user_input))
