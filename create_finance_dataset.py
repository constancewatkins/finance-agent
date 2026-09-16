from dotenv import load_dotenv
import pandas as pd
from phoenix.client import Client

load_dotenv()

queries = [
    (
        "Create a moderate-risk portfolio of exactly five U.S. stocks or ETFs "
        "focused on AI and clean energy. Include current evidence and risk metrics, "
        "and make the allocations total 100%."
    ),
    (
        "Create a conservative income portfolio of exactly five U.S. stocks or ETFs. "
        "Include at least two ETFs, keep every allocation at or below 30%, discuss "
        "dividend and volatility risks, and make the allocations total 100%."
    ),
    (
        "Create an aggressive-growth portfolio of exactly five U.S. stocks or ETFs "
        "focused on semiconductors, cloud computing, and cybersecurity. Include "
        "current evidence and downside risks, and make the allocations total 100%."
    ),
]

dataset_df = pd.DataFrame(data={"input": queries})

client = Client()
dataset = client.datasets.create_dataset(
    dataframe=dataset_df,
    name="finance-portfolio-questions",
    input_keys=["input"],
)

print(f"Created Phoenix dataset: {dataset.name}")
