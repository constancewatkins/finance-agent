from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.client.experiments import run_experiment

from finance_evaluators import allocation_evaluator, requirement_evaluator
from finance_portfolio_agent import run_finance_agent

load_dotenv()

client = Client()
dataset = client.datasets.get_dataset(dataset="finance-portfolio-questions")


def agent_task(input):
    query = input["input"]
    return run_finance_agent(query)


experiment = run_experiment(
    dataset=dataset,
    task=agent_task,
    experiment_name="improved finance risk alignment",
    evaluators=[requirement_evaluator, allocation_evaluator],
    timeout=180,
    retries=0,
)
