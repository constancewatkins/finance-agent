# Agent Cookbook: OpenAI Agents + Phoenix OSS

Educational examples for building finance agents with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) and tracing, evaluating, and iterating on them with locally hosted [Phoenix OSS](https://github.com/Arize-ai/phoenix).

The workflow-pattern scripts follow [Arize’s OpenAI Agents tutorials](https://arize.com/docs/phoenix/cookbook/agent-workflow-patterns/openai-agents). The finance evaluation loop adapts Arize’s [Iterative Evaluation & Experimentation](https://arize.com/docs/phoenix/cookbook/ai-engineering-workflows/iterative-evaluation-experimentation-workflow-python) cookbook to a portfolio-research agent instead of a travel agent.

## Disclaimer

This project is for learning agent workflows and observability. It is **not financial advice**, not a production trading or advisory system, and not an official Arize or OpenAI product. Outputs can be incomplete, outdated, or wrong. Do not use them to make investment decisions.

## What is here

**Agent workflow patterns**

| File | Pattern |
| --- | --- |
| `agent.py` | Simple agent with web search |
| `prompt_chaining.py` | Research agent → portfolio agent |
| `parallelization.py` | Concurrent tool calls across tickers |
| `routing.py` | Router hands off to research or Q&A |
| `evaluator_optimizer.py` | Generate a report, grade it, revise |
| `orchestrator.py` | Orchestrator chooses research, eval, then portfolio |

**Finance evaluation loop**

| File | Role |
| --- | --- |
| `finance_portfolio_agent.py` | Reusable portfolio agent with Yahoo Finance tools, risk metrics, and allocation checks |
| `create_finance_dataset.py` | Uploads three test prompts to Phoenix |
| `finance_evaluators.py` | LLM judges for requirement adherence and allocation consistency |
| `run_finance_experiment.py` | Runs the agent on the dataset and scores the outputs |
| `run_tool_eval_examples.py` | Schema check + tool-selection judge on existing TOOL spans; `--matrix` compares that to `tool_selection_human` labels |

## Setup

You need Python 3.12+, an [OpenAI API key](https://platform.openai.com/api-keys), and a local Phoenix server.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Put your OpenAI key in `.env`. Do not commit `.env`.

If `cryptography` fails to install on an Intel Mac, the pin in `requirements.txt` (`<46`) is there because newer versions may try to compile against OpenSSL and fail.

In a second terminal, start Phoenix and leave it running:

```bash
source .venv/bin/activate
phoenix serve
```

Open [http://localhost:6006](http://localhost:6006).

## Run an example

```bash
source .venv/bin/activate
python agent.py
```

Interactive examples (`prompt_chaining.py`, `routing.py`, `orchestrator.py`, `finance_portfolio_agent.py`) will prompt you in the terminal. Type `exit` to stop the routing loop.

## Run the evaluation experiment

1. Start Phoenix.
2. Upload the dataset once:

```bash
python create_finance_dataset.py
```

3. Run the experiment:

```bash
python run_finance_experiment.py
```

Phoenix will print a link to compare results. The first experiment is a baseline; later runs use the same dataset so you can compare prompt changes.

## Notes

- Phoenix stores local data under `~/.phoenix`.
- Live Yahoo Finance and web-search results can change between runs, so experiment comparisons are not perfectly reproducible.
- Evaluator scores measure rubric compliance, not factual accuracy. Human review still matters.
