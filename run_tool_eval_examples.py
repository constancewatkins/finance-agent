"""Thin tool-eval loop against live finance-experiments traces. Assignment-grade."""

import json
import sys

from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.evals import ClassificationEvaluator, LLM, evaluate_dataframe
from sklearn.metrics import classification_report, confusion_matrix

load_dotenv()

PROJECT = "finance-experiments"

TOOL_CHOICE_PROMPT = """
You are given a user request and the tool the agent called to answer it.

[Request]: {request}
[Tool called]: {tool_name}
[Tool description]: {tool_description}

Decide whether this tool can answer the request as asked.
Answer with one word, either "correct" or "incorrect".
"""


def tool_call_is_well_formed(span) -> bool:
    schema = span["attributes.tool.parameters"]
    if isinstance(schema, str):
        schema = json.loads(schema)
    raw = span["attributes.input.value"]
    args = json.loads(raw) if isinstance(raw, str) else raw

    for key in schema.get("required", []):
        if key not in args:
            return False

    types = {
        "string": str,
        "array": list,
        "object": dict,
        "number": (int, float),
        "boolean": bool,
    }
    for key, value in args.items():
        declared = schema["properties"].get(key)
        if declared and not isinstance(value, types.get(declared["type"], ())):
            return False
    return True


def user_request_for_trace(trace_spans) -> str:
    for _, row in trace_spans[trace_spans["span_kind"] == "LLM"].iterrows():
        msgs = row.get("attributes.llm.input_messages")
        if not isinstance(msgs, list):
            continue
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("message.role") or msg.get("role") or "").lower()
            content = msg.get("message.content") or msg.get("content")
            if role == "user" and content:
                return str(content)
    return ""


def parse_score(raw):
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, list):
        raw = raw[0]
    if hasattr(raw, "label"):
        return raw.label, raw.score, raw.explanation
    return raw.get("label"), raw.get("score"), raw.get("explanation")


def make_judge(model: str):
    return ClassificationEvaluator(
        name="tool_selection",
        prompt_template=TOOL_CHOICE_PROMPT,
        llm=LLM(provider="openai", model=model),
        choices={"correct": 1.0, "incorrect": 0.0},
    )


client = Client()
spans = client.spans.get_spans_dataframe(project_identifier=PROJECT, limit=1000)
tools = spans[spans["span_kind"] == "TOOL"].copy()
print(f"project={PROJECT} TOOL spans={len(tools)}")

if "--matrix" in sys.argv:
    annotations = client.spans.get_span_annotations_dataframe(
        spans_dataframe=spans, project_identifier=PROJECT
    )
    yours = annotations[annotations["annotation_name"] == "tool_selection_human"]
    judge = annotations[annotations["annotation_name"] == "tool_selection"]
    paired = yours[["result.label"]].join(
        judge[["result.label", "result.explanation"]],
        lsuffix="_yours",
        rsuffix="_judge",
        how="inner",
    ).join(spans[["attributes.input.value", "name", "attributes.tool.name"]])
    labels = ["correct", "incorrect"]
    y_true, y_pred = paired["result.label_yours"], paired["result.label_judge"]
    print("confusion_matrix (rows=human, cols=judge)", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    disagreements = paired[y_true != y_pred]
    print(f"disagreements: {len(disagreements)}")
    for span_id, row in disagreements.iterrows():
        tool = row.get("attributes.tool.name") or row.get("name")
        print("---")
        print("span_id:", span_id)
        print("tool:", tool)
        print("input:", row["attributes.input.value"])
        print("human:", row["result.label_yours"])
        print("judge:", row["result.label_judge"])
        print("explanation:", row["result.explanation"])
    raise SystemExit(0)

print("\n=== snippet 1: well-formed tool call ===")
for i, (span_id, row) in enumerate(tools.iterrows(), 1):
    name = row.get("attributes.tool.name") or row["name"]
    try:
        ok = tool_call_is_well_formed(row)
        print(f"{i:2d}. {name:28s} {ok}  input={str(row['attributes.input.value'])[:120]}")
    except Exception as exc:
        print(f"{i:2d}. {name:28s} ERROR {type(exc).__name__}: {exc}")

# Bind judge inputs from the trace's first user message (root AGENT input is empty).
eval_rows = []
for span_id, row in tools.iterrows():
    trace = spans[spans["context.trace_id"] == row["context.trace_id"]]
    eval_rows.append(
        {
            "span_id": span_id,
            "request": user_request_for_trace(trace),
            "tool_name": row.get("attributes.tool.name") or row["name"],
            "tool_description": row.get("attributes.tool.description") or "",
            "input_value": row.get("attributes.input.value"),
        }
    )

eval_df = __import__("pandas").DataFrame(eval_rows).head(20)
print(f"\n=== snippet 2: tool_selection judge on {len(eval_df)} spans ===")

judge = None
last_error = None
for model in ("gpt-5.5", "gpt-4o"):
    try:
        candidate = make_judge(model)
        probe = evaluate_dataframe(eval_df.head(1), [candidate], hide_tqdm_bar=True)
        score_col = [c for c in probe.columns if c.endswith("_score")][0]
        parse_score(probe.iloc[0][score_col])
        judge = candidate
        print(f"using model={model}")
        break
    except Exception as exc:
        last_error = exc
        print(f"model {model} failed: {type(exc).__name__}: {exc}")

if judge is None:
    raise SystemExit(f"no working judge model: {last_error}")

scored = evaluate_dataframe(eval_df, [judge], hide_tqdm_bar=False)
score_col = [c for c in scored.columns if c.endswith("_score")][0]

print("\nlabels:")
for _, row in scored.iterrows():
    label, score, explanation = parse_score(row[score_col])
    print(f"- {row['tool_name']:28s} {label}")
    print(f"  request: {row['request'][:140]}")
    print(f"  why: {str(explanation)[:240]}")
    client.spans.add_span_annotation(
        span_id=row["span_id"],
        annotation_name="tool_selection",
        annotator_kind="LLM",
        label=label,
        score=float(score) if score is not None else None,
        explanation=explanation,
    )

print("\nwrote LLM annotations name=tool_selection")
print("Next: label the same TOOL spans in Phoenix as tool_selection_human (correct/incorrect).")
