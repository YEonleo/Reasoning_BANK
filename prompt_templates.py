import json

REACT_PROMPT_TEMPLATE = """You are a tool-using agent.

You have access to the following tools:
1) python_exec(path): execute a Python script and extract the final numeric output.
2) xlsx_query(path, query): query an Excel spreadsheet and compute useful aggregates.

You should follow the ReAct style:
- Start with `Thought:` when you reason.
- When you want to use a tool, output a single line starting with `Action:`.
  The Action must be exactly one of the following forms:
    Action: python_exec("<path>")
    Action: xlsx_query("<path>", "<query>")
- When you are confident about the final result, output a line starting with `Answer:`.

Question: {question}
Associated file path: {file_path}

Previous steps:
{history}
Reflections used so far: {reflections_used}

Now continue with your next Thought (and possibly Action or Answer).
"""


def build_react_prompt(question, file_path, traj, reflections_used):
    history = ""
    for step_log in traj:
        history += "Step %d Thought:\n%s\n" % (step_log["step"], step_log["thought"])
        if step_log["action"] is not None:
            history += "Action: %s\n" % step_log["action"]
            history += "Observation: %s\n" % json.dumps(step_log["observation"], ensure_ascii=False)
        history += "\n"

    prompt = REACT_PROMPT_TEMPLATE.format(
        question=question,
        file_path=file_path,
        history=history,
        reflections_used=reflections_used,
    )
    return prompt


ENHANCED_REACT_PROMPT_TEMPLATE = """You are a tool-using agent.

You have access to the following tools:
1) python_exec(path): execute a Python script and extract the final numeric output.
2) xlsx_query(path, query): query an Excel spreadsheet and compute useful aggregates.

You should follow the ReAct style:
- Start with `Thought:` when you reason.
- When you want to use a tool, output a single line starting with `Action:`.
  The Action must be exactly one of the following forms:
    Action: python_exec("<path>")
    Action: xlsx_query("<path>", "<query>")
- When you are confident about the final result, output a line starting with `Answer:`.

Here are some past reasoning strategies you may find useful:
{rules_block}

Question: {question}
Associated file path: {file_path}

Previous steps:
{history}
Reflections used so far: {reflections_used}

Now continue with your next Thought (and possibly Action or Answer).
"""


def build_react_prompt_enhanced(question, file_path, traj, reflections_used, rules):
    history = ""
    for step_log in traj:
        history += "Step %d Thought:\n%s\n" % (step_log["step"], step_log["thought"])
        if step_log["action"] is not None:
            history += "Action: %s\n" % step_log["action"]
            history += "Observation: %s\n" % json.dumps(step_log["observation"], ensure_ascii=False)
        history += "\n"

    if not rules:
        rules_block = "(no prior rules available for this task)\n"
    else:
        lines = []
        for r in rules:
            title = r.get("title", "")
            tags = r.get("tags", [])
            lines.append("- %s [tags: %s]" % (title, ", ".join(tags)))
            for c in r.get("content", []):
                lines.append("  • %s" % c)
        rules_block = "\n".join(lines)

    prompt = ENHANCED_REACT_PROMPT_TEMPLATE.format(
        question=question,
        file_path=file_path,
        history=history,
        reflections_used=reflections_used,
        rules_block=rules_block,
    )
    return prompt
