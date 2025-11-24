import json
import os
import re
import argparse
from pathlib import Path

from openai import OpenAI

from tools import python_exec, xlsx_query
from prompt_templates import build_react_prompt


class ReActAgent:
    def __init__(
        self,
        mode="baseline",
        max_steps=8,
        max_reflections=2,
        model_name="gpt-4o-mini",
        api_key=None,
        mock=False,
    ):
        self.mode = mode
        self.max_steps = max_steps
        self.max_reflections = max_reflections
        self.model_name = model_name
        self.mock = mock

        if not self.mock:
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OpenAI API key is required unless mock mode is enabled.")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

    def call_model(self, prompt):
        if self.mock:
            question = None
            file_path = None
            has_observation = False

            for line in prompt.splitlines():
                if line.startswith("Question: "):
                    question = line[len("Question: "):].strip()
                elif line.startswith("Associated file path: "):
                    file_path = line[len("Associated file path: "):].strip()
                elif line.strip().startswith("Observation:"):
                    has_observation = True

            if has_observation:
                if question is None:
                    question = "the question"
                return "Thought: I have seen the tool result.\nAnswer: mock answer for %s." % question

            if file_path is None:
                return "Thought: mock mode but no file path found.\nAnswer: mock answer."

            if file_path.endswith(".py"):
                return (
                    "Thought: I should run the python script to get the numeric result.\n"
                    f"Action: python_exec(\"{file_path}\")"
                )
            elif file_path.endswith(".xlsx"):
                q = question if question is not None else "Query over the spreadsheet."
                return (
                    "Thought: I should query the spreadsheet using the question.\n"
                    f"Action: xlsx_query(\"{file_path}\", \"{q}\")"
                )
            else:
                return "Thought: unsupported file type in mock mode.\nAnswer: mock answer."

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful reasoning agent that uses tools via ReAct."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

    def parse_action(self, model_output):
        lines = model_output.splitlines()
        action_line = None
        for line in lines:
            if line.strip().lower().startswith("action:"):
                action_line = line.strip()
                break

        if action_line is None:
            return None

        action_part = action_line.split(":", 1)[1].strip()

        m_py = re.match(r'python_exec\(["\'](.+?)["\']\)', action_part)
        if m_py:
            return {"tool": "python_exec", "input": m_py.group(1)}

        m_xlsx = re.match(r'xlsx_query\(["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\)', action_part)
        if m_xlsx:
            return {
                "tool": "xlsx_query",
                "input": {"path": m_xlsx.group(1), "query": m_xlsx.group(2)},
            }

        return None

    def run_single(self, task_id, question, file_name, base_dir=".", run_id=0):
        traj = []
        reflections_used = 0
        final_answer = None

        file_path = str(Path(base_dir) / file_name)

        for step in range(1, self.max_steps + 1):
            prompt = build_react_prompt(question, file_path, traj, reflections_used)

            model_output = self.call_model(prompt)

            if "Answer:" in model_output:
                answer_part = model_output.split("Answer:", 1)[1].strip()
                final_answer = answer_part
                traj.append(
                    {
                        "step": step,
                        "thought": model_output,
                        "action": None,
                        "observation": None,
                        "retrieved_rules": [],
                    }
                )
                break

            action_spec = self.parse_action(model_output)

            if action_spec is None:
                traj.append(
                    {
                        "step": step,
                        "thought": model_output,
                        "action": None,
                        "observation": {"error": "no_action_parsed"},
                        "retrieved_rules": [],
                    }
                )
                break

            tool_name = action_spec["tool"]
            tool_input = action_spec["input"]

            if tool_name == "python_exec":
                observation = python_exec(tool_input)
            elif tool_name == "xlsx_query":
                observation = xlsx_query(tool_input["path"], tool_input["query"])
            else:
                observation = {"error": "unknown_tool"}

            traj.append(
                {
                    "step": step,
                    "thought": model_output,
                    "action": {"tool": tool_name, "input": tool_input},
                    "observation": observation,
                    "retrieved_rules": [],
                }
            )

            if self._should_reflect(observation, model_output) and reflections_used < self.max_reflections:
                reflections_used += 1
                reflection_note = self._reflect(traj)
                traj.append(
                    {
                        "step": step,
                        "thought": "Reflection: " + reflection_note,
                        "action": None,
                        "observation": None,
                        "retrieved_rules": [],
                    }
                )

        judgment = "answered" if final_answer else "failed"

        log_obj = {
            "task_id": task_id,
            "mode": self.mode,
            "run_id": run_id,
            "question": question,
            "file_name": file_name,
            "final_answer": final_answer,
            "judgment": judgment,
            "trajectory": traj,
        }

        self._save_traj(task_id, run_id, log_obj)
        return log_obj

    def _should_reflect(self, observation, model_output):
        if isinstance(observation, dict) and "error" in observation:
            return True
        lower = model_output.lower()
        if "not sure" in lower or "uncertain" in lower:
            return True
        return False

    def _reflect(self, traj):
        return "I should reconsider my previous tool choices and double-check the results."

    def _save_traj(self, task_id, run_id, log_obj):
        out_dir = Path("runs") / str(task_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / ("baseline_%d.json" % run_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(log_obj, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=False)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--question", type=str, default="Which city had the greater total sales: Wharvton or Algrimand?")
    parser.add_argument("--file_name", type=str, default="your_api")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    agent = ReActAgent(
        mode="baseline",
        max_steps=8,
        max_reflections=2,
        model_name=args.model,
        api_key=args.api_key,
        mock=args.mock,
    )

    log = agent.run_single(
        task_id=args.task_id,
        question=args.question,
        file_name=args.file_name,
        base_dir=args.base_dir,
        run_id=args.run_id,
    )
    print("final_answer:", log["final_answer"])
    print("judgment:", log["judgment"])
