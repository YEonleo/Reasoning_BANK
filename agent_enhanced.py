
import json
import os
import re
import argparse
from pathlib import Path

from openai import OpenAI

from tools import python_exec, xlsx_query
from prompt_templates import build_react_prompt_enhanced
from reasoning_bank import ReasoningBank


class EnhancedAgent:
    def __init__(
        self,
        mode="enhanced",
        max_steps=8,
        max_reflections=2,
        model_name="gpt-4o-mini",
        api_key=None,
        mock=False,
        bank_path="memory/bank.json",
    ):
        self.mode = mode
        self.max_steps = max_steps
        self.max_reflections = max_reflections
        self.model_name = model_name
        self.mock = mock

        self.bank = ReasoningBank(bank_path)

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
                return "Thought: I have seen the tool result.\nAnswer: mock enhanced answer for %s." % question

            if file_path is None:
                return "Thought: mock mode but no file path found.\nAnswer: mock enhanced answer."

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
                return "Thought: unsupported file type in mock mode.\nAnswer: mock enhanced answer."

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

    def _infer_tags(self, question, file_path):
        tags = []
        lower_q = question.lower()
        if file_path.endswith(".xlsx"):
            tags.append("xlsx")
            tags.append("spreadsheet")
        if file_path.endswith(".py"):
            tags.append("python")
        if "sales" in lower_q:
            tags.append("sales")
        if "city" in lower_q:
            tags.append("city")
        if "operating" in lower_q:
            tags.append("operating_status")
        return tags

    def run_single(self, task_id, question, file_name, base_dir=".", run_id=0):
        traj = []
        reflections_used = 0
        final_answer = None

        file_path = str(Path(base_dir) / file_name)

        for step in range(1, self.max_steps + 1):
            tags = self._infer_tags(question, file_path)
            rules = self.bank.retrieve_rules(tags=tags, max_rules=2)
            prompt = build_react_prompt_enhanced(question, file_path, traj, reflections_used, rules)

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
                        "retrieved_rules": [r.get("id") for r in rules],
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
                        "retrieved_rules": [r.get("id") for r in rules],
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
                    "retrieved_rules": [r.get("id") for r in rules],
                }
            )

            if self._should_reflect(observation, model_output, traj) and reflections_used < self.max_reflections:
                reflections_used += 1
                reflection_note = self._reflect(traj)
                new_rules = self._generate_rules(question, file_path, traj, reflection_note)
                for r in new_rules:
                    self.bank.add_rule(r)
                traj.append(
                    {
                        "step": step,
                        "thought": "Reflection: " + reflection_note,
                        "action": None,
                        "observation": None,
                        "retrieved_rules": [r.get("id") for r in rules],
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

    def _should_reflect(self, observation, model_output, traj):
        # 1) mock 모드에서는 항상 한 번은 Reflection 하도록 (테스트, bank.json 생성용)
        if self.mock:
            return True

        # 2) 툴이 에러를 반환하면 Reflection
        if isinstance(observation, dict) and "error" in observation:
            return True

        lower = model_output.lower()

        # 3) xlsx_query에 SQL 같은 쿼리를 던진 건 실패 패턴으로 간주
        #    (SELECT, GROUP BY 등을 포함하는 경우)
        if "select " in lower or "group by" in lower or " from " in lower:
            return True

        # 4) 같은 Action을 두 번 연속 반복하면 비효율/실패 시도로 간주
        if len(traj) >= 2:
            last = traj[-1].get("action")
            prev = traj[-2].get("action")
            if last and prev and last == prev:
                return True

        # 5) step이 너무 많아지면(예: 3스텝 이상) 한번 돌아보자
        if len(traj) >= 3:
            return True

        return False


    def _reflect(self, traj):
        return "I should reconsider my previous tool choices and double-check the results."

    def _build_trajectory_text(self, traj):
        lines = []
        for step_log in traj:
            s = step_log.get("step")
            thought = step_log.get("thought")
            action = step_log.get("action")
            observation = step_log.get("observation")
            lines.append("Step %s" % s)
            if thought is not None:
                lines.append("Thought: %s" % thought)
            if action is not None:
                lines.append("Action: %s" % action)
            if observation is not None:
                lines.append("Observation: %s" % json.dumps(observation, ensure_ascii=False))
            lines.append("")
        return "\n".join(lines)

    def _generate_rules(self, question, file_path, traj, reflection_note):
        # 태그는 질문/파일에서 간단히 추론
        tags = self._infer_tags(question, file_path)
        last_step = traj[-1]["step"] if traj else 0

        # 마지막 observation을 한 번 훑어서 error 여부로 polarity 결정
        last_obs = None
        for step_log in reversed(traj):
            obs = step_log.get("observation")
            if obs is not None:
                last_obs = obs
                break

        polarity = "success"
        if isinstance(last_obs, dict) and "error" in last_obs:
            polarity = "failure"

        # content는 체크리스트 형태로 1~3줄 정도로 단순히 고정
        content_lines = [
            "Always inspect the tool output (e.g., Python stdout or spreadsheet rows) carefully before answering.",
            "If the question asks for a comparison or probability, explicitly compute all relevant numeric values using tools first.",
        ]

        rule = {
            "title": "Strategy for %s task" % (", ".join(tags) if tags else "this"),
            "description": "Reusable rule distilled from a %s trajectory." % polarity,
            "content": content_lines,
            "tags": tags,
            "polarity": polarity,
            "evidence": ["traj_step#%d" % last_step],
            # id, created_at, use_count는 ReasoningBank.add_rule에서 자동으로 채워짐
        }

        return [rule]

    def _save_traj(self, task_id, run_id, log_obj):
        out_dir = Path("runs") / str(task_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / ("enhanced_%d.json" % run_id)
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
    parser.add_argument("--bank_path", type=str, default="memory/bank.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    agent = EnhancedAgent(
        mode="enhanced",
        max_steps=8,
        max_reflections=2,
        model_name=args.model,
        api_key=args.api_key,
        mock=args.mock,
        bank_path=args.bank_path,
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
