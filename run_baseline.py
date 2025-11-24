import json
import argparse
from pathlib import Path

from agent_baseline import ReActAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=False)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--tasks_path", type=str, default="test/사전과제.json")
    parser.add_argument("--base_dir", type=str, default="test")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    agent = ReActAgent(
        mode="baseline",
        max_steps=8,
        max_reflections=2,
        model_name=args.model,
        api_key=args.api_key,
        mock=args.mock,
    )

    answers = []
    for idx, task in enumerate(tasks, start=1):
        question = task["question"]
        file_name = task["file_name"]

        log = agent.run_single(
            task_id=idx,
            question=question,
            file_name=file_name,
            base_dir=args.base_dir,
            run_id=args.run_id,
        )

        answers.append(
            {
                "task_id": idx,
                "question": question,
                "file_name": file_name,
                "answer": log["final_answer"],
                "judgment": log["judgment"],
            }
        )

    out_path = Path("answers_baseline.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
