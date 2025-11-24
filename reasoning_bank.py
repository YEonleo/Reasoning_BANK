# reasoning_bank.py

import json
from pathlib import Path
from datetime import datetime


class ReasoningBank:
    def __init__(self, path="memory/bank.json"):
        self.path = Path(path)
        self.rules = []
        self._load()

    def _load(self):
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                try:
                    self.rules = json.load(f)
                except Exception:
                    self.rules = []
        else:
            self.rules = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)

    def _next_id(self):
        # rb_0001, rb_0002 형식
        if not self.rules:
            return "rb_0001"
        nums = []
        for r in self.rules:
            rid = r.get("id", "")
            if rid.startswith("rb_"):
                try:
                    n = int(rid.split("_", 1)[1])
                    nums.append(n)
                except Exception:
                    pass
        if not nums:
            return "rb_0001"
        n = max(nums) + 1
        return "rb_%04d" % n

    def add_rule(self, rule):
        if "id" not in rule:
            rule["id"] = self._next_id()
        if "use_count" not in rule:
            rule["use_count"] = 0
        if "created_at" not in rule:
            rule["created_at"] = datetime.utcnow().isoformat() + "Z"
        # evidence는 리스트로 강제
        ev = rule.get("evidence")
        if isinstance(ev, str):
            rule["evidence"] = [ev]
        elif isinstance(ev, list):
            rule["evidence"] = ev
        else:
            rule["evidence"] = []
        self.rules.append(rule)
        self._save()
        return rule

    def retrieve_rules(self, tags=None, polarity=None, max_rules=2):
        if not self.rules:
            return []

        if tags is None:
            tags = []
        tags = [t.lower() for t in tags]

        scored = []
        for r in self.rules:
            if polarity is not None:
                if r.get("polarity") != polarity:
                    continue
            r_tags = [str(t).lower() for t in r.get("tags", [])]
            score = 0
            for t in tags:
                if t in r_tags:
                    score += 1
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = []
        for score, r in scored:
            if len(picked) >= max_rules:
                break
            if score > 0 or not tags:
                picked.append(r)

        for r in picked:
            r["use_count"] = int(r.get("use_count", 0)) + 1
        if picked:
            self._save()

        return picked
