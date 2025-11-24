# AI Agent 사전과제 – ReAct + Reflection + ReasoningBank (라이트 버전)

## 1. 개요

이 프로젝트는 OpenAI Python SDK 기반으로 두 가지 에이전트 모드를 구현한 사전과제 제출용 코드입니다.

- **Baseline:** ReAct + Reflection  
- **Enhanced:** ReasoningBank(라이트 버전) + ReAct + Reflection  

Baseline은 문제를 해결할 수 있지만, 특정 문제(Task 2)에서 SQL-like 쿼리 반복 등  
**불필요한 시도(step 증가)** 가 나타났습니다.  
이는 ReasoningBank 논문에서 말하는 **soft failure**(정답은 맞지만 비효율적 추론 경로) 패턴입니다.

본 구현에서는 이러한 **비효율적 사고 패턴을 규칙(Memory Item)** 으로 저장하고  
다음 execution에서 이를 불러와 **에이전트가 더 안정적으로 Action을 선택하도록 유도**하는  
ReasoningBank 구조를 라이트 버전으로 재현했습니다.

---

## 2. 구현 방향 요약

### ✔ ReAct + Reflection (Baseline)
- Thought → Action → Observation 구조  
- python_exec / xlsx_query 두 도구 사용  
- 실패·불확실 시 Reflection (최대 2회)

### ✔ ReasoningBank (Enhanced)
- Reflection 시점에서 trajectory 기반 규칙 자동 생성  
- 2-step 이상 소요 · 같은 Action 반복 · SQL-like 쿼리 오용 등을 비효율 패턴으로 간주  
- 자동 생성되는 규칙(JSON 스키마) 예시:

```
{
    "title": "Strategy for python task",
    "description": "Reusable rule distilled from a success trajectory.",
    "content": [
      "Always inspect the tool output carefully.",
      "Explicitly compute numeric values before comparing."
    ],
    "tags": ["python"],
    "polarity": "success",
    "evidence": ["traj_step#1"],
    "id": "rb_0001",
    "use_count": 3,
    "created_at": "2025-11-24T03:50:54.226999Z"
}
```

- Enhanced 모드에서는 문제 태그(xlsx, python 등)에 따라 규칙을 검색(max 2개)  
- 규칙을 프롬프트 상단에 주입하여 ReAct 추론 품질 개선  
- 각 step에는 `retrieved_rules` 로 어떤 규칙이 참고되었는지 기록됨

---

## 3. 프로젝트 구조

```
.
├── agent_baseline.py
├── agent_enhanced.py
├── reasoning_bank.py
├── prompt_templates.py
├── tools.py
├── run_baseline.py
├── run_enhanced.py
├── test/
│   ├── 사전과제.json
│   ├── *.py / *.xlsx
├── memory/
│   └── bank.json
└── runs/
    ├── {task_id}/baseline_0.json
    ├── {task_id}/enhanced_0.json
```

---

## 4. 실행 방법

### Baseline 실행
```
python run_baseline.py --mock
```
또는 실제 API:
```
python run_baseline.py --api_key YOUR_KEY --model gpt-5-nano
```

### Enhanced 실행
```
python run_enhanced.py --mock
```
또는 실제 API:
```
python run_enhanced.py --api_key YOUR_KEY --model gpt-5-nano
```

실행 후 생성:
- answers_baseline.json  
- answers_enhanced.json  
- memory/bank.json  
- runs/{task_id}/*.json  

---

## 5. 실험 결과 요약

### Baseline (answers_baseline.json)
- Python: 2 step → 정답  
- Sales 비교: SQL-like query 반복 → 4 step  
- 기관차 문제: 2 step

### Enhanced (answers_enhanced.json)
- 규칙(rb_0001~rb_0003)을 프롬프트에 주입하여 안정적 추론 수행  
- Sales 문제에서 SQL-like 반복 제거되고 step 감소  
- 기관차 문제에서도 operating_status 규칙 재사용

규칙 예시:
```
{
  "title": "Strategy for xlsx, spreadsheet, sales, city task",
  "content": [
    "Always inspect tool output carefully.",
    "Compute relevant numeric values first."
  ],
  "tags": ["xlsx","sales","city"],
  "polarity": "success",
  "evidence": ["traj_step#1"]
}
```

---

## 6. 제출 요건 충족 여부

| 항목 | 충족 내용 |
|------|-----------|
| 기능(40) | ReAct, Reflection, ReasoningBank(생성·검색·주입) 구현 |
| 로깅(20) | runs/{task_id}/{mode}_{run_id}.json에 step 단위 trajectory 기록 |
| 보고(20) | README에 전체 구조·실행 가이드·결과 요약 포함 |
| 적절성(20) | 과제 요구사항 충족 + 복잡도 최소화 |

---

## 7. 요약

본 구현은 Baseline의 한계를 보완하여  
**“경험 기반 self-evolving 에이전트”** 구조를 간결한 형태로 재현했습니다.  
Enhanced 모드에서는 Memory Bank 기반의 전략 재사용을 통해  
더 안정적·일관된 ReAct 추론을 수행함을 확인했습니다.
