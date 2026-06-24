# Redrob Intelligent Candidate Ranker

## What this does

Ranks 100,000 candidates against the Senior AI Engineer JD using a **multi-dimensional recruiter-logic scorer** — not keyword matching.

The system explicitly models:
- What the JD *means* vs. what it *says* (career evidence outweighs skill-list claims)
- JD disqualifiers (consulting-only backgrounds, pure research, CV/speech primary)
- Behavioral availability signals (recency, response rate, open-to-work)
- Honeypot detection (timeline impossibilities, overlapping roles)

## Architecture

```
candidates.jsonl (100K)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  1. Honeypot filter                                 │
│     - Career timeline consistency check             │
│     - Overlapping full-time role detection          │
│     - Impossible skill proficiency/duration combos  │
└───────────────────┬─────────────────────────────────┘
                    │ (valid candidates)
                    ▼
┌─────────────────────────────────────────────────────┐
│  2. Three-component scorer                          │
│                                                     │
│  Skill Match (42%)                                  │
│  ├─ embeddings/dense retrieval  (25%)               │
│  ├─ vector DB / hybrid search   (20%)               │
│  ├─ IR / ranking / RAG          (20%)               │
│  ├─ evaluation frameworks       (15%)               │
│  ├─ LLM / NLP applied           (12%)               │
│  └─ Python / ML stack           (8%)                │
│     Career evidence > skill list > profile mention  │
│                                                     │
│  Career Trajectory (35%)                            │
│  ├─ YoE in ideal band (5–9yr)                       │
│  ├─ Product-company vs services ratio               │
│  ├─ Current role writes code (not pure architect)   │
│  ├─ Disqualify: consulting-only, pure research      │
│  └─ Title-chasing detection (short stints)          │
│                                                     │
│  Availability (23%)                                 │
│  ├─ Days since last active                          │
│  ├─ Open-to-work flag                               │
│  ├─ Recruiter response rate                         │
│  ├─ Notice period                                   │
│  ├─ Location (Pune/Noida preferred)                 │
│  ├─ Interview completion rate                       │
│  └─ GitHub activity score                           │
└───────────────────┬─────────────────────────────────┘
                    │ composite = 0.42*skill + 0.35*career + 0.23*avail
                    ▼
┌─────────────────────────────────────────────────────┐
│  3. Reasoning generator                             │
│     Specific facts: title, YoE, company, skills    │
│     JD-connected logic                              │
│     Honest concerns acknowledged                    │
│     No templating — varied per candidate            │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
              submission.csv (top 100)
```

## Key design decisions

**Why career evidence > skill lists?**
The JD explicitly warns about keyword stuffers. A candidate whose job description says "migrated from BM25 to a dense retrieval system" scores higher on embeddings than one who lists "Embeddings" in their skill list with no career evidence.

**Why 42/35/23 weights?**
Skill match dominates because this is a highly specialized role. Career trajectory is heavily weighted because the JD has very specific anti-patterns to avoid (consulting-only, pure research). Availability matters because the best candidate on paper who hasn't logged in in 6 months and has a 5% response rate is, in practice, not hirable.

**Honeypot detection:**
The system flags candidates with impossible timelines (total career months >> stated YoE), overlapping full-time roles (>90 day overlap), or excessive "expert" skills with sub-6-month durations. These are excluded before scoring.

## Requirements

```
python >= 3.10
No external ML libraries needed — pure Python + stdlib
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/redrob-ranker
cd redrob-ranker
# No pip install needed — uses Python stdlib only
```

## Reproduce submission

**Step 1: Pre-computation (none required)**
This ranker uses no pre-computed embeddings or indexes — it scores on parsed features directly. This keeps it well within the 5-minute, CPU-only, no-network constraints.

**Step 2: Ranking**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Expected runtime: ~45 seconds on any modern CPU for 100K candidates.

**Step 3: Validate**
```bash
python validate_submission.py ./submission.csv
```

## Compute constraints compliance

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤ 5 min | ~45 sec |
| Memory | ≤ 16 GB | < 500 MB (streaming) |
| Compute | CPU only | ✅ |
| Network | Off | ✅ (no API calls) |
| Disk | ≤ 5 GB | < 1 MB output |

## Files

```
rank.py                        # Main ranker
submission.csv                 # Output (top 100 ranked candidates)
submission_metadata.yaml       # Metadata template
README.md                      # This file
```
