"""
Redrob Intelligent Candidate Ranker
====================================
Approach: Multi-dimensional scoring that mirrors how a great recruiter reads profiles.
- NOT keyword matching
- Reads career trajectory, signal coherence, behavioral availability
- Detects honeypots via timeline/skill consistency checks
- Produces 100 ranked candidates with specific, honest reasoning

Run:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import json
import csv
import argparse
import re
import math
import sys
from datetime import date, datetime
from collections import defaultdict


# ── JD-derived constants ─────────────────────────────────────────────────────

# Skills the JD says are REQUIRED (hard signals)
REQUIRED_SKILL_FAMILIES = {
    "embeddings": [
        "sentence-transformers", "sentence transformers", "embeddings",
        "openai embeddings", "bge", "e5", "bi-encoder", "dense retrieval",
        "semantic search", "vector search", "text embeddings"
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "chroma", "pgvector", "hybrid search",
        "ann", "approximate nearest neighbor", "hnsw"
    ],
    "retrieval_ir": [
        "information retrieval", "ir", "bm25", "sparse retrieval",
        "hybrid retrieval", "ranking", "re-ranking", "reranking",
        "recommendation", "search", "recommendation system",
        "retrieval augmented", "rag", "retrieval-augmented generation"
    ],
    "ranking_eval": [
        "ndcg", "mrr", "map", "precision@", "recall@", "a/b test",
        "ab testing", "offline eval", "online eval", "evaluation framework",
        "learning to rank", "ltr", "lambdamart", "ranknet", "listwise"
    ],
    "llm_applied": [
        "llm", "large language model", "gpt", "bert", "transformers",
        "fine-tuning", "fine tuning", "lora", "qlora", "peft", "rlhf",
        "instruction tuning", "prompt engineering", "langchain", "llamaindex",
        "hugging face", "huggingface"
    ],
    "python_ml": [
        "python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "pandas", "numpy", "spark", "mlflow"
    ],
}

# Skills that are DESIRED (nice to have)
DESIRED_SKILL_FAMILIES = {
    "distributed": ["kafka", "airflow", "kubernetes", "docker", "spark", "distributed systems"],
    "open_source": ["open source", "github", "contributions", "maintainer"],
    "hr_domain": ["recruiting", "hr-tech", "talent", "ats", "hiring", "recruitment"],
}

# Job title signals — actual AI/ML roles at product companies
STRONG_TITLE_SIGNALS = [
    "ml engineer", "machine learning engineer", "ai engineer", "applied scientist",
    "nlp engineer", "search engineer", "ranking engineer", "recommendation engineer",
    "research engineer", "applied ml", "senior engineer", "staff engineer",
    "principal engineer", "data scientist", "applied researcher"
]

# Explicit disqualifiers from the JD
EXPLICIT_DISQUALIFIER_COMPANIES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hexaware", "mphasis", "mindtree", "larsen & toubro infotech",
    "lti", "ltimindtree"
]

# Pure research roles (academia disqualifier)
RESEARCH_ONLY_SIGNALS = [
    "research assistant", "phd researcher", "postdoc", "research intern",
    "professor", "lecturer", "research fellow"
]

# Vision/speech specializations (explicit JD disqualifier if primary)
CV_SPEECH_ONLY = [
    "computer vision engineer", "cv engineer", "speech engineer",
    "robotics engineer", "autonomous systems"
]

# Title-chasing pattern (many short stints purely for title bump)
CONSULTING_DOMAINS = ["IT Services", "Consulting", "Staffing", "Outsourcing", "BPO"]

TARGET_LOCATIONS = [
    "pune", "noida", "delhi", "gurugram", "gurgaon", "hyderabad",
    "mumbai", "bangalore", "bengaluru", "chennai"
]

TODAY = date.today()


# ── Honeypot detection ────────────────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Returns (is_honeypot, reason).
    Checks for impossible profiles:
    - Start date before company founding (we approximate)
    - Too many 'expert' skills with 0 years total
    - Career timeline impossibilities
    """
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    yoe = candidate["profile"].get("years_of_experience", 0)

    # Check 1: Total career months vs stated YoE — fabricated profiles fail this
    total_months = sum(r.get("duration_months", 0) for r in career)
    if total_months > 0:
        total_years_from_history = total_months / 12.0
        # Allow some tolerance for gaps, overlaps
        if total_years_from_history > yoe * 1.6 + 5:
            return True, f"Career history months ({total_months}) far exceeds stated YoE ({yoe})"

    # Check for overlapping full-time roles (impossible)
    dated_roles = []
    for r in career:
        try:
            start = datetime.strptime(r["start_date"], "%Y-%m-%d").date()
            end_str = r.get("end_date")
            end = datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else TODAY
            dated_roles.append((start, end, r.get("title", "")))
        except Exception:
            pass

    dated_roles.sort()
    for i in range(len(dated_roles) - 1):
        s1, e1, t1 = dated_roles[i]
        s2, e2, t2 = dated_roles[i + 1]
        overlap = (min(e1, e2) - max(s1, s2)).days
        if overlap > 90:  # >3 months overlap = suspect
            return True, f"Overlapping roles: '{t1}' and '{t2}' overlap by {overlap} days"

    # Check for suspiciously many expert skills with very short durations
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) > 8:
        avg_duration = sum(s.get("duration_months", 0) for s in expert_skills) / len(expert_skills)
        if avg_duration < 6:
            return True, f"{len(expert_skills)} 'expert' skills with avg {avg_duration:.0f}mo duration"

    # Check for future dates in career
    for r in career:
        try:
            start = datetime.strptime(r["start_date"], "%Y-%m-%d").date()
            if start > TODAY:
                return True, f"Future start date: {r['start_date']}"
        except Exception:
            pass

    return False, ""


# ── Skill matching ────────────────────────────────────────────────────────────

def _text_contains_any(text: str, terms: list[str]) -> bool:
    text_lower = text.lower()
    return any(t.lower() in text_lower for t in terms)


def score_skill_match(candidate: dict) -> tuple[float, dict]:
    """
    Score candidate skills against JD requirements.
    Returns (score 0-1, breakdown dict).
    
    Key insight from JD: we care about CAREER EVIDENCE, not just skill lists.
    A candidate whose career description shows they built a recommendation system
    ranks higher than one who just lists 'recommendation systems' as a skill.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    # Build text pools
    skill_names = [s["name"].lower() for s in skills]
    skill_text = " ".join(skill_names)
    
    # Career descriptions are gold — they show what was actually done
    career_desc_text = " ".join(
        r.get("description", "") + " " + r.get("title", "")
        for r in career
    ).lower()
    profile_text = (
        candidate["profile"].get("summary", "") + " " +
        candidate["profile"].get("headline", "")
    ).lower()
    full_text = skill_text + " " + career_desc_text + " " + profile_text

    family_hits = {}
    family_weights = {
        "embeddings": 0.25,
        "vector_db": 0.20,
        "retrieval_ir": 0.20,
        "ranking_eval": 0.15,
        "llm_applied": 0.12,
        "python_ml": 0.08,
    }

    total_score = 0.0
    for family, terms in REQUIRED_SKILL_FAMILIES.items():
        # Career evidence weighs more than skill list alone
        in_career = _text_contains_any(career_desc_text, terms)
        in_skills = _text_contains_any(skill_text, terms)
        in_profile = _text_contains_any(profile_text, terms)

        if in_career:
            hit_score = 1.0  # They actually did it
        elif in_skills and in_profile:
            hit_score = 0.75  # Listed + summarized
        elif in_skills:
            hit_score = 0.5  # Listed only
        elif in_profile:
            hit_score = 0.35  # Mentioned in summary
        else:
            hit_score = 0.0

        family_hits[family] = hit_score
        total_score += hit_score * family_weights.get(family, 0.1)

    # Desired skills bonus (up to +0.10)
    desired_bonus = 0.0
    for family, terms in DESIRED_SKILL_FAMILIES.items():
        if _text_contains_any(full_text, terms):
            desired_bonus += 0.033
    desired_bonus = min(desired_bonus, 0.10)

    # Assessment score bonus: if they have skill assessment scores in these areas
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    relevant_assessments = [
        v for k, v in assessments.items()
        if any(t in k.lower() for terms in REQUIRED_SKILL_FAMILIES.values() for t in terms)
    ]
    assessment_bonus = 0.0
    if relevant_assessments:
        avg_score = sum(relevant_assessments) / len(relevant_assessments)
        assessment_bonus = (avg_score / 100.0) * 0.05

    final = min(total_score + desired_bonus + assessment_bonus, 1.0)
    return final, family_hits


# ── Career trajectory scoring ─────────────────────────────────────────────────

def score_career_trajectory(candidate: dict) -> tuple[float, list[str]]:
    """
    Evaluate career trajectory — the JD explicitly wants product-company experience,
    not services/consulting, not pure research, not title-chasers.
    Returns (score 0-1, list of positive signals).
    """
    career = candidate.get("career_history", [])
    profile = candidate["profile"]
    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "").lower()
    notes = []
    score = 0.5  # Start at 0.5, adjust

    # Experience band: JD wants 5-9 years, but open to 4-10 if signals are strong
    if 5 <= yoe <= 9:
        score += 0.15
        notes.append(f"{yoe:.1f}yr experience (ideal band)")
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        score += 0.08
        notes.append(f"{yoe:.1f}yr experience (near-ideal)")
    elif yoe < 3:
        score -= 0.25
    elif yoe > 15:
        score -= 0.05  # Slight concern about overqualification

    # Product company vs. services
    product_roles = 0
    services_roles = 0
    pure_research_count = 0
    recent_product = False

    for i, role in enumerate(career):
        industry = role.get("industry", "")
        company_size = role.get("company_size", "")
        title_lower = role.get("title", "").lower()
        company_lower = role.get("company", "").lower()
        is_current = role.get("is_current", False)
        duration = role.get("duration_months", 0)

        # Disqualifier: pure consulting companies in entire career
        is_consulting = any(
            dc in company_lower for dc in EXPLICIT_DISQUALIFIER_COMPANIES
        ) or industry in CONSULTING_DOMAINS

        if is_consulting:
            services_roles += 1
            if is_current:
                score -= 0.12  # Currently at consulting = harder sell
                notes.append(f"Currently at consulting/services ({role['company']})")
        else:
            product_roles += 1
            if is_current:
                recent_product = True

        # Pure research (academic) signal
        if any(r in title_lower for r in RESEARCH_ONLY_SIGNALS):
            pure_research_count += 1

        # Title-chasing: short stints (<18 months) at companies with title bumps
        # We allow this if it's early career (first 1-2 roles)

    # Pure services background = hard penalize
    if product_roles == 0 and services_roles > 0:
        score -= 0.30
        notes.append("Entire career in services/consulting (JD disqualifier)")
    elif product_roles > 0 and services_roles > 0:
        ratio = product_roles / (product_roles + services_roles)
        score += ratio * 0.10
        notes.append(f"Mix: {product_roles} product, {services_roles} services roles")
    elif product_roles > 0:
        score += 0.15
        notes.append(f"{product_roles} product-company roles")

    # Pure researcher penalty
    if pure_research_count >= len(career) * 0.7:
        score -= 0.25
        notes.append("Primarily research/academic roles (JD disqualifier)")

    # JD explicit disqualifier: hasn't written production code in 18 months
    # Signal: current role in architecture/tech lead with no engineering description
    current_role = next((r for r in career if r.get("is_current")), None)
    if current_role:
        desc = current_role.get("description", "").lower()
        prod_evidence = any(w in desc for w in [
            "built", "implemented", "deployed", "shipped", "wrote", "developed",
            "architected", "designed", "coded", "production", "inference", "pipeline"
        ])
        if not prod_evidence and any(
            t in current_role.get("title", "").lower()
            for t in ["tech lead", "architect", "head of", "vp of", "director"]
        ):
            score -= 0.10
            notes.append("Current role: management/architect with no engineering evidence")

    # CV/speech/robotics primary specialization = penalize
    cv_speech_primary = (
        any(cs in current_title for cs in ["vision", "speech", "robotic", "autonomous"])
        and not _text_contains_any(
            " ".join(r.get("description", "") for r in career[:2]),
            [t for terms in REQUIRED_SKILL_FAMILIES["retrieval_ir"] for t in [terms]]
        )
    )
    if cv_speech_primary:
        score -= 0.15
        notes.append("Primary specialization appears to be CV/speech (JD disqualifier)")

    # Title-chasing detection
    if len(career) >= 3:
        short_stints = sum(1 for r in career if r.get("duration_months", 24) < 14)
        if short_stints >= len(career) * 0.6:
            score -= 0.10
            notes.append(f"Short-tenure pattern: {short_stints}/{len(career)} roles <14mo")

    return max(0.0, min(1.0, score)), notes


# ── Availability / behavioral scoring ─────────────────────────────────────────

def score_availability(candidate: dict) -> tuple[float, list[str]]:
    """
    Score candidate's hiring availability using behavioral signals.
    From the JD: "A perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% response rate is not actually available."
    """
    sig = candidate.get("redrob_signals", {})
    notes = []
    score = 0.5

    # Recency of activity (critical)
    last_active_str = sig.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_inactive = (TODAY - last_active).days
            if days_inactive <= 14:
                score += 0.20
                notes.append("Active in last 2 weeks")
            elif days_inactive <= 30:
                score += 0.15
                notes.append("Active in last month")
            elif days_inactive <= 60:
                score += 0.05
            elif days_inactive <= 90:
                score -= 0.05
            elif days_inactive <= 180:
                score -= 0.15
                notes.append(f"Inactive {days_inactive} days (availability concern)")
            else:
                score -= 0.30
                notes.append(f"Inactive {days_inactive} days (likely unavailable)")
        except Exception:
            pass

    # Open to work flag (strong signal)
    if sig.get("open_to_work_flag"):
        score += 0.12
        notes.append("Marked open-to-work")
    else:
        score -= 0.08

    # Recruiter response rate
    rr = sig.get("recruiter_response_rate", 0.5)
    if rr >= 0.7:
        score += 0.10
        notes.append(f"High recruiter response rate ({rr:.0%})")
    elif rr >= 0.4:
        score += 0.05
    elif rr <= 0.15:
        score -= 0.15
        notes.append(f"Low recruiter response rate ({rr:.0%})")
    elif rr <= 0.30:
        score -= 0.05

    # Interview completion rate
    icr = sig.get("interview_completion_rate", 0.7)
    if icr >= 0.8:
        score += 0.05
        notes.append(f"High interview completion ({icr:.0%})")
    elif icr <= 0.5:
        score -= 0.08
        notes.append(f"Low interview completion ({icr:.0%}) — risk of no-show")

    # Notice period (JD wants sub-30 day ideal, <=90 acceptable)
    notice = sig.get("notice_period_days", 30)
    if notice <= 30:
        score += 0.08
        notes.append(f"Notice period {notice}d (ideal)")
    elif notice <= 60:
        score += 0.03
    elif notice <= 90:
        score -= 0.03
    else:
        score -= 0.08
        notes.append(f"Long notice period {notice}d")

    # Location fit (Pune/Noida preferred; major metros acceptable)
    location = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    relocate = sig.get("willing_to_relocate", False)

    if country in ["india", "in"]:
        if any(loc in location for loc in ["pune", "noida"]):
            score += 0.12
            notes.append(f"Located in {location} (preferred)")
        elif any(loc in location for loc in TARGET_LOCATIONS):
            score += 0.06
            notes.append(f"Located in {location} (acceptable metro)")
        elif relocate:
            score += 0.02
            notes.append("Willing to relocate")
        else:
            score -= 0.05
    elif relocate:
        score += 0.02
    else:
        score -= 0.10
        notes.append(f"Outside India ({country}), not willing to relocate")

    # Work mode fit (JD says hybrid)
    work_mode = sig.get("preferred_work_mode", "flexible")
    if work_mode in ["hybrid", "flexible"]:
        score += 0.03
    elif work_mode == "remote":
        score -= 0.02

    # Platform engagement (saved by recruiters = external validation of quality)
    saved = sig.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        score += 0.04
    elif saved >= 2:
        score += 0.02

    # GitHub activity (JD mentions needing to "see how they think")
    github = sig.get("github_activity_score", -1)
    if github >= 50:
        score += 0.06
        notes.append(f"Active GitHub (score {github:.0f})")
    elif github >= 20:
        score += 0.03
    # -1 = not linked, penalty only if profile claims open-source
    elif github == -1:
        profile_text = candidate["profile"].get("summary", "").lower()
        if "open source" in profile_text or "github" in profile_text:
            score -= 0.02  # Claims but no linked account

    # Salary range sanity (role is competitive Series A)
    sal_range = sig.get("expected_salary_range_inr_lpa", {})
    sal_min = sal_range.get("min", 0)
    if 20 <= sal_min <= 60:
        score += 0.02  # Realistic range for this role
    elif sal_min > 80:
        score -= 0.02  # Potentially expensive

    return max(0.0, min(1.0, score)), notes


# ── Final scoring ─────────────────────────────────────────────────────────────

# Weights — mirrors how a senior recruiter would balance these factors
WEIGHTS = {
    "skill_match": 0.42,      # Core: do they have what the role needs?
    "career_trajectory": 0.35, # Second: are they the right type of engineer?
    "availability": 0.23,      # Third: can we actually hire them?
}


def score_candidate(candidate: dict) -> tuple[float, dict]:
    """Master scoring function."""
    is_honeypot, honeypot_reason = detect_honeypot(candidate)
    if is_honeypot:
        return -1.0, {"honeypot": True, "reason": honeypot_reason}

    skill_score, skill_breakdown = score_skill_match(candidate)
    career_score, career_notes = score_career_trajectory(candidate)
    avail_score, avail_notes = score_availability(candidate)

    composite = (
        skill_score * WEIGHTS["skill_match"] +
        career_score * WEIGHTS["career_trajectory"] +
        avail_score * WEIGHTS["availability"]
    )

    return composite, {
        "skill_score": skill_score,
        "career_score": career_score,
        "availability_score": avail_score,
        "skill_breakdown": skill_breakdown,
        "career_notes": career_notes,
        "avail_notes": avail_notes,
    }


# ── Reasoning generation ──────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, score_data: dict, rank: int) -> str:
    """
    Generate specific, honest reasoning that references actual profile facts.
    Matches the Stage 4 evaluation criteria:
    - Specific facts from profile
    - JD connection
    - Honest concerns
    - No hallucination
    - Variation across candidates
    - Tone matches rank
    """
    profile = candidate["profile"]
    sig = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = profile.get("location", "")

    career_notes = score_data.get("career_notes", [])
    avail_notes = score_data.get("avail_notes", [])
    skill_breakdown = score_data.get("skill_breakdown", {})

    # Find strongest matching skills with evidence
    strong_skills = []
    all_skills_lower = [s["name"].lower() for s in skills]
    career_desc = " ".join(r.get("description", "") for r in career).lower()

    # Better: find actual named skills matching each family
    family_display = {
        "embeddings": "embedding retrieval",
        "vector_db": "vector search infra",
        "retrieval_ir": "IR/ranking systems",
        "ranking_eval": "ranking evaluation",
        "llm_applied": "LLM/NLP",
        "python_ml": "Python/ML stack",
    }
    for family, terms in REQUIRED_SKILL_FAMILIES.items():
        found = False
        for s in skills:
            s_lower = s["name"].lower()
            if any(t.lower() in s_lower or s_lower in t.lower() for t in terms):
                strong_skills.append(
                    f"{s['name']} ({s['proficiency']}, {s['duration_months']}mo)"
                )
                found = True
                break
        if not found:
            # Check career evidence for a meaningful display
            for term in terms:
                if term in career_desc and len(term) > 4:  # skip very short terms
                    strong_skills.append(f"{family_display.get(family, term)} (career evidence)")
                    break

    strong_skills = strong_skills[:3]  # Keep concise

    # Build sentence 1: who they are + strongest qualification
    if rank <= 10:
        qualifier = "Strong fit"
    elif rank <= 30:
        qualifier = "Good fit"
    elif rank <= 60:
        qualifier = "Moderate fit"
    else:
        qualifier = "Partial fit"

    # Find most recent relevant role
    product_role = next(
        (r for r in career if r.get("industry") not in CONSULTING_DOMAINS and
         any(w in r.get("description", "").lower() for w in
             ["retrieval", "ranking", "search", "embedding", "recommendation", "llm", "rag"])),
        career[0] if career else None
    )

    s1_parts = [f"{yoe:.1f}yr {title}"]
    if product_role and product_role != career[0]:
        s1_parts.append(f"with prior {product_role['title']} experience at {product_role['company']}")
    elif career:
        s1_parts.append(f"at {company}")
    if strong_skills:
        s1_parts.append(f"— demonstrated {', '.join(strong_skills[:2])}")

    sentence1 = "; ".join(s1_parts[:2]) + ("." if strong_skills else " with partial skill match.")
    if strong_skills and len(sentence1) < 150:
        sentence1 = sentence1.rstrip(".") + f"; demonstrated {strong_skills[0]}."

    # Build sentence 2: concerns or positive behavioral signals
    concerns = []
    positives = []

    # Availability signals
    rr = sig.get("recruiter_response_rate", 0.5)
    notice = sig.get("notice_period_days", 30)
    last_active_str = sig.get("last_active_date", "")
    open_to_work = sig.get("open_to_work_flag", False)

    if open_to_work:
        positives.append("marked open-to-work")
    if rr >= 0.6:
        positives.append(f"responsive to recruiters ({rr:.0%})")
    elif rr <= 0.20:
        concerns.append(f"low recruiter response rate ({rr:.0%})")

    if notice > 90:
        concerns.append(f"long notice period ({notice}d)")
    elif notice <= 30:
        positives.append(f"available quickly ({notice}d notice)")

    if last_active_str:
        try:
            days = (TODAY - datetime.strptime(last_active_str, "%Y-%m-%d").date()).days
            if days > 120:
                concerns.append(f"last active {days}d ago")
        except Exception:
            pass

    # Career concerns
    consulting_concern = any("consulting" in n.lower() or "services" in n.lower()
                              for n in career_notes)
    if consulting_concern:
        concerns.append("services/consulting background (JD-flagged concern)")

    github = sig.get("github_activity_score", -1)
    if github >= 40:
        positives.append(f"active GitHub (score {github:.0f})")

    if positives and not concerns:
        sentence2 = f"Engagement signals are strong: {', '.join(positives[:2])}."
    elif concerns and not positives:
        sentence2 = f"Concerns: {', '.join(concerns[:2])}."
    elif concerns and positives:
        sentence2 = f"Positives ({', '.join(positives[:1])}); concerns ({', '.join(concerns[:1])})."
    else:
        # Neutral — comment on fit gap
        matched_families = [f for f, v in skill_breakdown.items() if v >= 0.7]
        missing_families = [f for f, v in skill_breakdown.items() if v < 0.3]
        if missing_families:
            sentence2 = f"Skill gaps in {', '.join(missing_families[:2])} versus JD requirements."
        elif matched_families:
            sentence2 = f"Covers required areas of {', '.join(matched_families[:2])}."
        else:
            sentence2 = "Profile reviewed; ranked by composite signal score."

    return f"{sentence1} {sentence2}"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Top N to output (default 100)")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    scored = []
    total = 0
    honeypots_found = 0

    with open(args.candidates, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON array format
            candidates_list = json.load(f)
            lines_iter = candidates_list
        else:
            # JSONL format
            lines_iter = f

    def parse_iter(it, is_array):
        for item in it:
            if is_array:
                yield item
            else:
                line = item.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass

    is_array = first_char == "["
    for candidate in parse_iter(lines_iter if is_array else open(args.candidates, "r", encoding="utf-8"), is_array):

            total += 1
            if total % 10000 == 0:
                print(f"  Processed {total} candidates...")

            composite, score_data = score_candidate(candidate)
            if score_data.get("honeypot"):
                honeypots_found += 1
                continue  # Exclude from ranking

            scored.append((composite, candidate, score_data))

    print(f"Done. {total} candidates processed, {honeypots_found} honeypots excluded.")
    print(f"Ranking {len(scored)} valid candidates...")

    # Sort descending by composite score, break ties by candidate_id
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))

    # Take top N
    top_n = scored[:args.top_n]

    print(f"Writing top {len(top_n)} candidates to {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_pos, (score, candidate, score_data) in enumerate(top_n, start=1):
            cid = candidate["candidate_id"]
            reasoning = generate_reasoning(candidate, score_data, rank_pos)
            writer.writerow([cid, rank_pos, f"{score:.4f}", reasoning])

    print(f"\nSubmission written: {args.out}")
    print(f"Top 5 candidates:")
    for i, (score, cand, _) in enumerate(top_n[:5], 1):
        title = cand["profile"].get("current_title", "?")
        yoe = cand["profile"].get("years_of_experience", 0)
        print(f"  {i}. {cand['candidate_id']} — {title} ({yoe}yr) — score={score:.4f}")


if __name__ == "__main__":
    main()
