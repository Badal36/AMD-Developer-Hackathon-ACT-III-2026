"""
inference_wrapper/difficulty_classifier.py
-------------------------------------------
Classifies an incoming prompt into (domain, difficulty_level) using a
fast, zero-latency heuristic stack. Replaces the hard-coded simplicity
gate categories with a richer 2D signal.

Output:  DifficultyResult(domain, level, confidence, signals)

Domains  : math | factual | code | language | reasoning | instruction
Levels   : L1 (trivial) | L2 (moderate) | L3 (hard) | L4 (frontier)

Speed: pure Python regex + dict look-ups → ~0–2 ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any

# ── Compiled patterns ─────────────────────────────────────────────────────────

# Domain detectors
_CODE_STRONG = re.compile(
    r"```|def |class |import |#include|function\s*\(|->|::\w|"
    r"\blambda\b|\bfor\s+\w+\s+in\b|\basync\s+def\b",
    re.I,
)
_CODE_WEAK = re.compile(
    r"\bcode\b|\bprogram\b|\bscript\b|\bfunction\b|\bimplementation\b|"
    r"\bdebug\b|\bcompile\b|\bapi\b|\bsql\b|\bbug\b",
    re.I,
)
_MATH_STRONG = re.compile(
    r"[=∑∫√π]|\\frac|\\sum|\$\$?|\bintegral\b|\bderivative\b|"
    r"\bproof\b|\btheorem\b|\bequation\b|\bsolve\b|\bmodulo\b|"
    r"\d+\s*[\+\-\*\/\^]\s*\d+",
    re.I,
)
_MATH_WEAK = re.compile(
    r"\bhow many\b|\bhow much\b|\btotal\b|\baverage\b|\bpercent\b|"
    r"\bprobability\b|\bcalculate\b|\bcompute\b|\bcount\b|\bnumber of\b",
    re.I,
)
_REASONING = re.compile(
    r"\bwhy\b|\bexplain\b|\banalyze\b|\banalyse\b|\bcompare\b|\bevaluate\b|"
    r"\bimplications?\b|\bconsequences?\b|\btradeoff\b|trade.off|"
    r"\bethic\b|\bphilosoph\b|\bargue\b|\bjustif\b|\bif\b.{0,40}\bthen\b|"
    r"\bgiven that\b|\bassuming\b|\bparadox\b",
    re.I,
)
_INSTRUCTION = re.compile(
    r"\bwrite\b|\bcreate\b|\bgenerate\b|\bcompose\b|\bdraft\b|\bsummariz\b|"
    r"\btranslat\b|\breformat\b|\bconvert\b|\blist\b.{0,20}\bstep\b|"
    r"\bstep.by.step\b|\bfirst.*then.*finally\b",
    re.I,
)
_FACTUAL = re.compile(
    r"\bwho\b|\bwhat is\b|\bwhen\b|\bwhere\b|\bwhich\b|\bdefine\b|"
    r"\bname the\b|\bwhat was\b|\bwhat are\b|\bcapital of\b|\byear\b",
    re.I,
)

# Difficulty up/down signals
_HARD_SIGNALS = re.compile(
    r"\bproof\b|\bderive\b|\boptimiz\b|\bcomplexity\b|\bO\(n\b|"
    r"\btime complexity\b|\bspace complexity\b|\bNP.hard\b|\bdynamic programming\b|"
    r"\brecurren[ct]\b|\bmulti.hop\b|\bcounterexample\b|\brefut\b|"
    r"\bformal\b.{0,20}\bproof\b|\bbig.picture\b|\bcomprehensive\b",
    re.I,
)
_FRONTIER_SIGNALS = re.compile(
    r"\bcompetition\b|\bOlympiad\b|\badvanced\b.{0,20}\bresearch\b|"
    r"\bstate.of.the.art\b|\bnovel\b.{0,20}\balgorithm\b|\bfrontier\b|"
    r"\bmultimodal\b|\bRL\b|\breinforcement learning\b|\blatent\b.{0,20}\bspace\b|"
    r"\bPhD.level\b|\bgraduate.level\b|\bcutting.edge\b",
    re.I,
)
_MCQ = re.compile(r"\b[A-D][.)]\s", re.I)
_CONDITIONAL_COUNT = re.compile(
    r"\bif\b|\bunless\b|\bgiven that\b|\bassuming\b|\bprovided that\b", re.I
)
_STEP_KEYWORDS = re.compile(
    r"\bstep.by.step\b|\bfirst[,.]?\s|\bsecondly[,.]?\s|\bfinally[,.]?\s|"
    r"\bthen\b.{0,50}\bthen\b",
    re.I,
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DifficultyResult:
    domain:     str    # one of DOMAINS
    level:      str    # L1 | L2 | L3 | L4
    confidence: float  # 0..1 classifier confidence
    signals:    dict   # raw debug signals

    def __str__(self):
        return f"{self.domain}/{self.level} (conf={self.confidence:.2f})"


# ── Main classifier ───────────────────────────────────────────────────────────

def classify(prompt: str, feats: Dict[str, Any] = None) -> DifficultyResult:
    """
    Classify a prompt into (domain, level).

    Args:
        prompt: raw user prompt
        feats:  already-extracted features dict (from feature_extractor.extract_features).
                If None, uses heuristics only.
    Returns:
        DifficultyResult
    """
    feats   = feats or {}
    p       = prompt.strip()
    lower   = p.lower()
    words   = p.split()
    n       = len(words)

    # ── Domain detection (scored voting) ─────────────────────────────────────
    domain_scores: Dict[str, float] = {
        "math":        0.0,
        "factual":     0.0,
        "code":        0.0,
        "language":    0.0,
        "reasoning":   0.0,
        "instruction": 0.0,
    }

    # Use feature extractor task type if available (fast path)
    task_type = feats.get("_task_type", "")
    if task_type == "code" or feats.get("has_code_block"):
        domain_scores["code"] += 1.8
    if task_type == "math" or feats.get("has_math_symbols"):
        domain_scores["math"] += 1.8
    if task_type == "mcq":
        domain_scores["factual"] += 1.2

    # Pattern-based votes
    if _CODE_STRONG.search(p):
        domain_scores["code"]    += 1.5
    if _CODE_WEAK.search(p):
        domain_scores["code"]    += 0.5
    if _MATH_STRONG.search(p):
        domain_scores["math"]    += 1.5
    if _MATH_WEAK.search(p):
        domain_scores["math"]    += 0.5
    if _REASONING.search(p):
        domain_scores["reasoning"]   += 1.2
    if _INSTRUCTION.search(p):
        domain_scores["instruction"] += 1.0
    if _FACTUAL.search(p):
        domain_scores["factual"]     += 1.0
    if n <= 10 and not domain_scores.get("code") and not domain_scores.get("math"):
        domain_scores["language"]    += 0.6

    # ── Cross-signal boosters ─────────────────────────────────────────────
    # "Write/Implement/Create" + code language name → code, not instruction
    _PL_KEYWORDS = re.compile(
        r"\b(python|java|javascript|typescript|c\+\+|c#|rust|go|kotlin|"
        r"swift|ruby|sql|bash|shell|html|css|react|node|django|flask|"
        r"function|class|method|module|script|algorithm|data structure|"
        r"linked list|binary tree|hash map|queue|stack|graph|sort|search)\b",
        re.I,
    )
    _CODE_VERBS = re.compile(
        r"\b(write|implement|build|create|code|program|develop|fix|debug|"
        r"refactor|optimize|design)\b",
        re.I,
    )
    if _CODE_VERBS.search(p) and _PL_KEYWORDS.search(p):
        domain_scores["code"]        += 1.8  # strong boost: clearly a code task
        domain_scores["instruction"] -= 0.5  # suppress instruction

    # "Prove / irrational / theorem / formal proof" → math even in short prompts
    _MATH_PROOF = re.compile(
        r"\b(prove|proof|irrational|theorem|lemma|corollary|formal|derive|"
        r"induction|contradiction|modular arithmetic|number theory)\b",
        re.I,
    )
    if _MATH_PROOF.search(p):
        domain_scores["math"]    += 1.5
        domain_scores["language"] -= 0.5

    # "lock-free / concurrent / hash map / in C++" → code
    _SYSTEM_CODE = re.compile(
        r"\b(lock.free|concurrent|thread.safe|race condition|mutex|semaphore|"
        r"memory.safe|garbage.collect|cache.evict|O\(n|time complexity|"
        r"space complexity|hash.?map|binary.?search|trie|heap|segment.tree)\b",
        re.I,
    )
    if _SYSTEM_CODE.search(p):
        domain_scores["code"] += 1.5

    # Break ties: factual is the safe default
    domain_scores["factual"] += 0.1

    domain = max(domain_scores, key=domain_scores.__getitem__)
    top    = domain_scores[domain]
    second = sorted(domain_scores.values(), reverse=True)[1]
    domain_confidence = round((top - second) / max(top, 1) + 0.5, 2)
    domain_confidence = min(domain_confidence, 0.98)

    # ── Difficulty level ──────────────────────────────────────────────────────
    difficulty_score = 0.0

    # Length signal (softer: length is a hint, not the deciding factor)
    if   n <= 6:   difficulty_score -= 0.8   # very short -> leans L1
    elif n <= 15:  difficulty_score -= 0.2   # short -> slightly L1
    elif n <= 40:  difficulty_score += 0.3
    elif n <= 80:  difficulty_score += 0.7
    elif n <= 150: difficulty_score += 1.2
    else:          difficulty_score += 1.8   # very long -> L3+

    # Structural complexity
    conds = len(_CONDITIONAL_COUNT.findall(p))
    difficulty_score += conds * 0.4

    steps = len(_STEP_KEYWORDS.findall(p)[:3])   # cap at 3 step markers
    difficulty_score += steps * 0.5

    if feats.get("num_sentences", 1) > 4:
        difficulty_score += 0.4

    # Hard / frontier markers — these SET A FLOOR, not just add.
    # A short "Prove X" must reach L3+ regardless of length penalty.
    if _HARD_SIGNALS.search(p):
        difficulty_score = max(difficulty_score, 1.5)   # floor at L3 boundary
    if _FRONTIER_SIGNALS.search(p):
        difficulty_score = max(difficulty_score, 2.8)   # floor at L4

    # Domain-specific difficulty floors
    if domain == "code":
        # Any code task starts at least L2
        difficulty_score = max(difficulty_score, 0.5)
        if feats.get("has_code_block"):
            difficulty_score = max(difficulty_score, 1.0)   # code in prompt -> L2+
        if re.search(r"\boptimiz|\bO\(|\bcomplexity\b|\block.free|\bconcurrent\b", lower):
            difficulty_score = max(difficulty_score, 2.8)   # L4 systems tasks
    if domain == "math":
        # Math starts at L2
        difficulty_score = max(difficulty_score, 0.5)
        if re.search(r"\bproof\b|\btheorem\b|\bderive\b|\birrational\b|\bintegral\b", lower):
            difficulty_score = max(difficulty_score, 2.8)   # L4 proof tasks
    if domain == "reasoning":
        difficulty_score = max(difficulty_score, 1.0)   # reasoning starts at L2/L3
        # Boost reasoning to L3 when analysis/ethical/step-by-step patterns are present
        _REASONING_HARD = re.compile(
            r"\banalyze\b|\banalyse\b|\bethic\b|\bimplication|\bconsequence|"
            r"\bstep.by.step\b|\bexplain.{0,15}why\b|\bgeopolit|"
            r"\bcompare and contrast\b|\btrade.off\b|\bpros and cons\b",
            re.I,
        )
        if _REASONING_HARD.search(p):
            difficulty_score = max(difficulty_score, 1.8)  # floor at L3

    # MCQ: floor at L2, cap at L3 (MCQs rarely need frontier model)
    if _MCQ.search(p) or task_type == "mcq":
        difficulty_score = max(difficulty_score, 0.5)
        difficulty_score = min(difficulty_score, 1.8)  # MCQs don't hit L4

    # Map score → level
    if   difficulty_score < 0:   level = "L1"
    elif difficulty_score < 1.2: level = "L2"
    elif difficulty_score < 2.4: level = "L3"
    else:                        level = "L4"

    # level confidence: how far from the next boundary
    boundaries = {"L1": 0.0, "L2": 1.2, "L3": 2.4, "L4": 3.6}
    boundary   = boundaries[level]
    margin     = abs(difficulty_score - boundary)
    level_conf = min(0.5 + margin * 0.15, 0.95)

    # Overall confidence = geometric mean of domain and level confidence
    confidence = round((domain_confidence * level_conf) ** 0.5, 3)

    signals = {
        "n_words":         n,
        "difficulty_score": round(difficulty_score, 2),
        "domain_scores":   {k: round(v, 2) for k, v in domain_scores.items()},
        "has_hard_signal": bool(_HARD_SIGNALS.search(p)),
        "has_frontier":    bool(_FRONTIER_SIGNALS.search(p)),
        "cond_count":      conds,
        "step_count":      steps,
        "task_type_feat":  task_type,
    }

    return DifficultyResult(
        domain=domain, level=level, confidence=confidence, signals=signals
    )


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_CASES = [
        # (prompt, expected_domain, expected_level)
        ("Hello",                                                       "language",    "L1"),
        ("What is 2 + 2?",                                             "math",        "L1"),
        ("What is the capital of France?",                             "factual",     "L1"),
        ("Define machine learning.",                                    "factual",     "L1"),
        ("How many planets in the solar system?",                      "math",        "L2"),
        ("If 3x - 7 = 11, find x.",                                    "math",        "L2"),
        ("Write a Python function that reverses a list.",               "code",        "L2"),
        ("Explain step by step how neural networks learn.",            "reasoning",   "L3"),
        ("Analyze the ethical implications of autonomous AI.",         "reasoning",   "L3"),
        ("Prove that sqrt(2) is irrational using a formal proof.",     "math",        "L4"),
        ("Implement a lock-free concurrent hash map in C++.",          "code",        "L4"),
        ("What are the geopolitical trade-offs of semiconductor "
         "export controls given current US-China tensions?",           "reasoning",   "L4"),
        ("Translate 'hello' to French.",                               "instruction", "L1"),
        ("Summarize this 5000-word document in bullet points, "
         "then compare its conclusions to Rawls' theory of justice.",  "instruction", "L4"),
    ]

    print(f"{'Prompt':<55} {'Domain':>12}  {'Level':>5}  {'Conf':>6}  {'Expected':>18}")
    print("-" * 110)
    passed = 0
    for prompt, exp_dom, exp_lv in TEST_CASES:
        r = classify(prompt)
        ok = (r.domain == exp_dom) and (r.level == exp_lv)
        tag = "✓" if ok else "✗"
        if ok:
            passed += 1
        print(f"{tag} {prompt[:54]:<54} {r.domain:>12}  {r.level:>5}  {r.confidence:>6.2f}  "
              f"(exp {exp_dom}/{exp_lv})")
    print(f"\n{passed}/{len(TEST_CASES)} correct")
