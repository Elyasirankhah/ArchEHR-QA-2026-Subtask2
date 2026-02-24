"""
Subtask 2: Evidence Identification — #1 recipe + optional TTS
- Few-shot: 2–3 dev examples in prompt (when dev key + XML available).
- Ensemble: o3 + gpt-5.2 + gpt-5.1 (or configurable), merge by ≥MIN_VOTES.
- Optional TTS (test-time scaling): one model runs N times at high temp, vote-aggregate;
  other models run once. Hybrid TTS+ensemble (different from single-model TTS).
- Post-hoc: keep only valid sentence IDs, sorted.
"""

import os
import re
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
from dotenv import load_dotenv

# Load .env from this repo root (standalone repo)
_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env", override=True)

# Azure OpenAI SDK (Key Authentication) — Chat Completions
AZURE_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/") + "/" if os.getenv("AZURE_OPENAI_ENDPOINT") else ""
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_GPT52_API_KEY") or os.getenv("AZURE_O3_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
# Single-model fallback
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT") or os.getenv("AZURE_DEPLOYMENT_BEST_1", "gpt-5.2")
# Ensemble: Exp 5 trio only (o3, gpt-5.2, gpt-5.1) — no DeepSeek-R1
_ensemble = (os.getenv("AZURE_ENSEMBLE_DEPLOYMENTS") or "o3,gpt-5.2,gpt-5.1").strip()
ENSEMBLE_DEPLOYMENTS = [d.strip() for d in _ensemble.split(",") if d.strip()]

# Merge: 1 = union (64/63.97 setup). 2 = majority.
MIN_VOTES = int(os.getenv("AZURE_ENSEMBLE_MIN_VOTES", "1"))
# Few-shot: 10 = fixed cases 1–10 (64/63.97 setup). No bias: always 1,2,…,10.
FEW_SHOT_N = int(os.getenv("TASK2_FEW_SHOT_N", "10"))
# Fixed 1–10 only (no random). Set to 1 to sample from 1–20; we keep 0 for 64 setup.
FEW_SHOT_RANDOM_SAMPLE = os.getenv("TASK2_FEW_SHOT_RANDOM_SAMPLE", "0").strip().lower() in ("1", "true", "yes")
FEW_SHOT_RANDOM_SEED = int(os.getenv("TASK2_FEW_SHOT_RANDOM_SEED", "42"))
# o3 hits finish_reason=length with long prompt; use fewer few-shot for o3 so it returns content (restore 3-model union → 64).
O3_FEW_SHOT_N = int(os.getenv("TASK2_O3_FEW_SHOT_N", "3"))
# Use silver evidence from test key (21–120) when clinician_answer_sentences have "citations". Current v1.4 test key has no citations.
TASK2_USE_SILVER_FEW_SHOT = os.getenv("TASK2_USE_SILVER_FEW_SHOT", "0").strip().lower() in ("1", "true", "yes")
TASK2_FEW_SHOT_DEV_N = int(os.getenv("TASK2_FEW_SHOT_DEV_N", "10"))  # from dev (gold)
TASK2_FEW_SHOT_SILVER_N = int(os.getenv("TASK2_FEW_SHOT_SILVER_N", "0"))  # from test (silver), 0 = off
# Strict-only few-shot: show only essential (not supplementary) IDs in examples to bias toward strict metric.
STRICT_ONLY_FEW_SHOT = os.getenv("TASK2_STRICT_ONLY_FEW_SHOT", "0").strip().lower() in ("1", "true", "yes")
# Tier 1: Evidence-level post-filter after union (drop single-vote short sentences with no clinical entity).
TASK2_POST_FILTER = os.getenv("TASK2_POST_FILTER", "0").strip().lower() in ("1", "true", "yes")
TASK2_POST_FILTER_SINGLE_VOTE_MAX_TOKENS = int(os.getenv("TASK2_POST_FILTER_SINGLE_VOTE_MAX_TOKENS", "10"))
TASK2_POST_FILTER_REQUIRE_ENTITY = os.getenv("TASK2_POST_FILTER_REQUIRE_ENTITY", "1").strip().lower() in ("1", "true", "yes")
# Tier 1: Two-stage citation tightening (second LLM pass: remove citations not strictly required; only remove, never add).
TASK2_TIGHTEN_CITATIONS = os.getenv("TASK2_TIGHTEN_CITATIONS", "0").strip().lower() in ("1", "true", "yes")
TASK2_TIGHTEN_DEPLOYMENT = os.getenv("TASK2_TIGHTEN_DEPLOYMENT", "").strip() or (ENSEMBLE_DEPLOYMENTS[0] if ENSEMBLE_DEPLOYMENTS else "gpt-5.2")
# Test-time scaling (TTS): multiple high-temp samples from one model, vote-aggregate. Hybrid: TTS on one model + rest of ensemble deterministic.
TASK2_TTS = os.getenv("TASK2_TTS", "0").strip().lower() in ("1", "true", "yes")
TASK2_TTS_N = int(os.getenv("TASK2_TTS_N", "5"))
TASK2_TTS_TEMPERATURE = float(os.getenv("TASK2_TTS_TEMPERATURE", "0.7"))
TASK2_TTS_MIN_VOTES = int(os.getenv("TASK2_TTS_MIN_VOTES", "2"))
TASK2_TTS_DEPLOYMENT = (os.getenv("TASK2_TTS_DEPLOYMENT", "").strip() or (ENSEMBLE_DEPLOYMENTS[1] if len(ENSEMBLE_DEPLOYMENTS) > 1 else "gpt-5.2"))
# Enhanced prompt: richer clinical evidence guidance
TASK2_ENHANCED_PROMPT = os.getenv("TASK2_ENHANCED_PROMPT", "0").strip().lower() in ("1", "true", "yes")
# Chain-of-thought: ask model to reason before selecting
TASK2_COT = os.getenv("TASK2_COT", "0").strip().lower() in ("1", "true", "yes")
# Multi-run stabilization: run full ensemble N times, vote across runs at sentence level
TASK2_MULTI_RUN = int(os.getenv("TASK2_MULTI_RUN", "1"))
# Experiment tag: set output folder to submission_Subtask2_{tag}
TASK2_EXP_TAG = os.getenv("TASK2_EXP_TAG", "").strip()
# 80/20 dev split: hold-out case IDs (e.g. "4,9,14,19") excluded from few-shot pool
_HOLDOUT_RAW = (os.getenv("TASK2_HOLDOUT_IDS") or "").strip()
TASK2_HOLDOUT_IDS = set(x.strip() for x in _HOLDOUT_RAW.split(",") if x.strip())

try:
    from openai import AzureOpenAI
    azure_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    ) if AZURE_API_KEY else None
    # o3 often needs 2025-01-01-preview and higher max_completion_tokens to avoid empty (finish_reason=length)
    o3_api_version = os.getenv("AZURE_OPENAI_O3_API_VERSION", "2025-01-01-preview")
    azure_client_o3 = AzureOpenAI(
        api_version=o3_api_version,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    ) if AZURE_API_KEY and AZURE_ENDPOINT else None
except Exception as e:
    azure_client = None
    azure_client_o3 = None
    print(f"Azure OpenAI init: {e}")


def load_key_gold_evidence(key_path: Path, strict_only: bool = False) -> Dict[str, List[str]]:
    """Load gold evidence sentence IDs per case. strict_only=True => essential only (for strict metric)."""
    if not key_path.exists():
        return {}
    with open(key_path, "r", encoding="utf-8") as f:
        key_json = json.load(f)
    out = {}
    for case in key_json:
        case_id = case["case_id"]
        ids = []
        for a in case.get("answers", []):
            rel = a.get("relevance", "")
            if strict_only:
                if rel == "essential":
                    ids.append(a["sentence_id"])
            else:
                if rel in ("essential", "supplementary"):
                    ids.append(a["sentence_id"])
        out[case_id] = sorted(ids, key=lambda x: int(x) if x.isdigit() else 0)
    return out


def load_silver_evidence_from_key(key_path: Path) -> Dict[str, List[str]]:
    """Extract silver evidence from key when clinician_answer_sentences have 'citations'. Returns case_id -> list of sentence IDs. If no citations, returns {}."""
    if not key_path.exists():
        return {}
    with open(key_path, "r", encoding="utf-8") as f:
        key_json = json.load(f)
    out = {}
    for case in key_json:
        case_id = case["case_id"]
        ids = []
        for sent in case.get("clinician_answer_sentences", []):
            for sid in sent.get("citations", []):
                ids.append(str(sid).strip())
        if ids:
            out[case_id] = sorted(set(ids), key=lambda x: int(x) if x.isdigit() else 0)
    return out


def load_few_shot_examples(
    key_path: Path,
    xml_path: Path,
    case_ids: List[str],
    cases_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    strict_only: bool = False,
    silver_map: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Build few-shot examples: list of {case, gold_evidence_ids} for given case IDs. Uses gold_map from key (answers) or silver_map when provided for a case."""
    gold_map = load_key_gold_evidence(key_path, strict_only=strict_only) if key_path.exists() else {}
    if silver_map:
        # Merge: prefer gold when present, else silver
        for cid, ids in silver_map.items():
            if cid not in gold_map:
                gold_map[cid] = ids
    if not gold_map:
        return []
    if cases_by_id is None:
        if not xml_path.exists():
            return []
        cases_list = parse_qa_xml(xml_path)
        cases_by_id = {c["case_id"]: c for c in cases_list}
    examples = []
    for cid in case_ids:
        if cid not in gold_map or cid not in cases_by_id:
            continue
        examples.append({
            "case": cases_by_id[cid],
            "gold_ids": gold_map[cid],
        })
    return examples


def parse_qa_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse archehr-qa.xml into list of cases with patient_question, clinician_question, sentences."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cases = []
    for case_el in root.findall(".//case"):
        case_id = case_el.get("id", "")
        patient_question_el = case_el.find("patient_question")
        patient_question = ""
        if patient_question_el is not None:
            phrase = patient_question_el.find("phrase")
            if phrase is not None and (phrase.text or "").strip():
                patient_question = (phrase.text or "").strip()
            else:
                patient_question = (patient_question_el.text or "").strip()
        def el_text(el: Optional[ET.Element]) -> str:
            return "".join(el.itertext()).strip() if el is not None else ""

        clinician_el = case_el.find("clinician_question")
        clinician_question = el_text(clinician_el)
        note_el = case_el.find("note_excerpt")
        note_excerpt = el_text(note_el)
        sentences_el = case_el.find("note_excerpt_sentences")
        sentences = []
        if sentences_el is not None:
            for s in sentences_el.findall("sentence"):
                sid = s.get("id", "")
                text = el_text(s)
                sentences.append({"id": sid, "text": text})
        cases.append({
            "case_id": case_id,
            "patient_question": patient_question,
            "clinician_question": clinician_question,
            "note_excerpt": note_excerpt,
            "sentences": sentences,
        })
    return cases


def build_evidence_prompt(case: Dict[str, Any], few_shot_examples: Optional[List[Dict[str, Any]]] = None, enhanced: bool = False, cot: bool = False) -> str:
    """Build prompt for evidence identification, with optional few-shot examples."""
    if enhanced:
        lines = [
            "You are an expert clinical evidence selector for patient-facing EHR question answering.",
            "Your task: given a patient's question, the clinician's interpretation of that question, and a clinical note excerpt with numbered sentences, identify every sentence that contains evidence essential to answering the question.",
            "",
            "Guidelines for selecting evidence sentences:",
            "- Include sentences with diagnoses, medications, lab values, vitals, procedures, or clinical findings that directly address the question.",
            "- Include sentences that provide necessary clinical context (e.g. dates, dosages, test results) without which the answer would be incomplete.",
            "- Err on the side of inclusion: if a sentence might be needed to fully answer the question, include it.",
            "- Do NOT include sentences that are purely administrative (e.g. headers, signatures) unless they contain relevant clinical information.",
            "- Do NOT include sentences that discuss unrelated conditions or treatments.",
            "",
        ]
    else:
        lines = [
            "You are a clinical evidence selector. Given a patient question, a clinician-interpreted question, and a clinical note excerpt with numbered sentences, output the minimal set of sentence IDs that provide evidence needed to answer the question.",
            "",
        ]
    if cot:
        lines.append("Think step by step: (1) understand what the patient is asking, (2) identify what clinical information is needed to answer, (3) for each sentence, decide if it contains that information. Then output your final answer as a JSON array of sentence IDs.")
        lines.append("")
    if few_shot_examples:
        lines.append("Examples:")
        for ex in few_shot_examples:
            c, gold = ex["case"], ex["gold_ids"]
            lines.append("---")
            lines.append("Patient question: " + c["patient_question"])
            lines.append("Clinician question: " + c["clinician_question"])
            lines.append("Sentences:")
            for s in c["sentences"]:
                lines.append(f"  {s['id']}: {s['text']}")
            lines.append("Evidence sentence IDs: " + json.dumps(gold))
            lines.append("")
        lines.append("Now do the same for the following case. Output only a JSON array of sentence IDs (strings).")
        lines.append("")
    lines.extend([
        "Patient question:",
        case["patient_question"],
        "",
        "Clinician-interpreted question:",
        case["clinician_question"],
        "",
        "Clinical note sentences (ID: text):",
    ])
    for s in case["sentences"]:
        lines.append(f"  {s['id']}: {s['text']}")
    if cot:
        lines.extend([
            "",
            "First, briefly reason about which sentences are relevant. Then on a new line output ONLY a JSON array of sentence IDs (strings), e.g. [\"1\", \"2\", \"5\"]. If no sentence is relevant, output [].",
        ])
    else:
        lines.extend([
            "",
            "Output only a JSON array of sentence IDs (strings), e.g. [\"1\", \"2\", \"5\"]. No explanation. If no sentence is relevant, output [].",
        ])
    return "\n".join(lines)


def call_azure_chat(prompt: str, max_tokens: int = 500, temperature: float = 0.0, deployment: Optional[str] = None) -> str:
    """Call Azure OpenAI Chat Completions (o3, gpt-5.1, gpt-5.2). o3 gets dedicated client + higher cap; retry on 429 and empty."""
    model = deployment or AZURE_DEPLOYMENT
    is_o3 = model.lower() in ("o3", "o3-pro")
    client = (azure_client_o3 if is_o3 and azure_client_o3 else azure_client)
    if not client:
        return ""
    use_temp = not is_o3
    cap = 2048 if is_o3 else max_tokens
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if use_temp:
        kwargs["temperature"] = temperature
    kwargs["max_completion_tokens"] = cap
    max_retries = 4
    empty_retries = 2
    for attempt in range(max_retries):
        try:
            try:
                response = client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("max_completion_tokens", None)
                if not is_o3:
                    kwargs["max_tokens"] = max_tokens
                else:
                    kwargs["max_completion_tokens"] = cap
                response = client.chat.completions.create(**kwargs)
            choice = response.choices[0] if response.choices else None
            content = (choice.message.content if choice and choice.message else None) or ""
            content = content.strip() if isinstance(content, str) else ""
            if content:
                return content
            # Diagnose why empty: finish_reason often explains (content_filter, length, etc.)
            finish_reason = getattr(choice, "finish_reason", None) if choice else None
            ref = getattr(response, "model", None) or model
            print(f"  Empty from {model}: finish_reason={finish_reason!r}, id={getattr(response, 'id', None)!r}")
            if empty_retries > 0:
                empty_retries -= 1
                wait = 5
                print(f"  Waiting {wait}s then retry ({2 - empty_retries}/2)")
                time.sleep(wait)
                continue
            return ""
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "ratelimit" in err_str) and attempt < max_retries - 1:
                wait = 10
                print(f"  Rate limit (429), waiting {wait}s then retry ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"  API error ({model}): {e}")
            return ""
    return ""


# Lightweight clinical-entity detection for post-filter (rule-based).
_CLINICAL_ENTITY_PATTERN = re.compile(
    r"\b(mg|mL|mg/kg|IU|mcg|units?|%\s*\d|\d+\.?\d*\s*(mg|mL|IU|units?))\b|"
    r"\b(diagnosis|procedure|medication|dose|drug|admitted|discharged|surgery|treatment|blood|lab|patient|hospital|allergy|allergic)\b|"
    r"\b\d+\.?\d*\s*(mg|mL|IU)\b",
    re.IGNORECASE,
)


def _has_clinical_entity(text: str) -> bool:
    """True if text contains a simple clinical cue (units, drugs, procedures, etc.)."""
    return bool(_CLINICAL_ENTITY_PATTERN.search(text)) or bool(re.search(r"\d+\.?\d*", text))


def _vote_counts_from_predictions(
    list_of_predictions: List[List[str]],
    valid_ids: List[str],
) -> Counter:
    """Return vote count per sentence ID (only for valid IDs)."""
    valid_set = set(valid_ids)
    counts = Counter()
    for pred in list_of_predictions:
        for sid in pred:
            if sid in valid_set:
                counts[sid] += 1
    return counts


def post_filter_evidence(
    prediction: List[str],
    vote_counts: Counter,
    case: Dict[str, Any],
    single_vote_max_tokens: int = 10,
    require_entity_if_single_vote: bool = True,
) -> List[str]:
    """Drop single-vote sentences that are short and contain no clinical entity. Keeps recall high, cuts FP."""
    sentences_by_id = {s["id"]: s["text"] for s in case.get("sentences", [])}
    kept = []
    for sid in prediction:
        votes = vote_counts.get(sid, 0)
        if votes >= 2:
            kept.append(sid)
            continue
        if votes != 1:
            kept.append(sid)
            continue
        text = sentences_by_id.get(sid, "")
        tokens = len(text.split())
        if tokens >= single_vote_max_tokens:
            kept.append(sid)
            continue
        if require_entity_if_single_vote and _has_clinical_entity(text):
            kept.append(sid)
            continue
        # Drop: single vote, short, no clinical entity
    return sorted(kept, key=lambda x: int(x) if x.isdigit() else 0)


def build_tighten_prompt(case: Dict[str, Any], prediction: List[str]) -> str:
    """Prompt for second pass: remove citations that clearly do not help answer. Only remove, never add."""
    sentences_by_id = {s["id"]: s["text"] for s in case.get("sentences", [])}
    cited = [(sid, sentences_by_id.get(sid, "")) for sid in prediction if sid in sentences_by_id]
    lines = [
        "You are a clinical evidence pruner. Given a question and a list of cited sentences, output the sentence IDs that help answer the question. Remove only citations that clearly do not support the answer. When in doubt, keep the citation. Do not add any new sentence IDs.",
        "",
        "Patient question:",
        case.get("patient_question", ""),
        "",
        "Clinician question:",
        case.get("clinician_question", ""),
        "",
        "Cited sentences (ID: text):",
    ]
    for sid, text in cited:
        lines.append(f"  {sid}: {text}")
    lines.extend([
        "",
        "Output only a JSON array of sentence IDs (strings) to KEEP. No explanation. Example: [\"1\", \"3\"]",
    ])
    return "\n".join(lines)


def build_verify_prompt(case: Dict[str, Any], prediction: List[str]) -> str:
    """Verification pass: given the question and candidate sentences, keep ONLY those that are
    essential evidence (contain diagnoses, test results, medications, procedures, or clinical
    findings that directly answer the question). Remove supplementary context, administrative
    details, and sentences about unrelated conditions."""
    sentences_by_id = {s["id"]: s["text"] for s in case.get("sentences", [])}
    cited = [(sid, sentences_by_id.get(sid, "")) for sid in prediction if sid in sentences_by_id]
    lines = [
        "You are a strict clinical evidence verifier. You will be given a patient question, a clinician question, and a set of candidate evidence sentences that were pre-selected from a clinical note.",
        "",
        "Your task: Review each candidate sentence and KEEP only those that are ESSENTIAL to answering the question. Remove sentences that are:",
        "- About unrelated conditions, treatments, or findings",
        "- Purely administrative (headers, signatures, dates without clinical value)",
        "- Providing background context that is not needed to answer the specific question",
        "- Duplicating information already covered by other kept sentences",
        "",
        "A sentence is ESSENTIAL if removing it would make the answer to the question incomplete or less accurate.",
        "",
        "Patient question:",
        case.get("patient_question", ""),
        "",
        "Clinician question:",
        case.get("clinician_question", ""),
        "",
        "Candidate sentences (ID: text):",
    ]
    for sid, text in cited:
        lines.append(f"  {sid}: {text}")
    lines.extend([
        "",
        "Output ONLY a JSON array of sentence IDs (strings) to KEEP. Do NOT add new IDs. Example: [\"1\", \"3\"]",
    ])
    return "\n".join(lines)


def merge_ensemble_predictions(
    list_of_predictions: List[List[str]],
    valid_ids: List[str],
    min_votes: int,
) -> List[str]:
    """Keep sentence ID if at least min_votes models selected it. Ties broken by numeric order."""
    valid_set = set(valid_ids)
    counts = Counter()
    for pred in list_of_predictions:
        for sid in pred:
            if sid in valid_set:
                counts[sid] += 1
    result = [sid for sid, cnt in counts.items() if cnt >= min_votes]
    return sorted(result, key=lambda x: int(x) if x.isdigit() else 0)


def parse_evidence_response(response: str, valid_ids: List[str]) -> List[str]:
    """Parse model output into list of sentence IDs; only allow IDs present in valid_ids."""
    valid_set = set(valid_ids)
    result = []
    # Try JSON array first
    try:
        # Strip markdown code block if present
        text = response.strip()
        for pattern in [r"\[[\s\d\",\']+\]", r"\[\s*\"[\d]+\"\s*(?:,\s*\"[\d]+\"\s*)*\]"]:
            m = re.search(pattern, text)
            if m:
                arr = json.loads(m.group(0))
                result = [str(x).strip() for x in arr if str(x).strip() in valid_set]
                return sorted(result, key=lambda x: int(x) if x.isdigit() else 0)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: numbers or "1", "2" in text
    for part in re.findall(r'"(\d+)"|\b(\d+)\b', response):
        sid = (part[0] or part[1]).strip()
        if sid in valid_set and sid not in result:
            result.append(sid)
    return sorted(result, key=lambda x: int(x) if x.isdigit() else 0)


def run_evidence_pipeline(
    xml_path: Path,
    out_path: Path,
    limit: Optional[int] = None,
    key_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run evidence ID: few-shot (when key available) + ensemble (o3 + gpt-5.2 + gpt-5.1) + ≥MIN_VOTES merge."""
    cases = parse_qa_xml(xml_path)
    if limit:
        cases = cases[:limit]
    valid_ids_per_case = {c["case_id"]: [s["id"] for s in c["sentences"]] for c in cases}
    cases_by_id = {c["case_id"]: c for c in cases}

    # Few-shot: dev (gold) + optional silver from test (21–120) when key has citations. Sample from full dev pool to avoid bias to first N.
    few_shot_examples_all: List[Dict[str, Any]] = []
    if data_dir is None:
        data_dir = xml_path.parent.parent if xml_path.parent.name != "v1.4" else xml_path.parent
    dev_key = key_path or (data_dir / "dev" / "archehr-qa_key.json")
    dev_xml = data_dir / "dev" / "archehr-qa.xml"
    use_silver = TASK2_USE_SILVER_FEW_SHOT and TASK2_FEW_SHOT_SILVER_N > 0
    dev_n = min(TASK2_FEW_SHOT_DEV_N if use_silver else FEW_SHOT_N, 20)
    if FEW_SHOT_N > 0 and dev_key.exists() and dev_xml.exists():
        if TASK2_HOLDOUT_IDS:
            # 80/20 split: use only tune (non-hold-out) cases for few-shot
            few_shot_case_ids = [str(i) for i in range(1, 21) if str(i) not in TASK2_HOLDOUT_IDS]
            print(f"Hold-out (excluded from few-shot): {sorted(TASK2_HOLDOUT_IDS)} -> {len(few_shot_case_ids)} tune examples")
        else:
            pool = list(range(1, 21))
            if FEW_SHOT_RANDOM_SAMPLE:
                rng = random.Random(FEW_SHOT_RANDOM_SEED)
                few_shot_case_ids = [str(i) for i in rng.sample(pool, min(dev_n, len(pool)))]
            else:
                few_shot_case_ids = [str(i) for i in range(1, dev_n + 1)]
        few_shot_examples_all = load_few_shot_examples(
            dev_key, dev_xml, few_shot_case_ids, cases_by_id=None, strict_only=STRICT_ONLY_FEW_SHOT
        )
        print(f"Few-shot (dev): {len(few_shot_examples_all)} examples (case IDs {few_shot_case_ids[:5]}{'...' if len(few_shot_case_ids) > 5 else ''}), strict_only={STRICT_ONLY_FEW_SHOT}, random_sample={FEW_SHOT_RANDOM_SAMPLE}")

    # Optional: add silver examples from test (21–120) when key has clinician_answer_sentences.citations
    if TASK2_USE_SILVER_FEW_SHOT and TASK2_FEW_SHOT_SILVER_N > 0:
        test_key = data_dir / "test" / "archehr-qa_key.json"
        test_xml = data_dir / "test" / "archehr-qa.xml"
        silver_map = load_silver_evidence_from_key(test_key)
        if silver_map and test_xml.exists():
            test_cases = parse_qa_xml(test_xml)
            test_by_id = {c["case_id"]: c for c in test_cases}
            silver_case_ids = list(silver_map.keys())[:TASK2_FEW_SHOT_SILVER_N]
            silver_examples = load_few_shot_examples(
                test_key, test_xml, silver_case_ids, cases_by_id=test_by_id, silver_map=silver_map
            )
            few_shot_examples_all = few_shot_examples_all + silver_examples
            print(f"Few-shot (silver): +{len(silver_examples)} from test (total {len(few_shot_examples_all)})")
        else:
            print(f"Few-shot (silver): none (test key has no citations or test XML missing)")

    use_ensemble = len(ENSEMBLE_DEPLOYMENTS) >= 1
    effective_min_votes = min(MIN_VOTES, len(ENSEMBLE_DEPLOYMENTS)) if ENSEMBLE_DEPLOYMENTS else 1
    if use_ensemble:
        print(f"Ensemble: {ENSEMBLE_DEPLOYMENTS}, min_votes={effective_min_votes}")
    if TASK2_TTS and use_ensemble:
        print(f"TTS: on (deployment={TASK2_TTS_DEPLOYMENT}, N={TASK2_TTS_N}, temp={TASK2_TTS_TEMPERATURE}, tts_min_votes={TASK2_TTS_MIN_VOTES})")
    if TASK2_POST_FILTER:
        print(f"Post-filter: on (single_vote_max_tokens={TASK2_POST_FILTER_SINGLE_VOTE_MAX_TOKENS}, require_entity={TASK2_POST_FILTER_REQUIRE_ENTITY})")
    if TASK2_TIGHTEN_CITATIONS:
        print(f"Tighten citations: on (deployment={TASK2_TIGHTEN_DEPLOYMENT})")
    if TASK2_ENHANCED_PROMPT:
        print("Enhanced prompt: on")
    if TASK2_COT:
        print("Chain-of-thought: on")
    if TASK2_MULTI_RUN > 1:
        print(f"Multi-run stabilization: {TASK2_MULTI_RUN} runs, majority vote across runs")

    # --- helper: run one full pass of the ensemble for a single case ---
    def _run_one_pass(case: Dict[str, Any], few_shot_for_case, valid_ids: List[str], cot_max_tokens: int = 1500) -> List[str]:
        prompt_full = build_evidence_prompt(case, few_shot_for_case if few_shot_for_case else None,
                                            enhanced=TASK2_ENHANCED_PROMPT, cot=TASK2_COT)
        few_shot_o3 = few_shot_for_case[:O3_FEW_SHOT_N] if few_shot_for_case and len(few_shot_for_case) > O3_FEW_SHOT_N else few_shot_for_case
        prompt_o3 = build_evidence_prompt(case, few_shot_o3 if few_shot_o3 else None,
                                          enhanced=TASK2_ENHANCED_PROMPT, cot=TASK2_COT)
        tok = cot_max_tokens if TASK2_COT else 500

        if use_ensemble:
            all_preds: List[List[str]] = []
            for dep in ENSEMBLE_DEPLOYMENTS:
                prompt = prompt_o3 if dep.lower() in ("o3", "o3-pro") else prompt_full
                # Test-time scaling: one deployment runs N times at high temp, aggregate by vote
                if TASK2_TTS and dep.strip().lower() == TASK2_TTS_DEPLOYMENT.strip().lower():
                    tts_preds: List[List[str]] = []
                    for _ in range(TASK2_TTS_N):
                        response = call_azure_chat(
                            prompt, max_tokens=tok, deployment=dep, temperature=TASK2_TTS_TEMPERATURE
                        )
                        if response and response.strip():
                            tts_preds.append(parse_evidence_response(response, valid_ids))
                    if tts_preds:
                        pred = merge_ensemble_predictions(
                            tts_preds, valid_ids, min(TASK2_TTS_MIN_VOTES, len(tts_preds))
                        )
                    else:
                        pred = []
                    all_preds.append(pred)
                else:
                    response = call_azure_chat(prompt, max_tokens=tok, deployment=dep)
                    if not (response or response.strip()):
                        print(f"  WARNING: {dep} returned empty for case {case['case_id']}")
                    pred = parse_evidence_response(response, valid_ids)
                    all_preds.append(pred)
            prediction = merge_ensemble_predictions(all_preds, valid_ids, effective_min_votes)
            # Tier 1: Evidence-level post-filter (drop single-vote short sentences with no clinical entity)
            if TASK2_POST_FILTER and prediction:
                vote_counts = _vote_counts_from_predictions(all_preds, valid_ids)
                prediction = post_filter_evidence(
                    prediction,
                    vote_counts,
                    case,
                    single_vote_max_tokens=TASK2_POST_FILTER_SINGLE_VOTE_MAX_TOKENS,
                    require_entity_if_single_vote=TASK2_POST_FILTER_REQUIRE_ENTITY,
                )
        else:
            response = call_azure_chat(prompt_full, max_tokens=tok)
            prediction = parse_evidence_response(response, valid_ids)
        return prediction

    submissions = []
    for i, case in enumerate(cases):
        cid = case["case_id"]
        t0 = time.time()
        print(f"[{i+1}/{len(cases)}] Case {cid}")
        # Exclude current case from few-shot to avoid leaking gold
        few_shot_for_case = [ex for ex in few_shot_examples_all if ex["case"]["case_id"] != cid]
        valid_ids = valid_ids_per_case.get(cid, [])

        if TASK2_MULTI_RUN > 1:
            # Multi-run: run full ensemble N times, majority-vote at sentence level across runs
            run_preds: List[List[str]] = []
            for run_idx in range(TASK2_MULTI_RUN):
                pred = _run_one_pass(case, few_shot_for_case, valid_ids)
                run_preds.append(pred)
                print(f"  Run {run_idx+1}/{TASK2_MULTI_RUN}: {len(pred)} IDs")
            # Majority vote: keep sentence if selected in > half the runs
            multi_min = (TASK2_MULTI_RUN // 2) + 1
            prediction = merge_ensemble_predictions(run_preds, valid_ids, multi_min)
            print(f"  Multi-run vote (>={multi_min}/{TASK2_MULTI_RUN}): {len(prediction)} IDs")
        else:
            prediction = _run_one_pass(case, few_shot_for_case, valid_ids)

        # Tier 1: Two-stage citation tightening (second LLM pass: only remove, never add)
        if TASK2_TIGHTEN_CITATIONS and prediction and azure_client:
            tighten_prompt = build_tighten_prompt(case, prediction)
            tighten_response = call_azure_chat(tighten_prompt, deployment=TASK2_TIGHTEN_DEPLOYMENT)
            valid_set = set(valid_ids_per_case.get(cid, []))
            kept_ids = parse_evidence_response(tighten_response, list(valid_set))
            prediction = [sid for sid in prediction if sid in kept_ids]
            prediction = sorted(prediction, key=lambda x: int(x) if x.isdigit() else 0)

        elapsed = time.time() - t0
        print(f"  -> {len(prediction)} IDs ({elapsed:.1f}s)")
        submissions.append({"case_id": cid, "prediction": prediction})
        # Incremental save: write after each case so progress is visible and not lost on interrupt
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(submissions, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(submissions)} cases")

    print(f"Done. Wrote {out_path} ({len(submissions)} cases)")
    return submissions


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    argv = (sys.argv[1:] or ["dev"])
    split = argv[0].lower() if argv else "dev"
    limit = None
    if len(argv) >= 2 and argv[1].isdigit():
        limit = int(argv[1])
        print(f"Limit: {limit} cases")
    if split == "test":
        xml_path = data_dir / "test" / "archehr-qa.xml"
    elif split == "test-2026":
        xml_path = data_dir / "test-2026" / "archehr-qa.xml"
    else:
        xml_path = data_dir / "dev" / "archehr-qa.xml"
    if not xml_path.exists():
        xml_path = data_dir / "dev" / "archehr-qa.xml"
    # Experiment tag: output to submission_Subtask2_{tag} if set, else TTS→test2, else default
    if TASK2_EXP_TAG:
        out_dir = base / f"submission_Subtask2_{TASK2_EXP_TAG}"
    elif TASK2_TTS:
        out_dir = base / "submission_Subtask2_test2"
    else:
        out_dir = base / "submission_Subtask2"
    out_path = out_dir / ("submission_test.json" if split == "test-2026" else "submission.json")
    key_path = data_dir / "dev" / "archehr-qa_key.json"
    run_evidence_pipeline(xml_path, out_path, key_path=key_path, data_dir=data_dir, limit=limit)
