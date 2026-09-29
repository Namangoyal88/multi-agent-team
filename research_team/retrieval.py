"""Query understanding, targeted literature query generation and relevance ranking."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

from dataclasses import dataclass, field


KNOWN_CANONICAL_TOPICS = {
    "nested machine learning": "Nested Learning",
    "nested ml": "Nested Learning",
    "agentic rag": "Agentic RAG",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "rag": "RAG",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with", "from", "new",
    "recent", "latest", "current", "research", "related", "paper", "papers", "study", "studies",
    "about", "find", "discover", "looking", "look", "into", "what", "is", "are", "can", "be",
    "this", "that", "how", "does", "do", "vs", "versus",
}


@dataclass
class QuerySpec:
    topic: str
    organization: str = ""
    time: str = ""
    intent: str = "research discovery"
    canonical_terms: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "topic": self.topic,
            "organization": self.organization,
            "time": self.time,
            "intent": self.intent,
            "canonical_terms": list(self.canonical_terms),
        }


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w:.'/-]+", " ", text or "")).strip()


def _title_case_topic(topic: str) -> str:
    key = topic.lower().strip()
    return KNOWN_CANONICAL_TOPICS.get(key, topic.strip())


def infer_query_spec(question: str) -> QuerySpec:
    q = _clean(question)
    low = q.lower()

    org = ""
    for pat, name in (
        (r"\bgoogle research\b", "Google Research"),
        (r"\bgoogle\b", "Google Research"),
        (r"\bdeepmind\b", "DeepMind"),
        (r"\bmeta ai\b", "Meta AI"),
        (r"\bmicrosoft research\b", "Microsoft Research"),
        (r"\bnvidia research\b", "NVIDIA Research"),
        (r"\bopenai\b", "OpenAI"),
    ):
        if re.search(pat, low):
            org = name
            break

    topic = ""

    # Prefer the semantic subject over the entire user sentence. These patterns cover the
    # common research-question forms used by the workflow and stop before the predicate.
    topic_patterns = (
        r"\bwhat\s+(?:is|are|does|do)\s+(.+?)(?:\?|$)",
        r"\bdefine\s+(.+?)(?:\?|$)",
        r"\b(?:investigate|explore|study|analy[sz]e|assess|evaluate)\s+whether\s+(.+?)\s+(?:can|could|will|would|does|do|is|are)\b",
        r"\bwhether\s+(.+?)\s+(?:can|could|will|would|does|do|is|are)\b",
        r"\bcompare\s+(.+?)\s+(?:with|versus|vs\.?|against)\b",
        r"\b(?:papers?|research|work|studies?)\s+(?:on|about|regarding)\s+(.+?)(?:\?|$)",
        r"\bfind\s+(?:papers?\s+)?(?:on|about)\s+(.+?)(?:\?|$)",
        r"\brelated to\s+(.+?)(?:\?|$)",
        r"\babou?t\s+(.+?)(?:\?|$)",
        r"\bon\s+(.+?)(?:\?|$)",
        r"\babout\s+(.+?)(?:\?|$)",
    )
    for pat in topic_patterns:
        m = re.search(pat, low)
        if m:
            candidate = m.group(1).strip(" .,:;\"")
            candidate = re.sub(r"^(?:the|a|an)\s+", "", candidate).strip()
            topic = candidate
            break

    # Objective/metric questions often end with the actual research subject.
    if not topic:
        m = re.search(r"\b(?:objective|loss|metric|benchmark|performance)\s+(?:for|of|in)\s+(.+?)(?:\?|$)", low)
        if m:
            topic = m.group(1).strip(" .,:;\"")

    topic = _title_case_topic(topic)
    if not topic:
        cleaned = re.sub(
            r"\b(new|recent|latest|current|research|papers?|discover|find|google research|google|investigate|explore|study|analyse|analyze|whether|compare|explain|describe|show|discuss)\b",
            " ", low)
        cleaned = re.sub(r"\b(can|could|will|would|does|do|is|are)\b.*$", " ", cleaned)
        parts = cleaned.split()
        topic = _title_case_topic(" ".join(parts[:6]))

    # Recover known canonical concepts embedded inside a noisy residual phrase.
    low_topic = topic.lower()
    for alias, canonical in KNOWN_CANONICAL_TOPICS.items():
        if alias in low_topic:
            topic = canonical
            break
    if "agentic rag" in low_topic and low_topic != "agentic rag":
        topic = "Agentic RAG"

    time = ""
    if re.search(r"\b(new|recent|latest|current|this year|today|202[5-9])\b", low):
        time = "recent/current"

    if any(x in low for x in ("derive", "equation", "objective", "theorem", "proof")):
        intent = "theory"
    elif any(x in low for x in ("implement", "architecture", "code", "pipeline")):
        intent = "system/implementation"
    elif any(x in low for x in ("experiment", "benchmark", "ablation", "evaluate")):
        intent = "experimental evaluation"
    else:
        intent = "research discovery"

    aliases = [topic] if topic else []
    if topic.lower() == "nested learning":
        aliases += ["nested machine learning", "continual learning", "Hope architecture"]

    return QuerySpec(topic=topic or q[:100], organization=org, time=time, intent=intent,
                     canonical_terms=list(dict.fromkeys(aliases)))


def targeted_queries(spec: QuerySpec, original_question: str = "") -> list[str]:
    topic = spec.topic.strip() or original_question.strip()
    org = spec.organization.strip()
    out: list[str] = []

    if topic and org:
        out.append(f'"{topic}" "{org}"')
        out.append(f'"{topic}" {org} research')
    elif topic:
        out.append(f'"{topic}" research')
        out.append(f'"{topic}" related work')

    aliases = [x for x in spec.canonical_terms if x and x.lower() != topic.lower()]
    for alias in aliases[:3]:
        out.append(f'"{alias}" {org}'.strip())

    if topic.lower() == "nested learning":
        out.extend([
            '"Nested Learning: The Illusion of Deep Learning Architectures"',
            '"Google" "Nested Learning" continual learning',
            '"Hope" architecture "Nested Learning"',
        ])
    elif spec.time:
        year = datetime.now(timezone.utc).year
        out.append(f'"{topic}" {year}')
        out.append(f'"{topic}" latest research')

    if original_question:
        out.append(" ".join(original_question.split()[:12]))
    return list(dict.fromkeys(x.strip() for x in out if x.strip()))[:8]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", (text or "").lower()) if t not in STOPWORDS}


def _phrase_present(phrase: str, text: str) -> bool:
    return bool(phrase and re.search(re.escape(phrase.strip()), text or "", re.I))


def _year(paper: dict) -> int | None:
    raw = str(paper.get("year") or "")[:4]
    return int(raw) if raw.isdigit() else None


def relevance_score(paper: dict, spec: QuerySpec, original_question: str = "") -> tuple[float, dict]:
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    url = paper.get("url") or ""
    blob = f"{title} {abstract} {url}".lower()
    score = 0.0

    topic = spec.topic.strip()
    topic_tokens = _tokens(topic)
    text_tokens = _tokens(blob)
    overlap = len(topic_tokens & text_tokens) / max(len(topic_tokens), 1)
    title_tokens = _tokens(title)
    title_overlap = len(topic_tokens & title_tokens) / max(len(topic_tokens), 1)

    if _phrase_present(topic, title):
        score += 20
    elif _phrase_present(topic, blob):
        score += 14
    score += 10 * overlap
    score += 8 * title_overlap

    org_match = _phrase_present(spec.organization, blob) if spec.organization else False
    if org_match:
        score += 9

    question_tokens = _tokens(original_question)
    q_overlap = len(question_tokens & text_tokens) / max(len(question_tokens), 1)
    score += 5 * q_overlap

    year = _year(paper)
    recent_match = None
    if spec.time and year:
        recent_match = year >= datetime.now(timezone.utc).year - 4
        score += 5 if recent_match else -4

    source_bonus = {"arxiv": 1.5, "semantic_scholar": 1.0, "pubmed": 0.5}.get(paper.get("source"), 0.0)
    score += source_bonus

    # Penalise results that share almost no topic signal. This removes common-but-related
    # terms such as "optimization" or "learning" that otherwise swamp entity-specific search.
    if topic_tokens and overlap < 0.20 and title_overlap == 0:
        score -= 12

    # Small title similarity bonus helps exact/named-paper recovery when a model-generated
    # query is very specific.
    if original_question:
        score += 2 * SequenceMatcher(None, title.lower(), original_question.lower()).ratio()

    return score, {
        "topic_overlap": round(overlap, 3),
        "title_overlap": round(title_overlap, 3),
        "organization_match": org_match,
        "question_overlap": round(q_overlap, 3),
        "year": year,
        "recent_match": recent_match,
        "score": round(score, 3),
    }


def deduplicate_candidates(papers: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: dict[str, int] = {}

    def keys(p: dict) -> list[str]:
        # Prefer stable scholarly identifiers, then URL, then normalized title. Title fallback
        # is important when the same paper is returned by different providers with different
        # landing URLs or missing DOI/arXiv metadata.
        title = re.sub(r"[^a-z0-9]+", " ", str(p.get("title") or "").lower()).strip()
        url = str(p.get("url") or "").strip().lower().rstrip("/")
        return [k for k in (
            str(p.get("doi") or "").strip().lower(),
            str(p.get("arxiv_id") or "").strip().lower(),
            str(p.get("pmid") or "").strip().lower(),
            url,
            f"title:{title}" if title else "",
        ) if k]

    for p in papers:
        pkeys = keys(p)
        if not pkeys:
            continue
        idx = next((seen[k] for k in pkeys if k in seen), None)
        if idx is not None:
            # Merge missing/better provider metadata without creating another evidence record.
            merged = dict(out[idx])
            for k, v in p.items():
                if v not in (None, "", []):
                    if not merged.get(k) or (k == "abstract" and len(str(v)) > len(str(merged.get(k) or ""))):
                        merged[k] = v
            out[idx] = merged
            for k in pkeys:
                seen[k] = idx
        else:
            idx = len(out)
            out.append(dict(p))
            for k in pkeys:
                seen[k] = idx
    return out


def rank_and_filter(papers: list[dict], spec: QuerySpec, original_question: str = "",
                    *, candidate_cap: int = 48, final_cap: int = 12) -> tuple[list[dict], list[dict]]:
    candidates = deduplicate_candidates(papers)
    ranked: list[tuple[float, dict, dict]] = []
    for p in candidates:
        score, details = relevance_score(p, spec, original_question)
        # For short canonical concepts, require the phrase itself rather than accepting a
        # generic shared word such as "learning". Longer topics can use proportionate token overlap.
        topic_terms = _tokens(spec.topic)
        phrase_signal = any(_phrase_present(term, f"{p.get('title','')} {p.get('abstract','')}")
                            for term in ([spec.topic] + list(spec.canonical_terms)) if term)
        topic_signal = phrase_signal if len(topic_terms) <= 2 else details["topic_overlap"] >= 0.67
        entity_signal = details["organization_match"]
        recent_ok = details["recent_match"] is not False

        # Entity-specific research: require real topic signal. Organization is a ranking gate,
        # not a hard requirement, because arXiv/Semantic Scholar often omit affiliations.
        if spec.topic and not topic_signal:
            continue
        if spec.time and not recent_ok:
            continue
        if spec.organization and entity_signal:
            score += 4
        ranked.append((score, p, details))

    ranked.sort(key=lambda x: (x[0], int(x[1].get("citations") or 0)), reverse=True)
    candidates_out = [{**p, "relevance_score": round(score, 3), "relevance": details}
                      for score, p, details in ranked[:candidate_cap]]
    selected = candidates_out[:final_cap]
    return selected, candidates_out
