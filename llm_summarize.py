import os, re, google.generativeai as genai

MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
API = os.environ.get("GOOGLE_API_KEY")

SYS = """You are the summarization brain for Ruwya AI (daily AI/robotics digest).
For each story, produce JSON with keys:
summary: 2–3 crisp sentences (include concrete facts: numbers, dates, names).
why: one short line on impact (business/research/engineering).
impact_score: integer 1–10 (higher = more significant).
tweet: one shareable line <= 280 chars, no hashtags, no emojis.
title_llm: punchier but accurate headline.
Return ONLY JSON.
"""

TOP3_SYS = """You pick the Top 3 most significant stories of the day.
Criteria: novelty, breadth of impact, reliability, likely reader interest.
Return ONLY a JSON object: {"top3_ids": ["id1","id2","id3"]} (use given ids).
"""

def _model():
    if not API: return None
    genai.configure(api_key=API)
    try: return genai.GenerativeModel(MODEL)
    except Exception: return None

_m = _model()

def summarize_one(item: dict) -> dict:
    """
    item keys expected: id, title, source, excerpt, url.
    Returns dict with summary/why/impact_score/tweet/title_llm (fallbacks if LLM unavailable).
    """
    if not _m:
        # Fallbacks if no API: just reuse excerpt
        return {
            "summary": item.get("excerpt","")[:600],
            "why": "",
            "impact_score": 5,
            "tweet": f"{item.get('title','')[:240]} {item.get('url','')}",
            "title_llm": item.get("title","")
        }
    prompt = f"""{SYS}
TITLE: {item.get('title','')}
SOURCE: {item.get('source','')}
URL: {item.get('url','')}
TEXT: {item.get('excerpt','')}
"""
    try:
        resp = _m.generate_content(prompt)
        text = (resp.text or "").strip()
        # cheap JSON sanitizer
        json_part = text[text.find("{"): text.rfind("}")+1]
        import json
        data = json.loads(json_part)
        # clamps
        data["summary"] = (data.get("summary",""))[:800]
        data["why"] = (data.get("why",""))[:240]
        data["impact_score"] = int(str(data.get("impact_score","5")).strip()[:2] or 5)
        data["tweet"] = (data.get("tweet",""))[:280]
        data["title_llm"] = (data.get("title_llm","") or item.get("title",""))[:160]
        return data
    except Exception:
        return {
            "summary": item.get("excerpt","")[:600],
            "why": "",
            "impact_score": 5,
            "tweet": f"{item.get('title','')[:240]} {item.get('url','')}",
            "title_llm": item.get("title","")
        }

def pick_top3(items: list) -> list:
    """
    items: list of dicts with 'id','title','source','impact_score' and maybe more.
    Returns list of 3 ids. Falls back to highest impact_score then recency.
    """
    if not _m or not items:
        ranked = sorted(items, key=lambda x: (x.get("impact_score",5), x.get("published_at","")), reverse=True)
        return [it["id"] for it in ranked[:3]]
    # compact context to avoid long prompts
    rows = []
    for it in items[:30]:  # cap
        rows.append({
            "id": it.get("id"),
            "title": it.get("title"),
            "source": it.get("source"),
            "impact_score": it.get("impact_score", 5),
            "bucket": it.get("bucket"),
        })
    import json
    prompt = TOP3_SYS + "\nDATA:\n" + json.dumps(rows, ensure_ascii=False)
    try:
        resp = _m.generate_content(prompt)
        text = (resp.text or "").strip()
        json_part = text[text.find("{"): text.rfind("}")+1]
        data = json.loads(json_part)
        if isinstance(data.get("top3_ids"), list) and len(data["top3_ids"]) >= 3:
            return data["top3_ids"][:3]
    except Exception:
        pass
    ranked = sorted(items, key=lambda x: (x.get("impact_score",5), x.get("published_at","")), reverse=True)
    return [it["id"] for it in ranked[:3]]
