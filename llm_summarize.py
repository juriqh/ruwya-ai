import os, json, google.generativeai as genai

MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
API_KEY = os.environ.get("GOOGLE_API_KEY")

SYS = """You are the summarization brain for Ruwya AI (daily AI/robotics digest).
Return ONLY JSON with keys:
summary: 2–3 crisp sentences (include key numbers/dates/names).
why: one short line on impact (business/research/engineering).
impact_score: integer 1–10 (higher = more significant).
tweet: one shareable line <= 280 chars, no hashtags, no emojis.
title_llm: punchier but accurate headline.
"""

TOP3_SYS = """Pick the Top 3 most significant stories of the day.
Criteria: novelty, breadth of impact, reliability, reader interest.
Return ONLY: {"top3_ids": ["id1","id2","id3"]} using the provided ids.
"""

def _get_model():
    if not API_KEY:
        return None
    genai.configure(api_key=API_KEY)
    try:
        return genai.GenerativeModel(MODEL)
    except Exception:
        return None

_model = _get_model()

def summarize_one(item: dict) -> dict:
    """item should have id,title,source,excerpt,url. Returns dict with summary/why/impact_score/tweet/title_llm."""
    # Fallbacks if no model or error
    fallback = {
        "summary": (item.get("excerpt","") or item.get("title",""))[:800],
        "why": "",
        "impact_score": 5,
        "tweet": f"{item.get('title','')[:240]} {item.get('url','')}",
        "title_llm": item.get("title","")[:160],
    }
    if not _model:
        return fallback

    prompt = f"""{SYS}
TITLE: {item.get('title','')}
SOURCE: {item.get('source','')}
URL: {item.get('url','')}
TEXT: {item.get('excerpt','')}
"""
    try:
        resp = _model.generate_content(prompt)
        txt = (resp.text or "").strip()
        j = txt[txt.find("{"): txt.rfind("}")+1]
        data = json.loads(j)
        data["summary"] = (data.get("summary",""))[:800]
        data["why"] = (data.get("why",""))[:240]
        data["impact_score"] = int(str(data.get("impact_score","5")).strip()[:2] or 5)
        data["tweet"] = (data.get("tweet",""))[:280]
        data["title_llm"] = (data.get("title_llm","") or item.get("title",""))[:160]
        return data
    except Exception:
        return fallback

def pick_top3(items: list) -> list:
    """Return 3 ids (prefers LLM ranking; falls back to impact_score + recency)."""
    if not items:
        return []
    if not _model:
        ranked = sorted(items, key=lambda x: (x.get("impact_score",5), x.get("published_at","")), reverse=True)
        return [it["id"] for it in ranked[:3]]
    # compact input to save tokens
    rows = [{"id": it["id"], "title": it.get("title",""), "source": it.get("source",""), "impact_score": it.get("impact_score",5), "bucket": it.get("bucket","")} for it in items[:30]]
    try:
        payload = json.dumps(rows, ensure_ascii=False)
        resp = _model.generate_content(TOP3_SYS + "\nDATA:\n" + payload)
        txt = (resp.text or "").strip()
        j = txt[txt.find("{"): txt.rfind("}")+1]
        data = json.loads(j)
        out = data.get("top3_ids", [])[:3]
        if len(out) == 3:
            return out
    except Exception:
        pass
    ranked = sorted(items, key=lambda x: (x.get("impact_score",5), x.get("published_at","")), reverse=True)
    return [it["id"] for it in ranked[:3]]
