from llm_summarize import summarize_one, pick_top3
import os, json, re, time, hashlib, pathlib, feedparser, yaml
from datetime import datetime, timezone
from dateutil import parser as dtp
from bs4 import BeautifulSoup
from huggingface_hub import HfApi

HF_REPO_ID = "juriqhqh/daily-digest"   # <- your HF dataset repo id
OUT_DIR = pathlib.Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(html_or_text):
    text = BeautifulSoup(html_or_text or "", "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def first_sentences(text, max_chars=320):
    s = re.split(r"(?<=[.!?])\s+", text)
    out = []
    total = 0
    for sent in s:
        if not sent: continue
        if total + len(sent) + 1 > max_chars: break
        out.append(sent)
        total += len(sent) + 1
    return " ".join(out) or text[:max_chars]

def norm_date(entry):
    for k in ("published", "updated", "created"):
        if k in entry:
            try: return dtp.parse(entry[k]).astimezone(timezone.utc).isoformat()
            except: pass
    if getattr(entry, "published_parsed", None):
        return datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()

def item_id(url):
    return hashlib.md5((url or str(time.time())).encode()).hexdigest()[:16]

def load_sources(path="sources.yaml"):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    return y["sources"]

def fetch_feed(src):
    d = feedparser.parse(src["url"])
    items = []
    for e in d.entries[:15]:
        title = clean_text(getattr(e, "title", ""))
        link = getattr(e, "link", "")
        summary = clean_text(getattr(e, "summary", "")) or clean_text(getattr(e, "description", ""))
        published_at = norm_date(e)
        why = first_sentences(summary) if summary else ""
        items.append({
            "id": item_id(link),
            "title": title,
            "url": link,
            "source": src["name"],
            "bucket": src["type"],
            "published_at": published_at,
            "excerpt": why
        })
    return items

def enforce_buckets(items, ratios={"research":0.35,"industry":0.40,"fun":0.25}, total=12):
    per = {k: max(1, round(total*v)) for k,v in ratios.items()}
    grouped = {"research":[],"industry":[],"fun":[]}
    for it in items:
        if it["bucket"] in grouped:
            grouped[it["bucket"]].append(it)
    for k in grouped: grouped[k].sort(key=lambda x: x["published_at"], reverse=True)
    picked=[]
    for k in ("research","industry","fun"):
        picked += grouped[k][:per[k]]
    if len(picked)<total:
        leftovers = [i for i in items if i not in picked]
        leftovers.sort(key=lambda x: x["published_at"], reverse=True)
        picked += leftovers[:(total-len(picked))]
    seen=set(); out=[]
    for it in picked:
        key=(it["url"] or it["title"]).lower()
        if key in seen: continue
        seen.add(key); out.append(it)
    return out[:total]

def save_and_push(items, top3_ids=None):
    today = datetime.now(timezone.utc).date().isoformat()
    day_file = OUT_DIR/f"{today}.json"
    latest_file = OUT_DIR/"latest.json"
    with open(day_file,"w",encoding="utf-8") as f: json.dump(items, f, ensure_ascii=False, indent=2)
    with open(latest_file,"w",encoding="utf-8") as f: json.dump(items, f, ensure_ascii=False, indent=2)

    meta = {"top3": top3_ids or []}
    meta_file = OUT_DIR/"meta.json"
    with open(meta_file,"w",encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, indent=2)


    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(path_or_fileobj=str(day_file), path_in_repo=f"daily/{day_file.name}", repo_id=HF_REPO_ID, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(latest_file), path_in_repo="latest.json", repo_id=HF_REPO_ID, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(meta_file), path_in_repo="meta.json", repo_id=HF_REPO_ID, repo_type="dataset")


def main():
    sources = load_sources()
    all_items=[]
    for src in sources:
        try:
            all_items.extend(fetch_feed(src))
        except Exception as ex:
            print("ERR source", src["name"], ex)
    all_items.sort(key=lambda x: x["published_at"], reverse=True)
    picked = enforce_buckets(all_items, total=12)

    # Enrich with Gemini fields
    enriched = []
    for it in picked:
        llm = summarize_one(it)
        it["summary"] = llm.get("summary","") or it.get("excerpt","")
        it["why"] = llm.get("why","")
        it["impact_score"] = llm.get("impact_score",5)
        it["tweet"] = llm.get("tweet","")
        it["title_llm"] = llm.get("title_llm", it.get("title",""))
        enriched.append(it)

    # Compute Top 3 ids
    top3_ids = pick_top3(enriched)

    save_and_push(enriched, top3_ids=top3_ids)
    print(f"Published {len(picked)} items.")

if __name__ == "__main__":
    main()
