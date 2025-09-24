#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import hashlib
import logging
import datetime as dt
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

import requests
from dotenv import load_dotenv
from plexapi.server import PlexServer
from plexapi.collection import Collection

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("recommendabot")

# ----------------------------
# Load .env relative to this file
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
    log.info("Loaded .env from %s", ENV_PATH)
else:
    log.warning("No .env found at %s — continuing with system environment variables.", ENV_PATH)

# ----------------------------
# Config (env)
# ----------------------------
PLEX_URL         = os.getenv("PLEX_URL", "http://127.0.0.1:32400")
PLEX_TOKEN       = os.getenv("PLEX_TOKEN", "")
PLEX_ACCOUNT_ID  = os.getenv("PLEX_ACCOUNT_ID", "")   # your Plex accountID (string)

LIBRARY_SECTIONS = [s.strip() for s in os.getenv("LIBRARY_SECTIONS", "Movies,TV Shows").split(",")]

LAT              = float(os.getenv("LAT", "39.9526"))
LON              = float(os.getenv("LON", "-75.1652"))

USE_LLM          = os.getenv("USE_LLM", "false").lower() == "true"
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

COLLECTION_BASE  = os.getenv("COLLECTION_BASE", "Recommended Today")
LOOKBACK_MONTHS  = int(os.getenv("LOOKBACK_MONTHS", "7"))
MAX_HISTORY      = int(os.getenv("MAX_HISTORY", "1000"))
TOPK_RECS        = int(os.getenv("TOPK_RECS", "10"))
DRY_RUN          = os.getenv("DRY_RUN", "false").lower() == "true"

# Caching
CACHE_DIR              = os.path.join(BASE_DIR, ".cache")
CACHE_TTL_MINUTES      = int(os.getenv("CACHE_TTL_MINUTES", "360"))  # 6h default
REFRESH_CACHE          = os.getenv("REFRESH_CACHE", "false").lower() == "true"
CACHE_KEY_PREFIX       = "inventory_v1"  # bump if you change structures
ITEM_CACHE_PATH        = os.path.join(CACHE_DIR, "item_cache_v1.json")

# ----------------------------
# Startup config dump
# ----------------------------
log.info("[CFG] PLEX_URL=%s", PLEX_URL)
log.info("[CFG] LIBRARY_SECTIONS=%s", LIBRARY_SECTIONS)
log.info("[CFG] LOOKBACK_MONTHS=%s MAX_HISTORY=%s TOPK_RECS=%s DRY_RUN=%s",
         LOOKBACK_MONTHS, MAX_HISTORY, TOPK_RECS, DRY_RUN)
log.info("[LLM] enabled=%s | key=%s | model=%s",
         USE_LLM, "yes" if OPENAI_API_KEY else "no", OPENAI_MODEL)
log.info("[CACHE] dir=%s ttl_min=%s refresh=%s", CACHE_DIR, CACHE_TTL_MINUTES, REFRESH_CACHE)

if not PLEX_TOKEN:
    log.error("Missing PLEX_TOKEN. Set it in .env or environment.")
    sys.exit(1)

# ----------------------------
# Plex connection
# ----------------------------
try:
    log.info("Connecting to Plex server...")
    plex = PlexServer(PLEX_URL, PLEX_TOKEN)
    log.info("Connected to Plex: %s", plex.friendlyName)
except Exception as e:
    log.exception("Failed to connect to Plex at %s: %s", PLEX_URL, e)
    sys.exit(1)

# ----------------------------
# Helpers
# ----------------------------
def now_local() -> dt.datetime:
    return dt.datetime.now()

def plex_get(path: str, params: Dict[str, str] = None, accept_json=True):
    headers = {"X-Plex-Token": PLEX_TOKEN}
    if accept_json:
        headers["Accept"] = "application/json"
    url = f"{PLEX_URL.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, headers=headers, params=params or {}, timeout=30)
    r.raise_for_status()
    if "application/json" in r.headers.get("Content-Type",""):
        return r.json()
    return r.text

def season_from_date(d: dt.date) -> str:
    m = d.month
    if m in (12, 1, 2): return "winter"
    if m in (3, 4, 5):  return "spring"
    if m in (6, 7, 8):  return "summer"
    return "autumn"

def fetch_weather(lat: float, lon: float) -> Dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def weather_bucket(current: Dict) -> str:
    try:
        code = int(current["weathercode"])
        temp = float(current["temperature"])
    except Exception:
        return "neutral"
    precipy = {51,53,55,61,63,65,80,81,82}
    snow = {71,73,75,77,85,86}
    if code in snow: return "snowy"
    if code in precipy: return "rainy"
    if temp >= 85: return "hot"
    if temp <= 40: return "cold"
    return "clear"

def months_ago_ts(n: int) -> int:
    cutoff = now_local() - dt.timedelta(days=int(n*30.437))  # ~avg month
    return int(cutoff.timestamp())

def extract_guids(item) -> Set[str]:
    out: Set[str] = set()
    try:
        for g in getattr(item, "guids", []) or []:
            gid = getattr(g, "id", None)
            if gid:
                out.add(str(gid))
    except Exception:
        pass
    try:
        g = getattr(item, "guid", None)
        if g:
            out.add(str(g))
    except Exception:
        pass
    return out

def stable_key_for_title(item) -> str:
    title = getattr(item, "title", "") or ""
    year = str(getattr(item, "year", "") or "")
    itype = getattr(item, "type", "") or ""
    show = ""
    try:
        if itype == "episode":
            show = getattr(item, "grandparentTitle", "") or ""
    except Exception:
        pass
    return f"{title}|{year}|{itype}|{show}".lower()

# ----------------------------
# History (paged with cutoff buffer)
# ----------------------------
def get_history_until_cutoff(max_fetch: int, cutoff_ts: int) -> List[Dict]:
    """
    Page history newest→older until:
      - read >= max_fetch rows, OR
      - crossed cutoff and read >= 3 pages (buffer to avoid boundary misses)
    """
    t0 = time.time()
    out: List[Dict] = []
    start = 0
    size = 200
    crossed_cutoff = False
    pages_read = 0

    log.info("Fetching recent history (paged) until cutoff=%s ...", cutoff_ts)
    while True:
        params = {"X-Plex-Container-Start": start, "X-Plex-Container-Size": size}
        data = plex_get("/status/sessions/history/all", params=params)
        if isinstance(data, dict):
            items = data.get("MediaContainer", {}).get("Metadata", []) or []
        else:
            log.warning("Non-JSON history; stopping after %d items.", len(out))
            break
        if not items:
            break

        out.extend(items)
        pages_read += 1

        try:
            oldest_page = min(int(x.get("viewedAt", 0) or 0) for x in items) if items else 0
        except Exception:
            oldest_page = 0
        if oldest_page and oldest_page < cutoff_ts:
            crossed_cutoff = True

        start += size
        if start >= max_fetch:
            break
        if crossed_cutoff and pages_read >= 3:
            break

    out.sort(key=lambda x: int(x.get("viewedAt", 0) or 0), reverse=True)
    log.info("History fetched (paged): %d rows (elapsed: %.2fs).", len(out), time.time() - t0)
    return out

def filter_history_by_account(items: List[Dict], account_id: str) -> List[Dict]:
    if not account_id:
        log.warning("PLEX_ACCOUNT_ID empty; using ALL users' history.")
        return items
    out = [x for x in items if str(x.get("accountID","")) == str(account_id)]
    log.info("After filtering by accountID=%s: %d rows remain.", account_id, len(out))
    if not out:
        log.warning("Zero rows after filtering by accountID=%s. Check your PLEX_ACCOUNT_ID.", account_id)
    return out

def build_last_seen_maps(history_rows: List[Dict]) -> Tuple[Dict[str,int], Dict[str,int], Dict[str,int]]:
    """
    Returns:
      - last_view_ts_by_rk: ratingKey -> last viewed ts
      - last_view_ts_by_guid: guid -> last viewed ts
      - last_view_ts_by_titlekey: fallback key -> last viewed ts
    """
    t0 = time.time()
    last_view_ts_by_rk: Dict[str,int] = {}
    last_view_ts_by_guid: Dict[str,int] = {}
    last_view_ts_by_titlekey: Dict[str,int] = {}

    # Fetch full items for unique ratingKeys
    unique_keys: List[str] = []
    for h in history_rows:
        rk = h.get("ratingKey")
        v  = int(h.get("viewedAt", 0) or 0)
        if not rk:
            continue
        if rk not in last_view_ts_by_rk or v > last_view_ts_by_rk[rk]:
            last_view_ts_by_rk[rk] = v
        if rk not in unique_keys:
            unique_keys.append(rk)

    for rk in unique_keys:
        try:
            it = plex.fetchItem(int(rk))
        except Exception:
            it = None
        if not it:
            continue

        for gid in extract_guids(it):
            prev = last_view_ts_by_guid.get(gid, 0)
            ts   = last_view_ts_by_rk.get(rk, 0)
            if ts > prev:
                last_view_ts_by_guid[gid] = ts

        tk = stable_key_for_title(it)
        if tk:
            prev = last_view_ts_by_titlekey.get(tk, 0)
            ts   = last_view_ts_by_rk.get(rk, 0)
            if ts > prev:
                last_view_ts_by_titlekey[tk] = ts

    log.info("History maps: %d unique items | %d GUIDs | %d titleKeys. (elapsed: %.2fs)",
             len(unique_keys), len(last_view_ts_by_guid), len(last_view_ts_by_titlekey), time.time() - t0)
    return last_view_ts_by_rk, last_view_ts_by_guid, last_view_ts_by_titlekey

def last_seen_timestamp(item, rk_map: Dict[str,int], guid_map: Dict[str,int], title_map: Dict[str,int]) -> Optional[int]:
    """
    Best-effort last-seen for the *current token user*:
      1) item.lastViewedAt (per-user)
      2) ratingKey map from filtered history
      3) any GUID map from filtered history
      4) title|year|type|show fallback
    """
    try:
        lva = getattr(item, "lastViewedAt", None)
        if lva:
            return int(lva.timestamp())
    except Exception:
        pass

    rk = str(getattr(item, "ratingKey", "") or "")
    if rk and rk in rk_map:
        return rk_map[rk]
    for gid in extract_guids(item):
        if gid in guid_map:
            return guid_map[gid]
    tk = stable_key_for_title(item)
    if tk in title_map:
        return title_map[tk]
    return None

def fetch_user_rating(item) -> Optional[float]:
    try:
        return getattr(item, "userRating", None)
    except Exception:
        return None

# ----------------------------
# Weather/Season → bonuses
# ----------------------------
WEATHER_GENRE_BONUS = {
    "rainy":  {"Drama":1.0, "Romance":0.6, "Mystery":0.6, "Comedy":0.3},
    "snowy":  {"Family":0.7, "Fantasy":0.6, "Holiday":1.2, "Drama":0.5},
    "hot":    {"Adventure":1.0, "Action":0.7, "Comedy":0.5, "Sports":0.5},
    "cold":   {"Thriller":0.6, "Drama":0.5, "Documentary":0.4},
    "clear":  {"Adventure":0.6, "Comedy":0.6},
    "neutral":{"Comedy":0.3}
}
SEASON_GENRE_BONUS = {
    "winter":{"Holiday":1.0, "Drama":0.4, "Fantasy":0.4},
    "spring":{"Romance":0.6, "Documentary":0.4},
    "summer":{"Adventure":0.8, "Action":0.5, "Sports":0.5, "Comedy":0.5},
    "autumn":{"Horror":0.9, "Mystery":0.6, "Drama":0.5}
}

def genre_bonus(genres: List[str], weather_b: Dict[str,float], season_b: Dict[str,float]) -> float:
    score = 0.0
    for g in genres:
        score += weather_b.get(g, 0.0)
        score += season_b.get(g, 0.0)
    return score

def recency_decay(days_since_watch: Optional[float]) -> float:
    if days_since_watch is None:
        return 1.0
    return min(1.0, max(0.1, days_since_watch/210.0))  # 7 months ~210 days

def _extract_genres_with_fallback(item) -> List[str]:
    genres: List[str] = []
    try:
        if getattr(item, "genres", None):
            genres = [g.tag for g in item.genres if getattr(g, "tag", None)]
        if not genres and getattr(item, "type", "") == "episode":
            try:
                show = item.show() if hasattr(item, "show") else None
                if show and getattr(show, "genres", None):
                    genres = [g.tag for g in show.genres if getattr(g, "tag", None)]
            except Exception:
                pass
    except Exception:
        genres = []
    return genres

def score_item(item, rk_map: Dict[str,int], guid_map: Dict[str,int], title_map: Dict[str,int],
               weather_b: Dict[str,float], season_b: Dict[str,float]) -> float:
    lv = last_seen_timestamp(item, rk_map, guid_map, title_map)
    days = None
    if lv:
        days = (now_local().timestamp() - lv)/86400.0
    base = 1.0
    ur = fetch_user_rating(item)  # 0..10 (or None)
    my_rating_term = 0.0 if ur is None else (ur/10.0)*2.5
    rec = recency_decay(days) * 1.0
    genres = _extract_genres_with_fallback(item)
    mood = genre_bonus(genres, weather_b, season_b)
    show_bonus = 0.0
    if getattr(item, "type", "") == "episode":
        try:
            show = item.show()
            s_rating = getattr(show, "userRating", None)
            if s_rating:
                show_bonus = (s_rating/10.0) * 0.5  # small nudge for shows you rated well
        except Exception:
            pass
    return base + my_rating_term + rec + mood + show_bonus

# ----------------------------
# Caching helpers
# ----------------------------
def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def cache_path(name: str) -> str:
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, name)

def is_cache_fresh(path: str, ttl_minutes: int) -> bool:
    if not os.path.exists(path):
        return False
    age = (time.time() - os.path.getmtime(path)) / 60.0
    return age <= ttl_minutes

def save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_item_cache() -> Dict[str, dict]:
    try:
        return load_json(ITEM_CACHE_PATH)
    except Exception:
        return {}

def save_item_cache(cache: Dict[str, dict]):
    ensure_cache_dir()
    save_json(ITEM_CACHE_PATH, cache)

def snapshot_item_fields(it) -> dict:
    genres = _extract_genres_with_fallback(it)
    try:
        lva = getattr(it, "lastViewedAt", None)
        last_seen_ts = int(lva.timestamp()) if lva else None
    except Exception:
        last_seen_ts = None

    return {
        "rk": int(getattr(it, "ratingKey", 0) or 0),
        "title": getattr(it, "title", ""),
        "year": getattr(it, "year", ""),
        "type": getattr(it, "type", ""),
        "genres": genres,
        "userRating": getattr(it, "userRating", None),
        "lastSeenTs": last_seen_ts,
        "updatedAt": int(time.time()),
    }

# ----------------------------
# Inventory (Movies + Episodes) with caching
# ----------------------------
def build_inventory_with_cache(sections: List[str],
                               rk_map: Dict[str,int],
                               guid_map: Dict[str,int],
                               title_map: Dict[str,int],
                               cutoff_ts: int):
    """
    Cache a snapshot of library inventory (ratingKeys only),
    then materialize with selective fetch & per-item cache.
    """
    key_raw = f"{CACHE_KEY_PREFIX}|{','.join(sections)}"
    cache_name = f"{hash_key(key_raw)}.json"
    path = cache_path(cache_name)

    if not REFRESH_CACHE and is_cache_fresh(path, CACHE_TTL_MINUTES):
        log.info("[CACHE] Using cached inventory: %s", path)
        inv = load_json(path)
    else:
        t0 = time.time()
        log.info("[CACHE] Rebuilding inventory for sections=%s ...", sections)
        inv = {"movies": [], "episodes": [], "meta": {"sections": sections, "built_at": time.time()}}

        for name in sections:
            try:
                section = plex.library.section(name)
            except Exception as e:
                log.warning("Skipping library '%s': %s", name, e)
                continue

            stype = getattr(section, "type", None)
            if stype == "movie":
                try:
                    movies = section.all()
                except Exception as e:
                    log.warning("Failed to list movies in '%s': %s", name, e)
                    continue
                for mv in movies:
                    if getattr(mv, "type", "") == "movie":
                        inv["movies"].append(int(getattr(mv, "ratingKey", 0) or 0))

            elif stype in ("show", "episode", "tv", "show-mixed"):
                try:
                    shows = section.all()
                except Exception as e:
                    log.warning("Failed to list shows in '%s': %s", name, e)
                    continue
                for sh in shows:
                    try:
                        eps = sh.episodes()
                    except Exception:
                        continue
                    for ep in eps:
                        if getattr(ep, "type", "") == "episode":
                            inv["episodes"].append(int(getattr(ep, "ratingKey", 0) or 0))
            else:
                log.info("Library '%s' (type=%s) not recognized as movie or TV; skipping.", name, stype)

        save_json(path, inv)
        log.info("[CACHE] Inventory built: %d movies, %d episodes (elapsed: %.2fs) → %s",
                 len(inv["movies"]), len(inv["episodes"]), time.time() - t0, path)

    def materialize(keys: List[int], label: str, rk_map, guid_map, title_map, cutoff_ts) -> List[object]:
        items = []
        item_cache = load_item_cache()
        updated = 0
        skipped = 0
        fetched = 0
        t0 = time.time()

        must_refresh: Set[int] = set(int(k) for k in rk_map.keys() if k is not None)

        for rk in keys:
            rk_str = str(rk)
            cached = item_cache.get(rk_str)
            need_fetch = REFRESH_CACHE or (rk in must_refresh) or (cached is None)

            it = None
            if need_fetch:
                try:
                    it = plex.fetchItem(int(rk))
                    fetched += 1
                except Exception:
                    it = None
                if not it:
                    continue
                snap = snapshot_item_fields(it)
                item_cache[rk_str] = snap
                updated += 1
            else:
                snap = cached

            # cutoff check using best available last-seen
            if snap.get("lastSeenTs") is None:
                if it is None:
                    try:
                        it = plex.fetchItem(int(rk))
                        fetched += 1
                    except Exception:
                        it = None
                lv = last_seen_timestamp(it, rk_map, guid_map, title_map) if it else None
                snap["lastSeenTs"] = lv
                if it is not None:
                    item_cache[rk_str] = snapshot_item_fields(it)

            if snap.get("lastSeenTs") and snap["lastSeenTs"] >= cutoff_ts:
                skipped += 1
                continue

            if it is not None:
                items.append(it)
            else:
                items.append(rk)

        save_item_cache(item_cache)
        log.info("Materialized %d %s (after cutoff) [fetched=%d, cache_updates=%d, skipped_cutoff=%d] (%.2fs)",
                 len(items), label, fetched, updated, skipped, time.time() - t0)
        return items

    movie_items = materialize(inv.get("movies", []), "movies", rk_map, guid_map, title_map, cutoff_ts)
    ep_items    = materialize(inv.get("episodes", []), "episodes", rk_map, guid_map, title_map, cutoff_ts)
    return movie_items, ep_items

# ----------------------------
# Collections (Update in-place)
# ----------------------------
def ensure_collection_update(section_name: str, title: str, items: List):
    if DRY_RUN:
        log.info("[DRY RUN] Would upsert collection '%s' in '%s' with %d items.", title, section_name, len(items))
        return

    section = plex.library.section(section_name)
    existing = None
    try:
        for c in section.collections():
            if c.title == title:
                existing = c
                break
    except Exception as e:
        log.warning("Could not enumerate collections in '%s': %s", section_name, e)

    if existing is None:
        try:
            Collection.create(plex, title=title, section=section, items=items, smart=False)
            log.info("Created new collection '%s' with %d items.", title, len(items))
        except Exception as e:
            log.exception("Failed creating collection '%s': %s", title, e)
            raise
        return

    try:
        current_items = list(existing.items())
        current_ids = {getattr(m, "ratingKey", None) for m in current_items}
        desired_ids = {getattr(m, "ratingKey", None) for m in items if getattr(m, "ratingKey", None) is not None}

        to_remove = [m for m in current_items if getattr(m, "ratingKey", None) not in desired_ids]
        to_add    = [m for m in items if getattr(m, "ratingKey", None) not in current_ids]

        if to_remove:
            existing.removeItems(to_remove)
            log.info("Removed %d stale items from '%s'.", len(to_remove), title)
        if to_add:
            existing.addItems(to_add)
            log.info("Added %d new items to '%s'.", len(to_add), title)

        try:
            existing.reload()
        except Exception:
            pass

        log.info("Updated collection '%s': now %d items.", title, len(list(existing.items())))
    except Exception as e:
        log.exception("Failed updating collection '%s': %s", title, e)
        raise

# ----------------------------
# LLM (optional) — per type
# ----------------------------
def llm_rerank_list(label: str,
                    candidates: List[Tuple[float,object]],
                    top_hi: List[object], top_lo: List[object],
                    weather: str, season: str):
    if not USE_LLM:
        log.info("[LLM] %s: Disabled by USE_LLM=false.", label)
        return candidates[:TOPK_RECS]
    if not OPENAI_API_KEY:
        log.warning("[LLM] %s: OPENAI_API_KEY not set; skipping.", label)
        return candidates[:TOPK_RECS]
    if not candidates:
        return candidates

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        send_n = min(50, len(candidates))
        log.info("[LLM] %s: Sending %d candidates to %s ...", label, send_n, OPENAI_MODEL)

        def item_line(o, s=None):
            genres = ",".join([g.tag for g in getattr(o, "genres", [])]) if getattr(o,"genres",None) else ""
            yr = getattr(o, "year", "")
            t  = getattr(o, "type", "")
            ttl= getattr(o, "title", "")
            if t == "episode":
                s_num = getattr(o, "seasonNumber", "?")
                e_num = getattr(o, "index", "?")
                ttl = f"{ttl} S{s_num:0>2}E{e_num:0>2}"
            return f"{ttl} ({yr}) [{t}] g={genres} baseScore={s if s is not None else ''}"

        lines = "\n".join(item_line(o, sc) for sc,o in candidates[:send_n])
        hi = "\n".join(item_line(o) for o in top_hi[:15])
        lo = "\n".join(item_line(o) for o in top_lo[:15])
        prompt = f"""You are picking the best {label.lower()} for tonight.
Weather: {weather}; Season: {season}.
I value: my own ratings first, then how long since last watch (older is better), then season/weather fit.
Here are candidates with a pre-score:
{lines}

Context: my 15 highest-rated {label.lower()}:
{hi}

Context: my 15 lowest-rated {label.lower()}:
{lo}

Return a JSON array of the top {TOPK_RECS} titles from the candidates (use exact titles), in ranked order.
"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        text = resp.choices[0].message.content.strip()
        log.info("[LLM] %s: Response received.", label)

        import re as _re
        import json as _json
        arr = _json.loads(_re.search(r"\[.*\]", text, _re.S).group(0))

        by_title = {getattr(o,"title",""): (sc,o) for sc,o in candidates}
        ordered = []
        for t in arr:
            pair = by_title.get(t)
            if pair:
                ordered.append(pair)

        if len(ordered) < TOPK_RECS:
            seen = set(id(o) for _,o in ordered)
            for sc,o in candidates:
                if id(o) not in seen:
                    ordered.append((sc,o))
                if len(ordered) >= TOPK_RECS:
                    break
        return ordered[:TOPK_RECS]
    except Exception as e:
        log.exception("[LLM] %s: Error during re-rank: %s", label, e)
        return candidates[:TOPK_RECS]

def ensure_real_objects(pairs: List[Tuple[float, object]]) -> List[Tuple[float, object]]:
    out = []
    for sc, o in pairs:
        if isinstance(o, int):
            try:
                o = plex.fetchItem(int(o))
            except Exception:
                continue
        out.append((sc, o))
    return out

# ----------------------------
# main
# ----------------------------
def main():
    try:
        # Weather + season
        log.info("Fetching current weather (lat=%s, lon=%s)...", LAT, LON)
        weather_json = fetch_weather(LAT, LON)
        cw = weather_json.get("current_weather", {}) or weather_json.get("current", {})
        wbucket = weather_bucket(cw)
        wb = WEATHER_GENRE_BONUS.get(wbucket, WEATHER_GENRE_BONUS["neutral"])
        season = season_from_date(now_local().date())
        sb = SEASON_GENRE_BONUS.get(season, {})
        log.info("Weather bucket: %s | Season: %s", wbucket, season)

        # History: fetch ALL within LOOKBACK, then filter by account, then limit
        cutoff = months_ago_ts(LOOKBACK_MONTHS)
        raw_history = get_history_until_cutoff(max_fetch=MAX_HISTORY, cutoff_ts=cutoff)
        ac_counts = Counter(str(x.get("accountID","")) for x in raw_history)
        log.info("History (pre-filter) accountIDs seen: %s", dict(ac_counts))
        if PLEX_ACCOUNT_ID:
            log.info("Configured PLEX_ACCOUNT_ID=%s", PLEX_ACCOUNT_ID)
        history = filter_history_by_account(raw_history, PLEX_ACCOUNT_ID)[:MAX_HISTORY]

        # Build last-seen maps from filtered history
        rk_map, guid_map, title_map = build_last_seen_maps(history)

        # Highest/lowest samplings per TYPE (from your filtered history)
        movies_hist = []
        eps_hist = []
        for rk in set(int(h.get("ratingKey")) for h in history if h.get("ratingKey")):
            try:
                it = plex.fetchItem(int(rk))
            except Exception:
                continue
            if getattr(it, "type", "") == "movie":
                movies_hist.append(it)
            elif getattr(it, "type", "") == "episode":
                eps_hist.append(it)

        def rated_sorted(items):
            pairs = [(fetch_user_rating(i) or -1, i) for i in items]
            pairs = [p for p in pairs if p[0] >= 0]
            pairs.sort(key=lambda x: x[0], reverse=True)
            top15 = [it for _,it in pairs[:15]]
            bot15 = [it for _,it in sorted(pairs, key=lambda x: x[0])[:15]]
            return top15, bot15

        top15_movies, bot15_movies = rated_sorted(movies_hist)
        top15_eps,    bot15_eps    = rated_sorted(eps_hist)

        # Inventory from cache (fast), filtered by cutoff using last-seen maps
        movie_candidates, ep_candidates = build_inventory_with_cache(
            LIBRARY_SECTIONS, rk_map, guid_map, title_map, cutoff
        )

        # Score separately
        log.info("Scoring %d movie candidates and %d episode candidates...", len(movie_candidates), len(ep_candidates))
        t0 = time.time()
        scored_movies: List[Tuple[float, object]] = [(score_item(it, rk_map, guid_map, title_map, wb, sb), it) for it in movie_candidates]
        scored_eps:    List[Tuple[float, object]] = [(score_item(it, rk_map, guid_map, title_map, wb, sb), it) for it in ep_candidates]
        scored_movies.sort(key=lambda x: x[0], reverse=True)
        scored_eps.sort(key=lambda x: x[0], reverse=True)
        log.info("Scoring complete (elapsed: %.2fs).", time.time() - t0)

        # Top K per type + optional LLM per type
        top_movies_initial = scored_movies[:max(TOPK_RECS, 10)]
        top_eps_initial    = scored_eps[:max(TOPK_RECS, 10)]

        final_movies = llm_rerank_list("Movies", top_movies_initial, top15_movies, bot15_movies, wbucket, season)
        final_eps    = llm_rerank_list("TV Episodes", top_eps_initial, top15_eps, bot15_eps, wbucket, season)

        # Ensure fetched objects for final lists (lazy ints → objects)
        final_movies = ensure_real_objects(final_movies)
        final_eps    = ensure_real_objects(final_eps)

        # Log outputs
        print("\n=== Final Recommendations — Movies (Top 10) ===")
        for sc, o in final_movies:
            lv = last_seen_timestamp(o, rk_map, guid_map, title_map)
            last_seen_str = dt.datetime.fromtimestamp(lv).strftime("%Y-%m-%d") if lv else "never"
            genres = ",".join(_extract_genres_with_fallback(o))
            print(f"- {getattr(o,'title','?')} ({getattr(o,'year','')}) [Movie] score={sc:.2f} last_seen={last_seen_str} genres={genres}")

        print("\n=== Final Recommendations — TV Episodes (Top 10) ===")
        for sc, o in final_eps:
            lv = last_seen_timestamp(o, rk_map, guid_map, title_map)
            last_seen_str = dt.datetime.fromtimestamp(lv).strftime("%Y-%m-%d") if lv else "never"
            genres = ",".join(_extract_genres_with_fallback(o))
            sea = getattr(o, "seasonNumber", "?")
            epi = getattr(o, "index", "?")
            print(f"- {getattr(o,'title','?')} ({getattr(o,'year','')}) [S{sea:0>2}E{epi:0>2}] score={sc:.2f} last_seen={last_seen_str} genres={genres}")

        # Context samplings (separate by type)
        print("\n=== Highest rated Movies (top 15) ===")
        for o in top15_movies:
            print(f"- {o.title} ({getattr(o,'year','')}) | userRating={fetch_user_rating(o)}")
        print("\n=== Lowest rated Movies (bottom 15) ===")
        for o in bot15_movies:
            print(f"- {o.title} ({getattr(o,'year','')}) | userRating={fetch_user_rating(o)}")

        print("\n=== Highest rated TV Episodes (top 15) ===")
        for o in top15_eps:
            sea = getattr(o, "seasonNumber", "?")
            epi = getattr(o, "index", "?")
            print(f"- {o.title} S{sea:0>2}E{epi:0>2} | userRating={fetch_user_rating(o)}")
        print("\n=== Lowest rated TV Episodes (bottom 15) ===")
        for o in bot15_eps:
            sea = getattr(o, "seasonNumber", "?")
            epi = getattr(o, "index", "?")
            print(f"- {o.title} S{sea:0>2}E{epi:0>2} | userRating={fetch_user_rating(o)}")

        # Find libraries
        movie_section_name = None
        tv_section_name = None
        for name in LIBRARY_SECTIONS:
            try:
                section = plex.library.section(name)
            except Exception:
                continue
            stype = getattr(section, "type", None)
            if stype == "movie" and movie_section_name is None:
                movie_section_name = name
            if stype in ("show", "episode", "tv", "show-mixed") and tv_section_name is None:
                tv_section_name = name

        # Upsert collections: both types
        if movie_section_name:
            ensure_collection_update(movie_section_name, f"{COLLECTION_BASE} (Movies)", [o for _,o in final_movies])
        else:
            log.warning("No Movie library found among LIBRARY_SECTIONS=%s; skipping Movies collection.", LIBRARY_SECTIONS)

        if tv_section_name:
            ensure_collection_update(tv_section_name, f"{COLLECTION_BASE} (TV)", [o for _,o in final_eps])
        else:
            log.warning("No TV library found among LIBRARY_SECTIONS=%s; skipping TV collection.", LIBRARY_SECTIONS)

        log.info("Done.")
    except Exception as e:
        log.exception("Fatal error in main(): %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()