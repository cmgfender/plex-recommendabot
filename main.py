#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import hashlib
import logging
import datetime as dt
import random
from collections import Counter, defaultdict
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
LLM_SEND_N       = int(os.getenv("LLM_SEND_N", "120"))   # larger candidate sampling for LLM

COLLECTION_BASE  = os.getenv("COLLECTION_BASE", "Recommended Today")
LOOKBACK_MONTHS  = int(os.getenv("LOOKBACK_MONTHS", "7"))
MAX_HISTORY      = int(os.getenv("MAX_HISTORY", "1000"))
TOPK_RECS        = int(os.getenv("TOPK_RECS", "10"))
DRY_RUN          = os.getenv("DRY_RUN", "false").lower() == "true"

# Re-recommend decay settings
MIN_REREC_DAYS      = int(os.getenv("MIN_REREC_DAYS", "14"))
FULL_RECOVERY_DAYS  = int(os.getenv("FULL_RECOVERY_DAYS", "45"))

# Episode cap = max EPISODES PER SHOW (not number of shows)
EPISODE_SHOW_CAP    = int(os.getenv("EPISODE_SHOW_CAP", "2"))

# Caching
CACHE_DIR              = os.path.join(BASE_DIR, ".cache")
CACHE_TTL_MINUTES      = int(os.getenv("CACHE_TTL_MINUTES", "360"))  # 6h default
REFRESH_CACHE          = os.getenv("REFRESH_CACHE", "false").lower() == "true"
CACHE_KEY_PREFIX       = "inventory_v1"  # bump if you change structures
ITEM_CACHE_PATH        = os.path.join(CACHE_DIR, "item_cache_v1.json")
RECOMMEND_HISTORY_PATH = os.path.join(CACHE_DIR, "recommend_history_v1.json")

# ----------------------------
# Startup config dump
# ----------------------------
log.info("[CFG] PLEX_URL=%s", PLEX_URL)
log.info("[CFG] LIBRARY_SECTIONS=%s", LIBRARY_SECTIONS)
log.info("[CFG] LOOKBACK_MONTHS=%s MAX_HISTORY=%s TOPK_RECS=%s DRY_RUN=%s",
         LOOKBACK_MONTHS, MAX_HISTORY, TOPK_RECS, DRY_RUN)
log.info("[LLM] enabled=%s | key=%s | model=%s | send_n=%s",
         USE_LLM, "yes" if OPENAI_API_KEY else "no", OPENAI_MODEL, LLM_SEND_N)
log.info("[DECAY] min_days=%s full_recovery_days=%s | EPISODE_SHOW_CAP=%s",
         MIN_REREC_DAYS, FULL_RECOVERY_DAYS, EPISODE_SHOW_CAP)
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
               weather_b: Dict[str,float], season_b: Dict[str,float]) -> Tuple[float, Dict[str, float]]:
    """Return (score, breakdown) so we can explain to the LLM."""
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
                show_bonus = (s_rating/10.0) * 0.5
        except Exception:
            pass
    total = base + my_rating_term + rec + mood + show_bonus
    breakdown = {
        "base": base,
        "my_rating_term": my_rating_term,
        "recency_term": rec,
        "mood_term": mood,
        "show_bonus": show_bonus,
        "days_since_watch": (days if days is not None else -1),
        "user_rating": (ur if ur is not None else -1),
    }
    return total, breakdown

# ----------------------------
# Recommendation history cache (decay)
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

def load_rechist() -> Dict[str, dict]:
    try:
        return load_json(RECOMMEND_HISTORY_PATH)
    except Exception:
        return {}

def save_rechist(hist: Dict[str, dict]):
    ensure_cache_dir()
    save_json(RECOMMEND_HISTORY_PATH, hist)

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

def decay_penalty(now_ts: int, last_rec_ts: int) -> Tuple[bool, float]:
    """
    Returns (blocked, penalty_factor). blocked=True if inside MIN_REREC_DAYS.
    Otherwise returns a 0..0.5 penalty (subtract from score) decaying to 0 by FULL_RECOVERY_DAYS.
    """
    delta_days = (now_ts - last_rec_ts) / 86400.0
    if delta_days < MIN_REREC_DAYS:
        return True, 0.5  # blocked window
    if delta_days >= FULL_RECOVERY_DAYS:
        return False, 0.0
    f = 1.0 - (delta_days - MIN_REREC_DAYS) / max(1.0, (FULL_RECOVERY_DAYS - MIN_REREC_DAYS))
    return False, 0.5 * max(0.0, min(1.0, f))

def adjust_for_rerec(score: float, item, _rk_map_unused: Dict[str,int], rechist: Dict[str, dict]) -> Tuple[float, bool]:
    """Apply decay/blocks if the item was recently recommended but not watched yet."""
    rk = str(getattr(item, "ratingKey", "") or "")
    if not rk:
        return score, False
    rec = rechist.get(rk)
    if not rec:
        return score, False

    last_rec_ts = rec.get("last_recommended_ts", 0)
    if not last_rec_ts:
        return score, False

    # If watched after recommended → clear history entry
    lv = None
    try:
        lva = getattr(item, "lastViewedAt", None)
        if lva:
            lv = int(lva.timestamp())
    except Exception:
        pass
    if lv and lv > last_rec_ts:
        rechist.pop(rk, None)
        return score, False

    blocked, pen = decay_penalty(int(time.time()), last_rec_ts)
    if blocked:
        return -1e9, True  # effectively exclude
    return score - pen, False

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
# Top/Bottom context from full library (no overlap)
# ----------------------------
def _iter_all_items_for_type(sections: List[str], kind: str) -> List[object]:
    """All items of a type from library (no cutoff)."""
    out = []
    for name in sections:
        try:
            section = plex.library.section(name)
        except Exception:
            continue
        stype = getattr(section, "type", None)
        if kind == "movie" and stype == "movie":
            try:
                out.extend([m for m in section.all() if getattr(m, "type", "") == "movie"])
            except Exception:
                pass
        elif kind == "episode" and stype in ("show", "episode", "tv", "show-mixed"):
            try:
                for sh in section.all():
                    try:
                        out.extend([ep for ep in sh.episodes() if getattr(ep, "type", "") == "episode"])
                    except Exception:
                        continue
            except Exception:
                pass
    return out

def build_rating_context_all(sections: List[str], kind: str, top_n: int = 15) -> Tuple[List[object], List[object]]:
    """
    True top/bottom lists (no overlap) from entire library by userRating.
    'kind' is 'movie' or 'episode'.
    """
    items = _iter_all_items_for_type(sections, kind)
    rated = [(getattr(i, "userRating", None) or -1, i) for i in items]
    rated = [p for p in rated if p[0] >= 0]  # only items you rated
    if not rated:
        return [], []

    # Highest — bias to 9–10 if available
    rated.sort(key=lambda x: x[0], reverse=True)
    highest = []
    seen = set()
    for r, it in rated:
        rk = getattr(it, "ratingKey", None)
        if rk in seen:
            continue
        highest.append(it)
        seen.add(rk)
        if len(highest) >= top_n:
            break

    high9 = [it for it in highest if (getattr(it, "userRating", 0) or 0) >= 9]
    if len(high9) >= 6:
        highest = high9[:min(len(high9), top_n)]

    # Lowest — exclude anything already in highest
    rated_lo = sorted(rated, key=lambda x: x[0])  # ascending
    lowest = []
    for r, it in rated_lo:
        rk = getattr(it, "ratingKey", None)
        if rk in seen:
            continue
        lowest.append(it)
        if len(lowest) >= top_n:
            break

    return highest, lowest

# ----------------------------
# LLM helpers
# ----------------------------
def _candidate_json_tuple(score: float, breakdown: Dict[str, float], obj) -> Dict:
    t = getattr(obj, "type", "")
    title = getattr(obj, "title", "")
    year = getattr(obj, "year", "")
    rk = getattr(obj, "ratingKey", None)
    genres = _extract_genres_with_fallback(obj)
    show_title = ""
    sea = None
    epi = None
    show_rating = None
    try:
        if t == "episode":
            show_title = getattr(obj, "grandparentTitle", "") or ""
            sea = getattr(obj, "seasonNumber", None)
            epi = getattr(obj, "index", None)
            sh = obj.show()
            if sh:
                show_rating = getattr(sh, "userRating", None)
    except Exception:
        pass
    return {
        "ratingKey": rk,
        "title": title,
        "year": year,
        "type": t,
        "genres": genres,
        "user_rating": breakdown.get("user_rating", -1),
        "days_since_watch": breakdown.get("days_since_watch", -1),
        "score_components": {
            "base": breakdown.get("base", 0.0),
            "my_rating_term": breakdown.get("my_rating_term", 0.0),
            "recency_term": breakdown.get("recency_term", 0.0),
            "mood_term": breakdown.get("mood_term", 0.0),
            "show_bonus": breakdown.get("show_bonus", 0.0),
        },
        "total_score": score,
        "episode_info": {
            "show_title": show_title,
            "season": sea,
            "episode": epi,
            "show_user_rating": show_rating
        }
    }

def stratified_pool(scored: List[Tuple[float, object]], send_n: int) -> List[Tuple[float, object]]:
    """
    Build a diverse pool: top / middle / long-tail slices.
    Deterministic-ish via date-based seed to vary day-to-day.
    """
    n = len(scored)
    if n <= send_n:
        return scored[:]

    today_seed = int(dt.datetime.now().strftime("%Y%m%d"))
    random.seed(today_seed)

    top_n = max(10, int(send_n * 0.4))
    mid_n = max(10, int(send_n * 0.35))
    tail_n = max(10, send_n - top_n - mid_n)

    top_slice = scored[:max(top_n*2, top_n+20)]
    mid_start = n // 3
    mid_end = 2 * n // 3
    mid_slice = scored[mid_start:mid_end]
    tail_slice = scored[-max(tail_n*6, tail_n+60):]

    def pick(sample, k):
        if len(sample) <= k:
            return sample[:]
        idxs = sorted(random.sample(range(len(sample)), k))
        return [sample[i] for i in idxs]

    out = []
    out.extend(pick(top_slice, top_n))
    out.extend(pick(mid_slice, mid_n))
    out.extend(pick(tail_slice, tail_n))

    seen = set()
    dedup = []
    for sc, o in out:
        rk = getattr(o, "ratingKey", None)
        if rk in seen:
            continue
        seen.add(rk)
        dedup.append((sc, o))
    return dedup[:send_n]

def llm_rerank_list(label: str,
                    candidates: List[Tuple[float,object]],
                    top_hi: List[object], top_lo: List[object],
                    weather: str, season: str,
                    scored_breakdowns: Dict[int, Dict[str, float]],
                    rechist: Dict[str, dict]):
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

        # Build a stratified, decay-adjusted pool
        pool = stratified_pool(candidates, min(LLM_SEND_N, len(candidates)))

        send_objs = []
        kept_pairs = []
        blocked_count = 0
        for sc, o in pool:
            rk = getattr(o, "ratingKey", None)
            sc_b = scored_breakdowns.get(int(rk)) if rk is not None else None
            if sc_b is None:
                sc_b = {"base": 0.0, "my_rating_term": 0.0, "recency_term": 0.0, "mood_term": 0.0,
                        "show_bonus": 0.0, "days_since_watch": -1, "user_rating": -1}
            adj_sc, blocked = adjust_for_rerec(sc, o, {}, rechist)
            if blocked:
                blocked_count += 1
                continue
            kept_pairs.append((adj_sc, o))
            send_objs.append(_candidate_json_tuple(adj_sc, sc_b, o))

        if not kept_pairs:
            log.info("[LLM] %s: All candidates blocked by re-recency window; falling back to original candidates.", label)
            kept_pairs = candidates[:LLM_SEND_N]
            send_objs = []
            for sc, o in kept_pairs:
                rk = getattr(o, "ratingKey", None)
                sc_b = scored_breakdowns.get(int(rk)) if rk is not None else {}
                send_objs.append(_candidate_json_tuple(sc, sc_b, o))

        log.info("[LLM] %s: Sending %d candidates (blocked=%d) to %s ...", label, len(send_objs), blocked_count, OPENAI_MODEL)

        def simple_line(o):
            t = getattr(o, "type", "")
            ttl = getattr(o, "title", "")
            yr = getattr(o, "year", "")
            if t == "episode":
                sea = getattr(o, "seasonNumber", "?")
                epi = getattr(o, "index", "?")
                ttl = f"{ttl} S{int(sea):0>2}E{int(epi):0>2}" if isinstance(sea, int) and isinstance(epi, int) else f"{ttl}"
            return f"{ttl} ({yr}) [{t}]"

        hi = "\n".join(simple_line(o) for o in top_hi[:15])
        lo = "\n".join(simple_line(o) for o in top_lo[:15])

        prompt = {
            "task": f"Select the top {TOPK_RECS} {label.lower()} to recommend tonight.",
            "preferences": [
                "Prioritize my user rating first.",
                "Prefer things I haven't seen in a long time (more days since last watch is better).",
                "Season & weather genre fit should influence ties.",
                "A mix of obvious favorites and some variety is welcome."
            ],
            "context": {
                "weather_bucket": weather,
                "season": season,
                "highest_rated_examples": hi,
                "lowest_rated_examples": lo
            },
            "candidates": send_objs,
            "output_format": {
                "type": "json",
                "fields": ["ratingKey"],
                "array_name": "picks"
            },
            "instructions": [
                "Return JSON only. Example: {\"picks\": [12345, 67890, ...]}",
                f"Pick exactly {TOPK_RECS} unique ratingKeys, in order of recommendation strength."
            ]
        }

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a sharp, decisive recommender. Output strictly valid JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
            ],
            temperature=0.2
        )
        text = resp.choices[0].message.content.strip()

        import re as _re
        data = json.loads(_re.search(r"\{.*\}", text, _re.S).group(0))
        rk_list = data.get("picks", [])

        by_rk = {int(getattr(o, "ratingKey", 0)): (sc, o) for sc, o in kept_pairs if getattr(o, "ratingKey", None) is not None}

        ordered = []
        for rk in rk_list:
            pair = by_rk.get(int(rk))
            if pair:
                ordered.append(pair)

        if len(ordered) < TOPK_RECS:
            seen = set(getattr(o, "ratingKey", None) for _, o in ordered)
            for sc, o in kept_pairs:
                rk = getattr(o, "ratingKey", None)
                if rk not in seen:
                    ordered.append((sc, o))
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

# Enforce MAX EPISODES PER SHOW (not number of shows)
def enforce_episode_show_cap(pairs: List[Tuple[float, object]], cap: int) -> List[Tuple[float, object]]:
    if cap <= 0:
        return pairs
    counts = defaultdict(int)
    out = []
    for sc, o in pairs:
        if getattr(o, "type", "") != "episode":
            out.append((sc, o))
            continue
        show = getattr(o, "grandparentTitle", "") or ""
        if counts[show] < cap:
            out.append((sc, o))
            counts[show] += 1
    return out

# ----------------------------
# Backfill to guaranteed minimums
# ----------------------------
def _not_in_selected(selected: List[Tuple[float, object]], cand_obj) -> bool:
    sel_ids = {getattr(o, "ratingKey", None) for _, o in selected}
    return getattr(cand_obj, "ratingKey", None) not in sel_ids

def _passes_show_cap(selected: List[Tuple[float, object]], cand_obj, cap: int) -> bool:
    if getattr(cand_obj, "type", "") != "episode" or cap <= 0:
        return True
    from collections import Counter
    counts = Counter(getattr(o, "grandparentTitle", "") or "" for _, o in selected if getattr(o, "type", "") == "episode")
    return counts[(getattr(cand_obj, "grandparentTitle", "") or "")] < cap

def backfill_to_min(
    label: str,
    selected: List[Tuple[float, object]],
    scored_all: List[Tuple[float, object]],
    desired_n: int,
    is_tv: bool = False,
    show_cap: int = 0,
    rechist: Optional[Dict[str, dict]] = None
) -> List[Tuple[float, object]]:
    """
    Ensure we reach desired_n items by walking the scored list and adding the next best
    that aren't selected yet; respect per-show episode cap and decay/blocks.
    """
    rechist = rechist or {}
    out = list(selected)
    for sc, o in scored_all:
        if len(out) >= desired_n:
            break
        adj_sc, blocked = adjust_for_rerec(sc, o, {}, rechist)
        if blocked:
            continue
        if not _not_in_selected(out, o):
            continue
        if is_tv and not _passes_show_cap(out, o, show_cap):
            continue
        out.append((adj_sc, o))
    if len(out) < desired_n:
        log.warning("[%s] Could not reach desired minimum %d (have %d).", label, desired_n, len(out))
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

        # Build true top/bottom contexts from full library (no overlap)
        top15_movies, bot15_movies = build_rating_context_all(LIBRARY_SECTIONS, "movie", top_n=15)
        top15_eps,    bot15_eps    = build_rating_context_all(LIBRARY_SECTIONS, "episode", top_n=15)

        # Inventory from cache (fast), filtered by cutoff using last-seen maps
        movie_candidates, ep_candidates = build_inventory_with_cache(
            LIBRARY_SECTIONS, rk_map, guid_map, title_map, cutoff
        )

        # Score separately (collect breakdowns for LLM)
        log.info("Scoring %d movie candidates and %d episode candidates...", len(movie_candidates), len(ep_candidates))
        t0 = time.time()
        movie_breakdowns: Dict[int, Dict[str, float]] = {}
        ep_breakdowns: Dict[int, Dict[str, float]] = {}
        scored_movies: List[Tuple[float, object]] = []
        scored_eps:    List[Tuple[float, object]] = []

        for it in movie_candidates:
            sc, br = score_item(it, rk_map, guid_map, title_map, wb, sb)
            rk = getattr(it, "ratingKey", None)
            if rk is not None:
                movie_breakdowns[int(rk)] = br
            scored_movies.append((sc, it))

        for it in ep_candidates:
            sc, br = score_item(it, rk_map, guid_map, title_map, wb, sb)
            rk = getattr(it, "ratingKey", None)
            if rk is not None:
                ep_breakdowns[int(rk)] = br
            scored_eps.append((sc, it))

        scored_movies.sort(key=lambda x: x[0], reverse=True)
        scored_eps.sort(key=lambda x: x[0], reverse=True)
        log.info("Scoring complete (elapsed: %.2fs).", time.time() - t0)

        # Load recommend history for decay
        rechist = load_rechist()

        # Oversample so LLM can choose
        top_movies_initial = scored_movies[:max(TOPK_RECS, 10)*20]
        top_eps_initial    = scored_eps[:max(TOPK_RECS, 10)*20]

        # LLM rerank per type
        final_movies = llm_rerank_list("Movies", top_movies_initial, top15_movies, bot15_movies,
                                       wbucket, season, movie_breakdowns, rechist)
        final_eps    = llm_rerank_list("TV Episodes", top_eps_initial, top15_eps, bot15_eps,
                                       wbucket, season, ep_breakdowns, rechist)

        # Ensure fetched objects, enforce episode-per-show cap, then backfill to minimums
        final_movies = ensure_real_objects(final_movies)
        final_eps    = ensure_real_objects(final_eps)
        final_eps    = enforce_episode_show_cap(final_eps, EPISODE_SHOW_CAP)

        # Backfill to guarantee TOPK_RECS for both collections
        final_movies = backfill_to_min(
            "Movies",
            final_movies,
            scored_movies,
            TOPK_RECS,
            is_tv=False,
            show_cap=0,
            rechist=rechist
        )
        final_eps = backfill_to_min(
            "TV Episodes",
            final_eps,
            scored_eps,
            TOPK_RECS,
            is_tv=True,
            show_cap=EPISODE_SHOW_CAP,
            rechist=rechist
        )

        # Record recommendations in history (for decay)
        now_ts = int(time.time())
        for _, o in (final_movies + final_eps):
            rk = str(getattr(o, "ratingKey", "") or "")
            if not rk:
                continue
            rec = rechist.get(rk, {})
            rec["last_recommended_ts"] = now_ts
            rec["times"] = int(rec.get("times", 0)) + 1
            rechist[rk] = rec
        save_rechist(rechist)

        # Log outputs
        print("\n=== Final Recommendations — Movies (Top 10) ===")
        for sc, o in final_movies[:TOPK_RECS]:
            lv = last_seen_timestamp(o, rk_map, guid_map, title_map)
            last_seen_str = dt.datetime.fromtimestamp(lv).strftime("%Y-%m-%d") if lv else "never"
            genres = ",".join(_extract_genres_with_fallback(o))
            print(f"- {getattr(o,'title','?')} ({getattr(o,'year','')}) [Movie] score={sc:.2f} last_seen={last_seen_str} genres={genres}")

        print(f"\n=== Final Recommendations — TV Episodes (Top 10, max {EPISODE_SHOW_CAP} episodes per show) ===")
        for sc, o in final_eps[:TOPK_RECS]:
            lv = last_seen_timestamp(o, rk_map, guid_map, title_map)
            last_seen_str = dt.datetime.fromtimestamp(lv).strftime("%Y-%m-%d") if lv else "never"
            genres = ",".join(_extract_genres_with_fallback(o))
            sea = getattr(o, "seasonNumber", "?")
            epi = getattr(o, "index", "?")
            show = getattr(o, "grandparentTitle", "") or ""
            print(f"- {show} — {getattr(o,'title','?')} [S{sea:0>2}E{epi:0>2}] score={sc:.2f} last_seen={last_seen_str} genres={genres}")

        # Context samplings (true highs/lows)
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
            ensure_collection_update(movie_section_name, f"{COLLECTION_BASE} (Movies)", [o for _,o in final_movies[:TOPK_RECS]])
        else:
            log.warning("No Movie library found among LIBRARY_SECTIONS=%s; skipping Movies collection.", LIBRARY_SECTIONS)

        if tv_section_name:
            ensure_collection_update(tv_section_name, f"{COLLECTION_BASE} (TV)", [o for _,o in final_eps[:TOPK_RECS]])
        else:
            log.warning("No TV library found among LIBRARY_SECTIONS=%s; skipping TV collection.", LIBRARY_SECTIONS)

        log.info("Done.")
    except Exception as e:
        log.exception("Fatal error in main(): %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()