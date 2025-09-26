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
PLEX_URL           = os.getenv("PLEX_URL", "http://127.0.0.1:32400")
PLEX_TOKEN         = os.getenv("PLEX_TOKEN", "")
PLEX_ACCOUNT_ID    = os.getenv("PLEX_ACCOUNT_ID", "")

LIBRARY_SECTIONS = [s.strip() for s in os.getenv("LIBRARY_SECTIONS", "Movies,TV Shows").split(",")]

LAT                = float(os.getenv("LAT", "39.9526"))
LON                = float(os.getenv("LON", "-75.1652"))

USE_LLM            = os.getenv("USE_LLM", "false").lower() == "true"
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_SEND_N         = int(os.getenv("LLM_SEND_N", "120"))

COLLECTION_BASE    = os.getenv("COLLECTION_BASE", "Recommended Today")
LOOKBACK_MONTHS    = int(os.getenv("LOOKBACK_MONTHS", "7"))
MAX_HISTORY        = int(os.getenv("MAX_HISTORY", "1000"))
DRY_RUN            = os.getenv("DRY_RUN", "false").lower() == "true"

# MODIFICATION: Changed TOPK_RECS to separate values for movies and episodes
TOPK_RECS_MOVIES   = int(os.getenv("TOPK_RECS_MOVIES", "10"))
TOPK_RECS_EPISODES = int(os.getenv("TOPK_RECS_EPISODES", "10"))

# Re-recommend decay settings
MIN_REREC_DAYS      = int(os.getenv("MIN_REREC_DAYS", "14"))
FULL_RECOVERY_DAYS  = int(os.getenv("FULL_RECOVERY_DAYS", "45"))

# Episode cap = max EPISODES PER SHOW
EPISODE_SHOW_CAP    = int(os.getenv("EPISODE_SHOW_CAP", "2"))

# Hard limits for never-watched items (0+). Default 1 each.
MAX_NEVER_WATCHED_MOVIES   = int(os.getenv("MAX_NEVER_WATCHED_MOVIES", "1"))
MAX_NEVER_WATCHED_EPISODES = int(os.getenv("MAX_NEVER_WATCHED_EPISODES", "1"))

# Performance knobs: prefilter how many items to fully fetch per run
PREFETCH_LIMIT_MOVIES   = int(os.getenv("PREFETCH_LIMIT_MOVIES", "1500"))
PREFETCH_LIMIT_EPISODES = int(os.getenv("PREFETCH_LIMIT_EPISODES", "2500"))

# Caching
CACHE_DIR              = os.path.join(BASE_DIR, ".cache")
CACHE_TTL_MINUTES      = int(os.getenv("CACHE_TTL_MINUTES", "360"))
REFRESH_CACHE          = os.getenv("REFRESH_CACHE", "false").lower() == "true"
CACHE_KEY_PREFIX       = "inventory_v1"
ITEM_CACHE_PATH        = os.path.join(CACHE_DIR, "item_cache_v2.json")  # v2 adds guids/titleKey
RECOMMEND_HISTORY_PATH = os.path.join(CACHE_DIR, "recommend_history_v1.json")

# ----------------------------
# Startup config dump
# ----------------------------
log.info("[CFG] PLEX_URL=%s", PLEX_URL)
log.info("[CFG] LIBRARY_SECTIONS=%s", LIBRARY_SECTIONS)
# MODIFICATION: Updated log message to show new separate min recommendation values
log.info("[CFG] LOOKBACK_MONTHS=%s MAX_HISTORY=%s MIN_RECS(M/TV)=%s/%s DRY_RUN=%s",
         LOOKBACK_MONTHS, MAX_HISTORY, TOPK_RECS_MOVIES, TOPK_RECS_EPISODES, DRY_RUN)
log.info("[LLM] enabled=%s | key=%s | model=%s | send_n=%s",
         USE_LLM, "yes" if OPENAI_API_KEY else "no", OPENAI_MODEL, LLM_SEND_N)
log.info("[DECAY] min_days=%s full_recovery_days=%s | EPISODE_SHOW_CAP=%s",
         MIN_REREC_DAYS, FULL_RECOVERY_DAYS, EPISODE_SHOW_CAP)
log.info("[NEVER] max_unwatched_movies=%s max_unwatched_eps=%s",
         MAX_NEVER_WATCHED_MOVIES, MAX_NEVER_WATCHED_EPISODES)
log.info("[PREFETCH] movies=%s episodes=%s", PREFETCH_LIMIT_MOVIES, PREFETCH_LIMIT_EPISODES)
log.info("[CACHE] dir=%s ttl_min=%s refresh=%s", CACHE_DIR, CACHE_TTL_MINUTES, REFRESH_CACHE)

if not PLEX_TOKEN:
    log.error("Missing PLEX_TOKEN. Set it in .env or environment.")
    sys.exit(1)

# ... (The rest of the script from line 102 to 1007 remains unchanged) ...
# I will skip the unchanged part for brevity and pick up where the next change is needed.

# ----------------------------
# LLM helpers
# ----------------------------
# ... (The _candidate_json_tuple and stratified_pool functions are unchanged) ...

# MODIFICATION: The function now accepts a 'top_k' parameter
def llm_rerank_list(label: str,
                    candidates: List[Tuple[float,object]],
                    top_hi: List[object], top_lo: List[object],
                    weather: str, season: str,
                    scored_breakdowns: Dict[int, Dict[str, float]],
                    rechist: Dict[str, dict],
                    top_k: int):
    if not USE_LLM:
        log.info("[LLM] %s: Disabled by USE_LLM=false.", label)
        # MODIFICATION: Use the 'top_k' parameter here
        return candidates[:top_k]
    if not OPENAI_API_KEY:
        log.warning("[LLM] %s: OPENAI_API_KEY not set; skipping.", label)
        # MODIFICATION: Use the 'top_k' parameter here
        return candidates[:top_k]
    if not candidates:
        return candidates
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        pool = stratified_pool(candidates, min(LLM_SEND_N, len(candidates)))
        send_objs = []
        kept_pairs = []
        blocked_count = 0
        for sc, o in pool:
            rk = getattr(o, "ratingKey", None)
            sc_b = scored_breakdowns.get(int(rk)) if rk is not None else None
            if sc_b is None:
                sc_b = {"base":0.0,"my_rating_term":0.0,"recency_term":0.0,"mood_term":0.0,"show_bonus":0.0,
                        "days_since_watch":-1,"user_rating":-1}
            adj_sc, blocked = adjust_for_rerec(sc, o, {}, rechist)
            if blocked:
                blocked_count += 1
                continue
            kept_pairs.append((adj_sc, o))
            send_objs.append(_candidate_json_tuple(adj_sc, sc_b, o))

        if not kept_pairs:
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
            # MODIFICATION: Use the 'top_k' parameter in the prompt
            "task": f"Select the top {top_k} {label.lower()} to recommend tonight.",
            "preferences": [
                "Prioritize my user rating first.",
                "Prefer items not watched recently.",
                "Season & weather genre fit should influence ties.",
                "Keep a balance of obvious favorites and some variety."
            ],
            "context": {
                "weather_bucket": weather,
                "season": season,
                "highest_rated_examples": hi,
                "lowest_rated_examples": lo
            },
            "candidates": send_objs,
            "output_format": {"type":"json","fields":["ratingKey"],"array_name":"picks"},
            "instructions": [
                f"Return JSON only. Example: {{\"picks\": [12345, 67890, ...]}}",
                # MODIFICATION: Use the 'top_k' parameter in the prompt instructions
                f"Pick exactly {top_k} unique ratingKeys, in order."
            ]
        }

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You are a sharp, decisive recommender. Output strictly valid JSON only."},
                {"role":"user","content": json.dumps(prompt, ensure_ascii=False)}
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
        
        # MODIFICATION: Use 'top_k' for backfilling
        if len(ordered) < top_k:
            seen = set(getattr(o, "ratingKey", None) for _, o in ordered)
            for sc, o in kept_pairs:
                rk = getattr(o, "ratingKey", None)
                if rk not in seen:
                    ordered.append((sc, o))
                if len(ordered) >= top_k:
                    break
        
        # MODIFICATION: Use 'top_k' for slicing
        return ordered[:top_k]
    except Exception as e:
        log.exception("[LLM] %s: Error during re-rank: %s", label, e)
        # MODIFICATION: Use 'top_k' as the fallback length
        return candidates[:top_k]

# ... (The rest of the script from line 1148 to 1344 remains unchanged) ...

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

        # History
        cutoff = months_ago_ts(LOOKBACK_MONTHS)
        raw_history = get_history_until_cutoff(max_fetch=MAX_HISTORY, cutoff_ts=cutoff)
        ac_counts = Counter(str(x.get("accountID","")) for x in raw_history)
        log.info("History (pre-filter) accountIDs seen: %s", dict(ac_counts))
        if PLEX_ACCOUNT_ID:
            log.info("Configured PLEX_ACCOUNT_ID=%s", PLEX_ACCOUNT_ID)
        history = filter_history_by_account(raw_history, PLEX_ACCOUNT_ID)[:MAX_HISTORY]

        rk_map, guid_map, title_map = build_last_seen_maps(history)

        # True top/bottom contexts
        top15_movies, bot15_movies = build_rating_context_all(LIBRARY_SECTIONS, "movie", top_n=15)
        top15_eps,    bot15_eps    = build_rating_context_all(LIBRARY_SECTIONS, "episode", top_n=15)

        # Inventory + PREFILTER fetch
        movie_candidates, ep_candidates = build_inventory_with_cache_and_prefilter(
            LIBRARY_SECTIONS, rk_map, guid_map, title_map, cutoff, wb, sb,
            PREFETCH_LIMIT_MOVIES, PREFETCH_LIMIT_EPISODES
        )

        # Score (collect breakdowns)
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

        # Recommend history
        rechist = load_rechist()

        # Oversample so LLM can choose
        # MODIFICATION: Use the new variables for oversampling logic
        top_movies_initial = scored_movies[:max(TOPK_RECS_MOVIES, 10)*20]
        top_eps_initial    = scored_eps[:max(TOPK_RECS_EPISODES, 10)*20]

        # LLM rerank
        # MODIFICATION: Pass the new TOPK_RECS_MOVIES variable
        final_movies = llm_rerank_list("Movies", top_movies_initial, top15_movies, bot15_movies,
                                       wbucket, season, movie_breakdowns, rechist,
                                       top_k=TOPK_RECS_MOVIES)
        # MODIFICATION: Pass the new TOPK_RECS_EPISODES variable
        final_eps    = llm_rerank_list("TV Episodes", top_eps_initial, top15_eps, bot15_eps,
                                       wbucket, season, ep_breakdowns, rechist,
                                       top_k=TOPK_RECS_EPISODES)

        # Ensure objects, per-show cap
        final_movies = ensure_real_objects(final_movies)
        final_eps    = ensure_real_objects(final_eps)
        final_eps    = enforce_episode_show_cap(final_eps, EPISODE_SHOW_CAP)

        # Enforce never-watched limits + backfill to minimums
        # MODIFICATION: Use TOPK_RECS_MOVIES for the desired number
        final_movies = limit_never_watched_and_backfill(
            "Movies",
            final_movies,
            scored_movies,
            TOPK_RECS_MOVIES,
            MAX_NEVER_WATCHED_MOVIES,
            rk_map, guid_map, title_map,
            is_tv=False,
            show_cap=0,
            rechist=rechist
        )
        # MODIFICATION: Use TOPK_RECS_EPISODES for the desired number
        final_eps = limit_never_watched_and_backfill(
            "TV Episodes",
            final_eps,
            scored_eps,
            TOPK_RECS_EPISODES,
            MAX_NEVER_WATCHED_EPISODES,
            rk_map, guid_map, title_map,
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
        # MODIFICATION: Update logging to use the new variables
        print(f"\n=== Final Recommendations — Movies (Top {TOPK_RECS_MOVIES}) ===")
        for sc, o in final_movies[:TOPK_RECS_MOVIES]:
            lv = last_seen_timestamp(o, rk_map, guid_map, title_map)
            last_seen_str = dt.datetime.fromtimestamp(lv).strftime("%Y-%m-%d") if lv else "never"
            genres = ",".join(_extract_genres_with_fallback(o))
            print(f"- {getattr(o,'title','?')} ({getattr(o,'year','')}) [Movie] score={sc:.2f} last_seen={last_seen_str} genres={genres}")

        print(f"\n=== Final Recommendations — TV Episodes (Top {TOPK_RECS_EPISODES}, max {EPISODE_SHOW_CAP} episodes per show) ===")
        for sc, o in final_eps[:TOPK_RECS_EPISODES]:
            lv = last_seen_timestamp(o, rk_map, guid_map, title_map)
            last_seen_str = dt.datetime.fromtimestamp(lv).strftime("%Y-%m-%d") if lv else "never"
            genres = ",".join(_extract_genres_with_fallback(o))
            sea = getattr(o, "seasonNumber", "?")
            epi = getattr(o, "index", "?")
            show = getattr(o, "grandparentTitle", "") or ""
            print(f"- {show} — {getattr(o,'title','?')} [S{sea:0>2}E{epi:0>2}] score={sc:.2f} last_seen={last_seen_str} genres={genres}")

        # Context samplings
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

        # Upsert collections
        if movie_section_name:
            # MODIFICATION: Use TOPK_RECS_MOVIES for slicing the final list
            ensure_collection_update(movie_section_name, f"{COLLECTION_BASE} (Movies)", [o for _,o in final_movies[:TOPK_RECS_MOVIES]])
        else:
            log.warning("No Movie library found among LIBRARY_SECTIONS=%s; skipping Movies collection.", LIBRARY_SECTIONS)

        if tv_section_name:
            # MODIFICATION: Use TOPK_RECS_EPISODES for slicing the final list
            ensure_collection_update(tv_section_name, f"{COLLECTION_BASE} (TV)", [o for _,o in final_eps[:TOPK_RECS_EPISODES]])
        else:
            log.warning("No TV library found among LIBRARY_SECTIONS=%s; skipping TV collection.", LIBRARY_SECTIONS)

        log.info("Done.")
    except Exception as e:
        log.exception("Fatal error in main(): %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()