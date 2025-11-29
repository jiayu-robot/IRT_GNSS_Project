# adjust_los.py
# Post-processing utility to rebalance LOS:MP ratio in the "actual" observations.
# Matching is always done by (gps_time, id). We never synthesize new Obs; we only swap
# an existing "actual" Obs with its LOS/MP counterpart taken from obs_dir/obs_mp.

import copy
import random
# from math import isclose  # uncomment if you switch to the "isclose" time-matching variant

ADJUST_LOS_DEFAULT_CONFIG = {
    "PL_PREFIXES": ("PL",),

    # If PL target is not specified, follow the GNSS target when True; otherwise leave PL unchanged.
    "mirror_pl_target_if_none": False,

    # Source-picking strategies when multiple candidates exist:
    # - from obs_mp: "random" | "strongest" | "first"
    # - from obs_dir: "random" | "first"
    "mp_select_strategy": "random",
    "dir_select_strategy": "random",

    # Set to an int for reproducibility; use None to leave randomness uncontrolled.
    "random_seed": 42,
}


def _is_pl_obs(o, cfg):
    """Return True if the Obs is PL. Prefer flag_GNSS_PL; fall back to id prefix."""
    flag = getattr(o, "flag_GNSS_PL", None)
    if flag in (0, 1):
        return (flag == 1)
    oid = getattr(o, "id", "")
    return any(str(oid).startswith(p) for p in cfg["PL_PREFIXES"])


def _group_match(o, group_name, cfg):
    """Return True when the Obs belongs to the requested group: 'GNSS' or 'PL'."""
    is_pl = _is_pl_obs(o, cfg)
    return (group_name == "PL" and is_pl) or (group_name == "GNSS" and not is_pl)


def _calc_los_ratio(obs_dict, cfg, group_name):
    """Compute LOS statistics for the given group; return (los_count, total_count, ratio_pct)."""
    total = 0
    los = 0
    for t, lst in obs_dict.items():
        for o in lst:
            if not _group_match(o, group_name, cfg):
                continue
            total += 1
            los += int(getattr(o, "multipath", 0) == 0)
    ratio = (100.0 * los / total) if total > 0 else 0.0
    return los, total, ratio


def _collect_candidates(obs_dict, cfg, group_name, want):
    """
    Collect candidate indices from obs_dict for replacement.
    want='MP'  → pick current multipath==1 entries (for MP→LOS)
    want='LOS' → pick current multipath==0 entries (for LOS→MP)
    Return a list of (time_key, index_in_list).
    """
    items = []
    for t, lst in obs_dict.items():
        for j, o in enumerate(lst):
            if not _group_match(o, group_name, cfg):
                continue
            if want == "MP" and getattr(o, "multipath", 0) == 1:
                items.append((t, j))
            elif want == "LOS" and getattr(o, "multipath", 0) == 0:
                items.append((t, j))
    return items


def _find_dir_match(obs_dir, time_key, sid, cfg):
    """
    Select the LOS counterpart from obs_dir for the same (time, id).
    Time matching strategy (aligned with read_multipath_file in read_obs.py):
      A) choose the nearest time key (no explicit tolerance)  ← default
         tk = min(obs_dir.keys(), key=lambda k: abs(k - time_key))
    The alternative "isclose" variant is kept below as a commented one-liner.
    """
    if not obs_dir:
        return None

    # A) nearest time key (no explicit tolerance)
    tk = min(obs_dir.keys(), key=lambda k: abs(k - time_key))

    # B) alternative (commented): near-equality via isclose (relative tolerance)
    # tk = next((k for k in obs_dir.keys() if isclose(time_key, k, rel_tol=1e-9)), None)

    if tk is None:
        return None
    cands = [o for o in obs_dir[tk] if getattr(o, "id", None) == sid]
    if not cands:
        return None

    strat = cfg.get("dir_select_strategy", "random")
    if strat == "first" or len(cands) == 1:
        return cands[0]
    return random.choice(cands)


def _find_mp_match(obs_mp, time_key, sid, cfg):
    """
    Select an MP counterpart from obs_mp for the same (time, id).
    """
    if not obs_mp:
        return None

    #  nearest time key (no explicit tolerance)
    tk = min(obs_mp.keys(), key=lambda k: abs(k - time_key))

    # near-equality via isclose (relative tolerance)
    # tk = next((k for k in obs_mp.keys() if isclose(time_key, k, rel_tol=1e-9)), None)

    if tk is None:
        return None
    cands = [o for o in obs_mp[tk] if getattr(o, "id", None) == sid]
    if not cands:
        return None

    strat = cfg.get("mp_select_strategy", "random")
    if strat == "strongest":
        with_pow = [o for o in cands if getattr(o, "rec_pow", None) is not None]
        return max(with_pow, key=lambda x: x.rec_pow) if with_pow else cands[0]
    if strat == "first":
        return cands[0]
    return random.choice(cands)


def adjust_los_percentage(
    obs_dir,
    obs_mp,
    obs_real,
    GNSS_LOS_percentage,
    PL_LOS_percentage=None,
    config=None,
):
    """
    Rebalance LOS ratio for GNSS and PL independently by swapping "actual" Obs entries:

    If target LOS > current:
      randomly pick required number of MP entries (multipath==1) from obs_real and
      replace each with its LOS counterpart taken from obs_dir at the same (time, id).

    If target LOS < current:
      randomly pick required number of LOS entries (multipath==0) from obs_real and
      replace each with an MP counterpart taken from obs_mp at the same (time, id).

    This function returns:
      obs_percent : deep-copied and modified version of obs_real
      report      : summary dict with before/after ratios and replacement counts
    """
    cfg = copy.deepcopy(ADJUST_LOS_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    if cfg["random_seed"] is not None:
        random.seed(cfg["random_seed"])

    def _clamp_pct(x):
        try:
            x = float(x)
        except Exception:
            return None
        if x < 0.0:
            return 0.0
        if x > 100.0:
            return 100.0
        return x

    target_gnss = _clamp_pct(GNSS_LOS_percentage)
    target_pl = _clamp_pct(PL_LOS_percentage)
    if target_pl is None and cfg.get("mirror_pl_target_if_none", False):
        target_pl = target_gnss

    obs_percent = copy.deepcopy(obs_real)

    gnss_los_before, gnss_total, gnss_ratio_before = _calc_los_ratio(obs_percent, cfg, "GNSS")
    pl_los_before, pl_total, pl_ratio_before = _calc_los_ratio(obs_percent, cfg, "PL")

    def _align(group_name, target_ratio):
        """
        Adjust one group. Return (replaced_count, direction, ratio_after)
        direction = +1 for MP→LOS (increase LOS), -1 for LOS→MP (decrease LOS), 0 for no-op.
        """
        if target_ratio is None:
            _, _, now = _calc_los_ratio(obs_percent, cfg, group_name)
            return 0, 0, now

        los_now, total_now, ratio_now = _calc_los_ratio(obs_percent, cfg, group_name)
        if total_now == 0:
            return 0, 0, ratio_now

        delta = target_ratio - ratio_now
        need_cnt = int(round(abs(delta) * total_now / 100.0))
        if need_cnt <= 0:
            return 0, 0, ratio_now

        replaced = 0

        if delta > 0:
            # Increase LOS: MP → LOS
            cands = _collect_candidates(obs_percent, cfg, group_name, want="MP")
            random.shuffle(cands)  # layer-1 randomness: which entries to replace
            for (t, j) in cands:
                if replaced >= need_cnt:
                    break
                sid = getattr(obs_percent[t][j], "id", None)
                o_dir = _find_dir_match(obs_dir, t, sid, cfg)  # layer-2 randomness: source selection
                if o_dir is None:
                    continue
                obs_percent[t][j] = copy.deepcopy(o_dir)
                replaced += 1
            _, _, ratio_after = _calc_los_ratio(obs_percent, cfg, group_name)
            return replaced, +1, ratio_after

        else:
            # Decrease LOS: LOS → MP
            cands = _collect_candidates(obs_percent, cfg, group_name, want="LOS")
            random.shuffle(cands)  # layer-1 randomness
            for (t, j) in cands:
                if replaced >= need_cnt:
                    break
                sid = getattr(obs_percent[t][j], "id", None)
                o_mp = _find_mp_match(obs_mp, t, sid, cfg)  # layer-2 selection: random/strongest/first
                if o_mp is None:
                    continue
                obs_percent[t][j] = copy.deepcopy(o_mp)
                replaced += 1
            _, _, ratio_after = _calc_los_ratio(obs_percent, cfg, group_name)
            return replaced, -1, ratio_after

    done_gnss, dir_gnss, gnss_ratio_after = _align("GNSS", target_gnss)
    done_pl, dir_pl, pl_ratio_after = _align("PL", target_pl)

    report = {
        "GNSS": {
            "before": {"los": gnss_los_before, "total": gnss_total, "ratio_pct": gnss_ratio_before},
            "after": {"ratio_pct": gnss_ratio_after, "replaced_count": done_gnss, "direction": dir_gnss},
            "target_ratio_pct": target_gnss,
        },
        "PL": {
            "before": {"los": pl_los_before, "total": pl_total, "ratio_pct": pl_ratio_before},
            "after": {"ratio_pct": pl_ratio_after, "replaced_count": done_pl, "direction": dir_pl},
            "target_ratio_pct": target_pl,
            "note": ("PL_LOS_percentage was None → unchanged"
                     if (PL_LOS_percentage is None and not cfg.get("mirror_pl_target_if_none", False))
                     else ""),
        },
        "config_used": cfg,
    }
    return obs_percent, report
