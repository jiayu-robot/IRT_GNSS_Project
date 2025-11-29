"""
mp_zero_check.py

Post-processing utility to apply a zero-increment policy on multipath observations.

- Obs containers (obs_dir, obs_real, obs_mp, mp_info) are built in read_obs.read_obs_data(...)
- This module ONLY uses read_obs.mp_info as the multipath source
  (no direct parsing of Multipath_U100.log here)
- Under no-LOS condition:
      mode = "keep"       → keep strongest multipath
      mode = "use_dir"    → use direct-path Obs when |delay| <= ZERO_TOL
      mode = "use_second" → use second-strongest multipath when |delay| <= ZERO_TOL

Returns:
- obs_real_ext: dict[gps_time] -> list[Obs]
"""

import os
from collections import defaultdict

from read_obs import (
    read_obs_data,
    Obs,
    format_satellite_id,
    mp_info,
)
from utils import find_file

# Policy configuration (used by library + main)

ZERO_TOL = 1e-6          # meters
MODE = "use_second"      # "keep" | "use_dir" | "use_second"
SECOND_NONZERO_ONLY = True
PRINT_SAMPLES = 8        # number of transitions printed for inspection


def _nearest_key(t, keys):
    """Return the nearest epoch key from obs_dir / obs_mp."""
    return min(keys, key=lambda k: abs(k - t))


def _rank_strongest(cands, meta):
    """
    Pick the strongest Obs based on rec_pow.
    If rec_pow is not available, fall back to minimal power_loss in meta.
    """
    if not cands:
        return None

    if any(o.rec_pow is not None for o in cands):
        return max(cands, key=lambda x: (-1e12 if x.rec_pow is None else x.rec_pow))

    def loss_of(o):
        pd_loss = meta.get(o.multipath_id)
        return 1e12 if pd_loss is None else pd_loss[1]

    return min(cands, key=lambda x: loss_of(x))


def _pick_second(best, cands, meta, zero_tol=ZERO_TOL, nonzero_only=True):
    """
    Pick the second-strongest multipath Obs.
    Optionally require |path_delay| > zero_tol.
    """
    alts = [o for o in cands if o.multipath_id != best.multipath_id]
    if not alts:
        return None

    # rec_pow-based second
    if any(o.rec_pow is not None for o in alts):
        second = max(alts, key=lambda x: (-1e12 if x.rec_pow is None else x.rec_pow))
        if nonzero_only:
            sid = second.multipath_id
            pd_loss = meta.get(sid)
            if pd_loss and abs(pd_loss[0]) <= zero_tol:
                nz = [o for o in alts
                      if meta.get(o.multipath_id)
                      and abs(meta[o.multipath_id][0]) > zero_tol]
                if nz:
                    second = max(nz, key=lambda x: (-1e12 if x.rec_pow is None else x.rec_pow))
        return second

    # loss-based second
    def loss_of(o):
        pd_loss = meta.get(o.multipath_id)
        return 1e12 if pd_loss is None else pd_loss[1]

    if nonzero_only:
        nz = [o for o in alts
              if meta.get(o.multipath_id)
              and abs(meta[o.multipath_id][0]) > zero_tol]
        if nz:
            return min(nz, key=lambda x: loss_of(x))

    return min(alts, key=lambda x: loss_of(x))


def build_obs_dir_only(rinex_path, sp3_paths):
    """
    Convenience wrapper: build obs_dir using read_obs.read_obs_data().
    Multipath and received power are not loaded here.
    """
    obs_dir, _mp, _all, _real = read_obs_data(
        rinex_path, sp3_paths,
        rec_pow_file_path=None,
        multipath_file_path=None
    )
    return obs_dir


def _iter_aligned_mp_entries(obs_dir):
    """
    Iterate over multipath entries in mp_info and align them to obs_dir.

    Yields:
        (t_mp, sat_raw, no_los, comps, od, meta_map)
    """
    if not mp_info:
        print("[WARN] mp_info is empty in _iter_aligned_mp_entries.")
        return

    if not obs_dir:
        print("[WARN] obs_dir is empty in _iter_aligned_mp_entries.")
        return

    time_keys = list(obs_dir.keys())

    for t_mp, sat_dict in mp_info.items():
        for sat_raw, meta in sat_dict.items():
            no_los = bool(meta.get("no_los_flag", False))
            comps = list(meta.get("comps", []))   # (sid, pd, dp, loss)
            if not comps:
                continue

            # time alignment
            tk = _nearest_key(t_mp, time_keys)
            dir_list = obs_dir.get(tk, [])
            if not dir_list:
                continue

            ids = [o.id for o in dir_list]

            sat_id = sat_raw
            if sat_id not in ids:
                sat_id_fmt, _ = format_satellite_id(sat_raw)
                sat_id = sat_id_fmt

            if sat_id not in ids:
                continue

            od = dir_list[ids.index(sat_id)]

            # prepare meta_map for power_loss queries
            meta_map = {}
            for sid, pd, dp, loss in comps:
                if loss >= 1000:
                    continue
                meta_map[sid] = (pd, loss)

            if not meta_map:
                continue

            yield t_mp, sat_raw, no_los, comps, od, meta_map


def _build_candidates_from_comps(od, comps):
    """
    Convert multipath components into Obs candidates.
    """
    candidates = []
    for sid, pd, dp, loss in comps:
        if loss >= 1000:
            continue

        rec_pow = od.rec_pow - loss if od.rec_pow is not None else 0.0
        candidates.append(
            Obs(
                od.gps_time,
                od.id,
                od.sv_id,
                od.pseudorange + pd,
                od.carrier_phase,
                od.doppler_shift + dp,
                multipath=True,
                multipath_id=sid,
                satellite_pos=od.satellite_pos,
                satellite_vel=od.satellite_vel,
                clock_bias=od.clock_bias,
                rec_pow=rec_pow,
            )
        )
    return candidates


def _update_zero_stats(best_tuple, zero_tol, checked, zero, nonzero):
    """
    Update zero/nonzero statistics for MP-only path delays.
    """
    if best_tuple:
        _, best_pd, _best_loss = best_tuple
        checked += 1
        if abs(best_pd) <= zero_tol:
            zero += 1
        else:
            nonzero += 1
    return checked, zero, nonzero


def _apply_policy_to_entry(
    od,
    candidates,
    meta_map,
    mode,
    zero_tol,
    second_nonzero_only
):
    """
    Apply zero-increment policy to a single (time, sat) entry.
    """
    if not candidates:
        return od, "dir_no_candidate", None

    # "strongest" candidate used for navigation
    best = _rank_strongest(candidates, meta_map)
    if best is None:
        return od, "dir_no_best", None

    # best_tuple used for zero statistics (by minimal loss)
    best_tuple = min(
        [(sid, pd, loss) for sid, (pd, loss) in meta_map.items()],
        key=lambda x: x[2]
    )
    delay = best.pseudorange - od.pseudorange

    if abs(delay) > zero_tol:
        # not a zero-increment case → keep strongest
        return best, "keep", best_tuple

    # zero-increment case: choose according to mode
    if mode == "use_dir":
        return od, "use_dir", best_tuple

    if mode == "use_second":
        second = _pick_second(
            best,
            candidates,
            meta_map,
            zero_tol=zero_tol,
            nonzero_only=second_nonzero_only,
        )
        if second is not None:
            return second, "use_second", best_tuple
        return od, "fallback_dir", best_tuple

    # mode == "keep"
    return best, "keep", best_tuple


def _print_samples(samples):
    """Print a few before/after transitions for manual inspection."""
    if not samples:
        return

    print("\n[Samples: before(best) -> after(chosen)]")
    for t, sat_raw, action, best, chosen in samples:
        b_sid, b_pd, b_loss = best
        if chosen is None:
            print(
                f"  t={t:.3f} {sat_raw}: "
                f"best(sid={b_sid}, pd={b_pd:.6f}, loss={b_loss:.3f}) "
                f"-> DIR ({action})"
            )
        else:
            c_sid, c_pd, c_loss = chosen
            print(
                f"  t={t:.3f} {sat_raw}: "
                f"best(sid={b_sid}, pd={b_pd:.6f}, loss={b_loss:.3f}) "
                f"-> chosen(sid={c_sid}, pd={c_pd:.6f}, loss={c_loss:.3f}) ({action})"
            )


def _print_summary(checked, zero, nonzero, zero_tol,
                   act_keep, act_dir, act_second, act_fallback,
                   obs_real_ext):
    """Print statistics and summary of obs_real_ext."""
    if checked > 0:
        ratio = zero / checked
        print(
            f"[MP-only path_delay] checked={checked}, zero={zero}, "
            f"nonzero={nonzero}, zero_ratio={ratio:.6f}, tol={zero_tol} m"
        )
        print(
            "[policy(applied here)] "
            f"keep={act_keep}, use_dir={act_dir}, "
            f"use_second={act_second}, fallback={act_fallback}"
        )
    else:
        print("[MP-only path_delay] No eligible no-LOS multipath entries found.")

    epochs = len(obs_real_ext)
    total = sum(len(v) for v in obs_real_ext.values())
    mp_count = sum(
        sum(1 for o in v if getattr(o, "multipath", False))
        for v in obs_real_ext.values()
    )
    print(f"\n[RESULT] obs_real_ext built: epochs={epochs}, total={total}, mp_obs={mp_count}")


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def apply_mp_policy(
    obs_dir,
    mp_path=None,          # deprecated, kept only for API compatibility
    *,
    mode=MODE,
    zero_tol=ZERO_TOL,
    second_nonzero_only=SECOND_NONZERO_ONLY,
):
    """
    Apply zero-increment policy using mp_info.
    """
    obs_real_ext = defaultdict(list)

    if not obs_dir:
        print("[WARN] obs_dir is empty; nothing to process.")
        return obs_real_ext

    if not mp_info:
        print("[WARN] mp_info is empty; did you call read_obs_data(..., multipath_file_path=...)?")
        return obs_real_ext

    checked = zero = nonzero = 0
    act_keep = act_dir = act_second = act_fallback = 0
    samples = []

    # Main loop over aligned multipath entries
    for t_mp, sat_raw, no_los, comps, od, meta_map in _iter_aligned_mp_entries(obs_dir):
        # Build candidate Obs
        candidates = _build_candidates_from_comps(od, comps)

        if no_los and candidates:
            # Apply policy for no-LOS case
            chosen, action, best_tuple = _apply_policy_to_entry(
                od,
                candidates,
                meta_map,
                mode,
                zero_tol,
                second_nonzero_only,
            )

            # Update zero stats
            checked, zero, nonzero = _update_zero_stats(
                best_tuple,
                zero_tol,
                checked,
                zero,
                nonzero,
            )

            obs_real_ext[od.gps_time].append(chosen)

            # Collect sample transitions (except pure "keep")
            if PRINT_SAMPLES and len(samples) < PRINT_SAMPLES and action != "keep" and best_tuple:
                b_sid, b_pd, b_loss = best_tuple
                if chosen is od:
                    samples.append((t_mp, sat_raw, action, best_tuple, None))
                else:
                    sid_c = chosen.multipath_id
                    pd_loss_c = meta_map.get(sid_c)
                    if pd_loss_c:
                        samples.append(
                            (
                                t_mp,
                                sat_raw,
                                action,
                                best_tuple,
                                (sid_c, pd_loss_c[0], pd_loss_c[1]),
                            )
                        )
                    else:
                        samples.append((t_mp, sat_raw, action, best_tuple, None))

            # Action counters
            if action == "keep":
                act_keep += 1
            elif action == "use_dir":
                act_dir += 1
            elif action == "use_second":
                act_second += 1
            elif action == "fallback_dir":
                act_fallback += 1
        else:
            # LOS visible or no candidate → keep direct-path
            obs_real_ext[od.gps_time].append(od)
            act_dir += 1

    # Print statistics and samples
    _print_summary(
        checked, zero, nonzero, zero_tol,
        act_keep, act_dir, act_second, act_fallback,
        obs_real_ext,
    )
    _print_samples(samples)

    return obs_real_ext


# ---------------------------------------------------------
# Script entry (only used when running this file directly)
# ---------------------------------------------------------
def main():
    """
    Script mode:
    - Use read_obs.read_obs_data() to load RINEX/SP3 and multipath
    - Apply zero-increment policy using mp_info
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = "Rectangle_2m"  # change this if you want to test another folder
    folder_path = os.path.join(dir_path, data)

    rinex_file_path = os.path.join(folder_path, find_file(folder_path, 'V', '.rnx')[0])
    sp3_file_path = [
        os.path.join(folder_path, find_file(folder_path, 'GPS', '.SP3')[0]),
        os.path.join(folder_path, find_file(folder_path, 'GAL', '.SP3')[0]),
    ]
    multipath_file_path = os.path.join(folder_path, "Multipath_U100.log")

    if not (os.path.isfile(rinex_file_path)
            and os.path.isfile(multipath_file_path)
            and all(os.path.isfile(p) for p in sp3_file_path)):
        print("[ERROR] Missing inputs. Check data folder and file paths.")
        return

    obs_dir, obs_mp, obs_all, obs_real = read_obs_data(
        rinex_file_path,
        sp3_file_path,
        rec_pow_file_path=None,
        multipath_file_path=multipath_file_path,
    )

    apply_mp_policy(
        obs_dir,
        mode=MODE,
        zero_tol=ZERO_TOL,
        second_nonzero_only=SECOND_NONZERO_ONLY,
    )


if __name__ == "__main__":
    main()
