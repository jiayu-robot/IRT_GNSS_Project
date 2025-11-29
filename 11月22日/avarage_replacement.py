"""
avarage_replacement.py

Post-processing utility for "average multipath replacement" under no-LOS condition.

- Obs containers are built by read_obs.read_obs_data(...)
- Multipath information is taken only from read_obs.mp_info
- For each (time, sat) with no-LOS:
      strongest-by-loss multipath = MP with minimal power_loss
      if |pd_strongest| <= ZERO_TOL:
          → replace LOS with average multipath:
                pd_avg, dp_avg, loss_avg over all valid MPs
      else:
          → keep strongest-by-loss single multipath

Returns:
- obs_real_ext: dict[gps_time] -> list[Obs]
"""

import os
from collections import defaultdict
from statistics import mean

from mp_zero_check import _nearest_key as nearest_key   # reuse alignment helper
from read_obs import read_obs_data, format_satellite_id, Obs, mp_info
from utils import find_file

# ---------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------
ZERO_TOL = 1e-6
PRINT_SAMPLES = 8
RECPOW_PLACEHOLDER = -130.0   # used when od.rec_pow is None


def _strongest_by_loss(meta_map):
    """
    Given meta_map = {sid: (pd, loss)}, return (sid, pd, loss) with minimal loss.
    """
    if not meta_map:
        return None
    return min(
        [(sid, tpl[0], tpl[1]) for sid, tpl in meta_map.items()],
        key=lambda x: x[2]
    )


def _iter_aligned_mp_entries(obs_dir):
    """
    Iterate over multipath entries from mp_info aligned to obs_dir.

    Yields:
        (t_mp, sat_raw, no_los_flag, valids, od, meta_map)
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
            no_los_flag = bool(meta.get("no_los_flag", False))
            comps = list(meta.get("comps", []))   # (sid, pd, dp, loss)
            if not comps:
                continue

            tk = nearest_key(t_mp, time_keys)
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

            # filter valid multipath components
            valids = [(sid, pd, dp, loss) for (sid, pd, dp, loss) in comps if loss < 1000]
            if not valids:
                yield t_mp, sat_raw, no_los_flag, [], od, {}
                continue

            meta_map = {sid: (pd, loss) for (sid, pd, _dp, loss) in valids}
            yield t_mp, sat_raw, no_los_flag, valids, od, meta_map


def _build_average_obs(od, valids, recpow_placeholder):
    """
    Build an Obs that represents the average multipath over all valid components.
    Returns:
        avg_obs: Obs
        (pd_avg, loss_avg)
    """
    pd_avg   = mean([pd   for (_sid, pd,  _dp, _loss) in valids])
    dp_avg   = mean([dp   for (_sid, _pd, dp,  _loss) in valids])
    loss_avg = mean([loss for (_sid, _pd, _dp, loss) in valids])

    rp = (od.rec_pow - loss_avg) if (od.rec_pow is not None) else recpow_placeholder

    avg_obs = Obs(
        od.gps_time,
        od.id,
        od.sv_id,
        od.pseudorange + pd_avg,
        od.carrier_phase,
        od.doppler_shift + dp_avg,
        multipath=True,
        multipath_id=-1,
        satellite_pos=od.satellite_pos,
        satellite_vel=od.satellite_vel,
        clock_bias=od.clock_bias,
        rec_pow=rp,
    )
    return avg_obs, (pd_avg, loss_avg)


def _build_strongest_obs(od, valids, recpow_placeholder):
    """
    Build an Obs from the strongest-by-loss multipath component.
    Returns:
        keep_obs: Obs
        (sid_k, pd_k, loss_k)
    """
    sid_k, pd_k, dp_k, loss_k = min(valids, key=lambda x: x[3])
    rp_k = (od.rec_pow - loss_k) if (od.rec_pow is not None) else recpow_placeholder

    keep_obs = Obs(
        od.gps_time,
        od.id,
        od.sv_id,
        od.pseudorange + pd_k,
        od.carrier_phase,
        od.doppler_shift + dp_k,
        multipath=True,
        multipath_id=sid_k,
        satellite_pos=od.satellite_pos,
        satellite_vel=od.satellite_vel,
        clock_bias=od.clock_bias,
        rec_pow=rp_k,
    )
    return keep_obs, (sid_k, pd_k, loss_k)


def _update_zero_stats(best_tuple, zero_tol, checked, zero, nonzero):
    """
    Update MP-only zero/nonzero counters using the strongest-by-loss tuple.
    """
    if best_tuple:
        _, pd_b, _loss_b = best_tuple
        checked += 1
        if abs(pd_b) <= zero_tol:
            zero += 1
        else:
            nonzero += 1
    return checked, zero, nonzero


def _print_samples(samples):
    """Print a few before/after transitions for manual inspection."""
    if not samples:
        return

    print("\n[Samples: before(best by loss) -> after(chosen)]")
    for t, sat_raw, best, chosen, action in samples:
        b_sid, b_pd, b_loss = best
        if chosen and chosen[0] == "AVG":
            _tag, apd, aloss = chosen
            print(
                f"  t={t:.3f} {sat_raw}: "
                f"best(sid={b_sid}, pd={b_pd:.6f}, loss={b_loss:.3f}) "
                f"-> chosen(AVG, pd={apd:.6f}, loss={aloss:.3f}) ({action})"
            )
        else:
            c_sid, c_pd, c_loss = chosen if chosen else ("DIR", 0.0, 0.0)
            print(
                f"  t={t:.3f} {sat_raw}: "
                f"best(sid={b_sid}, pd={b_pd:.6f}, loss={b_loss:.3f}) "
                f"-> chosen(sid={c_sid}, pd={c_pd:.6f}, loss={c_loss:.3f}) ({action})"
            )


def _print_summary(checked, zero, nonzero, zero_tol,
                   act_use_avg, act_keep, act_dir,
                   obs_real_ext):
    """Print statistics and obs_real_ext summary."""
    if checked > 0:
        ratio = zero / checked
        print(
            f"[MP-only path_delay] checked={checked}, zero={zero}, nonzero={nonzero}, "
            f"zero_ratio={ratio:.6f}, tol={zero_tol} m"
        )
    else:
        print("[MP-only path_delay] No eligible no-LOS multipath entries found.")

    print(f"[actions] use_avg={act_use_avg}, keep={act_keep}, use_dir={act_dir}")

    epochs = len(obs_real_ext)
    total = sum(len(v) for v in obs_real_ext.values())
    mp_count = sum(
        sum(1 for o in v if getattr(o, "multipath", False))
        for v in obs_real_ext.values()
    )
    print(f"\n[RESULT] obs_real_ext built: epochs={epochs}, total_obs={total}, mp_obs={mp_count}")


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def apply_avg_policy(
    obs_dir,
    mp_path=None,   # deprecated, kept for API compatibility
    *,
    zero_tol: float = ZERO_TOL,
    print_samples: int = PRINT_SAMPLES,
    recpow_placeholder: float = RECPOW_PLACEHOLDER,
):
    """
    Average-replacement policy using mp_info as the multipath source.
    """
    obs_real_ext = defaultdict(list)

    if not obs_dir:
        print("[WARN] obs_dir is empty; nothing to process.")
        return obs_real_ext

    if not mp_info:
        print("[WARN] mp_info is empty; please call read_obs_data(..., multipath_file_path=...).")
        return obs_real_ext

    checked = zero = nonzero = 0
    act_use_avg = act_keep = act_dir = 0
    samples = []

    # Iterate over aligned multipath entries
    for t_mp, sat_raw, no_los_flag, valids, od, meta_map in _iter_aligned_mp_entries(obs_dir):

        # LOS visible or no valid MPs → keep direct-path Obs
        if not no_los_flag or not valids:
            obs_real_ext[od.gps_time].append(od)
            act_dir += 1
            continue

        # strongest-by-loss
        best_tuple = _strongest_by_loss(meta_map)
        checked, zero, nonzero = _update_zero_stats(
            best_tuple,
            zero_tol,
            checked,
            zero,
            nonzero,
        )

        # zero-increment → use average multipath
        if best_tuple and abs(best_tuple[1]) <= zero_tol:
            avg_obs, (pd_avg, loss_avg) = _build_average_obs(
                od,
                valids,
                recpow_placeholder,
            )
            obs_real_ext[od.gps_time].append(avg_obs)
            act_use_avg += 1

            if print_samples and len(samples) < print_samples:
                sid_best, pd_best, loss_best = best_tuple
                samples.append(
                    (
                        t_mp,
                        sat_raw,
                        (sid_best, pd_best, loss_best),
                        ("AVG", pd_avg, loss_avg),
                        "use_avg",
                    )
                )
        else:
            # non-zero → keep strongest-by-loss single MP
            keep_obs, (sid_k, pd_k, loss_k) = _build_strongest_obs(
                od,
                valids,
                recpow_placeholder,
            )
            obs_real_ext[od.gps_time].append(keep_obs)
            act_keep += 1

            if print_samples and len(samples) < print_samples and best_tuple:
                sid_best, pd_best, loss_best = best_tuple
                samples.append(
                    (
                        t_mp,
                        sat_raw,
                        (sid_best, pd_best, loss_best),
                        (sid_k, pd_k, loss_k),
                        "keep",
                    )
                )

    _print_summary(
        checked,
        zero,
        nonzero,
        zero_tol,
        act_use_avg,
        act_keep,
        act_dir,
        obs_real_ext,
    )
    _print_samples(samples)

    return obs_real_ext


# ---------------------------------------------------------
# Script entry
# ---------------------------------------------------------
def main():
    """
    Script mode:
    - Load RINEX/SP3/multipath via read_obs.read_obs_data()
    - Apply average-replacement policy
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = "Rectangle_2m"   # change this when testing other datasets
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
        print("[ERROR] Missing inputs. Check DATA_FOLDER and paths.")
        return

    obs_dir, obs_mp, obs_all, obs_real = read_obs_data(
        rinex_file_path,
        sp3_file_path,
        rec_pow_file_path=None,
        multipath_file_path=multipath_file_path,
    )

    _ = apply_avg_policy(
        obs_dir,
        mp_path=None,
        zero_tol=ZERO_TOL,
        print_samples=PRINT_SAMPLES,
        recpow_placeholder=RECPOW_PLACEHOLDER,
    )


if __name__ == "__main__":
    main()

