"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its Exophase earn rate at the case's target, from the house's
recorded reads of the trophy page up to the case's checkpoint.

Two regimes, picked from the case's own history:

1. Flat history (every read identical).  The page is stale: Exophase has not
   refreshed the game's counts since the first probe, and the case's target
   falls after its first refresh.  The forecast therefore expects the
   refresh the kit documents (competition.yaml): the tracked-player count
   jumps 638 -> 3,868 and the trophies-earned total 943 -> 4,276, so every
   rate but the one at 100% falls.  The new players' earners are split
   across trophies by a power law in the old rate, calibrated so the split
   totals the documented new earners (3,333).  A trophy everyone holds
   stays at 100%.

2. Growing history (the final's shape).  Each trophy's logit rate is fitted
   against ln(1 + hours since 2026-11-19T00:00Z) over the last 24 hours and
   over the last 48 hours, the two slopes blended, then shrunk halfway
   toward the cross-trophy median slope (noisy per-trophy fits borrow
   strength from the pool) and toward zero when the fit is not significant.
   The shrunk slope is read at the target.  The spread combines the
   house's random-walk scale (0.5 logit per root day) with the slope's own
   standard error times the projection distance, so a trophy whose trend is
   uncertain gets an honestly wider band.  The platinum is capped at the
   lowest other trophy's rate (it cannot be earned without the rest).

Runs offline, standard library only, deterministic: the same case always
writes the same bytes.  `python3 main.py` reads $MAKEFUN_INPUT_DIR (else
/input) and writes $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# shipped beside this file would import.  (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Origin of the log-time axis, as the house curve baseline uses it.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- documented first refresh (kit competition.yaml) -----------------------
# Exophase's counts were unchanged from the first probe through the 04:00Z
# read on launch night (638 players tracked, 943 trophies earned), then
# refreshed between 04:00Z and 04:30Z: 3,868 players tracked, 4,276
# trophies earned.  The practice cases straddle that refresh.
REFRESH_PLAYERS_BEFORE = 638
REFRESH_PLAYERS_AFTER = 3868
REFRESH_EARNED_BEFORE = 943
REFRESH_EARNED_AFTER = 4276

# Random-walk spread of a rate's move, logit units per square-root day
# (the house baseline's scale, kept as the honest base uncertainty).
RW_SIGMA = 0.5


def clamp_pct(x):
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        e = math.exp(-lg)
        return 100.0 / (1.0 + e)
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as the house baseline rounds)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_epoch(dt):
    """Hours from the log-time origin, floored at 0 (a target before the
    origin still gets a defined coordinate)."""
    return max(0.0, (dt - EPOCH).total_seconds() / 3600.0)


def quantiles_from_logit(centre_logit, spread_logit):
    """Five quantiles around a logit centre with a given logit spread."""
    out = []
    for z in Z:
        q = inv_logit(centre_logit + z * spread_logit)
        out.append(round4(min(100.0, max(0.0, q))))
    # Keep them non-decreasing after rounding.
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def history_is_flat(history):
    """True when every read shows the same rates and player count."""
    if len(history) < 2:
        return True
    first = history[0]
    for read in history[1:]:
        if read.get("players_tracked") != first.get("players_tracked"):
            return False
        if read.get("rates") != first.get("rates"):
            return False
    return True


def solve_power_gamma(rates, new_players, total_new_earners):
    """Find gamma so sum(new_players * r**gamma) == total_new_earners.

    r is each trophy's old rate as a fraction (0..1).  The 100% trophy
    contributes new_players whatever gamma is, so the equation is solved
    on the remaining trophies.  Bisection; gamma lives in [0.5, 8].
    """
    others = [r / 100.0 for r in rates if r is not None and r < 100.0]
    need = total_new_earners - new_players  # the 100% trophy takes new_players
    if need <= 0 or not others:
        return 1.0

    def total(gamma):
        return sum(new_players * (p ** gamma) for p in others if p > 0)

    lo, hi = 0.5, 8.0
    if total(lo) < need:
        return lo
    if total(hi) > need:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if total(mid) < need:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def forecast_flat_history(case):
    """Expect the documented first refresh (practice cases' shape)."""
    trophies = case["trophies"]
    last = case["history"][-1]
    rates = [last["rates"][i] for i in range(len(trophies))]
    players_now = last.get("players_tracked") or REFRESH_PLAYERS_BEFORE
    earned_now = last.get("trophies_earned") or REFRESH_EARNED_BEFORE

    # The documented refresh, scaled to this case's own counts: players
    # grow by the documented factor, earners by theirs.
    players_after = players_now * REFRESH_PLAYERS_AFTER / REFRESH_PLAYERS_BEFORE
    earned_after = earned_now * REFRESH_EARNED_AFTER / REFRESH_EARNED_BEFORE
    new_players = players_after - players_now
    total_new_earners = earned_after - earned_now

    gamma = solve_power_gamma(rates, new_players, total_new_earners)

    # A refresh is a big, partly unpredictable jump: the band is wider
    # than a 90-minute random walk, and it grows a little with the
    # horizon (the target sits just after the refresh here).
    horizon_days = case["horizon_minutes"] / 1440.0
    spread = math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2 + 0.20 ** 2)

    rows = []
    for i, trophy in enumerate(trophies):
        r = rates[i]
        if r is None:
            r = 0.0
        if r >= 100.0:
            median = 100.0
        else:
            old_earners = r * players_now / 100.0
            new_earners = new_players * ((r / 100.0) ** gamma) if r > 0 else 0.0
            median = (old_earners + new_earners) / players_after * 100.0
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(logit(median), spread)})
    return {"trophies": rows}


def ols_slope(xs, ys):
    """Least-squares slope and its standard error for y ~ a + b*x.

    Returns (slope, standard_error).  With fewer than 3 points, or no
    spread in x, the slope is 0 with a large standard error.
    """
    n = len(xs)
    if n < 3:
        return 0.0, 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, 1.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    dof = n - 2
    var = resid / dof if dof > 0 else resid
    se = math.sqrt(var / sxx) if sxx > 0 else 1.0
    return slope, se


def forecast_growing_history(case):
    """Extrapolate each trophy's logit rate along log time (final's shape)."""
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_epoch(target))

    # Read times and log-time coordinates, oldest first.
    times = [parse_time(read["at"]) for read in history]
    xs = [math.log1p(hours_since_epoch(t)) for t in times]
    x_last = xs[-1]
    dx = max(0.0, x_target - x_last)

    # Recent windows, by clock time: last 24h and last 48h of reads.
    last_time = times[-1]
    recent_idx = [i for i, t in enumerate(times)
                  if (last_time - t).total_seconds() <= 24 * 3600]
    extended_idx = [i for i, t in enumerate(times)
                    if (last_time - t).total_seconds() <= 48 * 3600]
    if len(recent_idx) < 3:
        recent_idx = list(range(len(times)))
    if len(extended_idx) < 3:
        extended_idx = list(range(len(times)))

    # Per-trophy trend fits.  Each entry is (last_logit, slope, slope_se,
    # fallback_spread): the fallback spread is used when too few points
    # exist to fit at all.
    fits = []
    for i, trophy in enumerate(trophies):
        # Series of (history_index, x, logit rate), skipping nulls.
        series = [(k, xs[k], logit(history[k]["rates"][i]))
                  for k in range(len(history))
                  if history[k]["rates"][i] is not None]
        if len(series) < 3:
            last_rate = history[-1]["rates"][i]
            fits.append((logit(last_rate if last_rate is not None else 0.0),
                         0.0, 1.0,
                         RW_SIGMA * math.sqrt(horizon_days) + 0.5))
            continue
        recent = set(recent_idx)
        extended = set(extended_idx)
        rx = [x for (k, x, _) in series if k in recent]
        ry = [y for (k, _, y) in series if k in recent]
        ex = [x for (k, x, _) in series if k in extended]
        ey = [y for (k, _, y) in series if k in extended]
        b_recent, se_recent = ols_slope(rx, ry)
        b_ext, _ = ols_slope(ex, ey)
        # Blend: recent counts more, extended steadies it.
        slope = 0.6 * b_recent + 0.4 * b_ext
        fits.append((series[-1][2], slope, se_recent, None))

    # Cross-trophy shrinkage: pull each slope halfway toward the median
    # slope, and damp it toward zero when the fit is not significant.
    finite_slopes = sorted(f[1] for f in fits if math.isfinite(f[1]))
    pool = finite_slopes[len(finite_slopes) // 2] if finite_slopes else 0.0
    rows = []
    spreads_by_id = {}
    for i, trophy in enumerate(trophies):
        last_logit, slope, se, fallback_spread = fits[i]
        if fallback_spread is not None:
            spreads_by_id[trophy["id"]] = fallback_spread
            rows.append({"id": trophy["id"],
                         "q": quantiles_from_logit(last_logit,
                                                   fallback_spread)})
            continue
        # Significance damp: a slope within one standard error of zero
        # is halved; a clearly significant one is kept.
        t_stat = abs(slope) / se if se > 0 else 0.0
        damp = 1.0 if t_stat >= 2.0 else (0.5 + 0.25 * t_stat)
        shrunk = (0.5 * slope + 0.5 * pool) * damp
        centre = last_logit + shrunk * dx
        # Spread: random walk plus the slope's own uncertainty over dx.
        rw = RW_SIGMA * math.sqrt(horizon_days)
        total_spread = math.sqrt(rw * rw + (se * dx) ** 2)
        total_spread = min(2.0, max(0.6, total_spread))
        spreads_by_id[trophy["id"]] = total_spread
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(centre, total_spread)})

    # The platinum cannot be earned without the rest: cap it at the
    # lowest other trophy's median, keeping its own spread.
    by_id = {row["id"]: row for row in rows}
    platinum = next((t for t in trophies if t.get("tier") == "platinum"), None)
    if platinum is not None:
        others = [by_id[t["id"]]["q"][2] for t in trophies
                  if t["id"] != platinum["id"]]
        if others:
            cap = min(others)
            row = by_id[platinum["id"]]
            if row["q"][2] > cap:
                row["q"] = quantiles_from_logit(
                    logit(cap), spreads_by_id[platinum["id"]])
    return {"trophies": rows}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    if history_is_flat(case["history"]):
        return forecast_flat_history(case)
    return forecast_growing_history(case)


def fallback_forecast(case):
    """A valid forecast for a case the model cannot read: every trophy's
    median is its last known rate (or 50% with no data), with a wide band.
    Used only so a strange case still yields a valid output instead of a
    crash, which would forfeit the entry's prize eligibility."""
    trophies = case.get("trophies", [])
    history = case.get("history", [])
    last = history[-1] if history else None
    rows = []
    for i, trophy in enumerate(trophies):
        rate = None
        if last is not None:
            rates = last.get("rates") or []
            if i < len(rates):
                rate = rates[i]
        median = rate if rate is not None else 50.0
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(logit(median), 1.5)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = fallback_forecast(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
