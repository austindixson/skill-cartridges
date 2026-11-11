"""Paired comparison of two run_eval JSONs:  stats.py A.json B.json [--json]
Reports correct counts, gained (A wrong -> B right), lost, net; exact two-sided McNemar p (binomial on the
discordant pairs, p=0.5); and a 95% CI for the paired difference in proportions (B - A).

CI method = percentile bootstrap over items (200k resamples, seed 0, resampling the paired per-item
differences in {-1,0,+1} as a multinomial). It reproduces the reference "27 gained, 0 lost, n=97 -> +19.6 to
+37.1 pp" (19.59 / 37.11). Tango score interval does NOT (19.89 / 37.47) and neither does Newcombe paired
(18.6 / 36.2) or Wald (18.9 / 36.8). The Tango interval is also printed as `tango_pp` for reference."""
import argparse, json, math
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import binom

Z = 1.959964


def mcnemar_exact(lost, gained):
    n = lost + gained
    return 1.0 if n == 0 else min(1.0, 2 * float(binom.cdf(min(lost, gained), n, 0.5)))


def bootstrap_ci(lost, gained, n, B=200_000, seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [lost / n, gained / n, 1 - (lost + gained) / n], size=B)
    d = (draws[:, 1] - draws[:, 0]) / n
    return tuple(float(v) for v in np.percentile(d, [2.5, 97.5]))


def tango_ci(lost, gained, n):
    """Tango (1998) score interval for (gained-lost)/n."""
    def stat(delta):
        def nll(q):  # q = P(lost), q+delta = P(gained)
            pc, r = q + delta, 1 - 2 * q - delta
            if q <= 0 or pc <= 0 or r <= 0:
                return 1e9
            return -(lost * math.log(q) + gained * math.log(pc) + (n - lost - gained) * math.log(r))
        q = minimize_scalar(nll, bounds=(max(1e-12, -delta + 1e-12), (1 - delta) / 2 - 1e-12), method="bounded",
                            options={"xatol": 1e-14}).x
        return (gained - lost - n * delta) ** 2 / (n * (2 * q + delta - delta ** 2)) - Z * Z
    d0 = (gained - lost) / n
    lo = brentq(stat, -0.99, d0 - 1e-9) if stat(-0.99) > 0 else -1.0
    hi = brentq(stat, d0 + 1e-9, 0.99) if stat(0.99) > 0 else 1.0
    return lo, hi


def compare(A, B):
    ia, ib = A["items"], B["items"]
    assert len(ia) == len(ib) == A["n"] == B["n"], "n differs"
    assert [x["gold"] for x in ia] == [x["gold"] for x in ib], "golds differ"
    a = [bool(x["correct"]) for x in ia]
    b = [bool(x["correct"]) for x in ib]
    n = len(a)
    gained = sum((not x) and y for x, y in zip(a, b))
    lost = sum(x and (not y) for x, y in zip(a, b))
    return summarize(n, sum(a), sum(b), gained, lost)


def summarize(n, a_correct, b_correct, gained, lost):
    lo, hi = bootstrap_ci(lost, gained, n)
    tlo, thi = tango_ci(lost, gained, n)
    return dict(n=n, A_correct=a_correct, B_correct=b_correct, gained=gained, lost=lost, net=gained - lost,
                mcnemar_p=mcnemar_exact(lost, gained), diff_pp=100 * (gained - lost) / n,
                ci_method="paired percentile bootstrap", ci_pp=[100 * lo, 100 * hi], ci_items=[n * lo, n * hi],
                tango_pp=[100 * tlo, 100 * thi])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    r = compare(json.load(open(args.a)), json.load(open(args.b)))
    if args.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"n={r['n']}  A={r['A_correct']}  B={r['B_correct']}  gained={r['gained']} lost={r['lost']} net={r['net']:+d}")
        print(f"McNemar exact two-sided p = {r['mcnemar_p']:.3g}")
        print(f"diff {r['diff_pp']:+.1f} pp; 95% CI ({r['ci_method']}) {r['ci_pp'][0]:+.1f} to {r['ci_pp'][1]:+.1f} pp "
              f"= {r['ci_items'][0]:+.1f} to {r['ci_items'][1]:+.1f} items; Tango {r['tango_pp'][0]:+.1f} to {r['tango_pp'][1]:+.1f} pp")
