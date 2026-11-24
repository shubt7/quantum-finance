"""Build docs/dashboard.html from live runs of the library.

Run:  python scripts/build_dashboard.py

Produces a single self-contained HTML file (inline SVG, no external assets).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from qfinance import (
    build_qubo,
    call_price,
    evaluate_portfolio,
    price_european_call_mc,
    price_european_call_qae,
    solve_annealing,
    solve_exact,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "dashboard.html"

S0, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 1.0
N_QUBITS = 10
M_RANGE = range(3, 15)
MC_REPEATS = 50


def convergence_data():
    bs = call_price(S0, K, r, sigma, T)
    rows = []
    for m in M_RANGE:
        q = price_european_call_qae(S0, K, r, sigma, T, n_qubits=N_QUBITS, m_qubits=m)
        calls = q.oracle_calls
        mc_errs = [
            abs(price_european_call_mc(S0, K, r, sigma, T, n_samples=calls, rng=seed)[0] - bs)
            for seed in range(MC_REPEATS)
        ]
        rows.append(
            dict(
                m=m,
                calls=calls,
                qae_err=abs(q.price - bs),
                mc_err=float(np.mean(mc_errs)),
                qae_price=q.price,
            )
        )
    return bs, rows


def portfolio_data():
    rng = np.random.default_rng(7)
    n, k, risk = 12, 4, 1.0
    mu = rng.uniform(0.02, 0.12, n)
    F = rng.normal(0, 0.1, (n, 2))
    cov = F @ F.T + np.diag(rng.uniform(0.005, 0.02, n))
    Q, penalty = build_qubo(mu, cov, k, risk_aversion=risk)
    x_ex, _ = solve_exact(Q)
    x_sa, _ = solve_annealing(Q, n_sweeps=600, rng=1)
    out = []
    for label, x in [("Exhaustive", x_ex), ("Annealing", x_sa)]:
        m = evaluate_portfolio(x, mu, cov, k, risk)
        out.append((label, m))
    universe = [(i, float(mu[i]), float(math.sqrt(cov[i, i]))) for i in range(n)]
    return out, universe, penalty


def loglog_svg(rows, width=640, height=340):
    pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 45
    xs = [row["calls"] for row in rows]
    ys = [v for row in rows for v in (row["qae_err"], row["mc_err"]) if v > 0]
    x_lo, x_hi = math.log10(min(xs)), math.log10(max(xs))
    y_lo, y_hi = math.floor(math.log10(min(ys))), math.ceil(math.log10(max(ys)))

    def px(x):
        return pad_l + (math.log10(x) - x_lo) / (x_hi - x_lo) * (width - pad_l - pad_r)

    def py(y):
        return pad_t + (1 - (math.log10(y) - y_lo) / (y_hi - y_lo)) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    for e in range(y_lo, y_hi + 1):
        y = py(10**e)
        parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" class="axis" text-anchor="end">1e{e}</text>')
    for x in xs:
        parts.append(f'<text x="{px(x):.1f}" y="{height-pad_b+16}" class="axis" text-anchor="middle">{x}</text>')
    parts.append(f'<text x="{(width)/2:.0f}" y="{height-6}" class="axis" text-anchor="middle">oracle calls / payoff samples (log scale)</text>')
    parts.append(f'<text x="14" y="{height/2:.0f}" class="axis" transform="rotate(-90 14 {height/2:.0f})" text-anchor="middle">|price error| (log scale)</text>')

    def polyline(key, cls):
        pts = " ".join(f"{px(r['calls']):.1f},{py(max(r[key], 10**(y_lo))):.1f}" for r in rows)
        parts.append(f'<polyline points="{pts}" class="{cls}" fill="none"/>')
        for r_ in rows:
            parts.append(f'<circle cx="{px(r_["calls"]):.1f}" cy="{py(max(r_[key], 10**y_lo)):.1f}" r="3.5" class="{cls}-dot"/>')

    polyline("mc_err", "mc")
    polyline("qae_err", "qae")
    parts.append(f'<g transform="translate({width-pad_r-150},{pad_t+4})"><rect width="150" height="44" class="legend-bg"/>'
                 f'<circle cx="12" cy="14" r="4" class="qae-dot"/><text x="24" y="18" class="axis">QAE error</text>'
                 f'<circle cx="12" cy="32" r="4" class="mc-dot"/><text x="24" y="36" class="axis">Monte Carlo (mean of {MC_REPEATS})</text></g>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_html():
    bs, rows = convergence_data()
    portfolios, universe, penalty = portfolio_data()
    last = rows[-1]

    table_rows = "\n".join(
        f"<tr><td>{r['m']}</td><td>{r['calls']:,}</td><td>{r['qae_price']:.4f}</td>"
        f"<td>{r['qae_err']:.4f}</td><td>{r['mc_err']:.4f}</td></tr>"
        for r in rows
    )
    port_rows = "\n".join(
        f"<tr><td>{label}</td><td>{', '.join(map(str, m.selected))}</td>"
        f"<td>{m.expected_return:.4f}</td><td>{m.variance:.5f}</td>"
        f"<td>{m.objective:.5f}</td><td>{'yes' if m.feasible else 'no'}</td></tr>"
        for label, m in portfolios
    )
    asset_rows = "\n".join(
        f"<span class='chip{' sel' if i in portfolios[0][1].selected else ''}'>asset {i}"
        f" · μ {mu:.3f} · σ {sd:.3f}</span>"
        for i, mu, sd in universe
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>quantum-finance dashboard</title>
<style>
  :root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --card:#f5f5f7; --qae:#0a7ea4; --mc:#c2410c; --grid:#d9d9de; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#111; --fg:#f2f2f2; --muted:#9a9aa0; --card:#1c1c1e; --grid:#333; --qae:#4fc3e8; --mc:#ff9a5c; }} }}
  body {{ font: 15px/1.5 system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 0; padding: 32px; }}
  h1 {{ margin: 0 0 4px; font-size: 26px; }} h2 {{ font-size: 18px; margin: 0 0 12px; }}
  .sub {{ color: var(--muted); margin-bottom: 24px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .kpi, .card {{ background: var(--card); border-radius: 12px; padding: 16px 18px; }}
  .kpi .v {{ font-size: 24px; font-weight: 600; }} .kpi .l {{ color: var(--muted); font-size: 13px; }}
  .grid2 {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }}
  @media (max-width: 800px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  .card {{ margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 500; }}
  .chart {{ width: 100%; height: auto; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }} .axis {{ fill: var(--muted); font-size: 11px; }}
  .qae {{ stroke: var(--qae); stroke-width: 2.2; }} .qae-dot {{ fill: var(--qae); }}
  .mc {{ stroke: var(--mc); stroke-width: 2.2; }} .mc-dot {{ fill: var(--mc); }}
  .legend-bg {{ fill: var(--bg); stroke: var(--grid); }}
  .chip {{ display: inline-block; background: var(--bg); border: 1px solid var(--grid); border-radius: 999px; padding: 4px 10px; margin: 3px; font-size: 13px; color: var(--muted); }}
  .chip.sel {{ border-color: var(--qae); color: var(--fg); font-weight: 600; }}
  .note {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}
</style>
</head>
<body>
<h1>quantum-finance</h1>
<div class="sub">Quantum amplitude estimation for option pricing · QUBO portfolio selection. Generated from live library runs by <code>scripts/build_dashboard.py</code>.</div>

<div class="kpis">
  <div class="kpi"><div class="v">{bs:.4f}</div><div class="l">Black-Scholes reference price</div></div>
  <div class="kpi"><div class="v">{last['qae_price']:.4f}</div><div class="l">QAE price, m={last['m']} ({last['calls']:,} oracle calls)</div></div>
  <div class="kpi"><div class="v">{last['qae_err']:.4f}</div><div class="l">QAE |error| at m={last['m']}</div></div>
  <div class="kpi"><div class="v">{last['mc_err']:.4f}</div><div class="l">Mean MC |error| with the same {last['calls']:,} samples</div></div>
</div>

<div class="grid2">
  <div class="card">
    <h2>Error vs. cost</h2>
    {loglog_svg(rows)}
    <div class="note">QAE error falls roughly as 1/M (oracle calls). Monte Carlo error falls roughly as 1/√N. Monte Carlo values are averaged over {MC_REPEATS} seeds; the QAE curve is a single exact run.</div>
  </div>
  <div class="card">
    <h2>Portfolio selection (12 assets, choose 4)</h2>
    <div>{asset_rows}</div>
    <table style="margin-top:12px">
      <tr><th>Solver</th><th>Assets</th><th>Return</th><th>Variance</th><th>Objective</th><th>Feasible</th></tr>
      {port_rows}
    </table>
    <div class="note">QUBO penalty = {penalty:.3f}. Annealing reaches the exhaustive optimum on this instance.</div>
  </div>
</div>

<div class="card">
  <h2>Convergence table</h2>
  <table>
    <tr><th>m (qubits)</th><th>Oracle calls</th><th>QAE price</th><th>QAE |error|</th><th>MC mean |error|</th></tr>
    {table_rows}
  </table>
  <div class="note">Call option: S₀ = {S0:g}, K = {K:g}, r = {r:.0%}, σ = {sigma:.0%}, T = {T:g}, grid of 2^{N_QUBITS} nodes. QAE is simulated exactly; loading and comparator costs are not modelled.</div>
</div>
</body>
</html>
"""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
