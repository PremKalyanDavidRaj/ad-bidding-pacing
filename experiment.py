"""Compare bidding policies in paired, synthetic first-price auctions."""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

HOURS = np.arange(24)
BUDGET = 450.0
TARGET_CPA = 35.0
SCENARIOS = ("Normal", "Midday price spike", "Evening traffic surge")
POLICIES = ("Fixed bid", "ML bid", "ML bid + pacing")


@dataclass
class Auctions:
    hours: np.ndarray
    features: np.ndarray
    price_thresholds: np.ndarray
    conversions: np.ndarray


def generate_day(seed, scenario="Normal"):
    rng = np.random.default_rng(seed)
    counts = (120 + 180 * np.exp(-((HOURS - 19) / 5) ** 2)).astype(int)
    if scenario == "Evening traffic surge":
        counts[18:23] *= 3
    hours = np.repeat(HOURS, counts)
    mobile = rng.binomial(1, 0.65, len(hours))
    premium = rng.binomial(1, 0.27, len(hours))
    evening = np.exp(-((hours - 20) / 4) ** 2)
    features = np.column_stack((mobile, premium, evening))
    logits = -4.7 + 0.55 * mobile + 1.05 * premium + 0.75 * evening
    probability = 1 / (1 + np.exp(-logits))
    thresholds = rng.lognormal(
        -1.45 + 0.65 * premium + 0.45 * evening, 0.55
    )
    if scenario == "Midday price spike":
        thresholds[(hours >= 11) & (hours < 16)] *= 2.8
    return Auctions(hours, features, thresholds, rng.random(len(hours)) < probability)


def replay(day, base_bids, planned, paced=False):
    """Clear auctions in order. Bids above the remaining budget are capped."""
    spend = 0.0
    conversions = 0
    wins = 0
    hourly_spend = []
    for hour in HOURS:
        # Use only spend observed before the hour and the historical spend plan.
        gap = planned[hour] - spend
        multiplier = float(np.clip(1 + gap / (BUDGET / 12), 0.15, 2.5)) if paced else 1.0
        start, end = np.searchsorted(day.hours, [hour, hour + 1])
        auctions = zip(
            base_bids[start:end], day.price_thresholds[start:end], day.conversions[start:end]
        )
        for base_bid, threshold, converted in auctions:
            bid = min(base_bid * multiplier, BUDGET - spend)
            if bid >= threshold:
                spend += bid
                wins += 1
                conversions += int(converted)
        hourly_spend.append(spend)
    assert 0 <= spend <= BUDGET + 1e-8
    return wins, conversions, spend, np.array(hourly_spend)


def run(output, days):
    historical = [generate_day(seed) for seed in range(12)]
    model = LogisticRegression().fit(
        np.concatenate([day.features for day in historical]),
        np.concatenate([day.conversions for day in historical]),
    )
    historical_hours = np.concatenate([day.hours for day in historical])
    shares = np.bincount(historical_hours, minlength=24) / len(historical_hours)
    planned = np.r_[0, np.cumsum(shares) * BUDGET]

    validation = [generate_day(seed) for seed in range(100, 108)]
    validation_scores = [
        model.predict_proba(day.features)[:, 1] * TARGET_CPA for day in validation
    ]

    def validation_conversions(candidate, use_model):
        return sum(
            replay(day, score * candidate if use_model else np.full(len(score), candidate), planned)[1]
            for day, score in zip(validation, validation_scores)
        )

    fixed_bid = max((0.10, 0.15, 0.20, 0.30, 0.40, 0.60),
                    key=lambda bid: validation_conversions(bid, False))
    scale = max((0.20, 0.35, 0.50, 0.75, 1.00),
                key=lambda value: validation_conversions(value, True))

    rows = []
    curves = {}
    for scenario in SCENARIOS:
        for seed in range(1000, 1000 + days):
            day = generate_day(seed, scenario)
            ml_bids = model.predict_proba(day.features)[:, 1] * TARGET_CPA * scale
            policies = (np.full(len(day.hours), fixed_bid), ml_bids, ml_bids)
            for policy, bids in zip(POLICIES, policies):
                wins, conversions, spend, hourly = replay(
                    day, bids, planned, paced=policy == POLICIES[-1]
                )
                rows.append(dict(scenario=scenario, seed=seed, policy=policy,
                                 wins=wins, conversions=conversions, spend=spend,
                                 pacing_mae=np.mean(np.abs(hourly - planned[1:]))))
                curves.setdefault((scenario, policy), []).append(hourly)

    with (output / "daily_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Fixed bid ${fixed_bid:.2f} | ML bid scale {scale:.2f} | {days} evaluation days per scenario")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True, layout="constrained")
    fig.suptitle("Daily ad spend under changing auction conditions", fontweight="bold")
    for scenario, ax in zip(SCENARIOS, axes):
        ax.plot(np.arange(25), planned, "--", color="#303641", label="Historical plan")
        for policy, color in zip(POLICIES, ("#965014", "#3f65a8", "#087c69")):
            group = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            conversions = sum(r["conversions"] for r in group)
            spend = sum(r["spend"] for r in group)
            cpa = f"${spend / conversions:.2f}" if conversions else "n/a"
            mae = np.mean([r["pacing_mae"] for r in group])
            print(f"{scenario:22} | {policy:15} | {conversions/days:5.1f} conv/day | "
                  f"CPA {cpa:>6} | {spend/(days*BUDGET):6.1%} spent | ${mae:5.1f} pacing MAE")
            mean_curve = np.mean(curves[scenario, policy], axis=0)
            ax.plot(np.arange(25), np.r_[0, mean_curve], color=color, label=policy)
        ax.set(title=scenario, xlabel="Hour of day", xticks=[0, 6, 12, 18, 24], ylim=(0, 475))
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean cumulative spend ($)")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.savefig(output / "pacing.png", dpi=160)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True, layout="constrained")
    fig.suptitle("Daily conversions across paired simulated auction streams", fontweight="bold")
    labels = ["Fixed", "ML", "ML + pacing"]
    for scenario, ax in zip(SCENARIOS, axes):
        samples = [[r["conversions"] for r in rows
                    if r["scenario"] == scenario and r["policy"] == policy] for policy in POLICIES]
        for position, values in enumerate(samples):
            ax.scatter(np.full(len(values), position), values, alpha=0.26, color="#3f65a8")
            ax.scatter(position, np.mean(values), marker="D", color="#152d4b", label="Mean" if not position else None)
        ax.set(title=scenario, xticks=range(3), xticklabels=labels)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Conversions per day")
    axes[0].legend(frameon=False)
    fig.savefig(output / "conversions.png", dpi=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.output, args.days)
