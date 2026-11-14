# Ad Auction Bidding and Budget Pacing

A reproducible Python experiment that compares three policies under a $450 daily campaign budget. Each synthetic auction clears when the campaign bid meets the highest competing bid. A win costs the campaign its own bid.

## Run it

Use Python 3.11 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py
```

Use `--days 5 --output quick-results` for a shorter run. The default run recreates the two PNG charts and the per-day CSV in `results/`. The checked-in summary is in `benchmark.txt`.

## Experiment design

- Fit logistic regression on 12 synthetic historical days to estimate conversion probability conditional on showing the ad. Features are device, placement, and an evening time-of-day feature.
- Select the fixed bid and ML bid scale separately on eight validation days, maximizing realized conversions under the daily budget. An ML bid is predicted conversion probability times a $35 CPA target times the selected scale. The target sets the bid scale and does not guarantee realized CPA.
- Build the cumulative spend plan using the historical hourly distribution of auction opportunities. The pacing controller updates the ML bid multiplier hourly based on the gap between actual spend and the plan, with the multiplier bounded between 0.15 and 2.5.
- Evaluate on 30 held-out days per scenario. All policies in a scenario and seed see identical competing bids and potential conversion outcomes. Only auction winners receive conversion credit. A bid is capped at the remaining daily budget.
- Compare average daily conversions, aggregate CPA (total spend divided by conversions), budget utilization, and pacing error (mean absolute hourly deviation from the planned cumulative spend curve).

## Default run

| Scenario | Fixed bid | ML bid | ML bid + pacing | ML pacing error before → after |
| --- | ---: | ---: | ---: | ---: |
| Normal | 25.9 | 36.9 | 37.6 | $24.8 → $6.4 |
| Midday price spike | 29.6 | 37.5 | 30.1 | $37.5 → $15.0 |
| Evening traffic surge | 26.8 | 39.1 | 41.7 | $32.3 → $12.7 |

Conversion columns show mean conversions per day. During the price spike, the paced bidder raises bids when behind plan. That behavior can consume budget during expensive hours and leave less for later auctions. The observed trade-off suggests testing a pacing controller that accounts for both spend and expected auction value.

![Spend curves](results/pacing.png)

![Daily conversions](results/conversions.png)

The simulator supplies potential conversion outcomes for every historical impression. Real auction logs usually reveal conversion feedback only after a win and an ad exposure, and outcomes can arrive late. The replay assumes competing bids remain fixed when the campaign changes policy. Results describe these simulated conditions and need logged-data or online experimentation before any production interpretation.
