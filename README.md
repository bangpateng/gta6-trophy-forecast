# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bangpateng/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bangpateng/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five quantiles of its Exophase earn rate at the case's target, from the house's recorded reads of the trophy page up to the case's checkpoint.

The model reads the case's own history and picks one of two regimes. When the history is flat, the page is stale and the case straddles Exophase's first refresh, so it expects the refresh the kit documents (tracked players 638 → 3,868, trophies earned 943 → 4,276) and splits the new players' earners across trophies by a power law in the old rate, calibrated to the documented totals. When the history grows, it fits each trophy's logit rate against log time over the last 24 and 48 hours, blends and shrinks the slopes toward the cross-trophy median, and reads the shrunk slope at the target, with a spread that combines the house's random-walk scale with the slope's own uncertainty. Standard library only, deterministic, offline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=0h2zmzc4&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
