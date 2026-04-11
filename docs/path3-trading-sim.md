# Path 3: Financial Trading Simulation — Ultra-Fast Market Replay for RL

**Status:** EXPLORE AFTER PATH 1 AND PATH 2
**Priority:** Clearest commercial demand, lowest implementation effort
**Target customers:** Quant hedge funds, prop trading firms, fintech, quant researchers

---

## The Problem

RL for trading requires simulating millions of market episodes to train policies. Current tools:

- **FinRL:** Python/PyTorch. Achieved 1,746x speedup with GPU-parallel envs (227K samples/sec), but still Python overhead.
- **Proprietary simulators:** Fast but closed-source. Every fund builds their own.
- **Gym-based environments:** StockTradingEnv etc. Single-threaded Python. Painfully slow.

The environment is conceptually simple:
```
State:  price history window + portfolio positions + cash
Action: buy/sell/hold per asset (discrete or continuous sizing)
Reward: portfolio PnL, Sharpe ratio, or risk-adjusted return
Step:   advance one time period, update prices from historical data
```

This is a perfect `c_step()` — pure arithmetic on arrays.

## The Opportunity

A C/CUDA market replay environment in PufferLib would:
- Simulate millions of trading episodes per second (vs thousands in Python)
- Support multi-asset, multi-agent, multi-timeframe
- Be the fastest open-source trading simulator available
- Attract quant researchers → enterprise customers

**FinRL benchmark:** 227K samples/sec with 2048 parallel envs on GPU (Python/PyTorch).
**PufferLib target:** 10M-50M samples/sec with 4096+ parallel envs in C/CUDA.

## Data Requirements

### Free / Open Source Data
| Source | Data | Granularity | Coverage |
|---|---|---|---|
| Yahoo Finance (yfinance) | US equities, ETFs | Daily, 1-min | 20+ years |
| Binance API | Crypto | 1-min, tick | 5+ years |
| Alpaca Markets (free tier) | US equities | 1-min | 5+ years |
| Kaggle datasets | Various | Daily, hourly | Varies |
| LOBSTER | Limit order book | Tick-level | Academic (paid) |

### Data Pipeline
1. Download historical OHLCV data (Python script, one-time)
2. Convert to binary format (flat arrays of floats, memory-mapped)
3. Load into C environment via mmap at startup
4. Each `c_step()` advances the price pointer — zero parsing overhead

### Data Format
```c
// Pre-processed binary file: [num_assets × num_timesteps × num_features]
// Features per timestep: open, high, low, close, volume (5 floats)
// Total: num_assets * num_timesteps * 5 * sizeof(float)
```

## Validation Plan — Phase 1: Proof of Concept
### Step 1: Build minimal single-asset market replay env

```
ocean/market/
  market.h       — Core environment: price replay, portfolio tracking
  binding.c      — Python bindings with data loading
```

**Environment struct:**
```c
typedef struct {
    Log log;
    float* observations;    // price window + portfolio state
    int* actions;           // 0=hold, 1=buy, 2=sell (or continuous sizing)
    float* rewards;
    unsigned char* terminals;
    
    // Market data (pre-loaded, shared across envs)
    float* price_data;      // [num_timesteps × num_features]
    int num_timesteps;
    int window_size;        // observation lookback
    
    // Portfolio state
    float cash;
    float position;         // shares held
    float entry_price;
    int tick;               // current timestep
} Market;
```

**Observation:** last N prices (normalized) + position + cash + unrealized PnL
**Action:** discrete {sell, hold, buy} or continuous [-1, 1] sizing
**Reward:** step PnL or episode Sharpe ratio
**Terminal:** end of data window

### Step 2: Data pipeline

```bash
# Download S&P 500 daily data, 20 years
python scripts/download_market_data.py --ticker SPY --start 2004-01-01 --output data/spy.bin

# Convert to binary format for C
python scripts/convert_to_binary.py --input data/spy.csv --output data/spy.bin
```

### Step 3: Benchmark against FinRL

Compare:
- PufferLib C market env: samples/sec with 4096 parallel envs
- FinRL StockTradingEnv: samples/sec (CPU baseline)
- FinRL GPU-parallel: samples/sec (their best reported)

**Success criteria:**
- [ ] >10x faster than FinRL's GPU-parallel implementation
- [ ] Correct PnL calculation (validated against Python reference)
- [ ] Support 4096+ parallel episodes

### Step 4: Train a basic policy

Use PufferLib's PPO to train a trading agent on SPY daily data:
- Split: 2004-2020 train, 2020-2024 test
- Baseline: buy-and-hold
- Metric: Sharpe ratio, max drawdown, total return

**Success criteria:**
- [ ] Policy trains and converges
- [ ] Out-of-sample performance is reasonable (doesn't need to beat market, just show learning)
- [ ] Training completes in < 5 minutes on RTX 4090

## Validation Plan — Phase 2: Multi-Asset & Features
### Multi-asset support
- Trade N assets simultaneously (e.g., S&P 500 components)
- Portfolio allocation as action (N-dimensional continuous)
- Correlation-aware reward (Sharpe of portfolio, not individual assets)

### Technical indicators as observations
- Moving averages, RSI, MACD, Bollinger bands
- All computed in C during `c_step()` (no Python feature engineering)

### Transaction costs & slippage
- Commission per trade
- Spread modeling
- Market impact for large orders

### Realistic constraints
- Maximum position sizes
- Short-selling restrictions (configurable)
- Margin requirements

## Validation Plan — Phase 3: Production Product
### "PufferTrader" package

```python
import puffer_trader

# Load market data
env = puffer_trader.create_env(
    data="data/sp500_daily.bin",
    assets=["AAPL", "GOOGL", "MSFT"],
    window=60,           # 60-day lookback
    commission=0.001,    # 10 bps
    total_agents=4096
)

# Train with PufferLib
from pufferlib import pufferl
pufferl.train(env, total_timesteps=50_000_000)  # Minutes on RTX 4090
```

### Multi-agent / adversarial trading
- Multiple agents trading same market (market impact)
- Adversarial training (market maker vs taker)
- Zero-sum game formulation

### LLM integration (connects to Path 1)
- LLM generates trading strategies as code
- PufferLib evaluates them at millions of episodes/sec
- GRPO/Meta-Harness optimizes strategy code

## Key Risks

1. **"Past performance does not predict future results."** RL trading agents often overfit to historical data. Need proper train/test splits, walk-forward validation, and regime-change robustness.

2. **Data quality matters more than simulation speed.** Bad data → bad policies, no matter how fast. Need clean, adjusted price data (dividends, splits).

3. **Regulatory considerations.** If selling to hedge funds, may need compliance review. Open-source simulator itself is fine, but marketed trading tools may attract scrutiny.

4. **Competition.** Every quant fund has internal simulators. The value proposition is: "start with something fast and open-source instead of building from scratch." Need to be significantly better than FinRL.

5. **Market data licensing.** Some data sources have redistribution restrictions. Need to provide tools to download data, not redistribute data itself.

## Revenue Model

- **Open-source core:** PufferTrader simulator (MIT, drives adoption)
- **Enterprise:** Custom asset classes, alternative data integration, priority support
- **Data services:** Pre-processed binary market data files (subscription)
- **Consulting:** Custom strategy development, backtesting infrastructure
- **Cloud:** Hosted ultra-fast backtesting API

**Pricing benchmark:** QuantConnect charges $8-60/month for retail. Enterprise backtesting platforms charge $50K-500K/year.

## Decision Point

After Phase 1 (2-3 days), we will have data on:
1. Can we build a correct market simulator in C? (PnL validation)
2. Is it actually 10x+ faster than FinRL?
3. Can PufferLib's PPO train a reasonable trading policy?

If YES → proceed to Phase 2 (multi-asset, features).
If speed advantage is marginal → question whether C overhead is worth it vs FinRL's GPU approach.
If trading policies don't learn → may be a reward shaping problem, not a speed problem.
