# DDQN-RL-MT5

### A Memory-Augmented Reinforcement Learning Trading System for MetaTrader 5

DDQN-RL-MT5 is a personal research project exploring how reinforcement learning and neural networks can be used directly inside MetaTrader 5.

The system is written natively in **MQL5**. It does not require Python or an external machine-learning runtime during execution.

The main research question is relatively simple:

> **Can a trading system learn from its previous decisions, remember both good and bad trading experiences, and use that information to change future behaviour?**

The current implementation combines a Double Dueling DQN with persistent memory, experience replay, market-regime awareness and risk-sensitive decision support.

The execution layer is still largely based on a **grid / basket trading framework**, so I see this project as an experimental environment for studying neural decision-making rather than a finished general-purpose RL trading solution.

---

## Why I Shared It

This project started as a personal experiment in applying reinforcement learning to trading decisions inside MT5.

I previously ran the system on a demo account across different instruments and trading frequencies. That historical account record was later lost when the demo-server environment became unavailable, so I do not use those earlier results as evidence of performance here.

I decided to publish the project because I am more interested in the broader problem:

> **Where can neural networks actually contribute useful information to trading decisions?**

For example:

- how should market state be represented?
- how should trading outcomes be rewarded?
- can rare drawdown events be remembered more effectively?
- should historical experience influence the current Q-values?
- can different volatility regimes develop different behaviour?
- how much control should the neural model have over a conventional trading framework?

I am still experimenting with these questions, and feedback or alternative approaches are welcome.

---

# Getting Started

## Requirements

You need:

```text
MetaTrader 5
MetaEditor
An MT5 trading or demo account
Historical data for the instruments you want to test
```

The neural network and reinforcement-learning logic run directly inside MQL5.

No external Python environment is required.

---

## Repository Structure

The full implementation has been separated into smaller MQL5 modules so the system is easier to read and modify.

```text
DDQN_RL_MT5/
│
├── README.md
│
├── src/
│   ├── AdaptiveDDQN_MT5.mq5
│   │
│   └── modules/
│       ├── 00_InputsAndGlobals.mqh
│       ├── 01_DDQNNetworkCore.mqh
│       ├── 02_MemoryModels.mqh
│       ├── 03_CoreUtilities.mqh
│       ├── 04_RewardsAndRisk.mqh
│       ├── 05_DecisionSupport.mqh
│       ├── 06_GridRegimeIndicators.mqh
│       ├── 07_StateAndStructureFeatures.mqh
│       ├── 08_MemoryAndReplay.mqh
│       ├── 09_DDQNTraining.mqh
│       ├── 10_DDEventAndDangerLearning.mqh
│       ├── 11_PendingTransitions.mqh
│       ├── 12_TradeExecution.mqh
│       ├── 13_StrategyManager.mqh
│       ├── 14_ZonesAndVisualization.mqh
│       ├── 15_Persistence.mqh
│       └── 16_Runtime.mqh
│
├── doc/
│   ├── architecture.md
│   ├── learning-system.md
│   ├── memory-system.md
│   └── limitations.md
│
├── tested_assets/
│   ├── SP500/
│   ├── eurusd/
│   └── xauusd/
│
└── legacy/
    └── AdaptiveDDQN_MT5_monolithic_reference.mq5
```

The `.mqh` files are not separate EAs.

When the main `.mq5` file is compiled, MetaEditor includes all modules and produces **one `.ex5` Expert Advisor**.

Conceptually:

```text
AdaptiveDDQN_MT5.mq5
        │
        ├── Neural Network
        ├── Memory
        ├── Replay
        ├── Risk
        ├── Execution
        └── Runtime
        │
        ▼
      Compile
        │
        ▼
AdaptiveDDQN_MT5.ex5
```

---

# Installation

Copy the complete source folder into your MT5 Expert Advisor directory.

For example:

```text
MQL5/
└── Experts/
    └── Adaptive-DDQN-MT5/
        ├── AdaptiveDDQN_MT5.mq5
        └── modules/
            ├── 00_InputsAndGlobals.mqh
            ├── ...
            └── 16_Runtime.mqh
```

In MetaTrader 5:

```text
File
→ Open Data Folder
→ MQL5
→ Experts
```

Place the files there and open:

```text
AdaptiveDDQN_MT5.mq5
```

in MetaEditor.

Press:

```text
Compile
```

Only the main `.mq5` file needs to be compiled.

If the build is successful, the EA will appear under:

```text
Navigator
→ Expert Advisors
```

in MetaTrader 5.

---

# Running It in Strategy Tester

I strongly recommend starting with **Strategy Tester** rather than a live chart.

Open:

```text
View
→ Strategy Tester
```

Select the compiled EA and choose:

```text
Instrument
Timeframe
Testing period
Initial deposit
Execution / modelling mode
EA inputs
```

Then start the test.

For research purposes, I normally use the Strategy Tester to observe both the trading behaviour and how the learning system develops over time.

The system may:

```text
observe market state
        ↓
estimate Q-values
        ↓
select an action
        ↓
manage / open positions
        ↓
observe the result
        ↓
calculate reward
        ↓
store the experience
        ↓
replay historical experience
        ↓
update the neural network
```

This process repeats throughout the test.

---

## First Run

On a fresh run, the system may begin with:

- newly initialised neural-network weights,
- little or no replay experience,
- limited historical memory,
- and a relatively high exploration rate.

This means the early part of a training run should not be interpreted in the same way as a mature policy.

The agent needs experience before the learned behaviour becomes meaningful.

---

## Continuing Previous Learning

The system can persist parts of its neural and memory state to disk.

Depending on the enabled components, saved information can include neural-network state and different forms of learned memory.

MT5 normally stores these files inside its `MQL5\Files` or Strategy Tester agent environment.

A typical tester location may look similar to:

```text
MetaQuotes/
Tester/
<TESTER-ID>/
Agent-127.0.0.1-<PORT>/
MQL5/
Files/
```

The exact location depends on your MT5 installation and which tester agent is running.

### Important

Some persistence filenames use:

```cpp
MQLInfoString(MQL_PROGRAM_NAME)
```

so changing the name of the main `.mq5` program can also change the names of the files the EA tries to load.

If you want to continue using existing trained state, keeping the same main EA filename is recommended.

---

# How the System Is Organised

## 1. Market State

The agent does not look only at price.

Its state can include several groups of information:

```text
Basket / Exposure
Indicators
Volatility
Market Structure
Zone / Candle Context
```

These inputs describe both the market and the EA's own current position state.

---

## 2. Multi-Branch Neural Encoding

Different feature groups are processed through separate neural branches before being combined.

Conceptually:

```text
Basket State ─────────┐
Indicators ───────────┤
Volatility ───────────┤
Market Structure ─────┤
Zone / Candle Data ───┤
                      ▼
              Shared Representation
                      ▼
                Dueling DDQN
```

This allows different types of market information to develop their own internal representation before being fused together.

---

## 3. Double Dueling DQN

The policy estimates three main action values:

```text
Q(HOLD)
Q(BUY)
Q(SELL)
```

The dueling architecture separates:

```text
State Value
     +
Action Advantage
     ↓
Q-Values
```

The Double-DQN structure uses separate online and target-network logic to reduce some of the overestimation problems found in standard DQN training.

---

## 4. Experience Replay

The system does not learn only from the latest trade.

Previous transitions can be stored and replayed during training.

The broader memory architecture includes several types of experience, such as:

```text
Main Replay
Recent Experience
Danger Experience
Deep-Basket Experience
Efficient Experience
```

This lets the agent repeatedly revisit different classes of historical behaviour.

---

## 5. Longer-Term Memory

The project also experiments with memory beyond normal replay.

This includes concepts such as:

```text
Episode Memory
Pattern Memory
Regime-Event Memory
Drawdown-Event Memory
Persistent Q-Memory
```

The idea is to combine:

```text
Neural Generalisation
        +
Historical Recall
```

rather than expecting neural-network weights alone to represent every useful past experience.

---

## 6. Regime Awareness

The system can distinguish between different volatility environments.

Different model behaviour can therefore develop under different market conditions instead of assuming that one policy should behave identically all the time.

---

## 7. Decision Support

The final decision is not based only on raw neural-network output.

Before execution, the system can also consider:

```text
Current basket exposure
Historical Q-memory
Danger memory
Drawdown events
Regime context
Risk controls
```

This means the neural network remains the core learning mechanism, but it operates inside a wider decision-support framework.

---

## 8. Trade Execution

The current strategy framework uses basket and grid-style position management.

This is an important limitation of the project.

The neural network is learning inside an execution system whose behaviour is already influenced by:

```text
Basket exposure
Grid additions
Average entry price
Basket profit targets
Drawdown
Position management
```

The repository should therefore not be interpreted as evidence that DDQN itself provides a complete trading strategy.

I mainly use the framework as a practical environment to study how reinforcement learning behaves when it interacts with an actual trading and risk-management system.

---

# Where to Start Reading the Code

If you want to understand the implementation rather than immediately run it, I suggest starting with these modules:

```text
01_DDQNNetworkCore.mqh
```

Neural-network structures, forward propagation and branch encoders.

```text
07_StateAndStructureFeatures.mqh
```

How the market and basket state is converted into neural-network input.

```text
08_MemoryAndReplay.mqh
```

Replay buffers and memory systems.

```text
09_DDQNTraining.mqh
```

Double-DQN targets, network updates and training.

```text
10_DDEventAndDangerLearning.mqh
```

Drawdown-event and danger-state learning.

```text
13_StrategyManager.mqh
```

Where the learning system is connected to the trading logic.

```text
16_Runtime.mqh
```

The MT5 lifecycle, including `OnInit`, `OnTick`, `OnTimer` and shutdown behaviour.

For the broader architecture, see:

[**System Architecture →**](doc/architecture.md)

[**Learning System →**](doc/learning-system.md)

[**Memory System →**](doc/memory-system.md)

[**Limitations →**](doc/limitations.md)

---

# Learning Edition

A smaller implementation is also included for readers who want to understand the basic idea before going through the complete architecture.

```text
src/AdaptiveDQN_MT5_LearningEdition.mq5
```

The Learning Edition focuses on a simpler workflow:

```text
State
  ↓
DQN
  ↓
HOLD / BUY / SELL
  ↓
Reward
  ↓
Network Update
```

I recommend starting there if you are new to reinforcement learning or MQL5 neural-network programming.

The modular DDQN implementation is more suitable once the basic workflow is already familiar.

---

# Tested Markets

I have used the system experimentally across different markets and trading frequencies.

Current example results and configurations are available under:

- [`tested_assets/SP500/`](tested_assets/SP500/)
- [`tested_assets/eurusd/`](tested_assets/eurusd/)
- [`tested_assets/xauusd/`](tested_assets/xauusd/)

These results should be treated as **research observations**, not evidence of future profitability.

Different instruments also use different contract specifications and strategy settings, so results should not be compared as if they were identical experiments.

---

# Research Limitations

There are several important limitations.

The system includes averaging and basket behaviour, meaning exposure can increase when the market moves against existing positions.

Reinforcement learning and historical memory may help influence those decisions, but they do not remove:

```text
Market Risk
Model Risk
Execution Risk
Liquidity Risk
Regime-Change Risk
Tail Risk
Overfitting Risk
```

Backtesting also depends heavily on data quality, modelling assumptions, spread, execution and parameter selection.

For that reason, I currently see the project as more suitable for:

```text
Strategy Tester research
controlled experimentation
demo-account testing
supervised trading-system development
```

rather than unattended live deployment.

---

# Current Research Questions

Some questions I am still exploring include:

- Can neural models reduce harmful grid additions?
- Should the network control execution directly, or act as a decision layer?
- Can dangerous historical episodes improve current risk decisions?
- How much replay should come from recent versus rare events?
- How stable are learned policies across different instruments?
- Can regime-specific policies generalise better than one universal model?
- How should reward distinguish efficient profit from high-risk recovery?

I expect the architecture to continue changing as these questions are tested.

---

# Contributions and Discussion

This repository is mainly shared for technical discussion and experimentation.

I am particularly interested in alternative ideas around:

```text
State representation
Reward design
Replay sampling
Memory architecture
Regime modelling
Risk-aware reinforcement learning
Trading action spaces
```

If you are experimenting with similar ideas, feel free to open an issue or discuss possible improvements.

---

# Disclaimer

This repository is provided for research, educational and technical experimentation.

It does not constitute investment advice or a recommendation to trade.

Historical tests, reinforcement-learning behaviour and simulated results do not guarantee future performance.

Any market experimentation should use independent risk controls and appropriate supervision.

---

## Author

**CYR**

Research interests:

`Reinforcement Learning` · `Algorithmic Trading` · `Quantitative Finance` · `MQL5` · `Adaptive Systems`
