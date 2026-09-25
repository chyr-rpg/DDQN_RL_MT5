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

I decided to publish the project because I hope others who are interested in quantitative trading can also study this problem 

> **Where can neural networks actually contribute useful information to trading decisions?**

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
# Using the EA on a Live or Demo Account

The EA can also run on a normal MT5 chart after training has been completed in Strategy Tester.

For live use, I recommend treating the workflow as:

```text
Train / test in Strategy Tester
        ↓
Save the learned model and memory
        ↓
Move the saved .dat files to the live terminal
        ↓
Run the EA with TrainingMode = false
        ↓
Let the trained policy + memory handle inference
```

I would normally test this process on a **demo account first** before using it with real capital.

---

## 1. Train the model first

The system is designed to save its learned state so that a later MT5 session can continue from previously trained experience.

During training, keep:

```text
TrainingMode = true
SaveQTable = true
```

and, if you want to preserve the broader memory system:

```text
SaveArchiveMemory = true
SaveReplayBanks = true
SaveDangerMemory = true
SaveQMemory = true
SaveDDEventMemory = true
```

When the EA is removed, the tester stops, or MT5 deinitializes the EA, the current implementation saves the DQN together with enabled memory components. :contentReference[oaicite:0]{index=0}

The unified persistence system can save the main DQN, regime-specific networks, archive memories and replay banks. :contentReference[oaicite:1]{index=1}

---

## 2. Find the saved memory files

When the EA runs inside Strategy Tester, the generated files are stored inside the tester agent's:

```text
MQL5/Files/
```

directory.

Depending on which components are enabled, you may see files similar to:

```text
AdaptiveDDQN_MT5_DQN_MAIN_XAUUSD.dat

AdaptiveDDQN_MT5_DQN_MAIN_XAUUSD_R0.dat
AdaptiveDDQN_MT5_DQN_MAIN_XAUUSD_R1.dat
AdaptiveDDQN_MT5_DQN_MAIN_XAUUSD_R2.dat

AdaptiveDDQN_MT5_ARCHIVE_Episodes_XAUUSD.dat
AdaptiveDDQN_MT5_ARCHIVE_Patterns_XAUUSD.dat
AdaptiveDDQN_MT5_ARCHIVE_Regimes_XAUUSD.dat

AdaptiveDDQN_MT5_REPLAY_Main_XAUUSD.dat
AdaptiveDDQN_MT5_REPLAY_Danger_XAUUSD.dat
AdaptiveDDQN_MT5_REPLAY_DeepBasket_XAUUSD.dat
AdaptiveDDQN_MT5_REPLAY_Efficient_XAUUSD.dat
AdaptiveDDQN_MT5_REPLAY_Recent_XAUUSD.dat

AdaptiveDDQN_MT5_DangerMem_XAUUSD.dat
AdaptiveDDQN_MT5_QMem_XAUUSD.dat
AdaptiveDDQN_MT5_DDEvents_XAUUSD.dat
```

The exact filenames are built from the EA program name and symbol. :contentReference[oaicite:2]{index=2}

A local Strategy Tester directory normally looks similar to:

```text
MetaQuotes/
└── Tester/
    └── Agent-.../
        └── MQL5/
            └── Files/
```

If you cannot find the files directly, searching the MT5 data directory for:

```text
*.dat
```

or the EA name is usually the easiest approach.

---

## 3. Move the trained files to the live terminal

The Strategy Tester and the normal MT5 terminal use different file environments.

For normal demo or live execution, open:

```text
MT5
MetaQuotes\Tester\terminal\Agent(that runs the backtesting)\MQL5
→ File
→ Copy the trained `.dat` files into
MetaQuotes\Terminal\terminal\MQL5\Files

Or, you do not need to manually import the files from inside the EA.

When the EA starts, it automatically attempts to load the DQN and enabled memory systems. The current initialization process restores unified model persistence and then loads Danger Memory, Q-Memory and DD-event memory when those features are enabled. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## 4. Keep the same EA filename

This is important.

The EA uses:

```cpp
MQLInfoString(MQL_PROGRAM_NAME)
```

when constructing many persistence filenames.

For example:

```text
AdaptiveDDQN_MT5_DQN_MAIN_XAUUSD.dat
```

If the model was trained using:

```text
AdaptiveDDQN_MT5.mq5
```

but the live version is renamed to:

```text
AdaptiveDDQN_Live.mq5
```

the program name changes and the EA may look for a different set of `.dat` files.

For the easiest continuation from training to live inference, keep the same main EA filename.

---

## 5. Attach it to the correct chart

The current full implementation trades the **current chart symbol (`_Symbol`)**. :contentReference[oaicite:5]{index=5}

So if the model was trained for:

```text
XAUUSD
```

open an XAUUSD chart and attach the EA there.

Likewise:

```text
EURUSD model → EURUSD chart
SP500 model  → corresponding SP500 broker symbol
```

Broker symbol names can differ, so make sure the live symbol matches the one used when the persistence files were generated.

The default base timeframe is:

```cpp
BaseTF = PERIOD_CURRENT;
```

which means the chart timeframe can also affect the EA's base logic unless you explicitly select a fixed `BaseTF`. :contentReference[oaicite:6]{index=6}

For consistency, I normally keep the same timeframe configuration used during training.

---

## 6. You may switch off online training for live inference

*It is recommended to keep online training `TrainingMode = true` as the neural networks will update weights with new prices*

You can set `TrainingMode = false` to test the trained bot. 
This stops the normal online training and runs the network primarily as an inference policy.

When `TrainingMode = false`, the current implementation sets:

```cpp
currentEpsilon = MinExplorationRate;
```

rather than automatically setting exploration to zero. :contentReference[oaicite:7]{index=7}

This matters because action selection still performs an epsilon-random check before using the neural Q-values. :contentReference[oaicite:8]{index=8}

The current default is:

```text
MinExplorationRate = 0.02
```

so even with training disabled there can still be a small amount of random exploration. :contentReference[oaicite:9]{index=9}

If you want deterministic live inference, use:

```text
TrainingMode = false
MinExplorationRate = 0.0
```

If you intentionally want the agent to retain a small exploratory component, you can leave a non-zero minimum epsilon.

---

## 7. Keep the training and live configurations compatible

The trained DQN should be used with a compatible state architecture.

In particular, avoid changing major model settings between training and live use, such as:

```text
HiddenSize
HiddenSize2
ActionCount
feature branches
timeframe features
regime-bank configuration
state construction
indicator settings
structure settings
symbol
```

The loader checks the network input dimension against the state dimension. If a compatible model is not found, the EA can initialise a new network instead. :contentReference[oaicite:10]{index=10}

This means that successfully finding a `.dat` file does not necessarily guarantee that the model will be used if the live architecture has changed substantially.

---

## 8. Review the trading-risk settings

Before enabling live execution, review the execution settings separately from the neural model.

The current system is still a basket/grid-based trading framework and can hold multiple positions. The default configuration includes:

```text
Lots = 0.01
LotExponent = 1.4
MaxTrades = 10
```

so position exposure can increase as the basket develops. :contentReference[oaicite:11]{index=11}

The code also contains optional equity controls such as:

```text
UseEquityStop
EquityRiskPercent

UseEquityLossStop
EquityLossStopAmount

UseGlobalAccountWatchdog
```

These controls are configurable and some are disabled by default, so they should be reviewed rather than assumed to be active. :contentReference[oaicite:12]{index=12}

For the version published in this repository, I also normally keep:

```text
UseZScoreEmergencyHedge = false
```

if I do not want the risk guard to open an opposite hedge against an existing basket.

---

## 9. Start the EA in MT5

Once the `.dat` files are in the correct `MQL5/Files` folder:

```text
1. Open the intended symbol chart
2. Select the intended timeframe
3. Attach AdaptiveDDQN_MT5
4. Load or review the EA inputs
5. Set TrainingMode = false
6. Set MinExplorationRate = 0.0 if deterministic inference is wanted
7. Enable Algo Trading / AutoTrading in MT5
8. Allow algorithmic trading in the EA properties
```

After initialization, check the:

```text
Experts
Journal
```

tabs.

I recommend confirming that the persisted model was actually loaded before allowing the EA to trade.

If the model is not found or is incompatible, the EA can initialize a new DQN instead of continuing the trained model.

---

## 10. What happens during live operation

Once running, the live process is approximately:

```text
New market data
      ↓
Build current state
      ↓
Determine volatility regime
      ↓
Run DDQN inference
      ↓
Retrieve relevant historical memory
      ↓
Apply danger / decision-support context
      ↓
Q(HOLD), Q(BUY), Q(SELL)
      ↓
Risk and execution checks
      ↓
Open / manage basket
```

The system does not simply use the neural-network output directly.

The broader live decision process can also incorporate:

```text
Persistent Q-memory
Danger memory
Historical episodes
Replay-derived risk context
Regime information
Current basket exposure
Risk controls
```
This is why transferring the memory files together with the DQN is useful: the live system can continue using more than just the learned network weights.

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
