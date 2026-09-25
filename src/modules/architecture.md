# Adaptive-DDQN-MT5 — Modular Source Build

This package restructures the original large MQL5 implementation into smaller `.mqh` modules while keeping one thin `.mq5` entry point.

The split is organized around the actual research architecture: state construction, branch encoders, reward/risk logic, decision support, replay and memory, DDQN training, drawdown-event learning, execution, persistence and runtime orchestration.

See [`doc/modular-architecture.md`](doc/modular-architecture.md) for the module map and compile notes.

The original monolithic file is retained under `legacy/` as a reference during validation.

> This refactor is intended to preserve behavior, but it has not been compiled in MetaEditor in this environment. Compile the main file and run a controlled Strategy Tester comparison before treating it as the canonical build.
