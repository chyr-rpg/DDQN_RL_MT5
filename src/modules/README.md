# Source Modules

These `.mqh` files form the full Adaptive-DDQN-MT5 EA when included by the main `.mq5` entry file.

The numbering is intentional: it documents the dependency/order of the original program and makes the architecture easy to navigate on GitHub.

This first modularization pass avoids changing algorithms or moving individual functions across dependency boundaries. Once the modular build compiles and reproduces the monolithic Strategy Tester behavior, later refactors can introduce cleaner interfaces/classes with much lower regression risk.
