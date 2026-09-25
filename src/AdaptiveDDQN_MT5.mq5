
//+------------------------------------------------------------------+
//| Real DDQN (branch encoders + 2-layer fusion)                  |
//| Trades CURRENT chart symbol (_Symbol)                            |
//| Quality-learning + archive-aware risk reward + stale-input cleanup        |
//|  - Persistent Q-memory + replay + target net                     |
//|  - Pending transitions for delayed outcome learning              |
//|  - DD-event memory kept as archive / training context only       |
//|  - DangerBrain retained as live protective layer                 |
//+------------------------------------------------------------------+
#property copyright "2025, CYR"
#property strict

#include <Trade/Trade.mqh>
#include <Arrays/ArrayDouble.mqh>

CTrade trade;

// Modular build: each include below is a direct extraction of the original monolithic source.
// Keep the main program filename unchanged if you need compatibility with existing .dat persistence files,
// because the EA derives persistence filenames from MQL_PROGRAM_NAME.

#include "modules\\00_InputsAndGlobals.mqh"
#include "modules\\01_DDQNNetworkCore.mqh"
#include "modules\\02_MemoryModels.mqh"
#include "modules\\03_CoreUtilities.mqh"
#include "modules\\04_RewardsAndRisk.mqh"
#include "modules\\05_DecisionSupport.mqh"
#include "modules\\06_GridRegimeIndicators.mqh"
#include "modules\\07_StateAndStructureFeatures.mqh"
#include "modules\\08_MemoryAndReplay.mqh"
#include "modules\\09_DDQNTraining.mqh"
#include "modules\\10_DDEventAndDangerLearning.mqh"
#include "modules\\11_PendingTransitions.mqh"
#include "modules\\12_TradeExecution.mqh"
#include "modules\\13_StrategyManager.mqh"
#include "modules\\14_ZonesAndVisualization.mqh"
#include "modules\\15_Persistence.mqh"
#include "modules\\16_Runtime.mqh"
