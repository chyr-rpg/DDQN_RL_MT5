//+------------------------------------------------------------------+
//| 00_InputsAndGlobals.mqh                                         |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Inputs, limits, global runtime state, indicator handles, configu|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

//  LIMITS / GLOBALS
#define MAX_SYMBOLS   16
#define REGIME_COUNT  3
#define MINI_DIM      8

int          gSymbolCount                 = 0;
string       gSymbols[MAX_SYMBOLS];
int          gMagics[MAX_SYMBOLS];
int          gPositionsCount[MAX_SYMBOLS];
CArrayDouble gTrades[MAX_SYMBOLS];
CArrayDouble gTradeLots[MAX_SYMBOLS];
datetime     gFirstTradeTime[MAX_SYMBOLS];
bool         gDeepBasketStrongEntryArmed[MAX_SYMBOLS];
datetime     gDeepBasketStrongEntryArmTime[MAX_SYMBOLS];
int          gDeepBasketStrongEntryClosedCount[MAX_SYMBOLS];

struct SDZone
{
   string   symbol;
   double   high;
   double   low;
   datetime startTime;
   datetime endTime;
   datetime breakoutTime;
   bool     isDemand;
   bool     tested;
   bool     broken;
   string   name;
};
SDZone zones[];
input int MaxZonesTracked = 50;

//--- General
input string   General              = "==== General settings ====";
input int      InpMagic             = 220111;
input double   Lots                 = 0.01;
input double   LotExponent          = 1.4;
input int      MaxTrades            = 10;
input bool     UseDeepBasketStrongFirstTrade = true;
input int      DeepBasketStrongFirstTradeMinClosed = 4;
input double   DeepBasketStrongFirstTradeMult = 2.0;
input double   ChannelCloseAlpha    = 0.30;    // close at avg +/- alpha * channelStep
input bool     UseHybridBasketTP    = true;
input int      LegacyTakeProfitPts  = 100;     // old-version style reference points
input double   BasketTPDepthFactor  = 0.20;    // deeper baskets tighten TP toward breakeven
input double   BasketTPMinFactor    = 0.15;    // never tighter than 15% of channel target
input int      Slippage             = 30;

//--- Base timeframe
input string          BaseTFSettings = "==== Base timeframe (EA logic) ====";
input ENUM_TIMEFRAMES BaseTF         = PERIOD_CURRENT;

//--- Grid (single-step BaseTF grid)
input string   GridSettings              = "==== Grid settings (single-step BaseTF) ====";

input bool     UseDynamicPips            = true;
input int      DefaultPips               = 120;      // POINTS fallback
input int      Depth                     = 24;       // bars per selected TF
input double   PipsFactor                = 3.0;      // same divisor logic
input double   GridRangeSmooth           = 0.30;     // same smoothing logic
input bool     UseStdevGridExpansion    = true;     // widen later basket adds using rolling channel-width stdev
input double   GridStdevCoeff           = 0.35;     // per-leg channel-width sigma widening from 4th position onward
input double   GridStdevMaxSigmaMult    = 2.0;      // cap total sigma widening
input int      GridStdevStartPosition   = 4;        // first position index using sigma widening
input int      GridStdevMaxPosition     = 10;       // stop increasing beyond this total basket size
input int      GridWidthStatsBars       = 100;      // rolling sample count for channel-width mean/stdev
input double   GridRollingMeanMaxWeight = 0.75;     // max pull toward rolling mean when current width is too narrow

// leg allocation
// includes first trade
// persistent DD can force extreme state
input double   ExtremeDDMoneyTrigger     = 3000.0;
input int      ExtremeDDHoursTrigger     = 2;

// chart display
input bool     ShowGridStatusPanel       = true;
input int      GridPanelCorner           = CORNER_LEFT_UPPER;
input int      GridPanelX                = 10;
input int      GridPanelY                = 20;

//--- Indicators (BaseTF)
input string   IndicatorSettings    = "==== Indicator settings (BaseTF) ====";
input int      RSI_Period           = 14;
input bool     UseCCI               = false;
input int      CCI_Period           = 55;
input int      CCI_Level            = 500;

//--- Higher TF feature packs
input string          HTFFeatures    = "==== Optional HTF features ====";
input bool            UseH1Features  = true;
input ENUM_TIMEFRAMES H1_TF          = PERIOD_H1;
input bool            UseH4Features  = true;
input ENUM_TIMEFRAMES H4_TF          = PERIOD_H4;

input int             EMA_Period     = 50;
input int             MACD_Fast      = 12;
input int             MACD_Slow      = 26;
input int             MACD_Signal    = 9;
input int             RVI_Period     = 14;

//--- ATR for regimes
input string   ATRSettings          = "==== ATR settings ====";
input int      ATR_FastPeriod       = 14;
input int      ATR_SlowPeriod       = 100;

//--- Exit / risk
input string   ExitSettings         = "==== Exit & risk settings ====";
input bool     UseTrailingStop      = false;
input int      TrailStart           = 100;
input int      TrailStop            = 100;

input bool     UseEquityStop        = false;
input double   EquityRiskPercent    = 20.0;

input bool     UseEquityLossStop        = false;
input double   EquityLossStopAmount     = 4900.0;   // money-based floating loss limit
input bool     EquityLossStopClosePositions = true;
input int      EquityLossStopCooldownSeconds = 1; // pause after forced close, then resume
// strong terminal penalty to DQN when equity stop hits
input string   GlobalWatchdogSettings   = "==== Global account watchdog ====";
input bool     UseGlobalAccountWatchdog = true;
input bool     WatchAllAccountPositions = false;   // true = include every open position on account
input int      WatchdogMagicMin         = 220100;  // used when WatchAllAccountPositions=false
input int      WatchdogMagicMax         = 220199;  // used when WatchAllAccountPositions=false


input string   ProfitPauseSettings      = "==== Profit pause after profit target ====";
input bool     UseProfitPause           = false;
input double   ProfitPauseTargetAmount  = 300.0;
input int      ProfitPauseDurationHours = 24;


//--- Virtual equity budget
input string   BudgetSettings       = "==== Virtual equity budget ====";
input double   EquityBudget         = 100000.0;
input bool     UsePerSymbolVirtualBudget = true;

//--- RL / DQN
input string   DQNSettings          = "==== DQN settings ====";
input bool     UseDQN               = true;

input int      ActionCount          = 3;      // 0 hold,1 buy,2 sell
input double   ExplorationRate      = 0.30;
input double   ExplorationDecay     = 0.998;
input double   MinExplorationRate   = 0.02;
input int      TrainingFreq         = 10;
input bool     SaveQTable           = true;

// DQN network
input string   DQNNetSettings       = "==== DQN network ====";
input int      HiddenSize           = 36;
input int      HiddenSize2          = 18;
input double   DQNLearningRate      = 0.001;
input double   DQNGamma             = 0.95;
input bool     UseInputNorm         = false;
input int      BranchEncoderMinWidth = 6;
input int      BranchEncoderH1Cap    = 64;
input int      BranchEncoderH2Cap    = 32;

// Step 1 state-architecture upgrade
input string   DDQNBranchSettings   = "==== DDQN branch / timeframe scaffold ====";
input bool     UseStateV2SelfAwareness = true;
input bool     StateNormalizationL2   = true;
input bool     UseBranchScaffold       = true;   // routing scaffold only in this step

input ENUM_TIMEFRAMES TF_EXEC         = PERIOD_CURRENT;
input ENUM_TIMEFRAMES TF_MID          = PERIOD_M15;
input ENUM_TIMEFRAMES TF_LONG         = PERIOD_H1;
input ENUM_TIMEFRAMES TF_STRUCT_EXT   = PERIOD_H4;

input bool     UseTFExecFeatures      = true;
input bool     UseTFMidFeatures       = true;
input bool     UseTFLongFeatures      = true;
input bool     UseTFStructExtFeatures = false;

input bool     UseBasketBranch        = true;
input bool     UseIndicatorBranch     = true;
input bool     UseVolatilityBranch    = true;
input bool     UseStructureBranch     = true;
input bool     UseZoneCandleBranch    = true;
input int      BasketAgeNormBars      = 288;
input double   StateDistAtrClamp      = 8.0;
input double   StateBudgetRatioClamp  = 5.0;
input int      StructLookbackExec    = 24;
input int      StructLookbackMid     = 24;
input int      StructLookbackLong    = 24;
input double   StructDistClampAtr    = 8.0;
input double   StructAmpClampAtr     = 12.0;
input int      StructHistoryBarsExec = 2880;
input int      StructHistoryBarsMid  = 960;
input int      StructHistoryBarsLong = 720;
input int      StructMaxSwingsPerTF  = 80;
input double   SwingReversalAtrMult  = 1.25;

// Double Dueling / stability controls
input bool     UseDoubleDQN         = true;
input bool     UseHuberLoss         = true;
input double   HuberDelta           = 1.0;
input bool     UseGradientClipping  = true;
input double   GradientClipValue    = 1.0;

// Multi-regime DQN bank
input string   RegimeBankSettings   = "==== Multi-regime DQN bank (3 models) ====";
input bool     UseRegimeBank        = true;
input bool     TrainAllRegimes      = true;
input bool     TrainAllRegimesWarmupOnly = true;
input int      RegimeWarmupReplayCount = 1500;
input bool     UseSoftRegimeInference = true;
input double   RegimeInferenceBlendWidth = 0.22;
input double   DangerRegimeMatchBonus = 0.08;
input double   DangerRegimeMismatchPenalty = 0.05;
input double   RegimeLowThresh      = 0.85;
input double   RegimeHighThresh     = 1.25;


// Reward architecture (risk-first DDQN)
input string   RewardSettings       = "==== DDQN reward architecture (risk-first) ====";
input double   RewardV2BasketProfitScale = 1.00;
input double   RewardV2RecoveryQualityScale = 1.00;
input double   RewardV2ProfitToMaxDDScale  = 1.10;
input double   RewardV2OneRoundBonus       = 1.50;
input double   RewardV2CleanCycleBonus     = 0.90;
input double   RewardV2CleanCycleMaxDD     = 0.02;
input bool     UseQualityAdaptiveEntry  = true;

input double   RewardV2ReversalPenaltyScale= 0.70;
input double   RewardV2OpenBaseCost        = 0.08;

input double   RewardV2AddPenaltyScale     = 0.15;
input double   RewardV2AddPenaltyQuadratic = 0.05;
input int      RewardV2DeepBasketStartPositions = 4;
input double   RewardV2DeepBasketPenaltyScale = 0.60;
input double   RewardV2AgingAddPenaltyScale = 0.12;
input double   RewardV2FailedBasketPenaltyScale = 0.65;

input double   RewardV2DDPenaltyScale      = 1.80;
input double   RewardV2EpisodeMaxDDPenaltyScale = 3.80;
input double   RewardV2DangerPenaltyScale  = 1.00;
input double   RewardV2MarginPenaltyScale  = 0.85;
input double   RewardV2AgePenaltyPerDay    = 0.14;
input double   RewardV2RewardClamp         = 10.0;

input bool     UseArchiveAwareReward            = true;
input int      ArchiveRewardScanLimit           = 600;
input double   RewardV2ArchiveOpenRiskPenaltyScale  = 0.20;
input double   RewardV2ArchiveCloseRiskPenaltyScale = 1.10;
input double   RewardV2ArchiveCleanBonusScale       = 0.45;
input double   RewardV2ArchivePatternPenaltyScale   = 0.55;
input double   RewardV2ArchiveRegimePenaltyScale    = 0.35;
input double   RewardV2RiskyProfitPenaltyScale      = 1.30;

input string   ArchiveLiveSettings                = "==== Archive live similarity / retrieval ====";
input bool     UseArchiveLiveSimilarity          = true;
input int      ArchiveLiveScanLimit              = 800;
input double   ArchiveLiveBlendWeight            = 0.18;
input double   ArchiveLiveMinBlendWeight         = 0.05;
input double   ArchiveLiveRecentHalfLifeDays     = 14.0;
input double   ArchiveLivePatternCautionScale    = 0.18;
input double   ArchiveLiveRegimeCautionScale     = 0.14;
input bool     UseArchiveAddRiskGate             = false;
input int      ArchiveAddRiskGateMinPositions    = 4;
input double   ArchiveAddRiskGateThreshold       = 2.30;
input double   ArchiveAddRiskGateCleanOffset     = 0.50;
input bool     UseEfficientPeriodLiveSupport     = true;
input double   EfficientPeriodLiveSupportScale   = 0.16;
input double   DeepSequenceAddRiskScale          = 0.16;

input string   DecisionSupportSettings           = "==== Unified live decision support ====";
input bool     UseDecisionSupportCache          = true;
input int      DecisionSupportRefreshBars       = 1;

input string   SmartAddGateSettings             = "==== Smart basket add gate ====";
input bool     UseSmartAddGate                  = true;
input double   AddGateRiskWidenThreshold        = 0.50;
input double   AddGateRiskBlockThreshold        = 0.74;
input double   AddGateSpacingMaxMult            = 2.20;
input double   AddGateLotMinScale               = 0.35;
input double   AddGateAgeSoftDays               = 1.50;

// Regime weighting
input string   RegimeWeightSettings = "==== Regime weighting ====";
input double   ExtremeRewardBoost   = 3.0;
input double   MildRewardScale      = 0.85;
input bool     SeparateExtremeStates= true;

// Training mode
input string   TrainingSettings     = "==== Training mode ====";
input bool     TrainingMode         = true;

input string   QMemorySettings      = "==== Persistent Q-memory ====";
input bool     UseQMemory           = true;
input bool     SaveQMemory          = true;
input int      MaxQMemEntries       = 30000;
input double   QMemMergeSim         = 0.97;
input double   QMemFeatureEMA       = 0.10;
input double   QMemBlendWeight      = 0.40;
input double   QMemMinConfidence    = 0.08;
input double   QMemPruneMinScore    = 0.10;

// resume behavior
input string   ResumeSettings       = "==== Resume / continuation ====";
input string   UnifiedPersistenceSettings = "==== Unified model / archive / replay persistence ====";
input bool     SaveArchiveMemory = true;
input bool     SaveReplayBanks = true;
input bool     SaveMainReplayBank = true;
input bool     SaveRecentReplayBank = true;

input bool     ResumeLowEpsilon     = true;
input double   LoadedEpsilonFactor  = 0.10;

input string   Step2Settings         = "==== Step 2 replay/target settings ====";
input bool     UseReplayBuffer       = true;
input int      ReplayCapacity        = 5000;
input int      ReplayBatchSize       = 16;
input int      ReplayWarmup          = 300;
input int      ReplayTrainIters      = 2;
input int      TargetSyncFreq        = 500;
input bool     UseTargetNet          = true;
input double   ReplayPriorityEps      = 0.01;

input string   Step11Settings         = "==== Step 11 memory/replay architecture ====";
input int      MaxEpisodeMemory       = 50000;
input int      MaxPatternMemory       = 20000;
input int      MaxRegimeEventMemory   = 10000;

input bool     UseDangerReplayBank     = true;
input bool     UseDeepBasketReplayBank = true;
input bool     UseEfficientReplayBank  = true;
input int      DangerReplayCapacity    = 30000;
input int      DeepBasketReplayCapacity = 40000;
input int      EfficientReplayCapacity  = 60000;
input int      RecentReplayCapacity     = 10000;

input int      DeepBasketAddThreshold = 3;
input double   DangerReplayDDThresholdPct = 0.05;
input double   EfficientPeriodMinRewardEfficiency = 0.50;
input int      EfficientPeriodMinTrades = 3;


input bool     UseRecentReplayBank            = true;
input bool     UseBankAwareReplaySampling      = true;
input double   ReplaySampleWeightMain          = 0.46;
input double   ReplaySampleWeightRecent        = 0.18;
input double   ReplaySampleWeightDanger        = 0.08;
input double   ReplaySampleWeightDeepBasket    = 0.08;
input double   ReplaySampleWeightEfficient     = 0.20;
input bool     UseDangerBrainReplayHook        = false;
input double   DangerReplayObsAlpha            = 0.10;
input int      EfficientPeriodWindowMinutes = 240;
input double   EfficientPeriodMaxDDPct = 0.04;
input bool     EfficientSegmentByContext      = true;
input double   EfficientSegmentMinPerMin      = 0.0010;
input double   EfficientContextBreakTolerance = 0.60;
input double   DeepSequenceEarlyBoost         = 1.60;
input double   DangerReplayEarlyStateBoost    = 1.35;
input double   DangerReplayLateRescueWeight   = 0.65;
input double   DeepReplayLateRescueWeight     = 0.55;
input double   ReplayRiskyProfitThreshold     = 1.25;
input double   ReplayMatureEfficiencyBoost    = 0.60;
input double   ReplayAntiPatternDecayScale    = 0.35;
input bool     UseReplayAwareReward           = true;
input int      ReplayRewardScanLimit          = 400;
input double   ReplayRewardHalfLifeDays       = 10.0;
input double   RewardV2ReplayOpenAntiPatternScale = 0.20;
input double   RewardV2ReplayCloseAntiPatternScale = 1.20;
input double   RewardV2ReplayEfficientBonusScale   = 0.60;
input double   RewardV2ReplayRecentCautionScale    = 0.30;
input double   RewardV2ReplayDenseAntiPatternScale = 0.35;

input string   ReplayDiagnosticsSettings      = "==== Replay diagnostics ====";
input bool     UseReplayDiagnostics           = true;
input bool     ReplayDiagnosticsToLog         = true;
input int      ReplayDiagnosticsPrintEveryBars = 24;
input bool     ReplayDiagnosticsInStatusPanel = true;

input double   DangerBankPriorityAlpha        = 0.70;
input double   DeepBankPriorityAlpha          = 0.65;
input double   EfficientBankPriorityAlpha     = 0.60;


// soft target updates
input bool     UseSoftTargetUpdate   = true;
input double   SoftTargetTau         = 0.003;

// pending transitions
input string   PendingSettings      = "==== Pending transition learning ====";
input bool     UsePendingTransitions= true;
input int      MaxPendingTransitions= 128;
input bool     UseRealPostActionTransitions = true;
input int      PendingExpireMinutes = 240;
input double   PendingOpenRewardScale = 0.10;
input double   PendingCloseRewardScale= 1.00;
input double   PendingGoodCloseBonus  = 1.0;

input string   DenseBasketLearningSettings = "==== Dense basket-health feedback ====";
input bool     UseDenseBasketHealthReward  = true;
input double   DenseBasketPnLDeltaScale    = 28.0;
input double   DenseBasketDDDeltaPenaltyScale = 6.0;
input double   DenseBasketDDRecoveryScale  = 3.0;
input double   DenseBasketAddPenaltyScale  = 1.20;
input double   DenseBasketStallAgePenaltyScale = 0.10;
input double   DenseBasketOneRoundProgressScale = 0.40;
input double   DenseBasketOpenReturnClamp  = 0.08;
input double   DenseBasketRewardClamp      = 2.5;

input string   AddSpacingGateSettings      = "==== Add spacing gate ====";
input bool     UseMinAddSpacingGate        = true;
input double   MinAddSpacingPips           = 4.0;
input double   MinAddSpacingATRFrac        = 0.08;
input double   MinAddSpacingPricePct       = 0.0002;
input bool     UseMinOpenIntervalGate      = true;
input int      MinSecondsBetweenOpens      = 30;
input int      MinExecBarsBetweenOpens     = 1;


input string   ZScoreRiskSettings      = "==== Exact StatArb z-score risk guard ====";
input bool     UseZScoreRiskGuard      = true;

// Market 1 (always available)
input string   ZScoreSymbol2           = "XAGUSD";
input ENUM_TIMEFRAMES ZScoreTF         = PERIOD_M15;
input int      ZScorePeriod            = 100;
input double   ZScoreExtremeThreshold  = 2.0;

// Optional Market 2
input bool     UseSecondZScoreMarket   = true;
input string   ZScoreSymbol2_B         = "EURUSD";
input int      ZScorePeriod_B          = 100;
input double   ZScoreExtremeThreshold_B = 2.0;

// Combination mode:
// - if UseSecondZScoreMarket=false -> Market 1 only
// - if UseSecondZScoreMarket=true and UseZScoreOrRule=true -> pause when either market is extreme
input bool     UseZScoreOrRule         = true;

input int      ZScoreResumeQuietBars   = 4;
input bool     ZScoreBlockAllTrading   = true;
input bool     UseZScoreCloseSmallBasket = true;
input int      ZScoreCloseBasketMaxTrades = 2;
input bool     UseZScoreEmergencyHedge   = true;
input int      ZScoreEmergencyHedgeMagicOffset = 700001;
input double   ZScoreEmergencyHedgeLotFactor = 1.0;
input bool     UseRecoveryAfterEmergencyHedge = false;
input int      RecoveryRestartQuietBars  = 6;
input bool     RecoveryCloseAtBreakevenOnly = true;
input double   RecoveryCombinedCloseMoney = 0.0;


input string   Step3Settings          = "==== Step 3 DD-event memory ====";
input bool     UseDDEventMemory       = true;

// soft/hard trigger on EA virtual equity DD
input double   SoftDDTriggerMoney     = 200.0;
input double   HardDDTriggerMoney     = 400.0;

// how much context to preserve around DD event
input int      BasketsBeforeDD        = 15;
input int      BasketsAfterDD         = 15;

// rolling buffers
input int      BasketHistoryCapacity  = 5000;
input int      TickTraceCapacity      = 100000;

// archive limits
input int      MaxDDEventsStored      = 3000;
input int      MaxTicksPerEvent       = 6000;
input bool     DDEventTraceOnBars     = true;
input bool     DDEventTraceUseBaseTF  = true;
input ENUM_TIMEFRAMES DDEventTraceTF  = PERIOD_M5;
input bool     SaveDDEventMemory      = true;

// phase-aware DD-event decision layer
input bool     UseDDEventBias         = true;  // bias similar historical DD contexts toward safer choices
input double   DDEventMinSim          = 0.80;
input int      DDEventMaxEventsScan   = 300;
input double   DDEventPrePenalty      = 0.25;
input double   DDEventExpandPenalty   = 0.70;
input double   DDEventRecoveryBoost   = 0.30;
input double   DDEventHoldBias        = 0.20;
input double   DDEventSameDirPenalty  = 0.20;
input double   DDEventBlendWeight     = 0.30;
input bool     DDEventUseCautionOnly  = true;

input string   DecisionBiasControlSettings = "==== DDQN primacy / subordinate bias control ====";
input double   SubordinateBiasCapFrac      = 0.50;
input double   QMemConfirmScale            = 0.85;
input double   QMemConflictToHoldScale     = 0.90;
input double   QMemConflictDirPenaltyScale = 0.55;
input double   DDEventDrawdownBoostScale   = 1.00;
input double   DangerHoldVetoScale         = 1.00;
input double   DangerDirectionalLeakScale  = 0.15;


// Zones
input string   ZoneSettings         = "==== S/D zones (BaseTF) ====";
input int      MinZoneBaseBars      = 3;
input int      MaxZoneBaseBars      = 8;
input double   ZoneMaxHeightPoints  = 300;
input int      ZoneExtendBars       = 200;
input int      MaxZonesPerBar       = 2;
input double   ZoneInvalidationBufferAtr = 0.10;
input double   ZoneMinDisplacementAtr    = 1.20;
input int      ZonePivotClusterBars      = 1;
input int      ZoneMaxLookbackSwings     = 24;
input double   ZoneNearBufferAtr         = 0.35;
input double   ZoneConfluenceAtr         = 0.50;
input double   SwingConfluenceAtr        = 0.60;

input bool     DrawSwingStructureOnChart = false;
input bool     DrawSwingZonesOnChart     = false;
input bool     DrawZoneCandleFlagsOnChart= false;

input bool     FastTrainingMode           = true;
input bool     FastTrainingSkipIndicatorBranch   = false;
input bool     FastTrainingSkipVolatilityBranch  = false;
input bool     FastTrainingSkipStructureBranch   = false;
input bool     FastTrainingSkipZoneCandleBranch  = false;
input int      ReentryCooldownSeconds    = 60;
input int      ReentryCooldownExecBars   = 2;
input bool     ForceEntryBypassesReentryCooldown = false;
input bool     UseFastBacktestStateCache  = true;
input int      SwingZoneCacheSlots        = 64;
input bool     UseSlowFeatureCaches      = true;
input bool     CacheStructureZoneOnExecBar = true;
input int      TrainEveryNDecisionBars     = 1;
input bool     FastTrainingSparseQMemory = true;
input int      FastTrainingQMemoryRefreshDecisionBars = 4;
input int      FastTrainingQMemoryRefreshMinutes      = 60;
input int      MaxDrawnSwingsPerTF       = 8;
input int      MaxDrawnZonesPerTF        = 2;


// D1 EMA trend (state)
input string   D1TrendSettings      = "==== D1 EMA trend ====";
input int      D1_EMA_Period        = 100;
input int      D1_SlopeLookbackDays = 20;
input int      D1_SideLookbackDays  = 60;
input double   D1_MaxSideDurationDays = 60.0;
// Performance / viz
input string PerformanceSettings = "==== Performance / visualization ====";
input bool   DrawZonesOnChart    = false;
input bool   VerboseLogging      = false;

input string DangerBrainSettings  = "==== Danger Brain (Regime + Memory) ====";
input bool   UseDangerBrain       = true;

input ENUM_TIMEFRAMES FingerprintTF   = PERIOD_CURRENT;
input int    FingerprintBars          = 32;

// thresholds are copied into runtime vars (so we can calibrate)
input double T1_CautionEnter          = 0.55;
input double T1_CautionExit           = 0.45;
input double T2_DangerEnter           = 0.75;
input double T2_DangerExit            = 0.65;
input int    ModeCooldownMinutes      = 60;

input double LRScale_Caution          = 0.30;
input double LRScale_Danger           = 0.10;

input bool   DangerBlocksNewEntries   = false;
input double ScoreW_SimContrast       = 1.0;
input double ScoreW_Exposure          = 0.6;

input double ExposureW_Positions      = 0.10;
input double ExposureW_DD             = 2.0;

input bool   SaveDangerMemory         = true;

input string DangerSmartnessSettings  = "==== Danger smartness pack ====";

input bool   UseDeltaSimilarity       = true;
input double SimW_DeltaFingerprint    = 0.70;
input double SimW_DeltaMiniState      = 0.50;

// 0=original block rules, 1=danger hold only, 2=danger with-trend only
input int    DangerActionPolicy       = 0;

input string BadEpisodeSettings        = "==== Bad Episode Learning ====";
input double BadProtoMergeSim          = 0.96;
input double BadProtoFeatureEMA        = 0.20;
input double BadProtoBiasEMA           = 0.20;

input double BadBias_HoldBoost         = 0.70;
input double BadBias_AgainstPenalty    = 0.70;
input double BadBias_WithTrendBoost    = 0.25;
input double BadBias_RepeatPenalty     = 0.35;

input string SafeRecoverySettings      = "==== Safe Recovery Learning ====";
input string MemoryAgingSettings       = "==== Memory Aging / Pruning ====";
input double ProtoHalfLifeDays         = 360.0;
input int    MaxProtosStored           = 12000;
input double PruneMinKeepScore         = 0.1;
input bool   PruneOnDeinit             = true;

input string MiniStateSettings        = "==== Mini State Fingerprint ====";
input bool   UseMiniStateFingerprint  = true;

input double SimW_PriceFingerprint    = 1.0;
input double SimW_MiniState           = 0.8;

// indices based on BuildState push order
int gMiniIdx[MINI_DIM] = { 2, 6, 7, 1, 11, 12, 14, 15 };

input string ProfitRewardSettings      = "==== Profit reward shaping ====";
input double ProfitReturnClamp         = 0.80;
input double ProfitRewardScale         = 10.0;

input string RecoveryQualitySettings   = "==== Recovery quality shaping ====";
input double RecWinRatioScale          = 2.0;
input double RecLossCountPenalty       = 1.0;
input double RecTradesUsedPenalty      = 0.40;
input double RecQualityClamp           = 8.0;

input string PositionLearningSettings  = "==== Per-position basket learning ====";
input bool   UsePerPositionCloseLearning = true;
input double LegPnLPerLotScale         = 0.05;
input double LegDistancePointsScale    = 0.002;
input double LegIndexRewardScale       = 0.03;
input double LegHoldHoursPenaltyScale  = 0.01;
input int    MaxClosedLegHistory       = 3000;

input string   ForcedEntrySettings       = "==== Forced entry watchdog ====";
input bool     UseForcedEntryWatchdog    = true;
input int      ForcedEntryIdleMinutes    = 60;
input int      ForcedEntryWindowMinutes  = 15;
input bool     ForcedEntryIgnoreDanger   = true;

datetime gLastTradeOpenTime[MAX_SYMBOLS];
bool     gForcedEntryActive[MAX_SYMBOLS];
datetime gForcedEntryArmTime[MAX_SYMBOLS];
datetime gForcedEntryDeadline[MAX_SYMBOLS];

double   maxEquity             = 0.0;
double   gTickEquityBaseline   = 0.0;
double   gTickBalanceBaseline  = 0.0;
datetime gRewardBaselineTick   = 0;
double   gEAStartEquity        = 0.0;
double   gEAClosedProfit       = 0.0;

datetime gEquityLossStopResumeTime = 0;
datetime gProfitPauseResumeTime    = 0;
double   gProfitCycleClosedProfit  = 0.0;


double   gLastForcedStopLossMoney[MAX_SYMBOLS];
datetime gLastForcedStopTime[MAX_SYMBOLS];
bool     gLastForcedStopPendingLearn[MAX_SYMBOLS];

double   gForcedStopFP[MAX_SYMBOLS][6];
bool     gForcedStopHaveFP[MAX_SYMBOLS];

double   gForcedStopMini[MAX_SYMBOLS][MINI_DIM];
bool     gForcedStopHaveMini[MAX_SYMBOLS];

double   gForcedStopQ[MAX_SYMBOLS][8];
bool     gForcedStopHaveQ[MAX_SYMBOLS];

int      gForcedStopBasketDir[MAX_SYMBOLS];
int      gForcedStopTrendDir[MAX_SYMBOLS];
double   gForcedStopAtrRatio[MAX_SYMBOLS];
int      gForcedStopLabel[MAX_SYMBOLS];

// RL/runtime caches for efficiency
int      tickCounter          = 0;
double   currentEpsilon       = 0.0;
bool     isTraining           = true;
double   totalReward          = 0.0;
int      episodeCount         = 0;
int      gTargetSyncCounter   = 0;
double   gAccountEquityStopPeak = 0.0;

datetime gZScoreLastClosedBarTF[MAX_SYMBOLS];

// Effective / combined z-score state used by risk management and DDQN
double   gZScoreLastValue[MAX_SYMBOLS];
double   gZScoreLastAbs[MAX_SYMBOLS];
bool     gZScoreExtremeNow[MAX_SYMBOLS];
bool     gZScorePauseTrading[MAX_SYMBOLS];
int      gZScoreQuietBars[MAX_SYMBOLS];
datetime gZScoreLastExtremeBarTime[MAX_SYMBOLS];
datetime gZScoreCloseHandledBar[MAX_SYMBOLS];
bool     gZScoreEmergencyHedgeActive[MAX_SYMBOLS];
int      gZScoreEmergencyOriginalDir[MAX_SYMBOLS];
datetime gZScoreEmergencyHedgeBar[MAX_SYMBOLS];

// Per-market monitoring state
double   gZScoreLastValueA[MAX_SYMBOLS];
double   gZScoreLastAbsA[MAX_SYMBOLS];
bool     gZScoreExtremeA[MAX_SYMBOLS];
double   gZScoreLastValueB[MAX_SYMBOLS];
double   gZScoreLastAbsB[MAX_SYMBOLS];
bool     gZScoreExtremeB[MAX_SYMBOLS];


double   gCachedSymbolPnL[MAX_SYMBOLS];
bool     gHasPositionTypeCache[MAX_SYMBOLS];
int      gBasketDirCache[MAX_SYMBOLS];
double   gBasketAvgPriceCache[MAX_SYMBOLS];
bool     gBasketAvgValid[MAX_SYMBOLS];

