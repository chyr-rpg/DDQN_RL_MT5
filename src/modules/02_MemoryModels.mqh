//+------------------------------------------------------------------+
//| 02_MemoryModels.mqh                                             |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Replay, episode, pattern, regime, danger, DD-event and decision-|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

struct QMemEntry
{
   int      regime;
   datetime created;
   datetime lastUsed;
   uint     usedCount;

   double   stateKey[];
   double   qVals[];
   double   conf;
   double   score;
};

QMemEntry gQMem[];


struct ReplayItem
{
   int      symIdx;
   int      regime;
   int      action;
   double   reward;
   bool     done;
   double   state[];
   double   nextState[];
   double   priority;

   int      replayBankType;
   long     episodeId;
   long     patternId;
   long     regimeId;
   long     periodId;

   int      basketStateClass;
   int      addDepthClass;
   int      dangerClass;
   int      zoneContextClass;
   int      structureContextClass;
   int      liquidityClass;

   datetime eventTime;
   double   painSeverity;
   double   recurrenceScore;
   double   regimeBreakScore;
   double   macroSig[];
   double   microSig[];
};

enum ReplayBankType
{
   REPLAY_BANK_RECENT = 0,
   REPLAY_BANK_DANGER = 1,
   REPLAY_BANK_DEEP_BASKET = 2,
   REPLAY_BANK_EFFICIENT = 3
};

enum MemoryZoneContextClass
{
   MEM_ZONE_NONE = 0,
   MEM_ZONE_NEAR_DEMAND = 1,
   MEM_ZONE_NEAR_SUPPLY = 2,
   MEM_ZONE_INSIDE_DEMAND = 3,
   MEM_ZONE_INSIDE_SUPPLY = 4
};

enum MemoryStructureContextClass
{
   MEM_STRUCT_MIXED = 0,
   MEM_STRUCT_TREND_UP = 1,
   MEM_STRUCT_TREND_DOWN = 2,
   MEM_STRUCT_COMPRESSION = 3,
   MEM_STRUCT_EXPANSION = 4
};

struct BarSnapshot
{
   datetime time;
   ENUM_TIMEFRAMES tf;

   double open;
   double high;
   double low;
   double close;
   long   tickVolume;
   int    spreadPoints;

   double atr;
   double realizedVol;
   double adx;
   double rsi;
   double priceVsEMA;

   double structTendency;
   double swingHighDistAtr;
   double swingLowDistAtr;
   double zoneRelevance;
   double rejectionStrength;
   double acceptanceStrength;
   double indecisionState;
};

struct BarSpanRef
{
   ENUM_TIMEFRAMES tf;
   datetime startTime;
   datetime endTime;
   int startIndex;
   int endIndex;
};

struct EpisodeMemory
{
   long   episodeId;
   string symbol;
   int    symIdx;

   datetime startTime;
   datetime endTime;

   int basketDir;
   int openPositionsMax;
   int addCount;
   int actionsCount;

   double entryPriceFirst;
   double avgEntryAtWorst;
   double closePriceFinal;

   double pnlFinal;
   double rewardTotal;
   double rewardEfficiency;
   double maxDrawdownPct;
   double maxDangerScore;
   double maxMarginStress;

   int oneRoundTrade;
   int forcedStopLikeEvent;
   int inefficientRecovery;

   int sessionType;
   int liquidityType;
   int regimeType;
   int patternType;

   BarSpanRef execSpan;
   BarSpanRef midSpan;
   BarSpanRef longSpan;
   BarSpanRef structExtSpan;
};

struct PatternMemory
{
   long patternId;
   string symbol;
   int symIdx;

   int patternType;
   int strengthClass;
   int volatilityClass;
   int liquidityClass;

   datetime startTime;
   datetime endTime;

   double rewardEfficiencyMean;
   double ddMean;
   double addMean;

   BarSpanRef execSpan;
   BarSpanRef midSpan;
   BarSpanRef longSpan;

   long linkedEpisodeIds[];
};

struct RegimeEventMemory
{
   long regimeId;
   string symbol;
   int symIdx;

   int regimeType;
   int eventType;

   datetime startTime;
   datetime endTime;

   double avgVol;
   double avgSpread;
   double avgADX;
   double avgRewardEfficiency;

   BarSpanRef execSpan;
   BarSpanRef midSpan;
   BarSpanRef longSpan;

   long linkedEpisodeIds[];
   long linkedPatternIds[];
};

struct BasketSequenceRef
{
   long episodeId;
   int symIdx;

   datetime startTime;
   datetime endTime;

   int basketDir;
   int addCount;
   int maxPositions;
   double maxDD;
   double finalReward;

   int replayItemIndexes[];
};

struct EfficientPeriodRef
{
   long periodId;
   int symIdx;

   datetime startTime;
   datetime endTime;

   double rewardTotal;
   double rewardEfficiency;
   double ddMax;
   int oneRoundCount;
   int addCountTotal;

   int sessionType;
   int liquidityType;
   int regimeType;
   int patternType;

   int replayItemIndexes[];
};

struct ReplayBankStore
{
   ReplayItem items[];
   int count;
   int maxCount;
};

struct DeepBasketReplayStore
{
   ReplayItem items[];
   BasketSequenceRef sequences[];
   int itemCount;
   int seqCount;
   int maxItems;
   int maxSeqs;
};

struct EfficientReplayStore
{
   ReplayItem items[];
   EfficientPeriodRef periods[];
   int itemCount;
   int periodCount;
   int maxItems;
   int maxPeriods;
};

EpisodeMemory         gEpisodeMemory[];
PatternMemory         gPatternMemory[];
RegimeEventMemory     gRegimeEventMemory[];

ReplayBankStore       gDangerReplayBank;
DeepBasketReplayStore gDeepBasketReplayBank;
EfficientReplayStore  gEfficientReplayBank;
ReplayBankStore       gRecentReplayBank;

double                gDangerReplaySeverityEMA[MAX_SYMBOLS];
int                   gDangerReplayHits[MAX_SYMBOLS];
double                gDeepBasketReplaySeverityEMA[MAX_SYMBOLS];
int                   gDeepBasketReplayHits[MAX_SYMBOLS];
int                   gReplayDiagWindowRecentAdds[MAX_SYMBOLS];
int                   gReplayDiagWindowDangerAdds[MAX_SYMBOLS];
int                   gReplayDiagWindowDeepAdds[MAX_SYMBOLS];
int                   gReplayDiagWindowEfficientAdds[MAX_SYMBOLS];
int                   gReplayDiagBarsSincePrint[MAX_SYMBOLS];

struct ActiveBasketEpisode
{
   bool     active;
   long     episodeId;
   int      symIdx;
   int      basketDir;
   datetime startTime;
   datetime lastTime;
   int      addCount;
   int      maxPositions;
   double   maxDD;
   double   rewardAccum;
   int      replayItemIndexes[];
};

struct ActiveEfficientPeriod
{
   bool     active;
   long     periodId;
   int      symIdx;
   datetime startTime;
   datetime lastTime;
   double   rewardTotal;
   double   ddMax;
   int      oneRoundCount;
   int      addCountTotal;
   int      tradeCount;
   int      lastLiquidityClass;
   int      lastRegime;
   int      lastStructureClass;
   int      contextBreakCount;
   int      replayItemIndexes[];
};

ActiveBasketEpisode   gActiveBasketEpisodes[MAX_SYMBOLS];
ActiveEfficientPeriod gActiveEfficientPeriods[MAX_SYMBOLS];
long                  gNextEpisodeId = 1;
long                  gNextPeriodId  = 1;

ReplayItem gReplay[];

struct PendingTransition
{
   bool     active;
   int      symIdx;
   int      regime;
   int      action;
   datetime created;
   int      basketDir;
   int      positionsAtOpen;
   int      legIndex;
   datetime entryTime;
   datetime closeTime;
   double   entryPrice;
   double   closePrice;
   double   entryVolume;
   double   basketAvgAtEntry;
   double   individualPnL;
   double   denseRewardAccum;
   long     episodeId;
   double   state[];
};

PendingTransition gPending[];

struct DenseBasketHealthTracker
{
   bool     active;
   datetime lastBarTime;
   double   prevOpenReturn;
   double   prevDD;
   int      prevPositions;
   double   prevAgeDays;
};

DenseBasketHealthTracker gDenseBasketHealth[MAX_SYMBOLS];

struct PositionCloseItem
{
   ulong    ticket;
   int      basketDir;
   int      legIndex;
   datetime entryTime;
   datetime closeTime;
   double   entryPrice;
   double   closePrice;
   double   volume;
   double   profit;
};

struct BasketCloseResult
{
   int      attempted;
   int      closed;
   double   attemptedVolume;
   double   closedVolume;
   double   attemptedProfit;
   double   closedProfit;
   int      wins;
   int      losses;
   int      total;
   bool     allClosed;
};

struct ClosedLegLearningRecord
{
   datetime entryTime;
   datetime closeTime;
   string   symbol;
   int      regime;
   int      action;
   int      basketDir;
   int      legIndex;
   double   entryPrice;
   double   closePrice;
   double   entryVolume;
   double   basketAvgAtEntry;
   double   individualPnL;
   double   resolvedReward;
};

ClosedLegLearningRecord gClosedLegHistory[];

struct BasketSnapshot
{
   datetime timeStamp;
   string   symbol;
   int      magic;
   int      regime;
   int      basketDir;
   int      positionsCount;
   double   equity;
   double   openPnL;
   double   avgPrice;
   double   midPrice;
   double   entryPrice;
   double   gridStep;
   double   danger;
   double   stateKey[];
   double   qVals[];
};

struct TickTraceItem
{
   datetime timeStamp;
   double   bid;
   double   ask;
   double   mid;
   double   spreadPoints;
   double   equity;
   double   danger;
};

struct DDEventRecord
{
   datetime created;
   datetime triggerTime;
   string   symbol;
   int      magic;
   int      regimeAtTrigger;
   int      basketDirAtTrigger;

   double   ddAtTrigger;
   bool     hardTrigger;
   bool     completed;

   BasketSnapshot preBaskets[];
   BasketSnapshot postBaskets[];
   TickTraceItem  ticks[];

   double   triggerStateKey[];
   double   triggerQVals[];
   double   peakDD;
   int      basketDepthMax;
   double   timeUnderWaterNorm;
   double   recoveryFailureScore;
   double   painSeverity;
   int      eventType;
   double   macroSig[];
   double   microSig[];
};

BasketSnapshot gBasketHistory[];
TickTraceItem  gTickTrace[];
DDEventRecord  gDDEvents[];

bool     gDDEventActive            = false;
datetime gDDEventTriggerTime       = 0;
double   gDDEventTriggerDD         = 0.0;
bool     gDDEventHard              = false;
int      gDDEventStartBasketIndex  = -1;
int      gDDEventStartTickIndex    = -1;
int      gDDEventPostBasketCount   = 0;
datetime gDDEventLastTraceBarTime[MAX_SYMBOLS];

enum BrainMode { MODE_NORMAL=0, MODE_CAUTION=1, MODE_DANGER=2 };
enum EpisodeLabel { LBL_AGAINST_UPTREND=0, LBL_AGAINST_DOWNTREND=1 };

struct ProtoEntry
{
   int      label;
   int      basketDir;
   int      trendDir;
   double   atrRatio;
   datetime created;

   bool     isDanger;
   double   features[];
   double   stateMini[];
   double   deltaFP[];
   double   deltaMini[];

   double   adapterBias[];
   double   qSnap[];
   double   painMean;
   double   painCount;
   double   macroSig[];
   double   microSig[];
   double   deepBasketRate;
   double   regimeBreakRate;
   double   counterTrendFailureRate;
   double   reversalTrapRate;
   double   recoveryFailureRate;
   double   survivalScore;
   uint     usedCount;
   datetime lastUsed;
};

BrainMode gMode[MAX_SYMBOLS];
datetime  gModeLastChange[MAX_SYMBOLS];
double    gPDanger[MAX_SYMBOLS];
double    gLRScale[MAX_SYMBOLS];

double gT1Enter=0.0, gT1Exit=0.0, gT2Enter=0.0, gT2Exit=0.0;

int gCalibFalseDanger=0;
int gCalibMissedDanger=0;
int gCalibTrueDanger=0;

datetime gFPLastBar[MAX_SYMBOLS];
double   gFPCache[MAX_SYMBOLS][6];
double   gFPPrev[MAX_SYMBOLS][6];
bool     gFPPrevValid[MAX_SYMBOLS];

datetime gMiniLastBar[MAX_SYMBOLS];
double   gMiniCache[MAX_SYMBOLS][MINI_DIM];
bool     gMiniValid[MAX_SYMBOLS];
double   gMiniPrev[MAX_SYMBOLS][MINI_DIM];
bool     gMiniPrevValid[MAX_SYMBOLS];

ProtoEntry gProtos[];

struct DecisionSupportContext
{
   bool     valid;
   bool     forcedEntry;
   datetime barTime;
   datetime refreshTime;
   int      regime;
   int      positionsCount;
   int      basketDir;
   int      brainMode;

   double   actionDelta[3];
   double   qMemoryAgreement;
   double   qMemoryConflict;
   double   archiveCaution;
   double   ddEventRisk;
   double   dangerProbability;
   double   oneRoundPrior;
   double   addRiskPrior;
   double   supportConfidence;
   double   painRecurrenceRisk;
   double   macroReversalTrapPrior;
   double   counterTrendFailurePrior;
   double   regimeBreakPainPrior;
   double   recoveryFalseStartRisk;
   double   deepBasketPainPrior;
   double   painMemoryAgreement;
   double   macroMicroConflict;
   double   lateTrendFadePenalty;
   double   painConfidence;
   double   trendPersistenceProb;
   double   trendReversalProb;
   double   spikeRiskProb;
   double   expectedBasketDepth;
   double   trendContinuationQuality;
   double   breakoutReclaimQuality;
   double   reversalTransitionQuality;
   double   modeDominanceScore;
   double   modeConflictScore;
};

DecisionSupportContext gDecisionSupportCache[MAX_SYMBOLS];

datetime gBadDDStart[MAX_SYMBOLS];
bool     gBadDDConfirmed[MAX_SYMBOLS];
int      gBadLabel[MAX_SYMBOLS];
int      gBadBasketDir[MAX_SYMBOLS];
int      gBadTrendDir[MAX_SYMBOLS];
double   gBadAtrRatio[MAX_SYMBOLS];

double   gBadFPStart[MAX_SYMBOLS][6];
double   gBadFPBasketOpen[MAX_SYMBOLS][6];
bool     gBadHaveBasketFP[MAX_SYMBOLS];

double   gBadQStart[MAX_SYMBOLS][8];
bool     gBadHaveQStart[MAX_SYMBOLS];

double   gBadFPPre[MAX_SYMBOLS][6];
bool     gBadHavePreFP[MAX_SYMBOLS];

double   gBadFPEnd[MAX_SYMBOLS][6];
bool     gBadHaveEndFP[MAX_SYMBOLS];

double   gBadMiniStart[MAX_SYMBOLS][MINI_DIM];
double   gBadMiniBasketOpen[MAX_SYMBOLS][MINI_DIM];
bool     gBadHaveMiniStart[MAX_SYMBOLS];
bool     gBadHaveMiniBasket[MAX_SYMBOLS];
double   gBadMiniPre[MAX_SYMBOLS][MINI_DIM];
bool     gBadHaveMiniPre[MAX_SYMBOLS];
double   gBadMiniEnd[MAX_SYMBOLS][MINI_DIM];
bool     gBadHaveMiniEnd[MAX_SYMBOLS];

datetime gLastBarTime[MAX_SYMBOLS];
datetime gLastReplayDecisionBarTime[MAX_SYMBOLS];
int      gReplayDecisionBarCounter[MAX_SYMBOLS];
int      gReplayPendingTrainCount=0;
bool     gLoadedDQNFastModeMeta[MAX_SYMBOLS];
string   gLoadedDQNMetaNote[MAX_SYMBOLS];

datetime gDecisionCtxBarTime[MAX_SYMBOLS];
int      gDecisionCtxHourBucket[MAX_SYMBOLS];
int      gDecisionCtxRegimeType[MAX_SYMBOLS];
int      gDecisionCtxPatternType[MAX_SYMBOLS];
int      gDecisionCtxLiquidityType[MAX_SYMBOLS];
int      gDecisionCtxSessionType[MAX_SYMBOLS];
double   gDecisionCtxAtrRatio[MAX_SYMBOLS];
bool     gDecisionCtxValid[MAX_SYMBOLS];


datetime gSignalTrackBarTime[MAX_SYMBOLS];
int      gSignalTrackRegime[MAX_SYMBOLS];
int      gSignalTrackDirCandidate[MAX_SYMBOLS];
int      gSignalTrackPersistCount[MAX_SYMBOLS];
int      gSignalTrackMode[MAX_SYMBOLS];
double   gSignalTrackMacroBias[MAX_SYMBOLS];
double   gSignalTrackTransition[MAX_SYMBOLS];

#define DIR_STATE_DOWN   -1
#define DIR_STATE_NEUTRAL 0
#define DIR_STATE_UP      1


datetime gFastQMemLastRefreshTime[MAX_SYMBOLS];
datetime gFastQMemLastDecisionBarTime[MAX_SYMBOLS];
int      gFastQMemDecisionBarCounter[MAX_SYMBOLS];
int      gFastQMemCachedRegime[MAX_SYMBOLS];
bool     gFastQMemCachedValid[MAX_SYMBOLS];
double   gFastQMemCachedConf[MAX_SYMBOLS];
double   gFastQMemCachedQ[MAX_SYMBOLS][3];

datetime gD1LastBarTime[MAX_SYMBOLS];
double   gD1DistCache[MAX_SYMBOLS];
double   gD1SlopeCache[MAX_SYMBOLS];
double   gD1SideDurCache[MAX_SYMBOLS];

datetime gATRLastBarTime[MAX_SYMBOLS];
double   gATRratioCache[MAX_SYMBOLS];
datetime gGridLastBar[MAX_SYMBOLS];
double   gGridStepCache[MAX_SYMBOLS];
double   gGridWidthSigmaCache[MAX_SYMBOLS];

// kept only for compatibility with panel / snapshot code
int      gGridActiveChannel[MAX_SYMBOLS];
double   gGridActiveStep[MAX_SYMBOLS];

datetime gExtremeDDStart[MAX_SYMBOLS];
bool     gExtremeDDArmed[MAX_SYMBOLS];

string   gGridPanelName = "DQN_GRID_STATUS_PANEL";

// BaseTF indicator handles + caches
int      hRSI_Base[MAX_SYMBOLS];
int      hCCI_Base[MAX_SYMBOLS];
int      hMACD_Base[MAX_SYMBOLS];
int      hEMA_Base[MAX_SYMBOLS];
int      hRVI_Base[MAX_SYMBOLS];
int      hATRfast_Base[MAX_SYMBOLS];
int      hATRslow_Base[MAX_SYMBOLS];
datetime gIndLastBarTime[MAX_SYMBOLS];

double   gRSI_Base[MAX_SYMBOLS];
double   gCCI_Base[MAX_SYMBOLS];
double   gMACD_BaseMain[MAX_SYMBOLS];
double   gEMA_BaseVal[MAX_SYMBOLS];
double   gRVI_BaseMain[MAX_SYMBOLS];
double   gATRfast_BaseVal[MAX_SYMBOLS];
double   gATRslow_BaseVal[MAX_SYMBOLS];

// H1 handles + caches
int      hRSI_H1[MAX_SYMBOLS];
int      hMACD_H1[MAX_SYMBOLS];
int      hEMA_H1[MAX_SYMBOLS];
int      hRVI_H1[MAX_SYMBOLS];
int      hATRfast_H1[MAX_SYMBOLS];
int      hATRslow_H1[MAX_SYMBOLS];
datetime gH1LastBar[MAX_SYMBOLS];

double   gRSI_H1v[MAX_SYMBOLS];
double   gMACD_H1v[MAX_SYMBOLS];
double   gEMA_H1v[MAX_SYMBOLS];
double   gRVI_H1v[MAX_SYMBOLS];
double   gATRratio_H1[MAX_SYMBOLS];

// H4 handles + caches
int      hRSI_H4[MAX_SYMBOLS];
int      hMACD_H4[MAX_SYMBOLS];
int      hEMA_H4[MAX_SYMBOLS];
int      hRVI_H4[MAX_SYMBOLS];
int      hATRfast_H4[MAX_SYMBOLS];
int      hATRslow_H4[MAX_SYMBOLS];
datetime gH4LastBar[MAX_SYMBOLS];

double   gRSI_H4v[MAX_SYMBOLS];
double   gMACD_H4v[MAX_SYMBOLS];
double   gEMA_H4v[MAX_SYMBOLS];
double   gRVI_H4v[MAX_SYMBOLS];
double   gATRratio_H4[MAX_SYMBOLS];

