//+------------------------------------------------------------------+
//| 07_StateAndStructureFeatures.mqh                                |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| State-vector construction, branch layouts, swing/zone structure |
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

bool CacheMatchBasic(const SlowFeatureCacheEntry &c,const string symbol,const datetime execBar,const datetime midBar,const datetime longBar,const long avgQ,const long lastEntryQ,const bool needBasket)
{
   if(!c.valid) return false;
   if(c.symbol!=symbol) return false;
   if(c.execBar!=execBar || c.midBar!=midBar || c.longBar!=longBar) return false;
   if(needBasket)
   {
      if(c.avgQ!=avgQ || c.lastEntryQ!=lastEntryQ) return false;
   }
   return true;
}

void CloneDoubleArray(const double &src[], double &dst[])
{
   int n=ArraySize(src);
   ArrayResize(dst,n);
   for(int i=0;i<n;i++) dst[i]=src[i];
}


double SiLU(const double x)
{
   if(x>=0.0)
   {
      double e=MathExp(-x);
      return x/(1.0+e);
   }
   else
   {
      double e=MathExp(x);
      return x*e/(1.0+e);
   }
}

double SiLUDerivativeFromPreAct(const double x)
{
   double s;
   if(x>=0.0)
   {
      double e=MathExp(-x);
      s = 1.0/(1.0+e);
   }
   else
   {
      double e=MathExp(x);
      s = e/(1.0+e);
   }
   return s + x*s*(1.0-s);
}

void InitBranchLayoutForStateDim(const int totalDim)
{
   gBranchLayout.basketStart     = 0;
   gBranchLayout.basketCount     = 0;
   gBranchLayout.indicatorStart  = 0;
   gBranchLayout.indicatorCount  = 0;
   gBranchLayout.volatilityStart = 0;
   gBranchLayout.volatilityCount = 0;
   gBranchLayout.structureStart  = 0;
   gBranchLayout.structureCount  = 0;
   gBranchLayout.zoneCandleStart = 0;
   gBranchLayout.zoneCandleCount = 0;
   gBranchLayout.totalCount      = MathMax(totalDim,0);

   int cursor=0;

   if(UseBasketBranch && UseStateV2SelfAwareness)
   {
      gBranchLayout.basketStart = cursor;
      gBranchLayout.basketCount = ModuleAFeatureCount();
      cursor += gBranchLayout.basketCount;
   }

   if(UseIndicatorBranch)
   {
      gBranchLayout.indicatorStart = cursor;
      gBranchLayout.indicatorCount = ModuleBFeatureCount();
      cursor += gBranchLayout.indicatorCount;
   }

   if(UseVolatilityBranch)
   {
      gBranchLayout.volatilityStart = cursor;
      gBranchLayout.volatilityCount = ModuleCFeatureCount();
      cursor += gBranchLayout.volatilityCount;
   }

   if(UseStructureBranch)
   {
      gBranchLayout.structureStart = cursor;
      gBranchLayout.structureCount = ModuleDFeatureCount();
      cursor += gBranchLayout.structureCount;
   }

   if(UseZoneCandleBranch)
   {
      gBranchLayout.zoneCandleStart = cursor;
      gBranchLayout.zoneCandleCount = ModuleEFeatureCount();
      cursor += gBranchLayout.zoneCandleCount;
   }

   // Safety remainder handling only for unexpected dimension drift.
   int rem = MathMax(totalDim - cursor, 0);
   if(rem > 0)
   {
      if(gBranchLayout.indicatorCount > 0)
         gBranchLayout.indicatorCount += rem;
      else
      {
         gBranchLayout.indicatorStart = cursor;
         gBranchLayout.indicatorCount = rem;
      }
      cursor += rem;
   }

   gBranchLayout.totalCount = cursor;
}

int ModuleAFeatureCount()
{
   return 31; // basket/self risk + memory/meta + pain-memory priors + Option B estimators + 3-mode regime scores
}

int ModuleBFeatureCount()
{
   return 22; // setup + macro-trend / reversal-trap factors
}

int ModuleCFeatureCount()
{
   return 16; // volatility / execution / regime-break factors + exact z-score risk guard
}

int ModuleDFeatureCount()
{
   return 18; // 3 TF * 6 structure-quality factors
}

int ModuleEFeatureCount()
{
   return 15; // 3 TF * 5 zone / candle factors
}

int StructureLookbackForTF(const ENUM_TIMEFRAMES tf)
{
   if(tf == TF_MID)  return MathMax(StructLookbackMid, 8);
   if(tf == TF_LONG) return MathMax(StructLookbackLong, 8);
   return MathMax(StructLookbackExec, 8);
}

int StructureHistoryBarsForTF(const ENUM_TIMEFRAMES tf)
{
   if(tf == TF_MID)  return MathMax(StructHistoryBarsMid, 200);
   if(tf == TF_LONG) return MathMax(StructHistoryBarsLong, 200);
   return MathMax(StructHistoryBarsExec, 200);
}

double StructNormDist(const double delta,const double atr)
{
   return Clamp(delta / MathMax(atr,1e-8), -StructDistClampAtr, StructDistClampAtr) / MathMax(StructDistClampAtr, 1.0);
}

double StructNormAmp(const double amp,const double atr)
{
   return Clamp(amp / MathMax(atr,1e-8), 0.0, StructAmpClampAtr) / MathMax(StructAmpClampAtr, 1.0);
}

struct SwingPointMem
{
   double   price;
   int      barShift;
   datetime when;
   bool     isHigh;
};

struct SwingCacheEntry
{
   bool valid;
   string symbol;
   ENUM_TIMEFRAMES tf;
   datetime barTime;
   SwingPointMem swings[];
   int swingCount;
   bool ok;
};

struct SwingDerivedZone
{
   bool     valid;
   bool     isDemand;
   double   low;
   double   high;
   double   center;
   double   displacementAtr;
   double   freshness;
   int      mitigationCount;
   bool     invalidated;
   int      lifeState;      // -1 invalidated, 0 candidate, +1 active
   int      pivotShift;
   datetime pivotTime;
};


struct ZoneCacheEntry
{
   bool valid;
   string symbol;
   ENUM_TIMEFRAMES tf;
   datetime barTime;
   SwingDerivedZone demands[];
   int demandCount;
   SwingDerivedZone supplies[];
   int supplyCount;
   bool ok;
};

SwingCacheEntry gSwingCache[];
ZoneCacheEntry  gZoneCache[];

struct StateCacheEntry
{
   bool valid;
   string symbol;
   datetime baseBarTime;
   int positionsCount;
   int basketDir;
   long avgQ;
   long lastEntryQ;
   double state[];
};

StateCacheEntry gStateCache[MAX_SYMBOLS];

void InitPerfCaches()
{
   int n=MathMax(8, SwingZoneCacheSlots);
   ArrayResize(gSwingCache, n);
   ArrayResize(gZoneCache, n);
   for(int i=0;i<n;i++)
   {
      gSwingCache[i].valid=false;
      gZoneCache[i].valid=false;
   }
   for(int i=0;i<MAX_SYMBOLS;i++)
      gStateCache[i].valid=false;
}

bool ShouldUseFastStateCache()
{
   if(!UseFastBacktestStateCache) return false;
   return (bool)MQLInfoInteger(MQL_TESTER);
}
void ZeroDoubleArray(double &arr[], const int count)
{
   int n=MathMax(count,0);
   ArrayResize(arr,n);
   for(int i=0;i<n;i++) arr[i]=0.0;
}

string BuildCurrentDQNMetadataNote()
{
   return StringFormat("shape=full_state; zero_fill_shape_compatible=1; reward_architecture=ddqn_risk_first_v5_archive_replay_symbol_local; archive_live_similarity=%d; archive_live_blend=%g; archive_add_risk_gate=%d; archive_add_risk_threshold=%g; replay_aware_reward=%d; replay_reward_scan=%d; replay_diag=%d; post_action_next_state=%d; symbol_local_budget_reward=%d; regime_warmup_only=%d; soft_regime_inference=%d; dd_event_bias=%d; dd_event_caution_only=%d; main_replay_persistence=%d; recent_replay_persistence=%d; branch_caps=%d_%d_%d; fusion_hidden=%d_%d; fast_training=%d; fast_mask_total=83_including_self; fast_mask_modules=B21_C9_D27_E20; skip_indicator=%d; skip_volatility=%d; skip_structure=%d; skip_zone_candle=%d; structure_zone_cached_closed_bar=1; dense_basket_feedback=%d; archive_aware_reward=%d; efficient_period_live_support=%d; min_add_spacing_gate=%d; dd_event_trace_on_bars=%d; dd_event_trace_tf=%d; replay_every_n_decision_bars=%d; replay_train_iters=%d; replay_batch_samples=%d",
                       (UseArchiveLiveSimilarity ? 1 : 0),
                       ArchiveLiveBlendWeight,
                       (UseArchiveAddRiskGate ? 1 : 0),
                       ArchiveAddRiskGateThreshold,
                       (UseReplayAwareReward ? 1 : 0),
                       MathMax(50, ReplayRewardScanLimit),
                       (UseReplayDiagnostics ? 1 : 0),
                       (UseRealPostActionTransitions ? 1 : 0),
                       (UsePerSymbolVirtualBudget ? 1 : 0),
                       (TrainAllRegimesWarmupOnly ? 1 : 0),
                       (UseSoftRegimeInference ? 1 : 0),
                       (UseDDEventBias ? 1 : 0),
                       (DDEventUseCautionOnly ? 1 : 0),
                       (SaveMainReplayBank ? 1 : 0),
                       (SaveRecentReplayBank ? 1 : 0),
                       BranchEncoderMinWidth,
                       BranchEncoderH1Cap,
                       BranchEncoderH2Cap,
                       HiddenSize,
                       HiddenSize2,
                       (FastTrainingMode ? 1 : 0),
                       (FastTrainingSkipIndicatorBranch ? 1 : 0),
                       (FastTrainingSkipVolatilityBranch ? 1 : 0),
                       (FastTrainingSkipStructureBranch ? 1 : 0),
                       (FastTrainingSkipZoneCandleBranch ? 1 : 0),
                       (UseDenseBasketHealthReward ? 1 : 0),
                       (UseArchiveAwareReward ? 1 : 0),
                       (UseEfficientPeriodLiveSupport ? 1 : 0),
                       (UseMinAddSpacingGate ? 1 : 0),
                       (DDEventTraceOnBars ? 1 : 0),
                       (int)DDEventTraceTFForSymbol(_Symbol),
                       MathMax(1, TrainEveryNDecisionBars),
                       MathMax(1, ReplayTrainIters),
                       MathMax(1, ReplayBatchSize));
}

bool ShouldTrainReplayBatchNow(const int symIdx,const string symbol)
{
   if(!UseReplayBuffer || !isTraining) return false;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return false;
   if(ArraySize(gReplay) < ReplayWarmup || ReplayBatchSize<=0) return false;
   if(gReplayPendingTrainCount <= 0) return false;

   datetime barTime=iTime(symbol, BaseTF, 1);
   if(barTime<=0) return false;
   if(gLastReplayDecisionBarTime[symIdx]==barTime) return false;

   gLastReplayDecisionBarTime[symIdx]=barTime;
   gReplayDecisionBarCounter[symIdx]++;

   int everyN=MathMax(1, TrainEveryNDecisionBars);
   return ((gReplayDecisionBarCounter[symIdx] % everyN) == 0);
}

long QuantizePriceByPoint(const double price,const double point)
{
   double p=(point>0.0?point:0.00001);
   return (long)MathRound(price/p);
}

void CloneSwingArray(const SwingPointMem &src[], SwingPointMem &dst[])
{
   int n=ArraySize(src);
   ArrayResize(dst,n);
   for(int i=0;i<n;i++) dst[i]=src[i];
}

void CloneZoneArray(const SwingDerivedZone &src[], SwingDerivedZone &dst[])
{
   int n=ArraySize(src);
   ArrayResize(dst,n);
   for(int i=0;i<n;i++) dst[i]=src[i];
}

int FindOrAllocSwingCacheSlot(const string symbol,const ENUM_TIMEFRAMES tf)
{
   int freeIdx=-1;
   for(int i=0;i<ArraySize(gSwingCache);i++)
   {
      if(gSwingCache[i].valid && gSwingCache[i].symbol==symbol && gSwingCache[i].tf==tf)
         return i;
      if(!gSwingCache[i].valid && freeIdx<0)
         freeIdx=i;
   }
   if(freeIdx>=0) return freeIdx;
   return 0;
}

int FindOrAllocZoneCacheSlot(const string symbol,const ENUM_TIMEFRAMES tf)
{
   int freeIdx=-1;
   for(int i=0;i<ArraySize(gZoneCache);i++)
   {
      if(gZoneCache[i].valid && gZoneCache[i].symbol==symbol && gZoneCache[i].tf==tf)
         return i;
      if(!gZoneCache[i].valid && freeIdx<0)
         freeIdx=i;
   }
   if(freeIdx>=0) return freeIdx;
   return 0;
}

void AppendSwingMem(SwingPointMem &arr[], int &count, const int maxCount, const double price, const int barShift, const datetime when, const bool isHigh)
{
   if(maxCount <= 0) return;
   if(count < maxCount)
   {
      ArrayResize(arr, count + 1);
      arr[count].price    = price;
      arr[count].barShift = barShift;
      arr[count].when     = when;
      arr[count].isHigh   = isHigh;
      count++;
      return;
   }

   for(int i=1; i<count; ++i)
      arr[i-1] = arr[i];

   arr[count-1].price    = price;
   arr[count-1].barShift = barShift;
   arr[count-1].when     = when;
   arr[count-1].isHigh   = isHigh;
}

bool BuildSwingMemoryForTF_Core(const string symbol,
                           const ENUM_TIMEFRAMES tf,
                           SwingPointMem &swings[],
                           int &swingCount)
{
   swingCount = 0;
   ArrayResize(swings, 0);

   int barsWanted = StructureHistoryBarsForTF(tf);
   int need = MathMax(barsWanted, 120);
   if(Bars(symbol, tf) < need + 5)
      return false;

   double highs[], lows[], closes[], atrs[];
   datetime times[];
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(atrs, true);
   ArraySetAsSeries(times, true);

   int copiedH = CopyHigh(symbol, tf, 1, need, highs);
   int copiedL = CopyLow(symbol, tf, 1, need, lows);
   int copiedC = CopyClose(symbol, tf, 1, need, closes);
   int copiedT = CopyTime(symbol, tf, 1, need, times);

   int atrHandle = iATR(symbol, tf, 14);
   int copiedA = 0;
   if(atrHandle != INVALID_HANDLE)
   {
      copiedA = CopyBuffer(atrHandle, 0, 1, need, atrs);
      IndicatorRelease(atrHandle);
   }

   int n = MathMin(MathMin(copiedH, copiedL), MathMin(copiedC, copiedT));
   if(copiedA > 0) n = MathMin(n, copiedA);
   if(n < 40)
      return false;

   int oldest = n - 1;
   int mode = 0; // +1 up leg, -1 down leg, 0 unknown

   if(closes[oldest - 1] > closes[oldest]) mode = +1;
   else if(closes[oldest - 1] < closes[oldest]) mode = -1;

   double candidateHigh = highs[oldest];
   double candidateLow  = lows[oldest];
   int    candidateHighShift = oldest + 1; // because arrays start at shift=1
   int    candidateLowShift  = oldest + 1;
   datetime candidateHighTime = times[oldest];
   datetime candidateLowTime  = times[oldest];

   for(int i = oldest - 1; i >= 0; --i)
   {
      double atr = (i < ArraySize(atrs) ? atrs[i] : 0.0);
      if(atr <= 1e-8)
      {
         double close_i = closes[i];
         atr = MathMax(MathAbs(candidateHigh - close_i), MathAbs(close_i - candidateLow));
         atr = MathMax(atr, 1e-6);
      }
      double reversal = MathMax(atr * SwingReversalAtrMult, _Point * 10.0);

      if(mode >= 0)
      {
         if(highs[i] >= candidateHigh)
         {
            candidateHigh = highs[i];
            candidateHighShift = i + 1;
            candidateHighTime  = times[i];
         }

         if((candidateHigh - lows[i]) >= reversal)
         {
            AppendSwingMem(swings, swingCount, StructMaxSwingsPerTF, candidateHigh, candidateHighShift, candidateHighTime, true);
            mode = -1;
            candidateLow = lows[i];
            candidateLowShift = i + 1;
            candidateLowTime  = times[i];
            continue;
         }
      }

      if(mode <= 0)
      {
         if(lows[i] <= candidateLow)
         {
            candidateLow = lows[i];
            candidateLowShift = i + 1;
            candidateLowTime  = times[i];
         }

         if((highs[i] - candidateLow) >= reversal)
         {
            AppendSwingMem(swings, swingCount, StructMaxSwingsPerTF, candidateLow, candidateLowShift, candidateLowTime, false);
            mode = +1;
            candidateHigh = highs[i];
            candidateHighShift = i + 1;
            candidateHighTime  = times[i];
            continue;
         }
      }
   }

   
return (swingCount > 0);
}

bool BuildSwingMemoryForTF(const string symbol,
                           const ENUM_TIMEFRAMES tf,
                           SwingPointMem &swings[],
                           int &swingCount)
{
   swingCount=0;
   ArrayResize(swings,0);

   datetime barTime=iTime(symbol, tf, 1);
   int slot=FindOrAllocSwingCacheSlot(symbol, tf);
   if(slot>=0 && gSwingCache[slot].valid && gSwingCache[slot].symbol==symbol && gSwingCache[slot].tf==tf && gSwingCache[slot].barTime==barTime)
   {
      swingCount=gSwingCache[slot].swingCount;
      CloneSwingArray(gSwingCache[slot].swings, swings);
      return gSwingCache[slot].ok;
   }

   bool ok=BuildSwingMemoryForTF_Core(symbol, tf, swings, swingCount);

   if(slot>=0)
   {
      gSwingCache[slot].valid=true;
      gSwingCache[slot].symbol=symbol;
      gSwingCache[slot].tf=tf;
      gSwingCache[slot].barTime=barTime;
      gSwingCache[slot].swingCount=swingCount;
      gSwingCache[slot].ok=ok;
      CloneSwingArray(swings, gSwingCache[slot].swings);
   }
   return ok;
}


bool BuildSwingDerivedZonesForTF(const string symbol,
                                 const ENUM_TIMEFRAMES tf,
                                 SwingDerivedZone &demands[],
                                 int &demandCount,
                                 SwingDerivedZone &supplies[],
                                 int &supplyCount)
{
   demandCount = 0;
   supplyCount = 0;
   ArrayResize(demands, 0);
   ArrayResize(supplies, 0);

   datetime barTime=iTime(symbol, tf, 1);
   int slot=FindOrAllocZoneCacheSlot(symbol, tf);
   if(slot>=0 && gZoneCache[slot].valid && gZoneCache[slot].symbol==symbol && gZoneCache[slot].tf==tf && gZoneCache[slot].barTime==barTime)
   {
      demandCount=gZoneCache[slot].demandCount;
      supplyCount=gZoneCache[slot].supplyCount;
      CloneZoneArray(gZoneCache[slot].demands, demands);
      CloneZoneArray(gZoneCache[slot].supplies, supplies);
      return gZoneCache[slot].ok;
   }

   bool ok=BuildSwingDerivedZonesForTF_Core(symbol, tf, demands, demandCount, supplies, supplyCount);

   if(slot>=0)
   {
      gZoneCache[slot].valid=true;
      gZoneCache[slot].symbol=symbol;
      gZoneCache[slot].tf=tf;
      gZoneCache[slot].barTime=barTime;
      gZoneCache[slot].demandCount=demandCount;
      gZoneCache[slot].supplyCount=supplyCount;
      gZoneCache[slot].ok=ok;
      CloneZoneArray(demands, gZoneCache[slot].demands);
      CloneZoneArray(supplies, gZoneCache[slot].supplies);
   }
   return ok;
}

bool ExtractRecentSwingRefs(const SwingPointMem &swings[],
                            const int swingCount,
                            SwingPointMem &lastHigh,
                            SwingPointMem &prevHigh,
                            SwingPointMem &lastLow,
                            SwingPointMem &prevLow)
{
   bool hasLastHigh=false, hasPrevHigh=false, hasLastLow=false, hasPrevLow=false;
   for(int i=swingCount-1; i>=0; --i)
   {
      if(swings[i].isHigh)
      {
         if(!hasLastHigh) { lastHigh = swings[i]; hasLastHigh = true; }
         else if(!hasPrevHigh) { prevHigh = swings[i]; hasPrevHigh = true; }
      }
      else
      {
         if(!hasLastLow) { lastLow = swings[i]; hasLastLow = true; }
         else if(!hasPrevLow) { prevLow = swings[i]; hasPrevLow = true; }
      }
      if(hasLastHigh && hasPrevHigh && hasLastLow && hasPrevLow)
         break;
   }
   return (hasLastHigh || hasLastLow);
}

void FallbackStructureRefs(const string symbol,
                           const ENUM_TIMEFRAMES tf,
                           SwingPointMem &lastHigh,
                           SwingPointMem &prevHigh,
                           SwingPointMem &lastLow,
                           SwingPointMem &prevLow)
{
   int lookback = StructureLookbackForTF(tf);
   lastHigh.price = HighestInRangeTF(symbol, tf, 1, lookback);
   lastLow.price  = LowestInRangeTF(symbol, tf, 1, lookback);
   prevHigh.price = HighestInRangeTF(symbol, tf, lookback + 1, lookback);
   prevLow.price  = LowestInRangeTF(symbol, tf, lookback + 1, lookback);

   lastHigh.barShift = HighestIndexInRangeTF(symbol, tf, 1, lookback);
   lastLow.barShift  = LowestIndexInRangeTF(symbol, tf, 1, lookback);
   prevHigh.barShift = HighestIndexInRangeTF(symbol, tf, lookback + 1, lookback);
   prevLow.barShift  = LowestIndexInRangeTF(symbol, tf, lookback + 1, lookback);

   lastHigh.when = iTime(symbol, tf, lastHigh.barShift);
   lastLow.when  = iTime(symbol, tf, lastLow.barShift);
   prevHigh.when = iTime(symbol, tf, prevHigh.barShift);
   prevLow.when  = iTime(symbol, tf, prevLow.barShift);

   lastHigh.isHigh = true;
   prevHigh.isHigh = true;
   lastLow.isHigh  = false;
   prevLow.isHigh  = false;
}

double SwingTendencyFromMemory(const SwingPointMem &lastHigh,
                               const SwingPointMem &prevHigh,
                               const SwingPointMem &lastLow,
                               const SwingPointMem &prevLow,
                               const double atr)
{
   double highProg = Clamp((lastHigh.price - prevHigh.price) / MathMax(atr,1e-8), -StructDistClampAtr, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
   double lowProg  = Clamp((lastLow.price - prevLow.price) / MathMax(atr,1e-8), -StructDistClampAtr, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
   return Clamp(0.5 * (highProg + lowProg), -1.0, 1.0);
}

double SwingSequenceStrengthFromMemory(const SwingPointMem &lastHigh,
                                       const SwingPointMem &prevHigh,
                                       const SwingPointMem &lastLow,
                                       const SwingPointMem &prevLow,
                                       const double atr,
                                       const double tendency)
{
   double seqStrengthRaw = 0.5 * (MathAbs(lastHigh.price - prevHigh.price) + MathAbs(lastLow.price - prevLow.price));
   double seqStrength = StructNormAmp(seqStrengthRaw, atr);
   if(tendency < 0.0) seqStrength = -seqStrength;
   return seqStrength;
}



double BullBOSStrengthFromMemory(const double price,
                                 const SwingPointMem &lastHigh,
                                 const double atr,
                                 const double tendency)
{
   if(price <= lastHigh.price || tendency < -0.05)
      return 0.0;
   return Clamp((price - lastHigh.price) / MathMax(atr,1e-8), 0.0, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
}

double BearBOSStrengthFromMemory(const double price,
                                 const SwingPointMem &lastLow,
                                 const double atr,
                                 const double tendency)
{
   if(price >= lastLow.price || tendency > 0.05)
      return 0.0;
   return Clamp((lastLow.price - price) / MathMax(atr,1e-8), 0.0, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
}

double BullCHoCHStrengthFromMemory(const double price,
                                   const SwingPointMem &lastHigh,
                                   const double atr,
                                   const double tendency)
{
   if(price <= lastHigh.price || tendency >= 0.05)
      return 0.0;
   return Clamp((price - lastHigh.price) / MathMax(atr,1e-8), 0.0, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
}

double BearCHoCHStrengthFromMemory(const double price,
                                   const SwingPointMem &lastLow,
                                   const double atr,
                                   const double tendency)
{
   if(price >= lastLow.price || tendency <= -0.05)
      return 0.0;
   return Clamp((lastLow.price - price) / MathMax(atr,1e-8), 0.0, StructDistClampAtr) / MathMax(StructDistClampAtr,1.0);
}

double StrongWeakHighStateFromMemory(const double price,
                                     const SwingPointMem &lastHigh,
                                     const double atr,
                                     const double tendency,
                                     const double seqStrength)
{
   double proxHigh = 1.0 - Clamp(MathAbs(lastHigh.price - price) / MathMax(atr * MathMax(StructDistClampAtr,1.0), 1e-8), 0.0, 1.0);
   double weakScore = Clamp(0.60 * MathMax(tendency, 0.0) + 0.20 * MathMax(seqStrength, 0.0) + 0.20 * proxHigh, 0.0, 1.0);
   return Clamp(1.0 - 2.0 * weakScore, -1.0, 1.0); // +1 strong, -1 weak
}

double StrongWeakLowStateFromMemory(const double price,
                                    const SwingPointMem &lastLow,
                                    const double atr,
                                    const double tendency,
                                    const double seqStrength)
{
   double proxLow = 1.0 - Clamp(MathAbs(price - lastLow.price) / MathMax(atr * MathMax(StructDistClampAtr,1.0), 1e-8), 0.0, 1.0);
   double weakScore = Clamp(0.60 * MathMax(-tendency, 0.0) + 0.20 * MathMax(-seqStrength, 0.0) + 0.20 * proxLow, 0.0, 1.0);
   return Clamp(1.0 - 2.0 * weakScore, -1.0, 1.0); // +1 strong, -1 weak
}
struct ZoneBranchFeaturePack
{
   double nearestDemandDist;
   double nearestSupplyDist;
   double zoneSideState;
   double zoneDepthPosition;
   double zoneFreshness;
   double zoneMitigation;
   double zoneInvalidationState;
   double zoneRelevance;
   double zoneRetestBreakState;
};

void InitZoneBranchFeaturePack(ZoneBranchFeaturePack &z)
{
   z.nearestDemandDist=0.0;
   z.nearestSupplyDist=0.0;
   z.zoneSideState=0.0;
   z.zoneDepthPosition=0.0;
   z.zoneFreshness=0.0;
   z.zoneMitigation=0.0;
   z.zoneInvalidationState=0.0;
   z.zoneRelevance=0.0;
   z.zoneRetestBreakState=0.0;
}


void InitSwingDerivedZone(SwingDerivedZone &z)
{
   z.valid=false;
   z.isDemand=false;
   z.low=0.0;
   z.high=0.0;
   z.center=0.0;
   z.displacementAtr=0.0;
   z.freshness=0.0;
   z.mitigationCount=0;
   z.invalidated=false;
   z.lifeState=0;
   z.pivotShift=0;
   z.pivotTime=0;
}

double ZoneFreshnessFromShift(const ENUM_TIMEFRAMES tf,const int pivotShift)
{
   int barsBase = MathMax(StructureHistoryBarsForTF(tf), 200);
   return 1.0 - Clamp((double)pivotShift / (double)barsBase, 0.0, 1.0);
}

void GetPivotBodyBounds(const string symbol,
                        const ENUM_TIMEFRAMES tf,
                        const int pivotShift,
                        const int clusterBars,
                        double &bodyLow,
                        double &bodyHigh)
{
   bodyLow = DBL_MAX;
   bodyHigh = -DBL_MAX;
   int left = MathMax(clusterBars, 0);
   for(int d=-left; d<=left; ++d)
   {
      int sh = pivotShift + d;
      if(sh < 1) continue;
      double o = iOpen(symbol, tf, sh);
      double c = GetCloseSafe(symbol, tf, sh);
      double lo = MathMin(o, c);
      double hi = MathMax(o, c);
      if(lo < bodyLow) bodyLow = lo;
      if(hi > bodyHigh) bodyHigh = hi;
   }
   if(bodyLow >= DBL_MAX/2.0) bodyLow = MathMin(iOpen(symbol, tf, pivotShift), GetCloseSafe(symbol, tf, pivotShift));
   if(bodyHigh <= -DBL_MAX/2.0) bodyHigh = MathMax(iOpen(symbol, tf, pivotShift), GetCloseSafe(symbol, tf, pivotShift));
}

void FinalizeZoneTouches(const string symbol,
                         const ENUM_TIMEFRAMES tf,
                         SwingDerivedZone &z)
{
   if(!z.valid) return;
   double atr = MathMax(GetATRValueTF(symbol, tf, 14, z.pivotShift), 1e-8);
   double buffer = MathMax(MathAbs(ZoneInvalidationBufferAtr) * atr, SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);

   z.mitigationCount = 0;
   z.invalidated = false;
   z.lifeState = 0;

   for(int sh = z.pivotShift - 1; sh >= 1; --sh)
   {
      double hi = iHigh(symbol, tf, sh);
      double lo = iLow(symbol, tf, sh);
      double cl = GetCloseSafe(symbol, tf, sh);

      bool touched = (hi >= z.low && lo <= z.high);
      if(touched)
         z.mitigationCount++;

      if(z.isDemand)
      {
         if(cl < z.low - buffer)
         {
            z.invalidated = true;
            break;
         }
      }
      else
      {
         if(cl > z.high + buffer)
         {
            z.invalidated = true;
            break;
         }
      }
   }

   if(z.invalidated) z.lifeState = -1;
   else if(z.mitigationCount > 0) z.lifeState = +1;
   else z.lifeState = 0;
}

bool BuildSwingDerivedZonesForTF_Core(const string symbol,
                                 const ENUM_TIMEFRAMES tf,
                                 SwingDerivedZone &demands[],
                                 int &demandCount,
                                 SwingDerivedZone &supplies[],
                                 int &supplyCount)
{
   demandCount = 0;
   supplyCount = 0;
   ArrayResize(demands, 0);
   ArrayResize(supplies, 0);

   SwingPointMem swings[];
   int swingCount = 0;
   if(!BuildSwingMemoryForTF_Core(symbol, tf, swings, swingCount) || swingCount < 2)
      return false;

   int startIdx = MathMax(0, swingCount - MathMax(ZoneMaxLookbackSwings, 4));
   for(int i=startIdx; i < swingCount - 1; ++i)
   {
      SwingPointMem a = swings[i];
      SwingPointMem b = swings[i+1];
      double atrA = MathMax(GetATRValueTF(symbol, tf, 14, a.barShift), 1e-8);

      if(!a.isHigh && b.isHigh)
      {
         double disp = (b.price - a.price) / atrA;
         if(disp >= ZoneMinDisplacementAtr)
         {
            SwingDerivedZone z;
            InitSwingDerivedZone(z);
            z.valid = true;
            z.isDemand = true;
            z.pivotShift = a.barShift;
            z.pivotTime  = a.when;
            z.displacementAtr = disp;
            z.freshness = ZoneFreshnessFromShift(tf, a.barShift);

            double bodyLow, bodyHigh;
            GetPivotBodyBounds(symbol, tf, a.barShift, ZonePivotClusterBars, bodyLow, bodyHigh);
            z.low = a.price;
            z.high = MathMax(bodyLow, z.low + SymbolInfoDouble(symbol,SYMBOL_POINT) * 2.0);
            z.center = 0.5 * (z.low + z.high);

            FinalizeZoneTouches(symbol, tf, z);
            ArrayResize(demands, demandCount + 1);
            demands[demandCount++] = z;
         }
      }
      else if(a.isHigh && !b.isHigh)
      {
         double disp = (a.price - b.price) / atrA;
         if(disp >= ZoneMinDisplacementAtr)
         {
            SwingDerivedZone z;
            InitSwingDerivedZone(z);
            z.valid = true;
            z.isDemand = false;
            z.pivotShift = a.barShift;
            z.pivotTime  = a.when;
            z.displacementAtr = disp;
            z.freshness = ZoneFreshnessFromShift(tf, a.barShift);

            double bodyLow, bodyHigh;
            GetPivotBodyBounds(symbol, tf, a.barShift, ZonePivotClusterBars, bodyLow, bodyHigh);
            z.high = a.price;
            z.low = MathMin(bodyHigh, z.high - SymbolInfoDouble(symbol,SYMBOL_POINT) * 2.0);
            z.center = 0.5 * (z.low + z.high);

            FinalizeZoneTouches(symbol, tf, z);
            ArrayResize(supplies, supplyCount + 1);
            supplies[supplyCount++] = z;
         }
      }
   }

   return (demandCount > 0 || supplyCount > 0);
}



double DistanceToZoneBoundaryAtr(const SwingDerivedZone &z,const double px,const double atr)
{
   double a = MathMax(atr, 1e-8);
   if(px < z.low)  return (z.low - px) / a;
   if(px > z.high) return (px - z.high) / a;
   return 0.0;
}

double ZoneDepthForPrice(const SwingDerivedZone &z,const double px,const double atr)
{
   double h = MathMax(z.high - z.low, MathMax(atr,1e-8) * 0.10);
   double buf = MathMax(MathAbs(ZoneNearBufferAtr) * MathMax(atr,1e-8), SymbolInfoDouble(_Symbol,SYMBOL_POINT) * 5.0);
   if(px < z.low - buf || px > z.high + buf)
      return 0.0;

   if(z.isDemand)
   {
      double deepRef = z.low - buf;
      double shallowRef = z.high + buf;
      double depth = 1.0 - Clamp((px - deepRef) / MathMax(shallowRef - deepRef, 1e-8), 0.0, 1.0);
      return (depth * 2.0) - 1.0;
   }
   else
   {
      double shallowRef = z.low - buf;
      double deepRef = z.high + buf;
      double depth = Clamp((px - shallowRef) / MathMax(deepRef - shallowRef, 1e-8), 0.0, 1.0);
      return (depth * 2.0) - 1.0;
   }
}

double ZoneStateWeight(const SwingDerivedZone &z)
{
   if(z.lifeState > 0)  return 1.00; // active
   if(z.lifeState == 0) return 0.72; // candidate / first-touch
   return 0.30;                      // invalidated
}

bool IsTFEnabledForStructure(const ENUM_TIMEFRAMES tf)
{
   if(tf == TF_EXEC)       return UseTFExecFeatures;
   if(tf == TF_MID)        return UseTFMidFeatures;
   if(tf == TF_LONG)       return UseTFLongFeatures;
   if(tf == TF_STRUCT_EXT) return UseTFStructExtFeatures;
   return true;
}

double TFImportanceWeight(const ENUM_TIMEFRAMES tf)
{
   if(tf == TF_EXEC)       return 1.00;
   if(tf == TF_MID)        return 1.10;
   if(tf == TF_LONG)       return 1.20;
   if(tf == TF_STRUCT_EXT) return 1.30;
   return 1.00;
}

double ZoneOverlapDistanceAtr(const SwingDerivedZone &a,const SwingDerivedZone &b,const double atr)
{
   double lo = MathMax(a.low, b.low);
   double hi = MathMin(a.high, b.high);
   if(lo <= hi) return 0.0;
   if(a.high < b.low) return (b.low - a.high) / MathMax(atr,1e-8);
   return (a.low - b.high) / MathMax(atr,1e-8);
}

double ZoneConfluenceBoost(const string symbol,
                           const ENUM_TIMEFRAMES baseTf,
                           const SwingDerivedZone &z,
                           const double atr)
{
   ENUM_TIMEFRAMES tfs[4] = { TF_EXEC, TF_MID, TF_LONG, TF_STRUCT_EXT };
   double boost = 1.0;
   for(int i=0;i<4;i++)
   {
      ENUM_TIMEFRAMES tf2 = tfs[i];
      if(tf2 == baseTf) continue;
      if(!IsTFEnabledForStructure(tf2)) continue;

      SwingDerivedZone demands2[], supplies2[];
      int dc2=0, sc2=0;
      if(!BuildSwingDerivedZonesForTF(symbol, tf2, demands2, dc2, supplies2, sc2))
         continue;

      SwingDerivedZone bestOther;
      bool found = false;
      double bestDist = DBL_MAX;
      if(z.isDemand)
      {
         for(int j=0;j<dc2;j++)
         {
            if(!demands2[j].valid) continue;
            double d = ZoneOverlapDistanceAtr(z, demands2[j], atr);
            if(d < bestDist) { bestDist = d; bestOther = demands2[j]; found = true; }
         }
      }
      else
      {
         for(int j=0;j<sc2;j++)
         {
            if(!supplies2[j].valid) continue;
            double d = ZoneOverlapDistanceAtr(z, supplies2[j], atr);
            if(d < bestDist) { bestDist = d; bestOther = supplies2[j]; found = true; }
         }
      }

      if(found && bestDist <= MathAbs(ZoneConfluenceAtr))
      {
         double closeness = 1.0 - Clamp(bestDist / MathMax(MathAbs(ZoneConfluenceAtr),1e-8), 0.0, 1.0);
         boost += 0.12 * TFImportanceWeight(tf2) * closeness * ZoneStateWeight(bestOther);
      }
   }
   return Clamp(boost, 1.0, 1.6);
}

double SwingLevelConfluenceBoost(const string symbol,
                                 const ENUM_TIMEFRAMES baseTf,
                                 const double level,
                                 const bool isHigh,
                                 const double atr)
{
   ENUM_TIMEFRAMES tfs[4] = { TF_EXEC, TF_MID, TF_LONG, TF_STRUCT_EXT };
   double boost = 1.0;
   for(int i=0;i<4;i++)
   {
      ENUM_TIMEFRAMES tf2 = tfs[i];
      if(tf2 == baseTf) continue;
      if(!IsTFEnabledForStructure(tf2)) continue;

      SwingPointMem swings2[];
      int count2 = 0;
      if(!BuildSwingMemoryForTF(symbol, tf2, swings2, count2) || count2 < 1)
         continue;

      double best = DBL_MAX;
      for(int j=count2-1; j>=0 && j>=count2-12; --j)
      {
         if(swings2[j].isHigh != isHigh) continue;
         double d = MathAbs(swings2[j].price - level) / MathMax(atr,1e-8);
         if(d < best) best = d;
      }
      if(best <= MathAbs(SwingConfluenceAtr))
      {
         double closeness = 1.0 - Clamp(best / MathMax(MathAbs(SwingConfluenceAtr),1e-8), 0.0, 1.0);
         boost += 0.10 * TFImportanceWeight(tf2) * closeness;
      }
   }
   return Clamp(boost, 1.0, 1.5);
}

double SwingNearnessScore(const string symbol,const ENUM_TIMEFRAMES tf,const double px,const double atr)
{
   SwingPointMem swings[];
   int swingCount = 0;
   SwingPointMem lastHigh, prevHigh, lastLow, prevLow;
   bool ok = BuildSwingMemoryForTF(symbol, tf, swings, swingCount);
   ZeroMemory(lastHigh); ZeroMemory(prevHigh); ZeroMemory(lastLow); ZeroMemory(prevLow);
   bool hasRefs = false;
   if(ok)
      hasRefs = ExtractRecentSwingRefs(swings, swingCount, lastHigh, prevHigh, lastLow, prevLow);
   if(!hasRefs)
      FallbackStructureRefs(symbol, tf, lastHigh, prevHigh, lastLow, prevLow);

   double d = DBL_MAX;
   d = MathMin(d, MathAbs(px - lastHigh.price) / MathMax(atr,1e-8));
   d = MathMin(d, MathAbs(px - lastLow.price) / MathMax(atr,1e-8));
   d = MathMin(d, MathAbs(px - prevHigh.price) / MathMax(atr,1e-8));
   d = MathMin(d, MathAbs(px - prevLow.price) / MathMax(atr,1e-8));
   double nearBuf = MathMax(MathAbs(ZoneNearBufferAtr), 0.25);
   return 1.0 - Clamp(d / nearBuf, 0.0, 1.0);
}

double ZoneProximityScore(const SwingDerivedZone &z,const double px,const double avgPx,const double entryPx,const double atr)
{
   double wPrice = 0.50, wAvg = 0.30, wEntry = 0.20;

   double d1 = DistanceToZoneBoundaryAtr(z, px, atr);
   double d2 = DistanceToZoneBoundaryAtr(z, avgPx, atr);
   double d3 = DistanceToZoneBoundaryAtr(z, entryPx, atr);

   if(px >= z.low && px <= z.high)
      d1 = 0.0;
   if(avgPx >= z.low && avgPx <= z.high)
      d2 = 0.0;
   if(entryPx >= z.low && entryPx <= z.high)
      d3 = 0.0;

   double distAtr = (wPrice*d1 + wAvg*d2 + wEntry*d3);
   return 1.0 - Clamp(distAtr / 4.0, 0.0, 1.0);
}

bool SelectBestSwingZone(const SwingDerivedZone &zonesArr[],
                         const int count,
                         const string symbol,
                         const ENUM_TIMEFRAMES tf,
                         const double px,
                         const double avgPx,
                         const double entryPx,
                         const double atr,
                         int &bestIdx)
{
   bestIdx = -1;
   double bestScore = -DBL_MAX;
   for(int i=0; i<count; ++i)
   {
      if(!zonesArr[i].valid) continue;

      double prox = ZoneProximityScore(zonesArr[i], px, avgPx, entryPx, atr);
      double dispW = (0.65 + 0.35 * Clamp(zonesArr[i].displacementAtr / 3.0, 0.0, 1.0));
      double freshW = (0.55 + 0.45 * zonesArr[i].freshness);
      double mitiPenalty = (1.0 - 0.10 * MathMin(zonesArr[i].mitigationCount, 5));
      double stateW = ZoneStateWeight(zonesArr[i]);
      double confW = ZoneConfluenceBoost(symbol, tf, zonesArr[i], atr);

      double score = prox * dispW * freshW * mitiPenalty * stateW * confW;
      if(score > bestScore)
      {
         bestScore = score;
         bestIdx = i;
      }
   }
   return (bestIdx >= 0);
}

double ZoneRetestBreakStateForTF(const SwingDerivedZone &z,const string symbol,const ENUM_TIMEFRAMES tf,const double atr)
{
   double close1 = GetCloseSafe(symbol, tf, 1);
   double high1  = iHigh(symbol, tf, 1);
   double low1   = iLow(symbol, tf, 1);
   double open1  = iOpen(symbol, tf, 1);
   double buffer = MathMax(MathAbs(ZoneInvalidationBufferAtr) * MathMax(atr,1e-8), SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);

   if(z.isDemand)
   {
      if(close1 < z.low - buffer) return -1.0;
      if(low1 <= z.high + buffer && close1 > open1 && close1 >= z.low) return 0.5;
      if(low1 <= z.high + buffer) return 0.2;
      return 0.0;
   }
   else
   {
      if(close1 > z.high + buffer) return 1.0;
      if(high1 >= z.low - buffer && close1 < open1 && close1 <= z.high) return -0.5;
      if(high1 >= z.low - buffer) return -0.2;
      return 0.0;
   }
}

void ComputeZoneBranchFeatures(const string symbol,
                               const ENUM_TIMEFRAMES tf,
                               const double refPrice,
                               const double avgPrice,
                               const double lastEntryPrice,
                               ZoneBranchFeaturePack &outZ)
{
   InitZoneBranchFeaturePack(outZ);
   double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);

   SwingDerivedZone demands[], supplies[];
   int demandCount = 0, supplyCount = 0;
   if(!BuildSwingDerivedZonesForTF(symbol, tf, demands, demandCount, supplies, supplyCount))
      return;

   int idxDem = -1, idxSup = -1;
   bool hasDem = SelectBestSwingZone(demands, demandCount, symbol, tf, refPrice, avgPrice, lastEntryPrice, atr, idxDem);
   bool hasSup = SelectBestSwingZone(supplies, supplyCount, symbol, tf, refPrice, avgPrice, lastEntryPrice, atr, idxSup);

   if(hasDem)
      outZ.nearestDemandDist = Clamp(DistanceToZoneBoundaryAtr(demands[idxDem], refPrice, atr), 0.0, 5.0) / 5.0;
   if(hasSup)
      outZ.nearestSupplyDist = Clamp(DistanceToZoneBoundaryAtr(supplies[idxSup], refPrice, atr), 0.0, 5.0) / 5.0;

   int activeSide = 0;
   if(hasDem && hasSup)
   {
      double sd = ZoneProximityScore(demands[idxDem], refPrice, avgPrice, lastEntryPrice, atr) * ZoneStateWeight(demands[idxDem]);
      double ss = ZoneProximityScore(supplies[idxSup], refPrice, avgPrice, lastEntryPrice, atr) * ZoneStateWeight(supplies[idxSup]);
      activeSide = (sd >= ss ? +1 : -1);
   }
   else if(hasDem) activeSide = +1;
   else if(hasSup) activeSide = -1;
   else return;

   SwingDerivedZone z = (activeSide > 0 ? demands[idxDem] : supplies[idxSup]);

   double proxPrice = 1.0 - Clamp(DistanceToZoneBoundaryAtr(z, refPrice, atr) / MathMax(MathAbs(ZoneNearBufferAtr), 0.25), 0.0, 1.0);
   double proxAvg   = 1.0 - Clamp(DistanceToZoneBoundaryAtr(z, avgPrice, atr) / MathMax(MathAbs(ZoneNearBufferAtr), 0.25), 0.0, 1.0);
   double proxEntry = 1.0 - Clamp(DistanceToZoneBoundaryAtr(z, lastEntryPrice, atr) / MathMax(MathAbs(ZoneNearBufferAtr), 0.25), 0.0, 1.0);
   double combinedProx = Clamp(0.50*proxPrice + 0.30*proxAvg + 0.20*proxEntry, 0.0, 1.0);

   outZ.zoneSideState = (z.isDemand ? 1.0 : -1.0);
   outZ.zoneDepthPosition = ZoneDepthForPrice(z, refPrice, atr);
   outZ.zoneFreshness = z.freshness;
   outZ.zoneMitigation = Clamp((double)z.mitigationCount / 5.0, 0.0, 1.0);
   outZ.zoneInvalidationState = (z.lifeState > 0 ? 1.0 : (z.lifeState == 0 ? 0.0 : -1.0));

   double dispWeight = Clamp(z.displacementAtr / 3.0, 0.0, 1.0);
   double freshWeight = 0.50 + 0.50 * z.freshness;
   double mitiPenalty = 1.0 - 0.12 * MathMin(z.mitigationCount, 5);
   double stateWeight = ZoneStateWeight(z);
   double confWeight = ZoneConfluenceBoost(symbol, tf, z, atr);
   double signedRel = (z.isDemand ? 1.0 : -1.0) * combinedProx * freshWeight * (0.5 + 0.5*dispWeight) * mitiPenalty * stateWeight * confWeight;
   outZ.zoneRelevance = Clamp(signedRel, -1.0, 1.0);
   outZ.zoneRetestBreakState = ZoneRetestBreakStateForTF(z, symbol, tf, atr);
}
double HighestInRangeTF(const string symbol,const ENUM_TIMEFRAMES tf,const int startShift,const int count)
{
   double best = -DBL_MAX;
   for(int i=0;i<count;i++)
   {
      double v = iHigh(symbol, tf, startShift + i);
      if(v != EMPTY_VALUE && v > best) best = v;
   }
   if(best <= -DBL_MAX/2.0) best = GetCloseSafe(symbol, tf, startShift);
   return best;
}

double LowestInRangeTF(const string symbol,const ENUM_TIMEFRAMES tf,const int startShift,const int count)
{
   double best = DBL_MAX;
   for(int i=0;i<count;i++)
   {
      double v = iLow(symbol, tf, startShift + i);
      if(v != EMPTY_VALUE && v < best) best = v;
   }
   if(best >= DBL_MAX/2.0) best = GetCloseSafe(symbol, tf, startShift);
   return best;
}

int HighestIndexInRangeTF(const string symbol,const ENUM_TIMEFRAMES tf,const int startShift,const int count)
{
   double best = -DBL_MAX;
   int bestIdx = startShift;
   for(int i=0;i<count;i++)
   {
      int idx = startShift + i;
      double v = iHigh(symbol, tf, idx);
      if(v != EMPTY_VALUE && v > best) { best = v; bestIdx = idx; }
   }
   return bestIdx;
}

int LowestIndexInRangeTF(const string symbol,const ENUM_TIMEFRAMES tf,const int startShift,const int count)
{
   double best = DBL_MAX;
   int bestIdx = startShift;
   for(int i=0;i<count;i++)
   {
      int idx = startShift + i;
      double v = iLow(symbol, tf, idx);
      if(v != EMPTY_VALUE && v < best) { best = v; bestIdx = idx; }
   }
   return bestIdx;
}
double GetCloseSafe(const string symbol,const ENUM_TIMEFRAMES tf,const int shift)
{
   double v=iClose(symbol,tf,shift);
   if(v==0.0 || v==EMPTY_VALUE)
   {
      MqlTick t;
      if(SymbolInfoTick(symbol,t)) return (t.bid+t.ask)*0.5;
   }
   return v;
}

double RollingAvgSpreadPtsTF(const string symbol,const ENUM_TIMEFRAMES tf,const int bars)
{
   int n=MathMax(bars,1);
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   int copied=CopyRates(symbol,tf,1,n,rates);
   if(copied<=0) return 0.0;
   double sum=0.0;
   int cnt=0;
   for(int i=0;i<copied;i++)
   {
      if(rates[i].spread>=0)
      {
         sum += (double)rates[i].spread;
         cnt++;
      }
   }
   if(cnt<=0) return 0.0;
   return sum/(double)cnt;
}

double CopyLatestFromHandle(const int handle,const int buffer,const int shift,const double fallback=0.0)
{
   if(handle==INVALID_HANDLE) return fallback;
   double buf[];
   ArrayResize(buf,shift+1);
   ArraySetAsSeries(buf,true);
   if(CopyBuffer(handle,buffer,0,shift+1,buf) <= shift)
   {
      IndicatorRelease(handle);
      return fallback;
   }
   double v=buf[shift];
   IndicatorRelease(handle);
   if(v==EMPTY_VALUE) return fallback;
   return v;
}

double GetMAValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iMA(symbol,tf,period,0,MODE_EMA,PRICE_CLOSE);
   return CopyLatestFromHandle(h,0,shift,GetCloseSafe(symbol,tf,shift));
}

double GetRSIValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iRSI(symbol,tf,period,PRICE_CLOSE);
   return CopyLatestFromHandle(h,0,shift,50.0);
}

double GetCCIValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iCCI(symbol,tf,period,PRICE_TYPICAL);
   return CopyLatestFromHandle(h,0,shift,0.0);
}

double GetATRValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iATR(symbol,tf,period);
   return MathMax(CopyLatestFromHandle(h,0,shift,0.0),1e-8);
}

void GetMACDValuesTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift,double &mainVal,double &signalVal)
{
   int h=iMACD(symbol,tf,12,26,9,PRICE_CLOSE);
   mainVal   = CopyLatestFromHandle(h,0,shift,0.0);
   // handle was released above, recreate for signal
   h=iMACD(symbol,tf,12,26,9,PRICE_CLOSE);
   signalVal = CopyLatestFromHandle(h,1,shift,0.0);
}

double GetRVIValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift=1)
{
   int h=iRVI(symbol,tf,10);
   return CopyLatestFromHandle(h,0,shift,0.0);
}

double GetADXValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iADX(symbol,tf,period);
   return CopyLatestFromHandle(h,0,shift,20.0);
}

double GetPlusMinusDIValueTF(const string symbol,const ENUM_TIMEFRAMES tf,const int period,const int shift=1)
{
   int h=iADX(symbol,tf,period);
   double plus = CopyLatestFromHandle(h,1,shift,25.0);
   h=iADX(symbol,tf,period);
   double minus = CopyLatestFromHandle(h,2,shift,25.0);
   double denom = MathMax(plus + minus, 1e-8);
   return Clamp((plus - minus) / denom, -1.0, 1.0);
}

double ReturnBarsTF(const string symbol,const ENUM_TIMEFRAMES tf,const int barsBack)
{
   double c0=GetCloseSafe(symbol,tf,1);
   double cN=GetCloseSafe(symbol,tf,1+barsBack);
   if(cN==0.0) return 0.0;
   return (c0-cN)/cN;
}

double BarRangeAtrTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift=1)
{
   double hi=iHigh(symbol,tf,shift);
   double lo=iLow(symbol,tf,shift);
   double atr=GetATRValueTF(symbol,tf,14,shift);
   return Clamp((hi-lo)/MathMax(atr,1e-8),0.0,10.0)/10.0;
}

double CloseChangeAtrTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift=1)
{
   double c0=GetCloseSafe(symbol,tf,shift);
   double c1=GetCloseSafe(symbol,tf,shift+1);
   double atr=GetATRValueTF(symbol,tf,14,shift);
   return Clamp((c0-c1)/MathMax(atr,1e-8),-5.0,5.0)/5.0;
}

double KCDistFeatureTF(const string symbol,const ENUM_TIMEFRAMES tf,const int which)
{
   double close = GetCloseSafe(symbol,tf,1);
   double ema   = GetMAValueTF(symbol,tf,20,1);
   double atr   = GetATRValueTF(symbol,tf,20,1);
   double upper = ema + 2.0*atr;
   double lower = ema - 2.0*atr;
   if(which==0) return Clamp((upper-close)/MathMax(atr,1e-8),-5.0,5.0)/5.0; // to upper
   if(which==1) return Clamp((close-lower)/MathMax(atr,1e-8),-5.0,5.0)/5.0; // to lower
   if(which==2) return Clamp((close-ema)/MathMax(atr,1e-8),-5.0,5.0)/5.0;   // to mid
   if(close>upper) return 1.0;
   if(close<lower) return -1.0;
   return 0.0;
}

double EMAFeatTF(const string symbol,const ENUM_TIMEFRAMES tf,const int mode)
{
   double close1=GetCloseSafe(symbol,tf,1);
   double close2=GetCloseSafe(symbol,tf,2);
   double ema1=GetMAValueTF(symbol,tf,20,1);
   double ema2=GetMAValueTF(symbol,tf,20,2);
   double atr=GetATRValueTF(symbol,tf,14,1);
   if(mode==0) return Clamp((close1-ema1)/MathMax(atr,1e-8),-5.0,5.0)/5.0;
   return Clamp((ema1-ema2)/MathMax(atr,1e-8),-5.0,5.0)/5.0;
}

double ATRRatioTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   double fast=GetATRValueTF(symbol,tf,7,1);
   double slow=GetATRValueTF(symbol,tf,20,1);
   return Clamp(fast/MathMax(slow,1e-8),0.0,5.0)/5.0;
}

double StdReturnsTF(const string symbol,const ENUM_TIMEFRAMES tf,const int len,const int shiftBase=1)
{
   if(len<2) return 0.0;
   double vals[];
   ArrayResize(vals,len);
   int n=0;
   for(int i=0;i<len;i++)
   {
      double c0=GetCloseSafe(symbol,tf,shiftBase+i);
      double c1=GetCloseSafe(symbol,tf,shiftBase+i+1);
      if(c1==0.0) continue;
      vals[n++] = (c0-c1)/c1;
   }
   if(n<2) return 0.0;
   double mean=0.0;
   for(int i=0;i<n;i++) mean += vals[i];
   mean /= n;
   double var=0.0;
   for(int i=0;i<n;i++){ double d=vals[i]-mean; var += d*d; }
   var /= MathMax(n-1,1);
   return MathSqrt(MathMax(var,0.0));
}

double RealizedVolRatioTF(const string symbol,const ENUM_TIMEFRAMES tf,const int fastLen,const int slowLen)
{
   double f=StdReturnsTF(symbol,tf,fastLen,1);
   double s=StdReturnsTF(symbol,tf,slowLen,1);
   return Clamp(f/MathMax(s,1e-8),0.0,5.0)/5.0;
}

double RealizedVolLevelTF(const string symbol,const ENUM_TIMEFRAMES tf,const int len)
{
   return Clamp(StdReturnsTF(symbol,tf,len,1)*100.0,0.0,5.0)/5.0;
}

double RealizedVolDeltaTF(const string symbol,const ENUM_TIMEFRAMES tf,const int len)
{
   double cur=StdReturnsTF(symbol,tf,len,1);
   double prev=StdReturnsTF(symbol,tf,len,2);
   double base=MathMax(prev,1e-8);
   return Clamp((cur-prev)/base,-2.0,2.0)/2.0;
}

double RealizedVolGammaTF(const string symbol,const ENUM_TIMEFRAMES tf,const int len)
{
   double d1=RealizedVolDeltaTF(symbol,tf,len);
   double cur=StdReturnsTF(symbol,tf,len,2);
   double prev=StdReturnsTF(symbol,tf,len,3);
   double base=MathMax(prev,1e-8);
   double d0=Clamp((cur-prev)/base,-2.0,2.0)/2.0;
   return Clamp(d1-d0,-2.0,2.0)/2.0;
}

double RangeExpansionTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shortLen,const int longLen)
{
   double sumS=0.0,sumL=0.0;
   for(int i=1;i<=shortLen;i++) sumS += iHigh(symbol,tf,i)-iLow(symbol,tf,i);
   for(int i=1;i<=longLen;i++)  sumL += iHigh(symbol,tf,i)-iLow(symbol,tf,i);
   double aS=sumS/MathMax(shortLen,1);
   double aL=sumL/MathMax(longLen,1);
   return Clamp(aS/MathMax(aL,1e-8),0.0,5.0)/5.0;
}

double HighRangeBarStreakTF(const string symbol,const ENUM_TIMEFRAMES tf,const int lookback,const double mult)
{
   double avg=0.0;
   for(int i=2;i<2+lookback;i++) avg += iHigh(symbol,tf,i)-iLow(symbol,tf,i);
   avg /= MathMax(lookback,1);
   int streak=0;
   for(int i=1;i<1+lookback;i++)
   {
      double r=iHigh(symbol,tf,i)-iLow(symbol,tf,i);
      if(r > avg*mult) streak++;
      else break;
   }
   return Clamp((double)streak/(double)MathMax(lookback,1),0.0,1.0);
}

double GetStateBudgetBase()
{
   if(EquityBudget > 1e-9)  return EquityBudget;
   if(gEAStartEquity > 1e-9) return gEAStartEquity;
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > 1e-9) return eq;
   return 1.0;
}

int GetBasketDirStateFast(const string symbol,const int symIdx,const int magic)
{
   if(symIdx>=0 && symIdx<MAX_SYMBOLS)
   {
      if(gPositionsCount[symIdx] <= 0) return 0;
      if(gHasPositionTypeCache[symIdx]) return gBasketDirCache[symIdx];
   }

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      return (pt==POSITION_TYPE_BUY ? +1 : -1);
   }
   return 0;
}

double GetLastEntryPriceState(const string symbol,const int magic,const int basketDir,const double fallbackPrice)
{
   if(basketDir==0) return fallbackPrice;

   datetime lastT = 0;
   double   lastP = fallbackPrice;

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      int dir = (pt==POSITION_TYPE_BUY ? +1 : -1);
      if(dir != basketDir) continue;

      datetime t = (datetime)PositionGetInteger(POSITION_TIME);
      double   p = PositionGetDouble(POSITION_PRICE_OPEN);
      if(t >= lastT)
      {
         lastT = t;
         lastP = p;
      }
   }
   return lastP;
}

double GetBasketAgeBarsState(const int symIdx,const ENUM_TIMEFRAMES tf)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   if(gPositionsCount[symIdx] <= 0)    return 0.0;
   if(gFirstTradeTime[symIdx] <= 0)     return 0.0;

   int sec = PeriodSeconds(tf);
   if(sec <= 0) sec = PeriodSeconds(BaseTF);
   if(sec <= 0) sec = 60;

   double ageSec = (double)(TimeCurrent() - gFirstTradeTime[symIdx]);
   if(ageSec <= 0.0) return 0.0;
   return ageSec / (double)sec;
}


double RSIStateTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return Clamp((GetRSIValueTF(symbol, tf, 14, 1) - 50.0) / 30.0, -1.0, 1.0);
}

double CCIStateTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return Clamp(GetCCIValueTF(symbol, tf, 20, 1) / 200.0, -1.0, 1.0);
}

double MACDDiffStateTF(const string symbol,const ENUM_TIMEFRAMES tf,const double point)
{
   double mainV=0.0, signalV=0.0;
   GetMACDValuesTF(symbol, tf, 1, mainV, signalV);
   return Clamp((mainV - signalV) / MathMax(120.0 * point, 1e-8), -1.0, 1.0);
}

double MACDMainStateTF(const string symbol,const ENUM_TIMEFRAMES tf,const double point)
{
   double mainV=0.0, signalV=0.0;
   GetMACDValuesTF(symbol, tf, 1, mainV, signalV);
   return Clamp(mainV / MathMax(160.0 * point, 1e-8), -1.0, 1.0);
}

double RVIStateTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return Clamp(GetRVIValueTF(symbol, tf, 1), -1.0, 1.0);
}

double ADXStrengthTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return Clamp(GetADXValueTF(symbol, tf, 14, 1) / 35.0, 0.0, 1.0);
}

double VolumeRatioTF(const string symbol,const ENUM_TIMEFRAMES tf,const int lookback)
{
   long cur=iVolume(symbol, tf, 1);
   double acc=0.0;
   int used=0;
   for(int i=2;i<2+MathMax(lookback,4);i++)
   {
      long v=iVolume(symbol, tf, i);
      if(v<=0) continue;
      acc += (double)v;
      used++;
   }
   double avg=(used>0 ? acc/(double)used : (double)MathMax(cur,1));
   return Clamp(SafeDiv((double)cur, MathMax(avg,1.0), 1.0), 0.0, 3.0) / 3.0;
}

double BodyEfficiencyTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift)
{
   double openV=iOpen(symbol, tf, shift);
   double highV=iHigh(symbol, tf, shift);
   double lowV =iLow(symbol, tf, shift);
   double closeV=iClose(symbol, tf, shift);
   double range=MathMax(highV-lowV, SymbolInfoDouble(symbol,SYMBOL_POINT)*5.0);
   return Clamp(MathAbs(closeV-openV)/range, 0.0, 1.0);
}

double WickInstabilityTF(const string symbol,const ENUM_TIMEFRAMES tf,const int shift)
{
   double openV=iOpen(symbol, tf, shift);
   double highV=iHigh(symbol, tf, shift);
   double lowV =iLow(symbol, tf, shift);
   double closeV=iClose(symbol, tf, shift);
   double range=MathMax(highV-lowV, SymbolInfoDouble(symbol,SYMBOL_POINT)*5.0);
   double body=MathAbs(closeV-openV);
   return Clamp(1.0 - body/range, 0.0, 1.0);
}

void ComputeTrendFactorTF(const string symbol,
                          const ENUM_TIMEFRAMES tf,
                          const double point,
                          double &dir,
                          double &strength,
                          double &persistence,
                          double &accel,
                          double &overextension,
                          double &meanRev,
                          double &continuation,
                          double &reversalQual,
                          double &volumeAlign)
{
   double rsi=RSIStateTF(symbol,tf);
   double cci=CCIStateTF(symbol,tf);
   double macdDiff=MACDDiffStateTF(symbol,tf,point);
   double macdMain=MACDMainStateTF(symbol,tf,point);
   double rvi=RVIStateTF(symbol,tf);
   double emaDist=Clamp(EMAFeatTF(symbol,tf,0),-1.0,1.0);
   double emaSlope=Clamp(EMAFeatTF(symbol,tf,1),-1.0,1.0);
   double di=Clamp(GetPlusMinusDIValueTF(symbol,tf,14,1),-1.0,1.0);
   double adx=ADXStrengthTF(symbol,tf);
   double ret1=Clamp(ReturnBarsTF(symbol,tf,1)*100.0/2.0,-1.0,1.0);
   double ret3=Clamp(ReturnBarsTF(symbol,tf,3)*100.0/4.0,-1.0,1.0);
   double ret5=Clamp(ReturnBarsTF(symbol,tf,5)*100.0/6.0,-1.0,1.0);
   double volRatio=VolumeRatioTF(symbol,tf,10);
   double kcAbs=MathMax(MathMax(MathAbs(KCDistFeatureTF(symbol,tf,0)),MathAbs(KCDistFeatureTF(symbol,tf,1))),
                        MathMax(MathAbs(KCDistFeatureTF(symbol,tf,2)),MathAbs(KCDistFeatureTF(symbol,tf,3))));

   dir = Clamp(0.22*emaDist + 0.14*emaSlope + 0.18*macdDiff + 0.16*di + 0.10*rsi + 0.08*cci + 0.06*rvi + 0.06*ret3, -1.0, 1.0);
   strength = Clamp(0.45*MathAbs(dir) + 0.25*adx + 0.15*MathAbs(macdMain) + 0.15*MathAbs(di), 0.0, 1.0);

   double same1=(dir*ret1>0.0 ? 1.0 : 0.0);
   double same3=(dir*ret3>0.0 ? 1.0 : 0.0);
   double same5=(dir*ret5>0.0 ? 1.0 : 0.0);
   persistence = Clamp(0.35*same1 + 0.35*same3 + 0.15*same5 + 0.15*adx, 0.0, 1.0);

   double adxDelta=Clamp((GetADXValueTF(symbol,tf,14,1)-GetADXValueTF(symbol,tf,14,2))/20.0,-1.0,1.0);
   accel = Clamp(0.45*(ret1-ret3) + 0.30*adxDelta + 0.25*(macdDiff - rvi*0.5), -1.0, 1.0);

   overextension = Clamp(0.35*MathAbs(rsi) + 0.25*MathAbs(cci) + 0.25*kcAbs + 0.15*MathAbs(ret1), 0.0, 1.0);
   meanRev = Clamp(0.55*overextension + 0.20*(1.0-strength) + 0.25*MathMax(0.0, -dir*ret1), 0.0, 1.0);
   continuation = Clamp(0.42*strength + 0.20*persistence + 0.18*MathMax(0.0,accel) + 0.10*MathAbs(di) + 0.10*volRatio, 0.0, 1.0);
   reversalQual = Clamp(0.38*MathMax(0.0,-dir*ret1) + 0.24*overextension + 0.18*MathMax(0.0,-accel) + 0.10*(1.0-persistence) + 0.10*(1.0-adx), 0.0, 1.0);
   volumeAlign = Clamp(0.45*volRatio + 0.30*MathAbs(dir)*volRatio + 0.25*adx*volRatio, 0.0, 1.0);
}


double EMAReclaimStateTF(const string symbol,
                         const ENUM_TIMEFRAMES tf,
                         const int emaPeriod)
{
   if(emaPeriod<=1) return 0.0;

   int bars=iBars(symbol, tf);
   if(bars<5) return 0.0;

   int emaHandle=iMA(symbol, tf, emaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   if(emaHandle==INVALID_HANDLE) return 0.0;

   double emaBuf[];
   ArraySetAsSeries(emaBuf,true);
   int copied=CopyBuffer(emaHandle,0,0,4,emaBuf);
   IndicatorRelease(emaHandle);
   if(copied<3) return 0.0;

   double close1=iClose(symbol,tf,1);
   double close2=iClose(symbol,tf,2);
   double ema1=emaBuf[1];
   double ema2=emaBuf[2];
   double atr=MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);

   int side1=(close1>=ema1 ? 1 : -1);
   int side2=(close2>=ema2 ? 1 : -1);
   double distNorm=Clamp((close1-ema1)/MathMax(atr*2.0,1e-8), -1.0, 1.0);

   if(side1!=side2)
   {
      double strength=Clamp(MathAbs((close1-ema1)/MathMax(atr,1e-8)), 0.0, 1.0);
      return side1 * Clamp(0.65 + 0.35*strength, 0.0, 1.0);
   }

   return 0.35 * distNorm;
}

void ComputeMacroBiasContext(const string symbol,
                             const double point,
                             double &macroBias,
                             double &macroContinuation,
                             double &macroMaturity,
                             double &macroReclaim,
                             double &macroTransition,
                             double &lateTrendTrap)
{
   macroBias=0.0;
   macroContinuation=0.0;
   macroMaturity=0.0;
   macroReclaim=0.0;
   macroTransition=0.0;
   lateTrendTrap=0.0;

   double dirH1=0.0,strH1=0.0,persH1=0.0,accH1=0.0,overH1=0.0,mrH1=0.0,contH1=0.0,revH1=0.0,volH1=0.0;
   double dirH4=0.0,strH4=0.0,persH4=0.0,accH4=0.0,overH4=0.0,mrH4=0.0,contH4=0.0,revH4=0.0,volH4=0.0;

   ComputeTrendFactorTF(symbol, TF_LONG, point, dirH1,strH1,persH1,accH1,overH1,mrH1,contH1,revH1,volH1);
   ComputeTrendFactorTF(symbol, TF_STRUCT_EXT, point, dirH4,strH4,persH4,accH4,overH4,mrH4,contH4,revH4,volH4);

   double closeH1=GetCloseSafe(symbol, TF_LONG, 1);
   double closeH4=GetCloseSafe(symbol, TF_STRUCT_EXT, 1);

   double sDirH1=0.0,sStrH1=0.0,sContH1=0.0,sRevH1=0.0,sBreakH1=0.0,sTrapH1=0.0;
   double sDirH4=0.0,sStrH4=0.0,sContH4=0.0,sRevH4=0.0,sBreakH4=0.0,sTrapH4=0.0;
   ComputeStructureFactorTF(symbol, TF_LONG, closeH1, closeH1, closeH1, sDirH1,sStrH1,sContH1,sRevH1,sBreakH1,sTrapH1);
   ComputeStructureFactorTF(symbol, TF_STRUCT_EXT, closeH4, closeH4, closeH4, sDirH4,sStrH4,sContH4,sRevH4,sBreakH4,sTrapH4);

   double d1Dist=0.0,d1Slope=0.0,d1SideDur=0.0;
   GetD1TrendFeatures(symbol,d1Dist,d1Slope,d1SideDur);
   double d1Bias = Clamp(0.50*(d1Dist/5.0) + 0.35*(d1Slope/2.0) + 0.15*((d1Dist>=0.0 ? 1.0 : -1.0)*d1SideDur), -1.0, 1.0);
   double d1Continuation = Clamp(0.40*MathAbs(d1Bias) + 0.35*d1SideDur + 0.25*Clamp(MathAbs(d1Slope)/2.0,0.0,1.0), 0.0, 1.0);
   double d1Maturity = Clamp(0.45*Clamp(MathAbs(d1Dist)/5.0,0.0,1.0) + 0.35*d1SideDur + 0.20*(1.0- Clamp(MathAbs(d1Slope)/2.0,0.0,1.0)), 0.0, 1.0);

   double reclaimH1 = EMAReclaimStateTF(symbol, TF_LONG, 50);
   double reclaimH4 = EMAReclaimStateTF(symbol, TF_STRUCT_EXT, 50);
   double reclaimD1 = EMAReclaimStateTF(symbol, PERIOD_D1, MathMax(D1_EMA_Period,20));

   macroBias = Clamp(0.18*dirH1 + 0.12*sDirH1 + 0.20*dirH4 + 0.15*sDirH4 + 0.23*d1Bias + 0.12*Clamp(0.5*(reclaimH4+reclaimD1),-1.0,1.0), -1.0, 1.0);
   macroContinuation = Clamp(0.14*contH1 + 0.10*sContH1 + 0.20*contH4 + 0.14*sContH4 + 0.32*d1Continuation + 0.10*MathAbs(macroBias), 0.0, 1.0);
   macroMaturity = Clamp(0.14*overH1 + 0.08*MathAbs(reclaimH1) + 0.18*overH4 + 0.10*MathAbs(reclaimH4) + 0.36*d1Maturity + 0.14*MathAbs(reclaimD1), 0.0, 1.0);
   macroReclaim = Clamp(0.20*reclaimH1 + 0.35*reclaimH4 + 0.45*reclaimD1, -1.0, 1.0);
   macroTransition = Clamp(0.12*revH1 + 0.10*sRevH1 + 0.18*revH4 + 0.16*sRevH4 + 0.22*Clamp(MathAbs(macroReclaim),0.0,1.0) + 0.22*(1.0-macroContinuation), 0.0, 1.0);
   lateTrendTrap = Clamp(0.30*macroMaturity + 0.25*macroTransition + 0.20*Clamp(MathAbs(macroReclaim),0.0,1.0) + 0.15*(1.0-MathAbs(macroBias)) + 0.10*(1.0-macroContinuation), 0.0, 1.0);
}

double ComputeDirectionalMacroTrapRisk(const int action,
                                       const double macroBias,
                                       const double macroContinuation,
                                       const double macroMaturity,
                                       const double macroReclaim,
                                       const double macroTransition,
                                       const double lateTrendTrap)
{
   int dirSign=0;
   if(action==1) dirSign=1;
   else if(action==2) dirSign=-1;
   if(dirSign==0) return 0.0;

   double counterBias = MathMax(0.0, -dirSign*macroBias);
   double reclaimAgainst = MathMax(0.0, -dirSign*macroReclaim);
   double transitionAgainst = macroTransition * (0.55 + 0.45*counterBias);
   double lateMovePenalty = lateTrendTrap * (0.50 + 0.50*counterBias);

   return Clamp(0.35*counterBias +
                0.20*(macroContinuation*counterBias) +
                0.15*reclaimAgainst +
                0.15*transitionAgainst +
                0.15*lateMovePenalty, 0.0, 1.0);
}

void ComputeStructureFactorTF(const string symbol,
                              const ENUM_TIMEFRAMES tf,
                              const double price,
                              const double avgPrice,
                              const double lastEntryPrice,
                              double &dir,
                              double &strength,
                              double &continuation,
                              double &reversalQual,
                              double &breakFailureRisk,
                              double &counterTrapRisk)
{
   double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);
   double refPrice = GetCloseSafe(symbol, tf, 1);

   SwingPointMem swings[];
   int swingCount = 0;
   SwingPointMem lastHigh, prevHigh, lastLow, prevLow;
   ZeroMemory(lastHigh); ZeroMemory(prevHigh); ZeroMemory(lastLow); ZeroMemory(prevLow);
   bool ok = BuildSwingMemoryForTF(symbol, tf, swings, swingCount);
   bool hasRefs = false;
   if(ok)
      hasRefs = ExtractRecentSwingRefs(swings, swingCount, lastHigh, prevHigh, lastLow, prevLow);
   if(!hasRefs)
      FallbackStructureRefs(symbol, tf, lastHigh, prevHigh, lastLow, prevLow);

   double tendency = SwingTendencyFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr);
   double seqStrength = SwingSequenceStrengthFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr, tendency);
   double bullBOS   = BullBOSStrengthFromMemory(refPrice, lastHigh, atr, tendency);
   double bearBOS   = BearBOSStrengthFromMemory(refPrice, lastLow, atr, tendency);
   double bullCHoCH = BullCHoCHStrengthFromMemory(refPrice, lastHigh, atr, tendency);
   double bearCHoCH = BearCHoCHStrengthFromMemory(refPrice, lastLow, atr, tendency);
   double swingNearAvg = Clamp(MathAbs(avgPrice-refPrice)/MathMax(atr*3.0,1e-8),0.0,1.0);
   double swingNearEntry = Clamp(MathAbs(lastEntryPrice-refPrice)/MathMax(atr*3.0,1e-8),0.0,1.0);

   dir = Clamp(tendency, -1.0, 1.0);
   strength = Clamp(0.60*MathAbs(seqStrength) + 0.20*MathMax(bullBOS,bearBOS) + 0.20*(1.0-0.5*(swingNearAvg+swingNearEntry)), 0.0, 1.0);
   continuation = Clamp(0.45*MathAbs(seqStrength) + 0.35*MathMax(bullBOS,bearBOS) + 0.20*MathMax(0.0,dir*seqStrength), 0.0, 1.0);
   reversalQual = Clamp(0.50*MathMax(bullCHoCH,bearCHoCH) + 0.20*(1.0-MathMax(bullBOS,bearBOS)) + 0.15*(1.0-MathAbs(seqStrength)) + 0.15*(1.0-swingNearAvg), 0.0, 1.0);
   breakFailureRisk = Clamp(0.40*MathMax(bullCHoCH,bearCHoCH) + 0.30*MathMax(0.0,1.0-MathMax(bullBOS,bearBOS)) + 0.30*swingNearEntry, 0.0, 1.0);
   counterTrapRisk = Clamp(0.55*continuation + 0.25*(1.0-reversalQual) + 0.20*MathAbs(dir), 0.0, 1.0);
}

void ComputeZoneCandleFactorTF(const string symbol,
                               const ENUM_TIMEFRAMES tf,
                               const double avgPrice,
                               const double lastEntryPrice,
                               double &zoneRelevance,
                               double &zoneRespect,
                               double &zoneInvalidRisk,
                               double &entryTiming,
                               double &trapRisk)
{
   ZoneBranchFeaturePack zf;
   double closeRef = GetCloseSafe(symbol, tf, 1);
   ComputeZoneBranchFeatures(symbol, tf, closeRef, avgPrice, lastEntryPrice, zf);

   double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);
   double open1 = iOpen(symbol, tf, 1);
   double high1 = iHigh(symbol, tf, 1);
   double low1  = iLow(symbol, tf, 1);
   double close1= closeRef;
   double range = MathMax(high1 - low1, SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);
   double body  = MathAbs(close1 - open1);
   double upper = high1 - MathMax(open1, close1);
   double lower = MathMin(open1, close1) - low1;

   double closeLoc = Clamp((close1 - low1) / range, 0.0, 1.0);
   double rawRejectionDemand = Clamp((lower / range) * closeLoc, 0.0, 1.0);
   double rawRejectionSupply = Clamp((upper / range) * (1.0 - closeLoc), 0.0, 1.0);
   double rawRejection = (zf.zoneSideState >= 0.0 ? rawRejectionDemand : rawRejectionSupply);

   double rawAcceptanceDemand = Clamp((1.0 - closeLoc) * (body / range), 0.0, 1.0);
   double rawAcceptanceSupply = Clamp(closeLoc * (body / range), 0.0, 1.0);
   double rawAcceptance = (zf.zoneSideState >= 0.0 ? rawAcceptanceDemand : rawAcceptanceSupply);
   double indecision = Clamp(1.0 - body / range, 0.0, 1.0);
   double impulse = Clamp((body / atr), 0.0, 2.0) / 2.0;
   double activeDist = (zf.zoneSideState > 0.0 ? zf.nearestDemandDist : (zf.zoneSideState < 0.0 ? zf.nearestSupplyDist : 1.0));
   double zoneNear = 1.0 - Clamp(activeDist, 0.0, 1.0);

   zoneRelevance = Clamp(0.5 + 0.5*zf.zoneRelevance, 0.0, 1.0);
   zoneRespect = Clamp(0.45*rawRejection + 0.25*zoneNear + 0.15*(1.0-zf.zoneMitigation) + 0.15*MathAbs(zf.zoneRelevance), 0.0, 1.0);
   zoneInvalidRisk = Clamp(0.55*MathMax(0.0,-zf.zoneInvalidationState) + 0.20*rawAcceptance + 0.15*indecision + 0.10*(1.0-zoneNear), 0.0, 1.0);
   entryTiming = Clamp(0.35*zoneRespect + 0.25*impulse + 0.15*zoneNear + 0.15*(1.0-zoneInvalidRisk) + 0.10*(1.0-indecision), 0.0, 1.0);
   trapRisk = Clamp(0.35*rawAcceptance + 0.20*indecision + 0.20*zoneInvalidRisk + 0.15*(1.0-zoneRespect) + 0.10*(1.0-zoneNear), 0.0, 1.0);
}

void BuildModuleABlock(const string symbol,
                       const int symIdx,
                       const int positionsCount,
                       const int basketDirState,
                       const double avgPrice,
                       const double lastEntryPrice,
                       const double mid,
                       const double atrSlow,
                       double &out[])
{
   ArrayResize(out, ModuleAFeatureCount());
   int k=0;

   double addDepthNorm = (MaxTrades > 1 ? Clamp((double)MathMax(positionsCount - 1, 0) / (double)(MaxTrades - 1), 0.0, 1.0) : 0.0);
   double basketAgeBars = GetBasketAgeBarsState(symIdx, BaseTF);
   double basketAgeNorm = (BasketAgeNormBars > 0 ? Clamp(basketAgeBars / (double)BasketAgeNormBars, 0.0, 1.0) : 0.0);

   double distPriceToAvgAtr = 0.0;
   double distPriceToLastEntryAtr = 0.0;
   if(positionsCount > 0 && basketDirState != 0)
   {
      if(basketDirState > 0)
      {
         distPriceToAvgAtr = SafeDiv((avgPrice - mid), atrSlow, 0.0);
         distPriceToLastEntryAtr = SafeDiv((lastEntryPrice - mid), atrSlow, 0.0);
      }
      else
      {
         distPriceToAvgAtr = SafeDiv((mid - avgPrice), atrSlow, 0.0);
         distPriceToLastEntryAtr = SafeDiv((mid - lastEntryPrice), atrSlow, 0.0);
      }
   }
   distPriceToAvgAtr = Clamp(distPriceToAvgAtr, -StateDistAtrClamp, StateDistAtrClamp) / MathMax(StateDistAtrClamp, 1.0);
   distPriceToLastEntryAtr = Clamp(distPriceToLastEntryAtr, -StateDistAtrClamp, StateDistAtrClamp) / MathMax(StateDistAtrClamp, 1.0);

   double budgetBase = GetStateBudgetBase();
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double freeMarginBudgetRatio = Clamp(SafeDiv(freeMargin, budgetBase, 0.0), 0.0, StateBudgetRatioClamp) / MathMax(StateBudgetRatioClamp, 1.0);

   double worstGap = MathMax(MathAbs(distPriceToAvgAtr), MathAbs(distPriceToLastEntryAtr));
   double recoveryProgress = (positionsCount > 0 ? Clamp(1.0 - worstGap, 0.0, 1.0) : 0.0);
   double ddVelocity = Clamp(ComputeDrawdownWorseningScore(symIdx), 0.0, 1.0);
   double gridStep = GetGridStepCached(symIdx);
   double gridNorm = Clamp(SafeDiv(gridStep, MathMax(atrSlow,1e-8), 0.0), 0.0, 3.0) / 3.0;
   double addSpacingAdequacy = Clamp(0.60*gridNorm + 0.40*(1.0 - Clamp(MathAbs(distPriceToLastEntryAtr),0.0,1.0)), 0.0, 1.0);
   double rescueDependence = Clamp(0.45*addDepthNorm + 0.25*Clamp(MathAbs(distPriceToAvgAtr),0.0,1.0) + 0.15*ddVelocity + 0.15*(1.0-recoveryProgress), 0.0, 1.0);

   double recentQuality = ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate = ComputeRecentDeepBasketRate(symIdx);
   double ddCalm = ComputeRecentDrawdownCalmBonus(symIdx);

   DecisionSupportContext ds;
   if(symIdx>=0 && symIdx<MAX_SYMBOLS && gDecisionSupportCache[symIdx].valid)
      CopyDecisionSupportContext(gDecisionSupportCache[symIdx], ds);
   else
      ResetDecisionSupportContext(ds);

   double oneRoundPrior = Clamp((ds.valid ? ds.oneRoundPrior : recentQuality), 0.0, 1.0);
   double deepRiskPrior = Clamp((ds.valid ? ds.addRiskPrior : recentDeepRate), 0.0, 1.0);
   double supportConfidence = Clamp((ds.valid ? ds.supportConfidence : 0.0), 0.0, 1.0);
   double archiveCaution = Clamp((ds.valid ? ds.archiveCaution : recentDeepRate), 0.0, 1.0);
   double danger = Clamp(MathMax(gPDanger[symIdx], (ds.valid ? ds.dangerProbability : 0.0)), 0.0, 1.0);
   double replayRisk = Clamp(CurrentReplayRiskBias(symIdx), 0.0, 1.0);
   double painRecurrence = Clamp((ds.valid ? ds.painRecurrenceRisk : recentDeepRate), 0.0, 1.0);
   double macroTrapPrior = Clamp((ds.valid ? ds.macroReversalTrapPrior : 0.0), 0.0, 1.0);
   double macroMicroConflict = Clamp((ds.valid ? ds.macroMicroConflict : 0.0), 0.0, 1.0);
   double painConfidence = Clamp((ds.valid ? ds.painConfidence : 0.0), 0.0, 1.0);
   double lateTrendFadePenalty = Clamp((ds.valid ? ds.lateTrendFadePenalty : 0.0), 0.0, 1.0);
   double trendPersistenceProb = Clamp((ds.valid ? ds.trendPersistenceProb : (0.40*oneRoundPrior + 0.20*supportConfidence + 0.20*recentQuality + 0.20*(1.0-danger))), 0.0, 1.0);
   double trendReversalProb    = Clamp((ds.valid ? ds.trendReversalProb    : (0.35*(1.0-painRecurrence) + 0.25*supportConfidence + 0.20*ddCalm + 0.20*(1.0-deepRiskPrior))), 0.0, 1.0);
   double spikeRiskProb        = Clamp((ds.valid ? ds.spikeRiskProb        : (0.40*danger + 0.25*archiveCaution + 0.20*replayRisk + 0.15*macroMicroConflict)), 0.0, 1.0);
   double expectedBasketDepth  = Clamp((ds.valid ? ds.expectedBasketDepth  : (0.40*deepRiskPrior + 0.20*painRecurrence + 0.15*macroTrapPrior + 0.15*macroMicroConflict + 0.10*(1.0-oneRoundPrior))), 0.0, 1.0);
   double trendContinuationQuality = Clamp((ds.valid ? ds.trendContinuationQuality : (0.30*trendPersistenceProb + 0.20*oneRoundPrior + 0.20*(1.0-deepRiskPrior) + 0.15*supportConfidence + 0.15*(1.0-macroTrapPrior))),0.0,1.0);
   double breakoutReclaimQuality  = Clamp((ds.valid ? ds.breakoutReclaimQuality  : (0.25*trendReversalProb + 0.20*supportConfidence + 0.20*(1.0-macroMicroConflict) + 0.20*oneRoundPrior + 0.15*(1.0-danger))),0.0,1.0);
   double reversalTransitionQuality = Clamp((ds.valid ? ds.reversalTransitionQuality : (0.30*macroTrapPrior + 0.20*macroMicroConflict + 0.20*lateTrendFadePenalty + 0.15*trendReversalProb + 0.15*painRecurrence)),0.0,1.0);
   double modeDominanceScore = Clamp((ds.valid ? ds.modeDominanceScore : MathMax(trendContinuationQuality,MathMax(breakoutReclaimQuality,reversalTransitionQuality)) - MathMin(MathMax(trendContinuationQuality,breakoutReclaimQuality), MathMax(MathMin(trendContinuationQuality,breakoutReclaimQuality),reversalTransitionQuality))),0.0,1.0);
   double modeConflictScore  = Clamp((ds.valid ? ds.modeConflictScore  : (1.0-modeDominanceScore + 0.25*macroMicroConflict)),0.0,1.0);

   out[k++] = (double)basketDirState;
   out[k++] = addDepthNorm;
   out[k++] = basketAgeNorm;
   out[k++] = Clamp(MathAbs(distPriceToAvgAtr), 0.0, 1.0);
   out[k++] = Clamp(MathAbs(distPriceToLastEntryAtr), 0.0, 1.0);
   out[k++] = recoveryProgress;
   out[k++] = ddVelocity;
   out[k++] = addSpacingAdequacy;
   out[k++] = rescueDependence;
   out[k++] = freeMarginBudgetRatio;
   out[k++] = oneRoundPrior;
   out[k++] = deepRiskPrior;
   out[k++] = recentQuality;
   out[k++] = ddCalm;
   out[k++] = supportConfidence;
   out[k++] = Clamp(0.50*danger + 0.30*archiveCaution + 0.20*recentDeepRate, 0.0, 1.0);
   out[k++] = replayRisk;
   out[k++] = painRecurrence;
   out[k++] = macroTrapPrior;
   out[k++] = macroMicroConflict;
   out[k++] = painConfidence;
   out[k++] = lateTrendFadePenalty;
   out[k++] = trendPersistenceProb;
   out[k++] = trendReversalProb;
   out[k++] = spikeRiskProb;
   out[k++] = expectedBasketDepth;
   out[k++] = trendContinuationQuality;
   out[k++] = breakoutReclaimQuality;
   out[k++] = reversalTransitionQuality;
   out[k++] = modeDominanceScore;
   out[k++] = modeConflictScore;
}

void BuildModuleBBlock(const string symbol,const ENUM_TIMEFRAMES tfExec,const ENUM_TIMEFRAMES tfMid,const ENUM_TIMEFRAMES tfLong,
                       const double point,const double spreadNorm,const double spreadPts,const double hourNorm,const double dayNorm,
                       double &out[])
{
   ArrayResize(out, ModuleBFeatureCount());

   double dirE=0.0,strE=0.0,persE=0.0,accE=0.0,overE=0.0,mrE=0.0,contE=0.0,revE=0.0,volE=0.0;
   double dirM=0.0,strM=0.0,persM=0.0,accM=0.0,overM=0.0,mrM=0.0,contM=0.0,revM=0.0,volM=0.0;
   double dirL=0.0,strL=0.0,persL=0.0,accL=0.0,overL=0.0,mrL=0.0,contL=0.0,revL=0.0,volL=0.0;

   ComputeTrendFactorTF(symbol, tfExec, point, dirE,strE,persE,accE,overE,mrE,contE,revE,volE);
   ComputeTrendFactorTF(symbol, tfMid,  point, dirM,strM,persM,accM,overM,mrM,contM,revM,volM);
   ComputeTrendFactorTF(symbol, tfLong, point, dirL,strL,persL,accL,overL,mrL,contL,revL,volL);

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double setupConsensus = 1.0 - Clamp((MathAbs(dirE-dirM) + MathAbs(dirM-dirL) + MathAbs(dirE-dirL))/6.0, 0.0, 1.0);
   double bridgeAgreement = 1.0 - Clamp((MathAbs(dirM-dirL) + MathAbs(dirL-macroBias))/4.0, 0.0, 1.0);
   double consensus = Clamp(0.55*setupConsensus + 0.45*bridgeAgreement, 0.0, 1.0);
   double weightedDir = Clamp(0.28*dirE + 0.24*dirM + 0.18*dirL + 0.30*macroBias, -1.0, 1.0);
   double conflict = Clamp(0.60*(1.0-setupConsensus) + 0.40*(1.0-bridgeAgreement), 0.0, 1.0);
   double persistence = Clamp(0.35*persE + 0.25*persM + 0.15*persL + 0.25*macroContinuation, 0.0, 1.0);
   double accel = Clamp(0.45*accE + 0.25*accM + 0.10*accL + 0.20*macroBias, -1.0, 1.0);
   double maturity = Clamp(0.30*overE + 0.20*overM + 0.10*overL + 0.25*macroMaturity + 0.15*lateTrendTrap, 0.0, 1.0);
   double momentumStrength = Clamp(0.30*strE + 0.25*strM + 0.15*strL + 0.30*MathAbs(macroBias), 0.0, 1.0);
   double overextension = Clamp(0.40*overE + 0.20*overM + 0.10*overL + 0.20*macroMaturity + 0.10*MathAbs(macroReclaim), 0.0, 1.0);
   double meanRevPressure = Clamp(0.40*mrE + 0.25*mrM + 0.10*mrL + 0.15*macroTransition + 0.10*MathAbs(macroReclaim), 0.0, 1.0);
   double continuationPressure = Clamp(0.42*consensus*(0.50*contE + 0.30*contM + 0.20*contL) + 0.35*macroContinuation + 0.13*bridgeAgreement + 0.10*MathAbs(macroBias), 0.0, 1.0);
   double reversalConfirmation = Clamp(0.28*(0.45*revE + 0.35*revM + 0.20*revL) + 0.24*macroTransition + 0.18*MathAbs(macroReclaim) + 0.15*(1.0-macroContinuation) + 0.15*bridgeAgreement, 0.0, 1.0);
   double counterTrapRisk = Clamp(0.28*continuationPressure + 0.12*conflict + 0.10*maturity + 0.12*(1.0-reversalConfirmation) + 0.18*macroContinuation + 0.10*macroMaturity + 0.10*lateTrendTrap, 0.0, 1.0);
   double pullbackQuality = Clamp(0.30*meanRevPressure + 0.20*(1.0-counterTrapRisk) + 0.20*bridgeAgreement + 0.15*(1.0-maturity) + 0.15*MathAbs(macroReclaim), 0.0, 1.0);
   double directionalConviction = Clamp(MathAbs(weightedDir) * (0.55 + 0.25*consensus + 0.20*bridgeAgreement), 0.0, 1.0);
   double trendFreshness = Clamp(continuationPressure * (1.0 - MathMax(maturity,macroMaturity)), 0.0, 1.0);
   double volumeAlign = Clamp(0.35*volE + 0.30*volM + 0.15*volL + 0.20*MathAbs(macroBias), 0.0, 1.0);

   int k=0;
   out[k++] = dirE;
   out[k++] = dirM;
   out[k++] = dirL;
   out[k++] = consensus;
   out[k++] = conflict;
   out[k++] = persistence;
   out[k++] = accel;
   out[k++] = maturity;
   out[k++] = momentumStrength;
   out[k++] = overextension;
   out[k++] = meanRevPressure;
   out[k++] = continuationPressure;
   out[k++] = counterTrapRisk;
   out[k++] = reversalConfirmation;
   out[k++] = pullbackQuality;
   out[k++] = directionalConviction;
   out[k++] = trendFreshness;
   out[k++] = volumeAlign;
   out[k++] = macroBias;
   out[k++] = macroContinuation;
   out[k++] = macroReclaim;
   out[k++] = macroTransition;
}

void BuildModuleCBlock(const string symbol,const ENUM_TIMEFRAMES tfExec,const ENUM_TIMEFRAMES tfMid,const ENUM_TIMEFRAMES tfLong,double &out[])
{
   ArrayResize(out, ModuleCFeatureCount());
   int symIdx=SymbolIndex(symbol);
   double volExec = Clamp(0.55*RealizedVolLevelTF(symbol, tfExec, 8) + 0.45*RealizedVolLevelTF(symbol, tfExec, 24), 0.0, 1.0);
   double volMid  = Clamp(0.55*RealizedVolLevelTF(symbol, tfMid, 8)  + 0.45*RealizedVolLevelTF(symbol, tfMid, 24), 0.0, 1.0);
   double volLong = Clamp(0.60*RealizedVolLevelTF(symbol, tfLong, 8) + 0.40*RealizedVolLevelTF(symbol, tfLong, 24), 0.0, 1.0);

   double expExec = Clamp(0.45*RealizedVolDeltaTF(symbol, tfExec, 8) + 0.25*RealizedVolGammaTF(symbol, tfExec, 8) + 0.30*RangeExpansionTF(symbol, tfExec, 5, 20), 0.0, 1.0);
   double expMid  = Clamp(0.45*RealizedVolDeltaTF(symbol, tfMid, 8)  + 0.25*RealizedVolGammaTF(symbol, tfMid, 8)  + 0.30*RangeExpansionTF(symbol, tfMid, 5, 20), 0.0, 1.0);
   double expLong = Clamp(0.55*RealizedVolDeltaTF(symbol, tfLong, 8) + 0.20*RealizedVolDeltaTF(symbol, tfLong, 20) + 0.25*Clamp(StdReturnsTF(symbol, tfLong, 20, 1)*100.0,0.0,5.0)/5.0, 0.0, 1.0);

   double shock = Clamp(0.50*expExec + 0.25*expMid + 0.15*HighRangeBarStreakTF(symbol, tfExec, 5, 1.5) + 0.10*ComputeSpreadPressureScore(symbol,symIdx), 0.0, 1.0);
   double compression = Clamp(1.0 - (0.45*volExec + 0.35*expExec + 0.20*expMid), 0.0, 1.0);

   double dirE=0.0,strE=0.0,persE=0.0,accE=0.0,overE=0.0,mrE=0.0,contE=0.0,revE=0.0,volA=0.0;
   double dirM=0.0,strM=0.0,persM=0.0,accM=0.0,overM=0.0,mrM=0.0,contM=0.0,revM=0.0,volB=0.0;
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT); if(point<=0.0) point=0.00001;
   ComputeTrendFactorTF(symbol, tfExec, point, dirE,strE,persE,accE,overE,mrE,contE,revE,volA);
   ComputeTrendFactorTF(symbol, tfMid,  point, dirM,strM,persM,accM,overM,mrM,contM,revM,volB);

   double noiseToTrend = Clamp((0.45*WickInstabilityTF(symbol,tfExec,1) + 0.25*WickInstabilityTF(symbol,tfMid,1) + 0.30*volExec) / MathMax(0.20,0.60*MathAbs(dirE)+0.40*strE), 0.0, 1.0);
   double spreadPressure = ComputeSpreadPressureScore(symbol,symIdx);
   double executionFriction = Clamp(0.55*spreadPressure + 0.25*noiseToTrend + 0.20*shock, 0.0, 1.0);
   double regimeBreak = Clamp(0.40*shock + 0.20*MathMax(0.0, expExec-volExec) + 0.15*MathMax(0.0, expMid-volMid) + 0.15*WickInstabilityTF(symbol,tfExec,1) + 0.10*spreadPressure, 0.0, 1.0);

   double zSignedNorm = 0.0;
   double zAbsNorm = 0.0;
   double zPause = 0.0;
   double zQuietProg = 0.0;
   if(symIdx>=0 && symIdx<MAX_SYMBOLS && UseZScoreRiskGuard)
   {
      double denom = MathMax(ZScoreExtremeThreshold, 0.000001);
      zSignedNorm = Clamp(gZScoreLastValue[symIdx] / denom, -2.0, 2.0) / 2.0;
      zAbsNorm    = Clamp(gZScoreLastAbs[symIdx] / denom, 0.0, 2.0) / 2.0;
      zPause      = (gZScorePauseTrading[symIdx] ? 1.0 : 0.0);
      zQuietProg  = (ZScoreResumeQuietBars>0 ? Clamp((double)gZScoreQuietBars[symIdx] / (double)ZScoreResumeQuietBars, 0.0, 1.0) : 1.0);
   }

   int k=0;
   out[k++] = volExec;
   out[k++] = volMid;
   out[k++] = volLong;
   out[k++] = expExec;
   out[k++] = expMid;
   out[k++] = expLong;
   out[k++] = shock;
   out[k++] = compression;
   out[k++] = noiseToTrend;
   out[k++] = spreadPressure;
   out[k++] = executionFriction;
   out[k++] = regimeBreak;
   out[k++] = zSignedNorm;
   out[k++] = zAbsNorm;
   out[k++] = zPause;
   out[k++] = zQuietProg;
}

void BuildModuleDBlock(const string symbol,const ENUM_TIMEFRAMES tfExec,const ENUM_TIMEFRAMES tfMid,const ENUM_TIMEFRAMES tfLong,
                       const double price,const double avgPrice,const double lastEntryPrice,double &out[])
{
   ArrayResize(out, ModuleDFeatureCount());

   double dE=0.0,sE=0.0,cE=0.0,rE=0.0,bE=0.0,tE=0.0;
   double dM=0.0,sM=0.0,cM=0.0,rM=0.0,bM=0.0,tM=0.0;
   double dL=0.0,sL=0.0,cL=0.0,rL=0.0,bL=0.0,tL=0.0;

   ComputeStructureFactorTF(symbol, tfExec, price, avgPrice, lastEntryPrice, dE,sE,cE,rE,bE,tE);
   ComputeStructureFactorTF(symbol, tfMid,  price, avgPrice, lastEntryPrice, dM,sM,cM,rM,bM,tM);
   ComputeStructureFactorTF(symbol, tfLong, price, avgPrice, lastEntryPrice, dL,sL,cL,rL,bL,tL);

   int k=0;
   out[k++] = dE; out[k++] = sE; out[k++] = cE; out[k++] = rE; out[k++] = bE; out[k++] = tE;
   out[k++] = dM; out[k++] = sM; out[k++] = cM; out[k++] = rM; out[k++] = bM; out[k++] = tM;
   out[k++] = dL; out[k++] = sL; out[k++] = cL; out[k++] = rL; out[k++] = bL; out[k++] = tL;
}

void BuildModuleEBlock(const string symbol,const ENUM_TIMEFRAMES tfExec,const ENUM_TIMEFRAMES tfMid,const ENUM_TIMEFRAMES tfLong,
                       const double avgPrice,const double lastEntryPrice,double &out[])
{
   ArrayResize(out, ModuleEFeatureCount());

   double zrE=0.0,zsE=0.0,ziE=0.0,etE=0.0,trE=0.0;
   double zrM=0.0,zsM=0.0,ziM=0.0,etM=0.0,trM=0.0;
   double zrL=0.0,zsL=0.0,ziL=0.0,etL=0.0,trL=0.0;

   ComputeZoneCandleFactorTF(symbol, tfExec, avgPrice, lastEntryPrice, zrE,zsE,ziE,etE,trE);
   ComputeZoneCandleFactorTF(symbol, tfMid,  avgPrice, lastEntryPrice, zrM,zsM,ziM,etM,trM);
   ComputeZoneCandleFactorTF(symbol, tfLong, avgPrice, lastEntryPrice, zrL,zsL,ziL,etL,trL);

   int k=0;
   out[k++] = zrE; out[k++] = zsE; out[k++] = ziE; out[k++] = etE; out[k++] = trE;
   out[k++] = zrM; out[k++] = zsM; out[k++] = ziM; out[k++] = etM; out[k++] = trM;
   out[k++] = zrL; out[k++] = zsL; out[k++] = ziL; out[k++] = etL; out[k++] = trL;
}
void BuildModuleDFeaturesFastTF(const string symbol,
                                const ENUM_TIMEFRAMES tf,
                                const double price,
                                const double avgPrice,
                                const double lastEntryPrice,
                                const int offset,
                                double &out[])
{
   double atr = GetATRValueTF(symbol, tf, 14, 1);
   double refPrice = GetCloseSafe(symbol, tf, 1);

   SwingPointMem swings[];
   int swingCount = 0;
   SwingPointMem lastHigh, prevHigh, lastLow, prevLow;
   ZeroMemory(lastHigh); ZeroMemory(prevHigh); ZeroMemory(lastLow); ZeroMemory(prevLow);
   bool ok = BuildSwingMemoryForTF(symbol, tf, swings, swingCount);
   bool hasRefs = false;
   if(ok)
      hasRefs = ExtractRecentSwingRefs(swings, swingCount, lastHigh, prevHigh, lastLow, prevLow);

   if(!hasRefs)
      FallbackStructureRefs(symbol, tf, lastHigh, prevHigh, lastLow, prevLow);

   double tendency = SwingTendencyFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr);
   double seqStrength = SwingSequenceStrengthFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr, tendency);

   double confHigh = SwingLevelConfluenceBoost(symbol, tf, lastHigh.price, true, atr);
   double confLow  = SwingLevelConfluenceBoost(symbol, tf, lastLow.price, false, atr);
   double confMean = 0.5 * (confHigh + confLow);
   tendency = Clamp(tendency * (0.85 + 0.15 * confMean), -1.0, 1.0);
   seqStrength = Clamp(seqStrength * (0.70 + 0.30 * confMean), -1.0, 1.0);

   double bullBOS   = BullBOSStrengthFromMemory(refPrice, lastHigh, atr, tendency);
   double bearBOS   = BearBOSStrengthFromMemory(refPrice, lastLow, atr, tendency);
   double bullCHoCH = 0.0;
   double bearCHoCH = 0.0;
   if(offset == 0)
   {
      bullCHoCH = BullCHoCHStrengthFromMemory(refPrice, lastHigh, atr, tendency);
      bearCHoCH = BearCHoCHStrengthFromMemory(refPrice, lastLow, atr, tendency);
   }
   double strongWeakHigh = StrongWeakHighStateFromMemory(refPrice, lastHigh, atr, tendency, seqStrength);
   double strongWeakLow  = StrongWeakLowStateFromMemory(refPrice, lastLow, atr, tendency, seqStrength);

   out[offset + 0]  = StructNormDist(lastHigh.price - refPrice, atr);
   out[offset + 1]  = StructNormDist(refPrice - lastLow.price, atr);
   out[offset + 4]  = StructNormDist(lastHigh.price - avgPrice, atr);
   out[offset + 5]  = StructNormDist(avgPrice - lastLow.price, atr);
   out[offset + 13] = tendency;
   out[offset + 16] = bullBOS;
   out[offset + 17] = bearBOS;
   if(offset == 0)
   {
      out[offset + 18] = bullCHoCH;
      out[offset + 19] = bearCHoCH;
   }
   out[offset + 20] = strongWeakHigh;
   out[offset + 21] = strongWeakLow;
}
void BuildModuleEFeaturesFastTF(const string symbol,const ENUM_TIMEFRAMES tf,const double avgPrice,const double lastEntryPrice,const int offset,double &out[])
{
   ZoneBranchFeaturePack zf;
   double closeRef = GetCloseSafe(symbol, tf, 1);
   ComputeZoneBranchFeatures(symbol, tf, closeRef, avgPrice, lastEntryPrice, zf);

   double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);
   double open1 = iOpen(symbol, tf, 1);
   double high1 = iHigh(symbol, tf, 1);
   double low1  = iLow(symbol, tf, 1);
   double close1= closeRef;
   double range = MathMax(high1 - low1, SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);
   double body  = MathAbs(close1 - open1);
   double upper = high1 - MathMax(open1, close1);
   double lower = MathMin(open1, close1) - low1;

   double bodyAtr = Clamp(body / atr, 0.0, 5.0) / 5.0;
   double upperRatio = Clamp(upper / range, 0.0, 1.0);
   double lowerRatio = Clamp(lower / range, 0.0, 1.0);
   double closeLoc = Clamp((close1 - low1) / range, 0.0, 1.0);

   double activeDist = 1.0;
   if(zf.zoneSideState > 0.0) activeDist = zf.nearestDemandDist;
   else if(zf.zoneSideState < 0.0) activeDist = zf.nearestSupplyDist;
   double zoneNearness = 1.0 - Clamp(MathAbs(activeDist), 0.0, 1.0);
   double zoneDepthActive = 1.0 - MathAbs(zf.zoneDepthPosition);
   double swingNearness = SwingNearnessScore(symbol, tf, close1, atr);
   double structureContext = Clamp(0.50 * zoneNearness + 0.25 * zoneDepthActive + 0.15 * MathAbs(zf.zoneRelevance) + 0.10 * swingNearness, 0.0, 1.0);

   double rawRejectionDemand = Clamp((lower / range) * closeLoc, 0.0, 1.0);
   double rawRejectionSupply = Clamp((upper / range) * (1.0 - closeLoc), 0.0, 1.0);
   double rawRejection = (zf.zoneSideState >= 0.0 ? rawRejectionDemand : rawRejectionSupply);

   double rawAcceptanceDemand = Clamp((1.0 - closeLoc) * (body / range), 0.0, 1.0);
   double rawAcceptanceSupply = Clamp(closeLoc * (body / range), 0.0, 1.0);
   double rawAcceptance = (zf.zoneSideState >= 0.0 ? rawAcceptanceDemand : rawAcceptanceSupply);

   double rejection = Clamp(rawRejection * (0.55 + 0.45 * structureContext), 0.0, 1.0);
   double acceptance = Clamp(0.35 * rawAcceptance + 0.65 * rawAcceptance * structureContext, 0.0, 1.0);
   double indecisionBase = Clamp(1.0 - body / range, 0.0, 1.0);
   double indecision = Clamp(0.65 * indecisionBase + 0.35 * indecisionBase * structureContext, 0.0, 1.0);
   double impulseRaw = Clamp((body / atr) * ((close1 >= open1) ? 1.0 : -1.0), -3.0, 3.0) / 3.0;
   double impulse = Clamp(impulseRaw * (0.70 + 0.30 * structureContext), -1.0, 1.0);

   out[offset + 0] = zf.nearestDemandDist;
   out[offset + 1] = zf.nearestSupplyDist;
   out[offset + 7] = zf.zoneRelevance;
   out[offset + 8] = zf.zoneRetestBreakState;
   out[offset + 13] = rejection;
   out[offset + 14] = acceptance;
   out[offset + 15] = indecision;
   out[offset + 16] = impulse;

   if(offset == 17)
      out[offset + 15] = 0.0;
   if(offset == 34)
   {
      out[offset + 8] = 0.0;
      out[offset + 15] = 0.0;
      out[offset + 16] = 0.0;
   }
}
void BuildState(const string symbol,const int symIdx,const int positionsCount,CArrayDouble &trades,double &state[])
{
   int rawDim = 0;
   if(UseStateV2SelfAwareness) rawDim += ModuleAFeatureCount();
   if(UseIndicatorBranch)  rawDim += ModuleBFeatureCount();
   if(UseVolatilityBranch) rawDim += ModuleCFeatureCount();
   if(UseStructureBranch)  rawDim += ModuleDFeatureCount();
   if(UseZoneCandleBranch) rawDim += ModuleEFeatureCount();

   ArrayResize(state, rawDim);
   int k = 0;

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   double rsiB = gRSI_Base[symIdx];
   double cciB = gCCI_Base[symIdx];
   double macdB= gMACD_BaseMain[symIdx];
   double emaB = gEMA_BaseVal[symIdx];
   double rviB = gRVI_BaseMain[symIdx];

   double normRSI = Clamp(rsiB/100.0,0.0,1.0);
   double normCCI = Clamp((cciB+500.0)/1000.0,0.0,1.0);
   double normPos = (MaxTrades>0 ? Clamp((double)positionsCount/(double)MaxTrades,0.0,1.0) : 0.0);

   double avg=mid;
   if(positionsCount>0 && trades.Total()>0)
      avg=BasketAvgPrice(trades,gTradeLots[symIdx],mid);

   double priceDiff = SafeDiv((mid-avg),(100.0*point),0.0);

   double atrRatioBase = GetAtrRatioCached_Base(symbol);

   double eq = GetEAEquity();
   double bal= gEAStartEquity;
   double eqBal = (bal>1e-9 ? eq/bal : 1.0);

   double dd=0.0;
   if(maxEquity>1e-9 && eq<maxEquity)
      dd=Clamp((maxEquity-eq)/maxEquity,0.0,1.0);

   double lastClose=iClose(symbol,BaseTF,1);
   double zoneType=0.5, zoneLoc=0.5, zoneStatus=0.5;
   GetZoneFeatures(symbol,lastClose,zoneType,zoneLoc,zoneStatus);

   double d1Dist=0.0,d1Slope=0.0,d1Side=0.0;
   GetD1TrendFeatures(symbol,d1Dist,d1Slope,d1Side);

   UpdateMultiChannelGridStats(symIdx);

   double step = GetGridStepCached(symIdx);

   double atrSlow = gATRslow_BaseVal[symIdx];
   if(atrSlow<=1e-12) atrSlow=1e-12;

   double stepNorm          = Clamp(step/atrSlow,0.0,5.0)/5.0;
   double mediumStepNorm    = stepNorm;
   double extremeStepNorm   = stepNorm;
   double activeChannelNorm = 0.0;

   double spreadPts = SafeDiv((ask-bid),point,0.0);
   double spreadNorm= Clamp(spreadPts/10.0,0.0,10.0)/10.0;

   MqlDateTime tm;
   TimeToStruct(TimeCurrent(),tm);
   double hourNorm = (double)tm.hour/23.0;
   double dayNorm  = (double)tm.day_of_week/6.0;

   double emaDirBase = Clamp((lastClose-emaB)/atrSlow,-5.0,5.0);
   double macdScaled = SafeDiv(macdB,(100.0*point),0.0);
   double rviScaled  = Clamp(rviB,-2.0,2.0)/2.0;

   int basketDirState = GetBasketDirStateFast(symbol, symIdx, gMagics[symIdx]);
   double basketState = (double)basketDirState;
   double addDepthNorm = 0.0;
   if(MaxTrades > 1)
      addDepthNorm = Clamp((double)MathMax(positionsCount - 1, 0) / (double)(MaxTrades - 1), 0.0, 1.0);

   double basketAgeBars = GetBasketAgeBarsState(symIdx, BaseTF);
   double basketAgeNorm = (BasketAgeNormBars > 0 ? Clamp(basketAgeBars / (double)BasketAgeNormBars, 0.0, 1.0) : 0.0);

   double distPriceToAvgAtr = 0.0;
   if(positionsCount > 0 && basketDirState != 0)
   {
      if(basketDirState > 0) distPriceToAvgAtr = SafeDiv((avg - mid), atrSlow, 0.0);
      else                   distPriceToAvgAtr = SafeDiv((mid - avg), atrSlow, 0.0);
      distPriceToAvgAtr = Clamp(distPriceToAvgAtr, -StateDistAtrClamp, StateDistAtrClamp) / MathMax(StateDistAtrClamp, 1.0);
   }

   double lastEntryPrice = GetLastEntryPriceState(symbol, gMagics[symIdx], basketDirState, avg);
   double distPriceToLastEntryAtr = 0.0;
   if(positionsCount > 0 && basketDirState != 0)
   {
      if(basketDirState > 0) distPriceToLastEntryAtr = SafeDiv((lastEntryPrice - mid), atrSlow, 0.0);
      else                   distPriceToLastEntryAtr = SafeDiv((mid - lastEntryPrice), atrSlow, 0.0);
      distPriceToLastEntryAtr = Clamp(distPriceToLastEntryAtr, -StateDistAtrClamp, StateDistAtrClamp) / MathMax(StateDistAtrClamp, 1.0);
   }

   double budgetBase = GetStateBudgetBase();
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double freeMarginBudgetRatio = Clamp(SafeDiv(freeMargin, budgetBase, 0.0), 0.0, StateBudgetRatioClamp) / MathMax(StateBudgetRatioClamp, 1.0);


if(UseStateV2SelfAwareness)
{
   double featsA[];
   BuildModuleABlock(symbol, symIdx, positionsCount, basketDirState, avg, lastEntryPrice, mid, atrSlow, featsA);
   for(int i=0;i<ArraySize(featsA);i++) state[k++] = featsA[i];
}

   // ===== Module B/C slow-cache blocks =====
   ENUM_TIMEFRAMES tfExec = (TF_EXEC==PERIOD_CURRENT ? BaseTF : TF_EXEC);
   ENUM_TIMEFRAMES tfMid  = TF_MID;
   ENUM_TIMEFRAMES tfLong = TF_LONG;

   datetime execBar0 = iTime(symbol, tfExec, 0);
   datetime midBar0  = iTime(symbol, tfMid, 0);
   datetime longBar0 = iTime(symbol, tfLong, 0);
   long avgQSlow = QuantizePriceByPoint(avg, point);
   long lastEntryQSlow = QuantizePriceByPoint(lastEntryPrice, point);

   if(UseIndicatorBranch)
   {
      double featsB[];
      {
         bool hit = (UseSlowFeatureCaches && CacheMatchBasic(gIndicatorCache[symIdx], symbol, execBar0, midBar0, longBar0, 0, 0, false));
         if(hit) CloneDoubleArray(gIndicatorCache[symIdx].feats, featsB);
         else
         {
            BuildModuleBBlock(symbol, tfExec, tfMid, tfLong, point, spreadNorm, spreadPts, hourNorm, dayNorm, featsB);
            if(UseSlowFeatureCaches && symIdx>=0 && symIdx<MAX_SYMBOLS)
            {
               gIndicatorCache[symIdx].valid=true;
               gIndicatorCache[symIdx].symbol=symbol;
               gIndicatorCache[symIdx].execBar=execBar0;
               gIndicatorCache[symIdx].midBar=midBar0;
               gIndicatorCache[symIdx].longBar=longBar0;
               CloneDoubleArray(featsB, gIndicatorCache[symIdx].feats);
            }
         }
      }
      for(int i=0;i<ArraySize(featsB);i++) state[k++] = featsB[i];
   }

   if(UseVolatilityBranch)
   {
      double featsC[];
      {
         bool hit = (UseSlowFeatureCaches && CacheMatchBasic(gVolatilityCache[symIdx], symbol, execBar0, midBar0, longBar0, 0, 0, false));
         if(hit) CloneDoubleArray(gVolatilityCache[symIdx].feats, featsC);
         else
         {
            BuildModuleCBlock(symbol, tfExec, tfMid, tfLong, featsC);
            if(UseSlowFeatureCaches && symIdx>=0 && symIdx<MAX_SYMBOLS)
            {
               gVolatilityCache[symIdx].valid=true;
               gVolatilityCache[symIdx].symbol=symbol;
               gVolatilityCache[symIdx].execBar=execBar0;
               gVolatilityCache[symIdx].midBar=midBar0;
               gVolatilityCache[symIdx].longBar=longBar0;
               CloneDoubleArray(featsC, gVolatilityCache[symIdx].feats);
            }
         }
      }
      for(int i=0;i<ArraySize(featsC);i++) state[k++] = featsC[i];
   }

   if(UseStructureBranch)
   {
      double featsD[];
      {
         bool hit = (UseSlowFeatureCaches && CacheStructureZoneOnExecBar &&
                     CacheMatchBasic(gStructureCache[symIdx], symbol, execBar0, midBar0, longBar0, avgQSlow, lastEntryQSlow, true));
         if(hit) CloneDoubleArray(gStructureCache[symIdx].feats, featsD);
         else
         {
            BuildModuleDBlock(symbol, tfExec, tfMid, tfLong, mid, avg, lastEntryPrice, featsD);
            if(UseSlowFeatureCaches && CacheStructureZoneOnExecBar && symIdx>=0 && symIdx<MAX_SYMBOLS)
            {
               gStructureCache[symIdx].valid=true;
               gStructureCache[symIdx].symbol=symbol;
               gStructureCache[symIdx].execBar=execBar0;
               gStructureCache[symIdx].midBar=midBar0;
               gStructureCache[symIdx].longBar=longBar0;
               gStructureCache[symIdx].avgQ=avgQSlow;
               gStructureCache[symIdx].lastEntryQ=lastEntryQSlow;
               CloneDoubleArray(featsD, gStructureCache[symIdx].feats);
            }
         }
      }
      for(int i=0;i<ArraySize(featsD);i++) state[k++] = featsD[i];
   }

   if(UseZoneCandleBranch)
   {
      double featsE[];
      {
         bool hit = (UseSlowFeatureCaches && CacheStructureZoneOnExecBar &&
                     CacheMatchBasic(gZoneCandleCache[symIdx], symbol, execBar0, midBar0, longBar0, avgQSlow, lastEntryQSlow, true));
         if(hit) CloneDoubleArray(gZoneCandleCache[symIdx].feats, featsE);
         else
         {
            BuildModuleEBlock(symbol, tfExec, tfMid, tfLong, avg, lastEntryPrice, featsE);
            if(UseSlowFeatureCaches && CacheStructureZoneOnExecBar && symIdx>=0 && symIdx<MAX_SYMBOLS)
            {
               gZoneCandleCache[symIdx].valid=true;
               gZoneCandleCache[symIdx].symbol=symbol;
               gZoneCandleCache[symIdx].execBar=execBar0;
               gZoneCandleCache[symIdx].midBar=midBar0;
               gZoneCandleCache[symIdx].longBar=longBar0;
               gZoneCandleCache[symIdx].avgQ=avgQSlow;
               gZoneCandleCache[symIdx].lastEntryQ=lastEntryQSlow;
               CloneDoubleArray(featsE, gZoneCandleCache[symIdx].feats);
            }
         }
      }
      for(int i=0;i<ArraySize(featsE);i++) state[k++] = featsE[i];
   }

   for(int i=k;i<rawDim;i++)
      state[i]=0.0;

   if(StateNormalizationL2)
   {
      InitBranchLayoutForStateDim(rawDim);
      NormalizeVecSlice(state, gBranchLayout.basketStart,     gBranchLayout.basketCount);
      NormalizeVecSlice(state, gBranchLayout.indicatorStart,  gBranchLayout.indicatorCount);
      NormalizeVecSlice(state, gBranchLayout.volatilityStart, gBranchLayout.volatilityCount);
      NormalizeVecSlice(state, gBranchLayout.structureStart,  gBranchLayout.structureCount);
      NormalizeVecSlice(state, gBranchLayout.zoneCandleStart, gBranchLayout.zoneCandleCount);
   }
}

bool IsExtremeState(const int symIdx,const int positionsCount)
{
   if(IsPersistentExtremeDD(symIdx))
      return true;

   return false;
}

double ScaleRewardByRegime(double reward, bool isExtreme)
{
   if(isExtreme) return reward * ExtremeRewardBoost;
   if(reward < 0.0) return reward * MildRewardScale;
   return reward;
}

bool BuildMiniStateFast(const int symIdx, double &mini[])
{
   ArrayResize(mini, MINI_DIM);

   string sym=gSymbols[symIdx];
   double point=SymbolInfoDouble(sym,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double bid=SymbolInfoDouble(sym,SYMBOL_BID);
   double ask=SymbolInfoDouble(sym,SYMBOL_ASK);

   double normPos = (MaxTrades>0 ? Clamp((double)gPositionsCount[symIdx]/(double)MaxTrades,0.0,1.0) : 0.0);
   double atrRatio = GetAtrRatioCached_Base(sym);
   double eq=GetEAEquity();
   double dd=0.0;
   if(maxEquity>1e-9 && eq<maxEquity) dd=Clamp((maxEquity-eq)/maxEquity,0.0,1.0);

   double normCCI = Clamp((gCCI_Base[symIdx]+500.0)/1000.0,0.0,1.0);

   double d1Dist=0.0,d1Slope=0.0,d1Side=0.0;
   GetD1TrendFeatures(sym,d1Dist,d1Slope,d1Side);

   double step=GetGridStepCached(symIdx);
   double atrSlow=gATRslow_BaseVal[symIdx];
   if(atrSlow<=1e-12) atrSlow=1e-12;
   double stepNorm = Clamp(step/atrSlow,0.0,5.0)/5.0;

   double spreadPts = SafeDiv((ask-bid),point,0.0);
   double spreadNorm= Clamp(spreadPts/10.0,0.0,10.0)/10.0;

   mini[0]=normPos;
   mini[1]=atrRatio;
   mini[2]=dd;
   mini[3]=normCCI;
   mini[4]=d1Dist;
   mini[5]=d1Slope;
   mini[6]=stepNorm;
   mini[7]=spreadNorm;

   NormalizeVecN(mini);
   return true;
}

void UpdateMiniStateCache(const int symIdx)
{
   if(!UseMiniStateFingerprint){ gMiniValid[symIdx]=false; return; }

   string sym=gSymbols[symIdx];
   datetime bt=iTime(sym, BaseTF, 0);
   if(bt==0) return;
   if(bt==gMiniLastBar[symIdx] && gMiniValid[symIdx]) return;

   double mini[];
   if(!BuildMiniStateFast(symIdx, mini)){ gMiniValid[symIdx]=false; return; }

   if(gMiniValid[symIdx])
   {
      for(int k=0;k<MINI_DIM;k++) gMiniPrev[symIdx][k]=gMiniCache[symIdx][k];
      gMiniPrevValid[symIdx]=true;
   }

   for(int k=0;k<MINI_DIM;k++) gMiniCache[symIdx][k]=mini[k];
   gMiniValid[symIdx]=true;
   gMiniLastBar[symIdx]=bt;
}

bool BuildQMemoryKey(const int symIdx, const double &state[], double &key[])
{
   ArrayResize(key,0);

   int sz=ArraySize(state);
   if(sz<=0) return false;

   if(UseBranchScaffold)
      InitBranchLayoutForStateDim(sz);

   if(gBranchLayout.basketCount > 0)
   {
      int s=gBranchLayout.basketStart;
      if(gBranchLayout.basketCount>5)  Push(key, state[s+5]);   // recovery progress
      if(gBranchLayout.basketCount>10) Push(key, state[s+10]);  // one-round prior
      if(gBranchLayout.basketCount>11) Push(key, state[s+11]);  // deep-risk prior
      if(gBranchLayout.basketCount>12) Push(key, state[s+12]);  // recent quality
      if(gBranchLayout.basketCount>14) Push(key, state[s+14]);  // support confidence
      if(gBranchLayout.basketCount>15) Push(key, state[s+15]);  // danger / caution blend
      if(gBranchLayout.basketCount>17) Push(key, state[s+17]);  // pain recurrence
      if(gBranchLayout.basketCount>18) Push(key, state[s+18]);  // macro trap prior
      if(gBranchLayout.basketCount>20) Push(key, state[s+20]);  // pain confidence
      if(gBranchLayout.basketCount>22) Push(key, state[s+22]);  // persistence probability
      if(gBranchLayout.basketCount>23) Push(key, state[s+23]);  // reversal probability
      if(gBranchLayout.basketCount>24) Push(key, state[s+24]);  // spike risk probability
      if(gBranchLayout.basketCount>25) Push(key, state[s+25]);  // expected basket depth
   }

   if(gBranchLayout.indicatorCount > 0)
   {
      int s=gBranchLayout.indicatorStart;
      if(gBranchLayout.indicatorCount>3)  Push(key, state[s+3]);   // trend consensus
      if(gBranchLayout.indicatorCount>11) Push(key, state[s+11]);  // continuation pressure
      if(gBranchLayout.indicatorCount>12) Push(key, state[s+12]);  // counter-trend trap risk
      if(gBranchLayout.indicatorCount>13) Push(key, state[s+13]);  // reversal confirmation
      if(gBranchLayout.indicatorCount>15) Push(key, state[s+15]);  // directional conviction
      if(gBranchLayout.indicatorCount>18) Push(key, state[s+18]);  // macro bias
      if(gBranchLayout.indicatorCount>21) Push(key, state[s+21]);  // macro transition
   }

   if(gBranchLayout.volatilityCount > 0)
   {
      int s=gBranchLayout.volatilityStart;
      if(gBranchLayout.volatilityCount>6)  Push(key, state[s+6]);   // shock
      if(gBranchLayout.volatilityCount>9)  Push(key, state[s+9]);   // spread pressure
      if(gBranchLayout.volatilityCount>11) Push(key, state[s+11]);  // regime break
   }

   if(gBranchLayout.structureCount > 0)
   {
      int s=gBranchLayout.structureStart;
      if(gBranchLayout.structureCount>2)  Push(key, state[s+2]);    // exec continuation
      if(gBranchLayout.structureCount>3)  Push(key, state[s+3]);    // exec reversal
      if(gBranchLayout.structureCount>8)  Push(key, state[s+8]);    // mid continuation
      if(gBranchLayout.structureCount>14) Push(key, state[s+14]);   // long continuation
   }

   if(gBranchLayout.zoneCandleCount > 0)
   {
      int s=gBranchLayout.zoneCandleStart;
      if(gBranchLayout.zoneCandleCount>3)  Push(key, state[s+3]);   // exec entry timing
      if(gBranchLayout.zoneCandleCount>4)  Push(key, state[s+4]);   // exec trap risk
      if(gBranchLayout.zoneCandleCount>8)  Push(key, state[s+8]);   // mid entry timing
   }

   if(UseDangerBrain)
      Push(key, Clamp(gPDanger[symIdx],0.0,1.0));

   if(ArraySize(key)<=0) return false;

   NormalizeVecN(key);
   return true;
}


double StateBranchRelValue(const double &state[],
                           const int branchStart,
                           const int branchCount,
                           const int relIndex,
                           const double fallback=0.0)
{
   if(relIndex < 0 || branchStart < 0 || branchCount <= 0 || relIndex >= branchCount)
      return fallback;
   int idx=branchStart + relIndex;
   if(idx < 0 || idx >= ArraySize(state))
      return fallback;
   return state[idx];
}

double ComputeOptionBTrendPersistenceProb(const int symIdx,
                                          const double &state[],
                                          const double &qBase[],
                                          const DecisionSupportContext &ctx)
{
   if(ArraySize(state) <= 0) return 0.5;
   InitBranchLayoutForStateDim(ArraySize(state));

   double consensus      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,3,0.5);
   double persistence    = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,5,0.5);
   double continuation   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,11,0.5);
   double reversal       = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,13,0.5);
   double directionConv  = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,15,0.5);
   double volumeAlign    = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,17,0.5);
   double macroCont      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,19,0.5);
   double macroTransition= StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,21,0.5);

   double noiseToTrend   = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,8,0.5);
   double spreadPressure = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,9,0.5);
   double regimeBreak    = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);

   double structContE    = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,2,0.5);
   double structContM    = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,8,0.5);
   double entryTimingE   = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,3,0.5);

   double recentQuality  = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,12,0.5);
   double dangerBlend    = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,15,0.5);
   double qGap=Clamp(SafeDiv(ComputeDDQNActionGap(qBase,false),0.25,0.0),0.0,1.0);

   double logit=
      -0.55
      + 1.00*consensus
      + 0.88*continuation
      + 0.72*persistence
      + 0.56*macroCont
      + 0.28*directionConv
      + 0.18*structContE
      + 0.12*structContM
      + 0.10*entryTimingE
      + 0.12*volumeAlign
      + 0.10*qGap
      + 0.08*recentQuality
      - 0.70*reversal
      - 0.48*regimeBreak
      - 0.32*noiseToTrend
      - 0.20*spreadPressure
      - 0.14*macroTransition
      - 0.10*dangerBlend;

   return Clamp(Sigmoid(logit),0.0,1.0);
}

double ComputeOptionBTrendReversalProb(const int symIdx,
                                       const double &state[],
                                       const double &qBase[],
                                       const DecisionSupportContext &ctx)
{
   if(ArraySize(state) <= 0) return 0.5;
   InitBranchLayoutForStateDim(ArraySize(state));

   double continuation   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,11,0.5);
   double reversal       = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,13,0.5);
   double maturity       = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,7,0.5);
   double macroReclaim   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,20,0.0);
   double macroTransition= StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,21,0.5);

   double regimeBreak    = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);

   double structRevE     = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,3,0.5);
   double structRevM     = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,9,0.5);
   double structBreakE   = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,4,0.5);

   double zoneRespectE   = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,0,0.5);
   double zoneEntryE     = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,3,0.5);
   double zoneTrapE      = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,4,0.5);

   double macroMicroConflict = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,19,0.5);
   double lateTrendFade      = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,21,0.5);

   double qGap=Clamp(SafeDiv(ComputeDDQNActionGap(qBase,false),0.25,0.0),0.0,1.0);

   double logit=
      -0.75
      + 0.96*reversal
      + 0.82*macroTransition
      + 0.42*MathAbs(macroReclaim)
      + 0.32*maturity
      + 0.22*structRevE
      + 0.18*structRevM
      + 0.12*(1.0-structBreakE)
      + 0.10*zoneRespectE
      + 0.08*zoneEntryE
      + 0.10*macroMicroConflict
      + 0.08*lateTrendFade
      + 0.06*qGap
      - 0.70*continuation
      - 0.42*regimeBreak
      - 0.12*zoneTrapE;

   return Clamp(Sigmoid(logit),0.0,1.0);
}

double ComputeOptionBSpikeRiskProb(const int symIdx,
                                   const double &state[],
                                   const double &qBase[],
                                   const DecisionSupportContext &ctx)
{
   if(ArraySize(state) <= 0) return 0.5;
   InitBranchLayoutForStateDim(ArraySize(state));

   double shock          = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,6,0.5);
   double noiseToTrend   = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,8,0.5);
   double spreadPressure = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,9,0.5);
   double execFriction   = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,10,0.5);
   double regimeBreak    = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);

   double accel          = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,6,0.0);
   double volumeAlign    = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,17,0.5);

   double macroMicroConflict = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,19,0.5);

   double logit=
      -1.10
      + 1.05*shock
      + 0.92*regimeBreak
      + 0.54*spreadPressure
      + 0.50*execFriction
      + 0.44*noiseToTrend
      + 0.26*MathAbs(accel)
      + 0.20*macroMicroConflict
      + 0.18*ctx.ddEventRisk
      + 0.10*volumeAlign;

   return Clamp(Sigmoid(logit),0.0,1.0);
}

double ComputeOptionBExpectedBasketDepth(const int symIdx,
                                         const double &state[],
                                         const double &qBase[],
                                         const DecisionSupportContext &ctx,
                                         const double persistenceProb,
                                         const double reversalProb,
                                         const double spikeRiskProb)
{
   if(ArraySize(state) <= 0) return 0.5;
   InitBranchLayoutForStateDim(ArraySize(state));

   double recentQuality     = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,12,0.5);
   double ddCalm            = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,13,0.5);
   double dangerBlend       = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,15,0.5);
   double painRecurrence    = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,17,0.5);
   double macroTrapPrior    = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,18,0.5);
   double macroMicroConflict= StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,19,0.5);
   double lateTrendFade     = StateBranchRelValue(state,gBranchLayout.basketStart,gBranchLayout.basketCount,21,0.5);

   double trapRisk          = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,12,0.5);
   double continuation      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,11,0.5);

   double regimeBreak       = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);

   double oneRoundPrior=Clamp(ctx.oneRoundPrior,0.0,1.0);
   double addRiskPrior =Clamp(ctx.addRiskPrior,0.0,1.0);
   double deepPain     =Clamp(ctx.deepBasketPainPrior,0.0,1.0);

   double logit=
      -1.35
      + 1.10*addRiskPrior
      + 0.78*deepPain
      + 0.70*painRecurrence
      + 0.56*trapRisk
      + 0.44*spikeRiskProb
      + 0.38*regimeBreak
      + 0.32*macroTrapPrior
      + 0.28*macroMicroConflict
      + 0.22*continuation
      + 0.18*lateTrendFade
      + 0.16*MathMax(0.0,persistenceProb-reversalProb)
      + 0.12*dangerBlend
      - 0.62*oneRoundPrior
      - 0.38*recentQuality
      - 0.26*ddCalm;

   return Clamp(Sigmoid(logit),0.0,1.0);
}

void ComputeStrategyModeScores(const int symIdx,
                               const double &state[],
                               const DecisionSupportContext &ctx,
                               double &trendContinuationQuality,
                               double &breakoutReclaimQuality,
                               double &reversalTransitionQuality,
                               double &modeDominanceScore,
                               double &modeConflictScore)
{
   trendContinuationQuality=0.5;
   breakoutReclaimQuality=0.5;
   reversalTransitionQuality=0.5;
   modeDominanceScore=0.0;
   modeConflictScore=0.5;

   if(ArraySize(state) <= 0) return;
   InitBranchLayoutForStateDim(ArraySize(state));

   double persistence       = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,5,0.5);
   double maturity          = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,7,0.5);
   double continuation      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,11,0.5);
   double trapRisk          = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,12,0.5);
   double reversalConfirm   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,13,0.5);
   double pullbackQuality   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,14,0.5);
   double directionConv     = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,15,0.5);
   double macroBias         = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,18,0.0);
   double macroContinuation = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,19,0.5);
   double macroReclaim      = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,20,0.0);
   double macroTransition   = StateBranchRelValue(state,gBranchLayout.indicatorStart,gBranchLayout.indicatorCount,21,0.5);

   double regimeBreak       = StateBranchRelValue(state,gBranchLayout.volatilityStart,gBranchLayout.volatilityCount,11,0.5);
   double spikeRisk         = Clamp(ctx.spikeRiskProb,0.0,1.0);
   double expectedDepth     = Clamp(ctx.expectedBasketDepth,0.0,1.0);

   double structContM       = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,8,0.5);
   double structContL       = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,14,0.5);
   double structRevM        = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,9,0.5);
   double structRevL        = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,15,0.5);
   double breakFailM        = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,10,0.5);
   double breakFailL        = StateBranchRelValue(state,gBranchLayout.structureStart,gBranchLayout.structureCount,16,0.5);

   double zoneRespectM      = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,5,0.5);
   double zoneEntryM        = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,8,0.5);
   double zoneTrapM         = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,9,0.5);
   double zoneRespectL      = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,10,0.5);
   double zoneEntryL        = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,13,0.5);
   double zoneTrapL         = StateBranchRelValue(state,gBranchLayout.zoneCandleStart,gBranchLayout.zoneCandleCount,14,0.5);

   double oneRoundPrior     = Clamp(ctx.oneRoundPrior,0.0,1.0);
   double supportConf       = Clamp(ctx.supportConfidence,0.0,1.0);
   double macroConflict     = Clamp(ctx.macroMicroConflict,0.0,1.0);
   double painRecurrence    = Clamp(ctx.painRecurrenceRisk,0.0,1.0);
   double lateTrendFade     = Clamp(ctx.lateTrendFadePenalty,0.0,1.0);

   trendContinuationQuality =
      Clamp(0.16*continuation +
            0.12*pullbackQuality +
            0.12*persistence +
            0.12*macroContinuation +
            0.08*MathAbs(macroBias) +
            0.08*directionConv +
            0.08*MathMax(structContM,structContL) +
            0.06*(0.5*zoneEntryM + 0.5*zoneEntryL) +
            0.05*(0.5*zoneRespectM + 0.5*zoneRespectL) +
            0.06*(1.0-trapRisk) +
            0.04*(1.0-regimeBreak) +
            0.03*(1.0-spikeRisk) +
            0.04*(1.0-expectedDepth) +
            0.04*oneRoundPrior, 0.0, 1.0);

   double continuationRestart =
      Clamp(0.34*MathAbs(macroReclaim) +
            0.22*macroContinuation +
            0.12*continuation +
            0.10*(0.5*zoneRespectM + 0.5*zoneRespectL) +
            0.10*(0.5*zoneEntryM + 0.5*zoneEntryL) +
            0.06*MathMax(structContM,structContL) +
            0.06*(1.0-0.5*(breakFailM+breakFailL)), 0.0, 1.0);

   breakoutReclaimQuality =
      Clamp(0.28*continuationRestart +
            0.16*MathAbs(macroReclaim) +
            0.12*macroContinuation +
            0.08*directionConv +
            0.08*supportConf +
            0.06*(1.0-zoneTrapM) +
            0.05*(1.0-zoneTrapL) +
            0.06*(1.0-trapRisk) +
            0.05*(1.0-regimeBreak) +
            0.03*(1.0-spikeRisk) +
            0.03*(1.0-expectedDepth), 0.0, 1.0);

   reversalTransitionQuality =
      Clamp(0.18*macroTransition +
            0.14*maturity +
            0.14*reversalConfirm +
            0.12*MathAbs(macroReclaim) +
            0.10*macroConflict +
            0.08*MathMax(structRevM,structRevL) +
            0.06*(0.5*breakFailM + 0.5*breakFailL) +
            0.05*regimeBreak +
            0.04*spikeRisk +
            0.04*painRecurrence +
            0.03*lateTrendFade +
            0.02*(1.0-macroContinuation), 0.0, 1.0);

   double a=trendContinuationQuality, b=breakoutReclaimQuality, c=reversalTransitionQuality;
   double best=MathMax(a, MathMax(b,c));
   double second=MathMin(MathMax(a,b), MathMax(MathMin(a,b),c));
   modeDominanceScore=Clamp(best - second, 0.0, 1.0);
   modeConflictScore=Clamp(1.0 - modeDominanceScore + 0.25*macroConflict, 0.0, 1.0);
}



