//+------------------------------------------------------------------+
//| 08_MemoryAndReplay.mqh                                          |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Q-memory retrieval/update, replay metadata, episode archives, re|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

bool QueryQMemory(const int symIdx,
                  const int regime,
                  const double &key[],
                  double &qOut[],
                  double &confOut)
{
   ArrayResize(qOut, ActionCount);
   for(int a=0;a<ActionCount;a++) qOut[a]=0.0;
   confOut=0.0;

   if(!UseQMemory) return false;
   if(ArraySize(key)<=0) return false;

   double sumW=0.0;
   double bestSim=-1e9;
   int bestIdx=-1;

   int n=ArraySize(gQMem);
   for(int i=0;i<n;i++)
   {
      if(gQMem[i].regime != regime) continue;
      if(ArraySize(gQMem[i].stateKey)!=ArraySize(key)) continue;
      if(ArraySize(gQMem[i].qVals)!=ActionCount) continue;

      double sim = QMemSimilarity(key, gQMem[i].stateKey);
      if(sim < 0.0) continue;

      double w = sim * gQMem[i].conf * MathMax(0.0, gQMem[i].score) * QMemAgeFactor(gQMem[i]);
      if(w<=1e-12) continue;

      for(int a=0;a<ActionCount;a++)
         qOut[a] += w * gQMem[i].qVals[a];

      sumW += w;

      if(sim>bestSim)
      {
         bestSim=sim;
         bestIdx=i;
      }
   }

   if(sumW<=1e-12 || bestIdx<0) return false;

   for(int a=0;a<ActionCount;a++)
      qOut[a] /= sumW;

   confOut = Clamp(bestSim,0.0,1.0);

   gQMem[bestIdx].usedCount++;
   gQMem[bestIdx].lastUsed=TimeCurrent();

   return (confOut >= QMemMinConfidence);
}

datetime FastTrainingQMemoryDecisionBarTime(const string symbol)
{
   datetime barTime = iTime(symbol, BaseTF, 1);
   if(barTime<=0) barTime = iTime(symbol, BaseTF, 0);
   if(barTime<=0) barTime = TimeCurrent();
   return barTime;
}

bool ShouldUseSparseQMemoryRetrieval(const int symIdx)
{
   if(!FastTrainingMode) return false;
   if(!FastTrainingSparseQMemory) return false;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return false;
   return true;
}

bool GetQMemoryForDecision(const int symIdx,
                           const int regime,
                           const string symbol,
                           const double &stateKey[],
                           double &qOut[],
                           double &confOut)
{
   ArrayResize(qOut, ActionCount);
   for(int a=0;a<ActionCount;a++) qOut[a]=0.0;
   confOut=0.0;

   if(!UseQMemory) return false;
   if(ArraySize(stateKey)<=0) return false;

   if(!ShouldUseSparseQMemoryRetrieval(symIdx))
      return QueryQMemory(symIdx, regime, stateKey, qOut, confOut);

   datetime barTime = FastTrainingQMemoryDecisionBarTime(symbol);
   if(barTime<=0) barTime = TimeCurrent();

   bool barAdvanced = (gFastQMemLastDecisionBarTime[symIdx] != barTime);
   if(barAdvanced)
   {
      gFastQMemLastDecisionBarTime[symIdx] = barTime;
      gFastQMemDecisionBarCounter[symIdx]++;
   }

   int refreshBars = MathMax(1, FastTrainingQMemoryRefreshDecisionBars);
   int refreshMins = MathMax(1, FastTrainingQMemoryRefreshMinutes);

   bool needRefresh = (!gFastQMemCachedValid[symIdx] || gFastQMemCachedRegime[symIdx] != regime);
   if(!needRefresh && gFastQMemDecisionBarCounter[symIdx] >= refreshBars)
      needRefresh = true;
   if(!needRefresh && (TimeCurrent() - gFastQMemLastRefreshTime[symIdx]) >= (refreshMins * 60))
      needRefresh = true;

   if(needRefresh)
   {
      double qTmp[];
      double confTmp = 0.0;
      bool ok = QueryQMemory(symIdx, regime, stateKey, qTmp, confTmp);

      gFastQMemCachedValid[symIdx]   = ok;
      gFastQMemCachedRegime[symIdx]  = regime;
      gFastQMemCachedConf[symIdx]    = confTmp;
      gFastQMemLastRefreshTime[symIdx] = TimeCurrent();
      gFastQMemDecisionBarCounter[symIdx] = 0;

      for(int a=0;a<3;a++)
         gFastQMemCachedQ[symIdx][a] = (a < ArraySize(qTmp) ? qTmp[a] : 0.0);
   }

   if(!gFastQMemCachedValid[symIdx] || gFastQMemCachedRegime[symIdx] != regime)
      return false;

   for(int a=0;a<ActionCount && a<3;a++)
      qOut[a] = gFastQMemCachedQ[symIdx][a];
   confOut = gFastQMemCachedConf[symIdx];
   return (confOut >= QMemMinConfidence);
}

void UpdateQMemoryResolved(const int symIdx,
                           const int regime,
                           const double &state[],
                           const int action,
                           const double reward)
{
   if(!UseQMemory) return;
   if(action<0 || action>=ActionCount) return;

   double key[];
   if(!BuildQMemoryKey(symIdx,state,key)) return;

   double st[];
   ArrayResize(st,ArraySize(state));
   for(int i=0;i<ArraySize(state);i++) st[i]=state[i];

   double qBase[];
   ArrayResize(qBase,ActionCount);
   DQNForward(symIdx,regime,st,qBase);

   double qTarget[];
   ArrayResize(qTarget,ActionCount);
   for(int a=0;a<ActionCount;a++) qTarget[a]=qBase[a];

   double targetReward = reward;
   if(targetReward < 0.0 && QMemUseNegativeUpdates)
      targetReward *= QMemNegativePenaltyScale;
   else if(targetReward > 0.0)
      targetReward *= QMemPositiveBoostScale;

   qTarget[action] = targetReward;

   double bestSim=-1e9;
   int best=-1;

   int n=ArraySize(gQMem);
   for(int i=0;i<n;i++)
   {
      if(gQMem[i].regime!=regime) continue;
      if(ArraySize(gQMem[i].stateKey)!=ArraySize(key)) continue;
      if(ArraySize(gQMem[i].qVals)!=ActionCount) continue;

      double sim=QMemSimilarity(key,gQMem[i].stateKey);
      if(sim>bestSim){ bestSim=sim; best=i; }
   }

   if(best>=0 && bestSim>=QMemMergeSim)
   {
      for(int k=0;k<ArraySize(key);k++)
         gQMem[best].stateKey[k] = (1.0-QMemFeatureEMA)*gQMem[best].stateKey[k] + QMemFeatureEMA*key[k];
      NormalizeVecN(gQMem[best].stateKey);

      for(int a=0;a<ActionCount;a++)
         gQMem[best].qVals[a] = (1.0-QMemFeatureEMA)*gQMem[best].qVals[a] + QMemFeatureEMA*qTarget[a];

      if(reward>=0.0)
      {
         gQMem[best].conf  = Clamp(gQMem[best].conf + 0.05, 0.0, 1.0);
         gQMem[best].score = MathMax(0.0, gQMem[best].score + 0.20);
      }
      else
      {
         gQMem[best].conf  = Clamp(gQMem[best].conf - 0.03, 0.0, 1.0);
         gQMem[best].score = MathMax(0.0, gQMem[best].score - 0.10);
      }

      gQMem[best].usedCount++;
      gQMem[best].lastUsed=TimeCurrent();
   }
   else
   {
      QMemEntry e;
      e.regime=regime;
      e.created=TimeCurrent();
      e.lastUsed=TimeCurrent();
      e.usedCount=0;

      ArrayResize(e.stateKey,ArraySize(key));
      for(int k=0;k<ArraySize(key);k++) e.stateKey[k]=key[k];

      ArrayResize(e.qVals,ActionCount);
      for(int a=0;a<ActionCount;a++) e.qVals[a]=qTarget[a];

      e.conf  = (reward>=0.0 ? 0.35 : 0.20);
      e.score = (reward>=0.0 ? 1.0 : 0.4);

      int m=ArraySize(gQMem);
      ArrayResize(gQMem,m+1);
      gQMem[m]=e;
   }

   PruneQMemoryIfNeeded();
   if(symIdx>=0 && symIdx<MAX_SYMBOLS)
      gDecisionSupportCache[symIdx].valid=false;
}



double CurrentReplayDDPct()
{
   double eq = GetEAEquity();
   double peak = MathMax(maxEquity, 1e-8);
   return MathMax(0.0, (peak - eq) / peak);
}

double GetCurrentMidPriceForSymbol(const string symbol)
{
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   if(bid<=0.0 && ask<=0.0) return 0.0;
   if(bid<=0.0) return ask;
   if(ask<=0.0) return bid;
   return 0.5*(bid+ask);
}

double GetLastEntryPriceFromCache(const int symIdx, const double fallbackPrice)
{
   if(symIdx<0 || symIdx>=gSymbolCount) return fallbackPrice;
   int n = gTrades[symIdx].Total();
   if(n<=0) return fallbackPrice;
   return gTrades[symIdx].At(n-1);
}

int ClassifyReplayBasketState(const int symIdx)
{
   if(symIdx<0 || symIdx>=gSymbolCount) return 0;
   int pc = gPositionsCount[symIdx];
   if(pc<=0) return 0;
   if(pc==1) return 1;
   if(pc<=3) return 2;
   return 3;
}

int ClassifyReplayAddDepth(const int symIdx)
{
   if(symIdx<0 || symIdx>=gSymbolCount) return 0;
   int adds = MathMax(gPositionsCount[symIdx]-1, 0);
   if(adds<=0) return 0;
   if(adds<=2) return 1;
   if(adds<=DeepBasketAddThreshold) return 2;
   return 3;
}

int ClassifyReplayDanger(const int symIdx, const double reward)
{
   double dd = GetSymbolFloatingDDPct(symIdx);
   if(dd >= DangerReplayDDThresholdPct || reward <= -3.0) return 2;
   if(dd >= 0.5*DangerReplayDDThresholdPct || reward <= -1.0) return 1;
   return 0;
}

int ClassifyReplayLiquidity(const string symbol)
{
   double spr = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   double avgSpr = RollingAvgSpreadPtsTF(symbol, TF_EXEC, 64);
   if(avgSpr <= 0.0) avgSpr = MathMax(spr, 1.0);
   double ratio = spr / MathMax(avgSpr, 1e-8);
   if(ratio >= 1.75) return 2;
   if(ratio >= 1.20) return 1;
   return 0;
}


long EnsureActiveBasketEpisode(const int symIdx,
                              const int basketDir,
                              const int positionsAtOpen,
                              const datetime entryTime)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0;

   if(!gActiveBasketEpisodes[symIdx].active)
   {
      gActiveBasketEpisodes[symIdx].active=true;
      gActiveBasketEpisodes[symIdx].episodeId=gNextEpisodeId++;
      gActiveBasketEpisodes[symIdx].symIdx=symIdx;
      gActiveBasketEpisodes[symIdx].basketDir=basketDir;
      gActiveBasketEpisodes[symIdx].startTime=(entryTime>0 ? entryTime : TimeCurrent());
      gActiveBasketEpisodes[symIdx].lastTime=TimeCurrent();
      gActiveBasketEpisodes[symIdx].addCount=MathMax(positionsAtOpen-1,0);
      gActiveBasketEpisodes[symIdx].maxPositions=MathMax(positionsAtOpen,1);
      gActiveBasketEpisodes[symIdx].maxDD=GetSymbolFloatingDDPct(symIdx);
      gActiveBasketEpisodes[symIdx].rewardAccum=0.0;
      ArrayResize(gActiveBasketEpisodes[symIdx].replayItemIndexes,0);
   }
   else
   {
      gActiveBasketEpisodes[symIdx].lastTime=TimeCurrent();
      gActiveBasketEpisodes[symIdx].basketDir=basketDir;
      gActiveBasketEpisodes[symIdx].addCount=MathMax(gActiveBasketEpisodes[symIdx].addCount, MathMax(positionsAtOpen-1,0));
      gActiveBasketEpisodes[symIdx].maxPositions=MathMax(gActiveBasketEpisodes[symIdx].maxPositions, MathMax(positionsAtOpen,1));
      gActiveBasketEpisodes[symIdx].maxDD=MathMax(gActiveBasketEpisodes[symIdx].maxDD, GetSymbolFloatingDDPct(symIdx));
   }
   return gActiveBasketEpisodes[symIdx].episodeId;
}

long GetCurrentEpisodeIdForSymbol(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0;
   if(!gActiveBasketEpisodes[symIdx].active) return 0;
   return gActiveBasketEpisodes[symIdx].episodeId;
}

void AppendReplayIndexToActiveEpisode(const int symIdx,const int replayIdx,const double reward)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(!gActiveBasketEpisodes[symIdx].active) return;
   int n=ArraySize(gActiveBasketEpisodes[symIdx].replayItemIndexes);
   ArrayResize(gActiveBasketEpisodes[symIdx].replayItemIndexes,n+1);
   gActiveBasketEpisodes[symIdx].replayItemIndexes[n]=replayIdx;
   gActiveBasketEpisodes[symIdx].rewardAccum += reward;
   gActiveBasketEpisodes[symIdx].maxDD=MathMax(gActiveBasketEpisodes[symIdx].maxDD, GetSymbolFloatingDDPct(symIdx));
   gActiveBasketEpisodes[symIdx].lastTime=TimeCurrent();
}

int EstimateSessionType(const datetime t)
{
   MqlDateTime dt;
   TimeToStruct((t>0 ? t : TimeCurrent()), dt);
   int h=dt.hour;
   if(h<8) return 0;
   if(h<16) return 1;
   return 2;
}

bool GetDecisionContextCached(const int symIdx,
                              const int regimeHint,
                              int &regimeType,
                              int &patternType,
                              int &liquidityType,
                              int &sessionType,
                              double &atrRatio)
{
   if(symIdx<0 || symIdx>=gSymbolCount) return false;

   string symbol = gSymbols[symIdx];
   datetime barTime = iTime(symbol, BaseTF, 1);
   datetime nowT = TimeCurrent();
   int hourBucket = (int)(nowT / 3600);

   bool refresh = (!gDecisionCtxValid[symIdx]);
   if(!refresh && gDecisionCtxBarTime[symIdx] != barTime) refresh = true;
   if(!refresh && gDecisionCtxHourBucket[symIdx] != hourBucket) refresh = true;

   if(refresh)
   {
      atrRatio = GetAtrRatioCached_Base(symbol);
      gDecisionCtxAtrRatio[symIdx] = atrRatio;
      gDecisionCtxRegimeType[symIdx] = RegimeIndexFromRatio(atrRatio);
      gDecisionCtxPatternType[symIdx] = ClassifyReplayStructureContext(symbol);
      gDecisionCtxLiquidityType[symIdx] = ClassifyReplayLiquidity(symbol);
      gDecisionCtxSessionType[symIdx] = EstimateSessionType(nowT);
      gDecisionCtxBarTime[symIdx] = barTime;
      gDecisionCtxHourBucket[symIdx] = hourBucket;
      gDecisionCtxValid[symIdx] = true;
   }

   atrRatio = gDecisionCtxAtrRatio[symIdx];
   regimeType = gDecisionCtxRegimeType[symIdx];
   if(regimeHint>=0 && regimeHint<REGIME_COUNT)
      regimeType = regimeHint;
   patternType = gDecisionCtxPatternType[symIdx];
   liquidityType = gDecisionCtxLiquidityType[symIdx];
   sessionType = gDecisionCtxSessionType[symIdx];
   return true;
}


BarSpanRef MakeBarSpanRef(const string symbol,const ENUM_TIMEFRAMES tf,const datetime startT,const datetime endT)
{
   BarSpanRef r;
   r.tf=tf;
   r.startTime=startT;
   r.endTime=endT;
   r.startIndex=iBarShift(symbol, tf, startT, false);
   r.endIndex=iBarShift(symbol, tf, endT, false);
   return r;
}

void PruneArchiveMemoriesIfNeeded()
{
   while(ArraySize(gEpisodeMemory) > MaxEpisodeMemory) ArrayRemove(gEpisodeMemory,0,1);
   while(ArraySize(gPatternMemory) > MaxPatternMemory) ArrayRemove(gPatternMemory,0,1);
   while(ArraySize(gRegimeEventMemory) > MaxRegimeEventMemory) ArrayRemove(gRegimeEventMemory,0,1);
}

long NextPatternMemoryId()
{
   int n=ArraySize(gPatternMemory);
   return (n>0 ? (gPatternMemory[n-1].patternId + 1) : 1);
}

long NextRegimeEventMemoryId()
{
   int n=ArraySize(gRegimeEventMemory);
   return (n>0 ? (gRegimeEventMemory[n-1].regimeId + 1) : 1);
}

int PatternStrengthClassFromEpisode(const EpisodeMemory &ep)
{
   if(ep.rewardEfficiency > 0.0 && ep.maxDrawdownPct <= RewardV2CleanCycleMaxDD && ep.addCount <= 1)
      return 2;
   if(ep.rewardEfficiency < 0.0 || ep.maxDrawdownPct >= DangerReplayDDThresholdPct || ep.addCount >= DeepBasketAddThreshold)
      return 0;
   return 1;
}

void AppendArchiveRecordsFromBasketSequence(const int symIdx,const BasketSequenceRef &seq)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;

   string symbol = gSymbols[symIdx];
   datetime startT = (seq.startTime>0 ? seq.startTime : TimeCurrent());
   datetime endT   = (seq.endTime>=startT ? seq.endTime : TimeCurrent());

   EpisodeMemory ep;
   ep.episodeId         = seq.episodeId;
   ep.symbol            = symbol;
   ep.symIdx            = symIdx;
   ep.startTime         = startT;
   ep.endTime           = endT;
   ep.basketDir         = seq.basketDir;
   ep.openPositionsMax  = MathMax(seq.maxPositions,1);
   ep.addCount          = MathMax(seq.addCount,0);
   ep.actionsCount      = MathMax(ArraySize(seq.replayItemIndexes), ep.addCount + 1);
   ep.entryPriceFirst   = 0.0;
   ep.avgEntryAtWorst   = 0.0;
   ep.closePriceFinal   = 0.0;
   ep.pnlFinal          = 0.0;
   ep.rewardTotal       = seq.finalReward;
   double durationMin   = MathMax((double)(endT - startT) / 60.0, 1.0);
   ep.rewardEfficiency  = seq.finalReward / durationMin;
   ep.maxDrawdownPct    = MathMax(seq.maxDD, 0.0);
   ep.maxDangerScore    = CurrentReplayRiskBias(symIdx);
   ep.maxMarginStress   = CurrentMarginStressRatio();
   ep.oneRoundTrade     = (ep.openPositionsMax <= 1 ? 1 : 0);

   bool riskyProfit = (ep.rewardTotal > 0.0 && (ep.addCount > 0 || ep.openPositionsMax > 2 || ep.maxDrawdownPct > RewardV2CleanCycleMaxDD));
   ep.forcedStopLikeEvent = ((ep.rewardTotal <= 0.0 && ep.maxDrawdownPct >= DangerReplayDDThresholdPct) ||
                             ep.openPositionsMax >= MathMax(DeepBasketAddThreshold + 1, 4) ? 1 : 0);
   ep.inefficientRecovery = ((ep.addCount > 0) &&
                             (ep.rewardTotal <= 0.0 || ep.maxDrawdownPct > RewardV2CleanCycleMaxDD ||
                              ep.openPositionsMax > 2 || riskyProfit) ? 1 : 0);

   ep.sessionType       = EstimateSessionType(endT);
   ep.liquidityType     = ClassifyReplayLiquidity(symbol);
   ep.regimeType        = RegimeIndexFromRatio(GetAtrRatioCached_Base(symbol));
   ep.patternType       = ClassifyReplayStructureContext(symbol);

   ep.execSpan          = MakeBarSpanRef(symbol, TF_EXEC, startT, endT);
   ep.midSpan           = MakeBarSpanRef(symbol, TF_MID, startT, endT);
   ep.longSpan          = MakeBarSpanRef(symbol, TF_LONG, startT, endT);
   ep.structExtSpan     = MakeBarSpanRef(symbol, TF_STRUCT_EXT, startT, endT);

   int ne=ArraySize(gEpisodeMemory);
   ArrayResize(gEpisodeMemory, ne+1);
   gEpisodeMemory[ne]=ep;

   PatternMemory pm;
   pm.patternId            = NextPatternMemoryId();
   pm.symbol               = symbol;
   pm.symIdx               = symIdx;
   pm.patternType          = ep.patternType;
   pm.strengthClass        = PatternStrengthClassFromEpisode(ep);
   pm.volatilityClass      = ep.regimeType;
   pm.liquidityClass       = ep.liquidityType;
   pm.startTime            = startT;
   pm.endTime              = endT;
   pm.rewardEfficiencyMean = ep.rewardEfficiency;
   pm.ddMean               = ep.maxDrawdownPct;
   pm.addMean              = (double)ep.addCount;
   pm.execSpan             = ep.execSpan;
   pm.midSpan              = ep.midSpan;
   pm.longSpan             = ep.longSpan;
   ArrayResize(pm.linkedEpisodeIds,1);
   pm.linkedEpisodeIds[0]  = ep.episodeId;

   int np=ArraySize(gPatternMemory);
   ArrayResize(gPatternMemory, np+1);
   gPatternMemory[np]=pm;

   RegimeEventMemory rm;
   rm.regimeId            = NextRegimeEventMemoryId();
   rm.symbol              = symbol;
   rm.symIdx              = symIdx;
   rm.regimeType          = ep.regimeType;
   rm.eventType           = ((ep.forcedStopLikeEvent!=0 || ep.inefficientRecovery!=0 || riskyProfit) ? 1 : 0);
   rm.startTime           = startT;
   rm.endTime             = endT;
   rm.avgVol              = GetAtrRatioCached_Base(symbol);
   rm.avgSpread           = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   rm.avgADX              = GetADXValueTF(symbol, TF_EXEC, 14, 1);
   rm.avgRewardEfficiency = ep.rewardEfficiency;
   rm.execSpan            = ep.execSpan;
   rm.midSpan             = ep.midSpan;
   rm.longSpan            = ep.longSpan;
   ArrayResize(rm.linkedEpisodeIds,1);
   rm.linkedEpisodeIds[0] = ep.episodeId;
   ArrayResize(rm.linkedPatternIds,1);
   rm.linkedPatternIds[0] = pm.patternId;

   int nr=ArraySize(gRegimeEventMemory);
   ArrayResize(gRegimeEventMemory, nr+1);
   gRegimeEventMemory[nr]=rm;

   PruneArchiveMemoriesIfNeeded();
}


void FinalizeActiveBasketEpisode(const int symIdx,const double finalReward)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(!gActiveBasketEpisodes[symIdx].active) return;

   BasketSequenceRef seq;
   seq.episodeId   = gActiveBasketEpisodes[symIdx].episodeId;
   seq.symIdx      = symIdx;
   seq.startTime   = gActiveBasketEpisodes[symIdx].startTime;
   seq.endTime     = gActiveBasketEpisodes[symIdx].lastTime;
   seq.basketDir   = gActiveBasketEpisodes[symIdx].basketDir;
   seq.addCount    = gActiveBasketEpisodes[symIdx].addCount;
   seq.maxPositions= gActiveBasketEpisodes[symIdx].maxPositions;
   seq.maxDD       = gActiveBasketEpisodes[symIdx].maxDD;
   seq.finalReward = gActiveBasketEpisodes[symIdx].rewardAccum + finalReward;

   ArrayResize(seq.replayItemIndexes, ArraySize(gActiveBasketEpisodes[symIdx].replayItemIndexes));
   for(int i=0;i<ArraySize(seq.replayItemIndexes);i++)
      seq.replayItemIndexes[i]=gActiveBasketEpisodes[symIdx].replayItemIndexes[i];

   if(DeepBasketSequenceQualifies(seq))
   {
      int n=ArraySize(gDeepBasketReplayBank.sequences);
      ArrayResize(gDeepBasketReplayBank.sequences,n+1);
      gDeepBasketReplayBank.sequences[n]=seq;
      if(ArraySize(gDeepBasketReplayBank.sequences) > MathMax(DeepBasketReplayCapacity/4, 256))
         ArrayRemove(gDeepBasketReplayBank.sequences,0,1);
      gDeepBasketReplayBank.seqCount = ArraySize(gDeepBasketReplayBank.sequences);
   }

   AppendArchiveRecordsFromBasketSequence(symIdx, seq);

   gActiveBasketEpisodes[symIdx].active=false;
   gActiveBasketEpisodes[symIdx].episodeId=0;
   ArrayResize(gActiveBasketEpisodes[symIdx].replayItemIndexes,0);
}

void EnsureActiveEfficientPeriod(const int symIdx,const datetime t)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(gActiveEfficientPeriods[symIdx].active)
      return;

   gActiveEfficientPeriods[symIdx].active=true;
   gActiveEfficientPeriods[symIdx].periodId=gNextPeriodId++;
   gActiveEfficientPeriods[symIdx].symIdx=symIdx;
   gActiveEfficientPeriods[symIdx].startTime=t;
   gActiveEfficientPeriods[symIdx].lastTime=t;
   gActiveEfficientPeriods[symIdx].rewardTotal=0.0;
   gActiveEfficientPeriods[symIdx].ddMax=GetSymbolFloatingDDPct(symIdx);
   gActiveEfficientPeriods[symIdx].oneRoundCount=0;
   gActiveEfficientPeriods[symIdx].addCountTotal=0;
   gActiveEfficientPeriods[symIdx].tradeCount=0;
   gActiveEfficientPeriods[symIdx].lastLiquidityClass=-1;
   gActiveEfficientPeriods[symIdx].lastRegime=-1;
   gActiveEfficientPeriods[symIdx].lastStructureClass=-1;
   gActiveEfficientPeriods[symIdx].contextBreakCount=0;
   ArrayResize(gActiveEfficientPeriods[symIdx].replayItemIndexes,0);
}

void FinalizeActiveEfficientPeriod(const int symIdx,const int sessionType,const int liquidityClass,const int regimeType,const int patternType)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(!gActiveEfficientPeriods[symIdx].active) return;
   if(gActiveEfficientPeriods[symIdx].tradeCount < EfficientPeriodMinTrades)
   {
      gActiveEfficientPeriods[symIdx].active=false;
      gActiveEfficientPeriods[symIdx].periodId=0;
      ArrayResize(gActiveEfficientPeriods[symIdx].replayItemIndexes,0);
      return;
   }

   EfficientPeriodRef pr;
   pr.periodId = gActiveEfficientPeriods[symIdx].periodId;
   pr.symIdx   = symIdx;
   pr.startTime= gActiveEfficientPeriods[symIdx].startTime;
   pr.endTime  = gActiveEfficientPeriods[symIdx].lastTime;
   pr.rewardTotal = gActiveEfficientPeriods[symIdx].rewardTotal;
   double durationMin = MathMax((double)(pr.endTime - pr.startTime)/60.0, 1.0);
   pr.rewardEfficiency = pr.rewardTotal / durationMin;
   pr.ddMax = gActiveEfficientPeriods[symIdx].ddMax;
   pr.oneRoundCount = gActiveEfficientPeriods[symIdx].oneRoundCount;
   pr.addCountTotal = gActiveEfficientPeriods[symIdx].addCountTotal;
   pr.sessionType = sessionType;
   pr.liquidityType = liquidityClass;
   pr.regimeType = regimeType;
   pr.patternType = patternType;
   ArrayResize(pr.replayItemIndexes, ArraySize(gActiveEfficientPeriods[symIdx].replayItemIndexes));
   for(int i=0;i<ArraySize(pr.replayItemIndexes);i++)
      pr.replayItemIndexes[i]=gActiveEfficientPeriods[symIdx].replayItemIndexes[i];

   if(EfficientPeriodQualifies(pr))
   {
      int n=ArraySize(gEfficientReplayBank.periods);
      ArrayResize(gEfficientReplayBank.periods,n+1);
      gEfficientReplayBank.periods[n]=pr;
      if(ArraySize(gEfficientReplayBank.periods) > MathMax(EfficientReplayCapacity/4, 256))
         ArrayRemove(gEfficientReplayBank.periods,0,1);
      gEfficientReplayBank.periodCount = ArraySize(gEfficientReplayBank.periods);
   }

   gActiveEfficientPeriods[symIdx].active=false;
   gActiveEfficientPeriods[symIdx].periodId=0;
   ArrayResize(gActiveEfficientPeriods[symIdx].replayItemIndexes,0);
}

void UpdateActiveEfficientPeriodWithItem(const ReplayItem &item,const int replayIdx)
{
   int symIdx=item.symIdx;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   datetime nowT = (item.eventTime>0 ? item.eventTime : TimeCurrent());

   if(gActiveEfficientPeriods[symIdx].active)
   {
      int elapsedMin = (int)((nowT - gActiveEfficientPeriods[symIdx].startTime)/60);
      bool contextBreak=false;
      if(EfficientSegmentByContext)
      {
         if(gActiveEfficientPeriods[symIdx].lastLiquidityClass>=0 && item.liquidityClass != gActiveEfficientPeriods[symIdx].lastLiquidityClass)
            contextBreak=true;
         if(gActiveEfficientPeriods[symIdx].lastRegime>=0 && item.regime != gActiveEfficientPeriods[symIdx].lastRegime)
            contextBreak=true;
         if(gActiveEfficientPeriods[symIdx].lastStructureClass>=0 && item.structureContextClass != gActiveEfficientPeriods[symIdx].lastStructureClass)
            contextBreak=true;
      }

      if(contextBreak)
         gActiveEfficientPeriods[symIdx].contextBreakCount++;
      else
         gActiveEfficientPeriods[symIdx].contextBreakCount=0;

      double elapsedMinD = MathMax((double)elapsedMin, 1.0);
      double curEffPerMin = gActiveEfficientPeriods[symIdx].rewardTotal / elapsedMinD;
      bool degrade = (elapsedMin >= 30 && curEffPerMin < EfficientSegmentMinPerMin);
      bool ddBreak = (gActiveEfficientPeriods[symIdx].ddMax > EfficientPeriodMaxDDPct);

      if(elapsedMin >= EfficientPeriodWindowMinutes ||
         gActiveEfficientPeriods[symIdx].contextBreakCount >= EfficientContextBreakTolerance ||
         degrade || ddBreak)
      {
         FinalizeActiveEfficientPeriod(symIdx, 0, item.liquidityClass, item.regime, item.structureContextClass);
      }
   }

   EnsureActiveEfficientPeriod(symIdx, nowT);
   gActiveEfficientPeriods[symIdx].lastTime = nowT;
   gActiveEfficientPeriods[symIdx].rewardTotal += item.reward;
   gActiveEfficientPeriods[symIdx].ddMax = MathMax(gActiveEfficientPeriods[symIdx].ddMax, GetSymbolFloatingDDPct(symIdx));
   gActiveEfficientPeriods[symIdx].tradeCount++;
   if(item.addDepthClass==0 && item.reward>0.0) gActiveEfficientPeriods[symIdx].oneRoundCount++;
   gActiveEfficientPeriods[symIdx].addCountTotal += MathMax(item.addDepthClass,0);
   gActiveEfficientPeriods[symIdx].lastLiquidityClass = item.liquidityClass;
   gActiveEfficientPeriods[symIdx].lastRegime = item.regime;
   gActiveEfficientPeriods[symIdx].lastStructureClass = item.structureContextClass;
   int n=ArraySize(gActiveEfficientPeriods[symIdx].replayItemIndexes);
   ArrayResize(gActiveEfficientPeriods[symIdx].replayItemIndexes,n+1);
   gActiveEfficientPeriods[symIdx].replayItemIndexes[n]=replayIdx;
}

int ClassifyReplayZoneContext(const string symbol,
                              const double px,
                              const double avgPx,
                              const double entryPx)
{
   if(!UseZoneCandleBranch) return (int)MEM_ZONE_NONE;

   double atr = MathMax(GetATRValueTF(symbol, TF_EXEC, 14, 1), 1e-8);

   SwingDerivedZone demands[], supplies[];
   int dc=0, sc=0;
   if(!BuildSwingDerivedZonesForTF(symbol, TF_EXEC, demands, dc, supplies, sc))
      return (int)MEM_ZONE_NONE;

   int bestD=-1, bestS=-1;
   bool hasD = SelectBestSwingZone(demands, dc, symbol, TF_EXEC, px, avgPx, entryPx, atr, bestD);
   bool hasS = SelectBestSwingZone(supplies, sc, symbol, TF_EXEC, px, avgPx, entryPx, atr, bestS);

   double nearBuf = 0.35;
   if(hasD && bestD>=0)
   {
      bool inside = (px >= demands[bestD].low && px <= demands[bestD].high);
      double dist = DistanceToZoneBoundaryAtr(demands[bestD], px, atr);
      if(inside) return (int)MEM_ZONE_INSIDE_DEMAND;
      if(dist <= nearBuf) return (int)MEM_ZONE_NEAR_DEMAND;
   }
   if(hasS && bestS>=0)
   {
      bool inside = (px >= supplies[bestS].low && px <= supplies[bestS].high);
      double dist = DistanceToZoneBoundaryAtr(supplies[bestS], px, atr);
      if(inside) return (int)MEM_ZONE_INSIDE_SUPPLY;
      if(dist <= nearBuf) return (int)MEM_ZONE_NEAR_SUPPLY;
   }
   return (int)MEM_ZONE_NONE;
}

int ClassifyReplayStructureContext(const string symbol)
{
   if(!UseStructureBranch) return (int)MEM_STRUCT_MIXED;
   double atr = MathMax(GetATRValueTF(symbol, TF_LONG, 14, 1), 1e-8);
   SwingPointMem swings[];
   int swingCount=0;
   SwingPointMem lastHigh, prevHigh, lastLow, prevLow;
   bool ok = BuildSwingMemoryForTF(symbol, TF_LONG, swings, swingCount) &&
             ExtractRecentSwingRefs(swings, swingCount, lastHigh, prevHigh, lastLow, prevLow);
   if(!ok)
   {
      FallbackStructureRefs(symbol, TF_LONG, lastHigh, prevHigh, lastLow, prevLow);
   }
   double tendency = SwingTendencyFromMemory(lastHigh, prevHigh, lastLow, prevLow, atr);
   double lastAmp = MathMax(lastHigh.price - lastLow.price, 0.0);
   double prevAmp = MathMax(prevHigh.price - prevLow.price, 0.0);
   double compExp = Clamp(SafeDiv(lastAmp, MathMax(prevAmp,1e-8), 1.0) - 1.0, -2.0, 2.0) / 2.0;

   if(MathAbs(tendency) >= 0.20)
      return (tendency > 0.0 ? (int)MEM_STRUCT_TREND_UP : (int)MEM_STRUCT_TREND_DOWN);
   if(compExp <= -0.20) return (int)MEM_STRUCT_COMPRESSION;
   if(compExp >= 0.20) return (int)MEM_STRUCT_EXPANSION;
   return (int)MEM_STRUCT_MIXED;
}

void FillReplayMetadata(ReplayItem &item,
                        const int symIdx,
                        const int regime,
                        const double &state[],
                        const int action,
                        const double reward,
                        const bool done)
{
   item.replayBankType = (int)REPLAY_BANK_RECENT;
   item.episodeId = GetCurrentEpisodeIdForSymbol(symIdx);
   item.patternId = 0;
   item.regimeId = (long)regime;
   item.periodId = 0;
   item.eventTime = TimeCurrent();

   string symbol = ((symIdx>=0 && symIdx<gSymbolCount) ? gSymbols[symIdx] : _Symbol);
   double px = GetCurrentMidPriceForSymbol(symbol);
   double avgPx = ((symIdx>=0 && symIdx<gSymbolCount && gBasketAvgValid[symIdx]) ? gBasketAvgPriceCache[symIdx] : px);
   double entryPx = GetLastEntryPriceFromCache(symIdx, px);

   item.basketStateClass    = ClassifyReplayBasketState(symIdx);
   item.addDepthClass       = ClassifyReplayAddDepth(symIdx);
   item.dangerClass         = ClassifyReplayDanger(symIdx, reward);
   item.zoneContextClass    = ClassifyReplayZoneContext(symbol, px, avgPx, entryPx);
   item.structureContextClass = ClassifyReplayStructureContext(symbol);
   item.liquidityClass      = ClassifyReplayLiquidity(symbol);
}

void ReplayBankStorePush(ReplayBankStore &bank, const ReplayItem &item, const int capacity)
{
   if(capacity<=0) return;
   bank.maxCount = capacity;
   int n = ArraySize(bank.items);
   ArrayResize(bank.items, n+1);
   bank.items[n] = item;
   if(ArraySize(bank.items) > capacity)
      ArrayRemove(bank.items, 0, 1);
   bank.count = ArraySize(bank.items);
}

void DeepBasketReplayStorePush(DeepBasketReplayStore &bank, const ReplayItem &item, const int capacity)
{
   if(capacity<=0) return;
   bank.maxItems = capacity;
   int n = ArraySize(bank.items);
   ArrayResize(bank.items, n+1);
   bank.items[n] = item;
   if(ArraySize(bank.items) > capacity)
      ArrayRemove(bank.items, 0, 1);
   bank.itemCount = ArraySize(bank.items);
}

void EfficientReplayStorePush(EfficientReplayStore &bank, const ReplayItem &item, const int capacity)
{
   if(capacity<=0) return;
   bank.maxItems = capacity;
   int n = ArraySize(bank.items);
   ArrayResize(bank.items, n+1);
   bank.items[n] = item;
   if(ArraySize(bank.items) > capacity)
      ArrayRemove(bank.items, 0, 1);
   bank.itemCount = ArraySize(bank.items);
}

double ReplayRiskyProfitScore(const ReplayItem &item)
{
   if(item.reward <= 0.0) return 0.0;

   double score = 0.0;
   if(item.addDepthClass >= 1) score += 0.55 + 0.35 * MathMin(item.addDepthClass, 3);
   if(item.basketStateClass >= 2) score += 0.45;
   if(item.dangerClass > 0) score += 0.55 * MathMin(item.dangerClass, 2);
   if(item.action != 0) score += 0.10;
   return score;
}

bool ReplayIsRiskyProfit(const ReplayItem &item)
{
   return (item.reward > 0.0 && ReplayRiskyProfitScore(item) >= ReplayRiskyProfitThreshold);
}

bool ReplayIsEfficientCandidate(const ReplayItem &item)
{
   if(item.reward <= 0.0) return false;
   if(item.dangerClass > 0) return false;
   if(item.addDepthClass > 0) return false;
   if(item.basketStateClass > 1) return false;
   return true;
}

bool ReplayIsDeepAntiPattern(const ReplayItem &item)
{
   bool basketHeavy = (item.addDepthClass >= 1 || item.basketStateClass >= 2);
   if(!basketHeavy) return false;
   if(item.reward < 0.0) return true;
   return ReplayIsRiskyProfit(item);
}

bool ReplayIsDangerAntiPattern(const ReplayItem &item)
{
   if(item.dangerClass >= 2 && item.reward <= 0.0) return true;
   if(item.dangerClass >= 1 && ReplayIsRiskyProfit(item)) return true;
   if(item.reward <= -1.5 && item.dangerClass >= 1) return true;
   return false;
}

double ReplayEfficiencyProgress()
{
   double eff = (double)ArraySize(gEfficientReplayBank.items) + 2.0 * (double)ArraySize(gEfficientReplayBank.periods);
   double anti = (double)ArraySize(gDangerReplayBank.items) + (double)ArraySize(gDeepBasketReplayBank.items) + 2.0 * (double)ArraySize(gDeepBasketReplayBank.sequences);
   return Clamp(SafeDiv(eff, eff + anti + 1.0, 0.0), 0.0, 1.0);
}

double ReplayAntiPatternPressure()
{
   double eff = (double)ArraySize(gEfficientReplayBank.items) + 2.0 * (double)ArraySize(gEfficientReplayBank.periods);
   double anti = (double)ArraySize(gDangerReplayBank.items) + (double)ArraySize(gDeepBasketReplayBank.items) + 2.0 * (double)ArraySize(gDeepBasketReplayBank.sequences);
   return Clamp(SafeDiv(anti, MathMax(1.0, eff), 0.0), 0.0, 3.0);
}

double ReplayRewardRecencyWeight(const datetime t)
{
   if(t<=0) return 1.0;
   if(ReplayRewardHalfLifeDays <= 0.0) return 1.0;
   double ageSec = (double)(TimeCurrent() - t);
   if(ageSec <= 0.0) return 1.0;
   double halfSec = ReplayRewardHalfLifeDays * 86400.0;
   if(halfSec <= 1.0) return 1.0;
   return Clamp(MathExp(-0.6931471805599453 * (ageSec / halfSec)), 0.10, 1.0);
}

double ReplayRewardContextWeight(const ReplayItem &it,
                                 const int symIdx,
                                 const int basketDir,
                                 const int positionsCount,
                                 const int regimeType,
                                 const int structureType,
                                 const int liquidityType)
{
   if(it.symIdx != symIdx) return 0.0;

   double w = 0.15;
   if(it.regime == regimeType) w += 0.22;
   else if(MathAbs(it.regime - regimeType) == 1) w += 0.10;
   if(it.structureContextClass == structureType) w += 0.16;
   if(it.liquidityClass == liquidityType) w += 0.12;

   int targetAction = (basketDir>0 ? 1 : (basketDir<0 ? 2 : -1));
   if(targetAction>0 && it.action == targetAction) w += 0.10;

   if(positionsCount <= 1 && it.addDepthClass <= 0) w += 0.07;
   if(positionsCount > 1 && it.addDepthClass >= 1) w += 0.10;

   w *= ReplayRewardRecencyWeight(it.eventTime);
   return w;
}

double ReplayRewardDangerScore(const ReplayItem &it)
{
   double s = Clamp(MathMax(0.0, -it.reward), 0.0, 3.0);
   s += 0.35 * (double)MathMax(0, it.dangerClass);
   s += 0.18 * (double)MathMax(0, it.addDepthClass);
   if(ReplayIsRiskyProfit(it)) s += 0.85;
   return Clamp(s, 0.0, 5.0);
}

double ReplayRewardDeepScore(const ReplayItem &it)
{
   double s = 0.25 * (double)MathMax(0, it.addDepthClass);
   s += 0.18 * (double)MathMax(0, it.basketStateClass);
   if(it.reward < 0.0) s += Clamp(-it.reward, 0.0, 3.0);
   if(ReplayIsRiskyProfit(it)) s += 0.90;
   return Clamp(s, 0.0, 5.0);
}

double ReplayRewardEfficientScore(const ReplayItem &it)
{
   double s = Clamp(it.reward, 0.0, 3.0);
   if(ReplayIsEfficientCandidate(it)) s += 0.75;
   if(it.addDepthClass <= 0) s += 0.20;
   if(it.dangerClass <= 0) s += 0.20;
   if(it.reward <= 0.0) s *= 0.25;
   return Clamp(s, 0.0, 5.0);
}

void ComputeReplayRewardProfile(const int symIdx,
                                const int basketDir,
                                const int positionsCount,
                                double &dangerRiskOut,
                                double &deepRiskOut,
                                double &efficientOut,
                                double &recentCautionOut)
{
   dangerRiskOut=0.0;
   deepRiskOut=0.0;
   efficientOut=0.0;
   recentCautionOut=0.0;

   if(!UseReplayAwareReward) return;
   if(symIdx<0 || symIdx>=gSymbolCount) return;

   string symbol = gSymbols[symIdx];
   int regimeType=0, structureType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, -1, regimeType, structureType, liquidityType, sessionType, atrRatio);
   int scan = MathMax(50, ReplayRewardScanLimit);

   double sumDanger=0.0, accDanger=0.0;
   int nd=ArraySize(gDangerReplayBank.items);
   int sd=MathMax(0, nd-scan);
   for(int i=sd;i<nd;i++)
   {
      double w = ReplayRewardContextWeight(gDangerReplayBank.items[i], symIdx, basketDir, positionsCount, regimeType, structureType, liquidityType);
      if(w<=0.0) continue;
      accDanger += w * ReplayRewardDangerScore(gDangerReplayBank.items[i]);
      sumDanger += w;
   }
   if(sumDanger>1e-9) dangerRiskOut = Clamp(accDanger / sumDanger, 0.0, 4.0);

   double sumDeep=0.0, accDeep=0.0;
   int nb=ArraySize(gDeepBasketReplayBank.items);
   int sb=MathMax(0, nb-scan);
   for(int i=sb;i<nb;i++)
   {
      double w = ReplayRewardContextWeight(gDeepBasketReplayBank.items[i], symIdx, basketDir, positionsCount, regimeType, structureType, liquidityType);
      if(w<=0.0) continue;
      if(positionsCount>1 && gDeepBasketReplayBank.items[i].addDepthClass>=1) w *= 1.15;
      accDeep += w * ReplayRewardDeepScore(gDeepBasketReplayBank.items[i]);
      sumDeep += w;
   }
   int ns=ArraySize(gDeepBasketReplayBank.sequences);
   int ss=MathMax(0, ns-scan);
   for(int i=ss;i<ns;i++)
   {
      if(gDeepBasketReplayBank.sequences[i].symIdx != symIdx) continue;
      double w = 0.20;
      if(gDeepBasketReplayBank.sequences[i].basketDir == basketDir) w += 0.22;
      if(positionsCount>1) w += 0.12;
      w *= ReplayRewardRecencyWeight(gDeepBasketReplayBank.sequences[i].endTime);

      double s = Clamp(gDeepBasketReplayBank.sequences[i].maxDD / MathMax(DangerReplayDDThresholdPct, 1e-6), 0.0, 4.0);
      s += 0.25 * (double)MathMax(0, gDeepBasketReplayBank.sequences[i].addCount);
      s += 0.18 * (double)MathMax(0, gDeepBasketReplayBank.sequences[i].maxPositions - 1);
      if(gDeepBasketReplayBank.sequences[i].finalReward < 0.0) s += Clamp(-gDeepBasketReplayBank.sequences[i].finalReward, 0.0, 2.0);
      if(gDeepBasketReplayBank.sequences[i].finalReward > 0.0 &&
         (gDeepBasketReplayBank.sequences[i].addCount >= 2 || gDeepBasketReplayBank.sequences[i].maxDD >= 0.5*DangerReplayDDThresholdPct))
         s += 0.75;

      accDeep += w * Clamp(s, 0.0, 5.0);
      sumDeep += w;
   }
   if(sumDeep>1e-9) deepRiskOut = Clamp(accDeep / sumDeep, 0.0, 5.0);

   double sumEff=0.0, accEff=0.0;
   int ne=ArraySize(gEfficientReplayBank.items);
   int se=MathMax(0, ne-scan);
   for(int i=se;i<ne;i++)
   {
      double w = ReplayRewardContextWeight(gEfficientReplayBank.items[i], symIdx, basketDir, positionsCount, regimeType, structureType, liquidityType);
      if(w<=0.0) continue;
      accEff += w * ReplayRewardEfficientScore(gEfficientReplayBank.items[i]);
      sumEff += w;
   }
   int np=ArraySize(gEfficientReplayBank.periods);
   int sp=MathMax(0, np-scan);
   for(int i=sp;i<np;i++)
   {
      if(gEfficientReplayBank.periods[i].symIdx != symIdx) continue;
      double w = 0.18;
      if(gEfficientReplayBank.periods[i].regimeType == regimeType) w += 0.18;
      else if(MathAbs(gEfficientReplayBank.periods[i].regimeType - regimeType) == 1) w += 0.08;
      if(gEfficientReplayBank.periods[i].patternType == structureType) w += 0.14;
      if(gEfficientReplayBank.periods[i].liquidityType == liquidityType) w += 0.10;
      if(positionsCount <= 1) w += 0.10;
      w *= ReplayRewardRecencyWeight(gEfficientReplayBank.periods[i].endTime);

      double s = Clamp(gEfficientReplayBank.periods[i].rewardEfficiency, -2.0, 2.0);
      s += 0.20 * Clamp((double)gEfficientReplayBank.periods[i].oneRoundCount, 0.0, 6.0);
      s -= 0.35 * Clamp(gEfficientReplayBank.periods[i].ddMax / MathMax(EfficientPeriodMaxDDPct, 1e-6), 0.0, 4.0);
      s -= 0.15 * Clamp((double)gEfficientReplayBank.periods[i].addCountTotal, 0.0, 8.0);
      accEff += w * Clamp(s, -2.0, 3.0);
      sumEff += w;
   }
   if(sumEff>1e-9) efficientOut = Clamp(accEff / sumEff, -1.0, 3.0);

   double sumRecent=0.0, accRecent=0.0;
   int nr=ArraySize(gRecentReplayBank.items);
   int sr=MathMax(0, nr-scan);
   for(int i=sr;i<nr;i++)
   {
      double w = ReplayRewardContextWeight(gRecentReplayBank.items[i], symIdx, basketDir, positionsCount, regimeType, structureType, liquidityType);
      if(w<=0.0) continue;
      double c = 0.0;
      if(ReplayIsDangerAntiPattern(gRecentReplayBank.items[i])) c += 0.70 * ReplayRewardDangerScore(gRecentReplayBank.items[i]);
      if(ReplayIsDeepAntiPattern(gRecentReplayBank.items[i]))   c += 0.85 * ReplayRewardDeepScore(gRecentReplayBank.items[i]);
      if(ReplayIsEfficientCandidate(gRecentReplayBank.items[i])) c -= 0.45 * ReplayRewardEfficientScore(gRecentReplayBank.items[i]);
      accRecent += w * c;
      sumRecent += w;
   }
   if(sumRecent>1e-9) recentCautionOut = Clamp(accRecent / sumRecent, -1.5, 3.0);
}

string BuildReplayDiagnosticsText(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return "";
   double effProg = ReplayEfficiencyProgress();
   double antiP = ReplayAntiPatternPressure();
   double risk = CurrentReplayRiskBias(symIdx);
   return StringFormat("Replay main=%d recent=%d danger=%d deep=%d eff=%d seq=%d per=%d | new R/D/DP/E=%d/%d/%d/%d | prog=%.2f anti=%.2f risk=%.2f",
                       ArraySize(gReplay),
                       ArraySize(gRecentReplayBank.items),
                       ArraySize(gDangerReplayBank.items),
                       ArraySize(gDeepBasketReplayBank.items),
                       ArraySize(gEfficientReplayBank.items),
                       ArraySize(gDeepBasketReplayBank.sequences),
                       ArraySize(gEfficientReplayBank.periods),
                       gReplayDiagWindowRecentAdds[symIdx],
                       gReplayDiagWindowDangerAdds[symIdx],
                       gReplayDiagWindowDeepAdds[symIdx],
                       gReplayDiagWindowEfficientAdds[symIdx],
                       effProg, antiP, risk);
}

void MaybePrintReplayDiagnostics(const int symIdx)
{
   if(!UseReplayDiagnostics || !ReplayDiagnosticsToLog) return;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   gReplayDiagBarsSincePrint[symIdx]++;
   int every = MathMax(1, ReplayDiagnosticsPrintEveryBars);
   if(gReplayDiagBarsSincePrint[symIdx] < every) return;

   Print(gSymbols[symIdx], " | ", BuildReplayDiagnosticsText(symIdx));
   gReplayDiagBarsSincePrint[symIdx]=0;
   gReplayDiagWindowRecentAdds[symIdx]=0;
   gReplayDiagWindowDangerAdds[symIdx]=0;
   gReplayDiagWindowDeepAdds[symIdx]=0;
   gReplayDiagWindowEfficientAdds[symIdx]=0;
}

bool EfficientPeriodQualifies(const EfficientPeriodRef &pr)
{
   if(pr.rewardEfficiency < EfficientPeriodMinRewardEfficiency) return false;
   if(pr.ddMax > EfficientPeriodMaxDDPct) return false;
   if(pr.addCountTotal > MathMax(1, pr.oneRoundCount + 1)) return false;
   if(pr.oneRoundCount <= 0 && pr.addCountTotal > 0) return false;
   return true;
}

bool DeepBasketSequenceQualifies(const BasketSequenceRef &seq)
{
   bool riskyProfit = (seq.finalReward > 0.0 &&
                       (seq.maxDD >= 0.5*DangerReplayDDThresholdPct || seq.addCount >= 2 || seq.maxPositions > 2));
   if(seq.finalReward < 0.0) return true;
   if(seq.maxPositions > 2) return true;
   if(seq.addCount >= 1 && seq.maxDD >= MathMax(0.0125, 0.50*DangerReplayDDThresholdPct)) return true;
   if(riskyProfit) return true;
   return false;
}

int ChooseReplayBankType(const ReplayItem &item)
{
   if(ReplayIsDeepAntiPattern(item))
      return (int)REPLAY_BANK_DEEP_BASKET;
   if(ReplayIsDangerAntiPattern(item))
      return (int)REPLAY_BANK_DANGER;
   if(ReplayIsEfficientCandidate(item))
      return (int)REPLAY_BANK_EFFICIENT;
   if(item.reward < 0.0 && item.dangerClass >= 1)
      return (int)REPLAY_BANK_DANGER;
   return (int)REPLAY_BANK_RECENT;
}

void RouteReplayItemToBanks(ReplayItem &item,const int replayIdx)
{
   ReplayBankStorePush(gRecentReplayBank, item, RecentReplayCapacity);
   if(item.symIdx>=0 && item.symIdx<MAX_SYMBOLS)
      gReplayDiagWindowRecentAdds[item.symIdx]++;

   int bankType = ChooseReplayBankType(item);
   item.replayBankType = bankType;

   if(bankType == (int)REPLAY_BANK_DANGER && UseDangerReplayBank)
   {
      ReplayBankStorePush(gDangerReplayBank, item, DangerReplayCapacity);
      if(item.symIdx>=0 && item.symIdx<MAX_SYMBOLS) gReplayDiagWindowDangerAdds[item.symIdx]++;
   }
   else if(bankType == (int)REPLAY_BANK_DEEP_BASKET && UseDeepBasketReplayBank)
   {
      DeepBasketReplayStorePush(gDeepBasketReplayBank, item, DeepBasketReplayCapacity);
      if(item.symIdx>=0 && item.symIdx<MAX_SYMBOLS) gReplayDiagWindowDeepAdds[item.symIdx]++;
   }
   else if(bankType == (int)REPLAY_BANK_EFFICIENT && UseEfficientReplayBank)
   {
      EfficientReplayStorePush(gEfficientReplayBank, item, EfficientReplayCapacity);
      if(item.symIdx>=0 && item.symIdx<MAX_SYMBOLS) gReplayDiagWindowEfficientAdds[item.symIdx]++;
   }

   DangerBrainObserveReplayBankItem(item);

   if(item.episodeId>0)
      AppendReplayIndexToActiveEpisode(item.symIdx, replayIdx, item.reward);

   UpdateActiveEfficientPeriodWithItem(item, replayIdx);
}

void ReplayPush(const int symIdx,
                const int regime,
                const double &state[],
                const int action,
                const double reward,
                const double &nextState[],
                const bool done)
{
   if(!UseReplayBuffer) return;
   if(action<0 || action>=ActionCount) return;

   ReplayItem item;
   item.symIdx=symIdx;
   item.regime=regime;
   item.action=action;
   item.reward=reward;
   item.done=done;
   item.priority=MathAbs(reward) + ReplayPriorityReward(item.reward);
   if(item.priority<0.01) item.priority=0.01;

   ArrayResize(item.state,ArraySize(state));
   for(int i=0;i<ArraySize(state);i++) item.state[i]=state[i];

   ArrayResize(item.nextState,ArraySize(nextState));
   for(int i=0;i<ArraySize(nextState);i++) item.nextState[i]=nextState[i];

   FillReplayMetadata(item, symIdx, regime, state, action, reward, done);

   int n=ArraySize(gReplay);
   ArrayResize(gReplay,n+1);
   gReplay[n]=item;

   RouteReplayItemToBanks(gReplay[n], n);
   gReplayPendingTrainCount++;

   if(ArraySize(gReplay)>ReplayCapacity)
      ArrayRemove(gReplay,0,1);
}

double ReplayPriorityReward(const double reward)
{
   return ReplayPriorityRewardK * MathAbs(reward);
}

bool ReplaySampleIndex(int &idxOut)
{
   idxOut=-1;
   int n=ArraySize(gReplay);
   if(n<=0) return false;

   double sumP=0.0;
   for(int i=0;i<n;i++) sumP += MathMax(0.0001, gReplay[i].priority);
   if(sumP<=1e-12) return false;

   double r=((double)MathRand()/32767.0)*sumP;
   double c=0.0;
   for(int i=0;i<n;i++)
   {
      c += MathMax(0.0001, gReplay[i].priority);
      if(r<=c){ idxOut=i; return true; }
   }

   idxOut=n-1;
   return true;
}

double DangerReplaySampleWeight(const ReplayItem &it)
{
   double w = MathMax(0.0001, it.priority);
   if(it.reward < 0.0) w *= (1.0 + 0.20 * MathMin(-it.reward, 3.0));
   if(it.addDepthClass <= 1) w *= DangerReplayEarlyStateBoost;
   if(it.addDepthClass >= 3) w *= DangerReplayLateRescueWeight;
   if(ReplayIsRiskyProfit(it)) w *= 1.10;
   w *= (1.0 + 1.15*Clamp(it.painSeverity,0.0,1.0) + 0.45*Clamp(it.recurrenceScore,0.0,1.0) + 0.30*Clamp(it.regimeBreakScore,0.0,1.0));
   return w;
}

double DeepBasketSampleWeight(const ReplayItem &it)
{
   double w = MathMax(0.0001, it.priority);
   if(it.addDepthClass<=1 && it.episodeId>0) w *= DeepSequenceEarlyBoost;
   else if(it.addDepthClass>=3) w *= DeepReplayLateRescueWeight;
   if(it.reward < 0.0) w *= (1.0 + 0.15 * MathMin(-it.reward, 3.0));
   if(ReplayIsRiskyProfit(it)) w *= 1.15;
   if(it.done && it.reward > 0.0) w *= 0.85;
   w *= (1.0 + 1.00*Clamp(it.painSeverity,0.0,1.0) + 0.55*Clamp(it.recurrenceScore,0.0,1.0));
   return w;
}

double EfficientReplaySampleWeight(const ReplayItem &it)
{
   double w = MathMax(0.0001, it.priority);
   if(ReplayIsEfficientCandidate(it)) w *= 1.35;
   else if(it.reward > 0.0 && it.addDepthClass<=1 && it.dangerClass==0) w *= 1.10;
   if(it.dangerClass>0) w *= 0.65;
   if(it.reward<=0.0) w *= 0.25;
   return w;
}

bool ReplaySampleFromArrayWeighted(const ReplayItem &arr[], ReplayItem &itemOut, int &idxOut, const int source)
{
   idxOut=-1;
   int n=ArraySize(arr);
   if(n<=0) return false;

   double sumP=0.0;
   for(int i=0;i<n;i++)
   {
      double w=MathMax(0.0001, arr[i].priority);
      if(source==2) w = DangerReplaySampleWeight(arr[i]);
      else if(source==3) w = DeepBasketSampleWeight(arr[i]);
      else if(source==4) w = EfficientReplaySampleWeight(arr[i]);
      sumP += MathMax(0.0001, w);
   }
   if(sumP<=1e-12) return false;

   double r=((double)MathRand()/32767.0)*sumP;
   double c=0.0;
   for(int i=0;i<n;i++)
   {
      double w=MathMax(0.0001, arr[i].priority);
      if(source==2) w = DangerReplaySampleWeight(arr[i]);
      else if(source==3) w = DeepBasketSampleWeight(arr[i]);
      else if(source==4) w = EfficientReplaySampleWeight(arr[i]);
      c += MathMax(0.0001, w);
      if(r<=c){ idxOut=i; itemOut=arr[i]; return true; }
   }

   idxOut=n-1;
   itemOut=arr[n-1];
   return true;
}

enum ReplaySampleSource
{
   REPLAY_SRC_MAIN=0,
   REPLAY_SRC_RECENT=1,
   REPLAY_SRC_DANGER=2,
   REPLAY_SRC_DEEP=3,
   REPLAY_SRC_EFFICIENT=4
};

bool SampleReplayItemMixed(ReplayItem &itemOut,int &srcOut,int &idxOut)
{
   srcOut=REPLAY_SRC_MAIN;
   idxOut=-1;

   if(!UseBankAwareReplaySampling)
   {
      if(!ReplaySampleIndex(idxOut)) return false;
      itemOut=gReplay[idxOut];
      return true;
   }

   double wMain = MathMax(0.0, ReplaySampleWeightMain);
   double wRecent = (UseRecentReplayBank?MathMax(0.0, ReplaySampleWeightRecent):0.0);
   double wDanger = (UseDangerReplayBank?MathMax(0.0, ReplaySampleWeightDanger):0.0);
   double wDeep = (UseDeepBasketReplayBank?MathMax(0.0, ReplaySampleWeightDeepBasket):0.0);
   double wEff = (UseEfficientReplayBank?MathMax(0.0, ReplaySampleWeightEfficient):0.0);

   double replayRisk = 0.0;
   replayRisk = MathMax(replayRisk, SafeDiv(CurrentReplayDDPct(), MathMax(DangerReplayDDThresholdPct, 1e-6), 0.0));
   double dangerEMA = 0.0, deepEMA = 0.0;
   for(int s=0;s<gSymbolCount;s++)
   {
      dangerEMA = MathMax(dangerEMA, gDangerReplaySeverityEMA[s]);
      deepEMA   = MathMax(deepEMA,   gDeepBasketReplaySeverityEMA[s]);
   }
   replayRisk = MathMax(replayRisk, Clamp(dangerEMA / 3.0, 0.0, 2.0));
   replayRisk = MathMax(replayRisk, Clamp(deepEMA   / 3.0, 0.0, 2.0));

   if(replayRisk > 1.0)
   {
      double boost = Clamp(replayRisk - 1.0, 0.0, 1.0);
      wDanger *= (1.0 + 0.75 * boost);
      wDeep   *= (1.0 + 0.65 * boost);
      wEff    *= (1.0 - 0.30 * boost);
      wRecent *= (1.0 + 0.15 * boost);
      wMain   *= (1.0 - 0.10 * boost);
   }
   else
   {
      double calm = Clamp(1.0 - replayRisk, 0.0, 1.0);
      wEff    *= (1.0 + 0.50 * calm);
      wRecent *= (1.0 + 0.25 * calm);
   }

   double effProgress = ReplayEfficiencyProgress();
   double antiPressure = ReplayAntiPatternPressure();
   wEff    *= (1.0 + ReplayMatureEfficiencyBoost * effProgress);
   wRecent *= (1.0 + 0.20 * effProgress);
   wDanger *= MathMax(0.25, 1.0 - ReplayAntiPatternDecayScale * effProgress);
   wDeep   *= MathMax(0.25, 1.0 - ReplayAntiPatternDecayScale * effProgress);
   if(antiPressure > 1.0)
   {
      double p = Clamp(antiPressure - 1.0, 0.0, 1.5);
      wDanger *= (1.0 + 0.20 * p);
      wDeep   *= (1.0 + 0.25 * p);
   }

   bool okMain   = (ArraySize(gReplay)>0);
   bool okRecent = (ArraySize(gRecentReplayBank.items)>0);
   bool okDanger = (ArraySize(gDangerReplayBank.items)>0);
   bool okDeep   = (ArraySize(gDeepBasketReplayBank.items)>0);
   bool okEff    = (ArraySize(gEfficientReplayBank.items)>0);

   if(!okMain)   wMain=0.0;
   if(!okRecent) wRecent=0.0;
   if(!okDanger) wDanger=0.0;
   if(!okDeep)   wDeep=0.0;
   if(!okEff)    wEff=0.0;

   double wSum = wMain+wRecent+wDanger+wDeep+wEff;
   if(wSum<=1e-12)
   {
      if(!ReplaySampleIndex(idxOut)) return false;
      itemOut=gReplay[idxOut];
      srcOut=REPLAY_SRC_MAIN;
      return true;
   }

   double r=((double)MathRand()/32767.0)*wSum;
   double c=0.0;

   c += wMain;
   if(r<=c && okMain)
   {
      if(!ReplaySampleIndex(idxOut)) return false;
      itemOut=gReplay[idxOut];
      srcOut=REPLAY_SRC_MAIN;
      return true;
   }
   c += wRecent;
   if(r<=c && okRecent)
   {
      srcOut=REPLAY_SRC_RECENT;
      return ReplaySampleFromArrayWeighted(gRecentReplayBank.items, itemOut, idxOut, REPLAY_SRC_RECENT);
   }
   c += wDanger;
   if(r<=c && okDanger)
   {
      srcOut=REPLAY_SRC_DANGER;
      return ReplaySampleFromArrayWeighted(gDangerReplayBank.items, itemOut, idxOut, REPLAY_SRC_DANGER);
   }
   c += wDeep;
   if(r<=c && okDeep)
   {
      srcOut=REPLAY_SRC_DEEP;
      return ReplaySampleFromArrayWeighted(gDeepBasketReplayBank.items, itemOut, idxOut, REPLAY_SRC_DEEP);
   }
   if(okEff)
   {
      srcOut=REPLAY_SRC_EFFICIENT;
      return ReplaySampleFromArrayWeighted(gEfficientReplayBank.items, itemOut, idxOut, REPLAY_SRC_EFFICIENT);
   }

   if(!ReplaySampleIndex(idxOut)) return false;
   itemOut=gReplay[idxOut];
   srcOut=REPLAY_SRC_MAIN;
   return true;
}

void DangerBrainObserveReplayBankItem(const ReplayItem &item)
{
   if(!UseDangerBrainReplayHook) return;
   int s=item.symIdx;
   if(s<0 || s>=MAX_SYMBOLS) return;

   double sev = MathMax(MathAbs(item.reward), 0.10);
   sev *= (1.0 + 0.25*MathMax(0,item.dangerClass) + 0.10*MathMax(0,item.addDepthClass));
   sev *= (1.0 + 0.85*Clamp(item.painSeverity,0.0,1.0) + 0.35*Clamp(item.recurrenceScore,0.0,1.0));

   if(item.replayBankType == (int)REPLAY_BANK_DANGER)
   {
      gDangerReplaySeverityEMA[s] = (1.0-DangerReplayObsAlpha)*gDangerReplaySeverityEMA[s] + DangerReplayObsAlpha*sev;
      gDangerReplayHits[s]++;
      if(item.done && item.reward < 0.0 && gBadLabel[s] >= 0)
         LearnBadEpisodeSmart(s,false);
   }
   else if(item.replayBankType == (int)REPLAY_BANK_DEEP_BASKET)
   {
      gDeepBasketReplaySeverityEMA[s] = (1.0-DangerReplayObsAlpha)*gDeepBasketReplaySeverityEMA[s] + DangerReplayObsAlpha*sev;
      gDeepBasketReplayHits[s]++;
      if(item.done && item.reward < 0.0 && gBadLabel[s] >= 0)
         LearnBadEpisodeSmart(s,false);
   }
   else if(item.replayBankType == (int)REPLAY_BANK_EFFICIENT)
   {
      gDangerReplaySeverityEMA[s]     *= (1.0 - 0.50*DangerReplayObsAlpha);
      gDeepBasketReplaySeverityEMA[s] *= (1.0 - 0.55*DangerReplayObsAlpha);
      if(gDangerReplayHits[s] > 0) gDangerReplayHits[s]--;
      if(gDeepBasketReplayHits[s] > 0) gDeepBasketReplayHits[s]--;
   }
}


double QMemPositiveBoostScale = 1.00;
double QMemNegativePenaltyScale = 1.00;
bool   QMemUseNegativeUpdates = true;
double ReplayPriorityRewardK = 0.25;

//  Next part starts at: double CombinedSim(...)
double CombinedSim(const int symIdx,
                   const double &fp_now[], const double &mini_now[],
                   const double &dfp_now[], const bool haveDFP,
                   const double &dmini_now[], const bool haveDMini,
                   const ProtoEntry &p)
{
   double s_fp = CosSim(fp_now, p.features);
   double s = s_fp;

   if(UseMiniStateFingerprint && gMiniValid[symIdx] && ArraySize(p.stateMini)==MINI_DIM)
   {
      double s_m = CosSim(mini_now, p.stateMini);
      double w1 = MathMax(0.0, SimW_PriceFingerprint);
      double w2 = MathMax(0.0, SimW_MiniState);
      double den = w1+w2;
      if(den>1e-9) s = (w1*s_fp + w2*s_m)/den;
   }

   if(UseDeltaSimilarity && haveDFP && ArraySize(p.deltaFP)==6)
   {
      double s_dfp = CosSim(dfp_now, p.deltaFP);
      double w = MathMax(0.0, SimW_DeltaFingerprint);
      s += w * s_dfp;
   }
   if(UseDeltaSimilarity && haveDMini && ArraySize(p.deltaMini)==MINI_DIM)
   {
      double s_dm = CosSim(dmini_now, p.deltaMini);
      double w = MathMax(0.0, SimW_DeltaMiniState);
      s += w * s_dm;
   }

   double ageF = ProtoAgeFactor(p);
   s *= ageF;

   double u = (double)p.usedCount;
   if(u>0.0) s += 0.02 * MathLog(1.0 + u);

   return Clamp(s, -2.0, 2.0);
}

double ComputeExposureScore(const int symIdx)
{
   double eq=GetEAEquity();
   double dd=0.0;
   if(maxEquity>1e-9 && eq<maxEquity) dd=Clamp((maxEquity-eq)/maxEquity,0.0,1.0);

   double s=0.0;
   s += ExposureW_Positions * gPositionsCount[symIdx];
   s += ExposureW_DD * dd;
   return s;
}

int DangerPredict(const double &f_now[], const int symIdx, double &P_danger)
{
   double mini_now[];
   ArrayResize(mini_now, MINI_DIM);
   for(int k=0;k<MINI_DIM;k++) mini_now[k]=gMiniCache[symIdx][k];

   double dfp_now[];
   bool haveDFP=false;
   if(gFPPrevValid[symIdx])
   {
      double tmpCur[]; ArrayResize(tmpCur,6);
      double tmpPrev[]; ArrayResize(tmpPrev,6);
      for(int k=0;k<6;k++){ tmpCur[k]=f_now[k]; tmpPrev[k]=gFPPrev[symIdx][k]; }
      haveDFP = BuildDeltaVec(tmpCur, tmpPrev, 6, dfp_now);
   }

   double dmini_now[];
   bool haveDMini=false;
   if(UseMiniStateFingerprint && gMiniPrevValid[symIdx] && gMiniValid[symIdx])
   {
      double cur[]; ArrayResize(cur,MINI_DIM);
      double prev[]; ArrayResize(prev,MINI_DIM);
      for(int k=0;k<MINI_DIM;k++){ cur[k]=gMiniCache[symIdx][k]; prev[k]=gMiniPrev[symIdx][k]; }
      haveDMini = BuildDeltaVec(cur, prev, MINI_DIM, dmini_now);
   }

   double bestDanger=-1e9, bestSafe=-1e9;
   int bestDangerIdx=-1;

   int n=ArraySize(gProtos);
   for(int i=0;i<n;i++)
   {
      if(ArraySize(gProtos[i].features)!=ArraySize(f_now)) continue;
      double sim = CombinedSim(symIdx, f_now, mini_now, dfp_now, haveDFP, dmini_now, haveDMini, gProtos[i])
                  + RegimeContextSimilarityBonus(GetAtrRatioCached_Base(gSymbols[symIdx]), gProtos[i].atrRatio);
      if(gProtos[i].isDanger)
      {
         if(sim>bestDanger){ bestDanger=sim; bestDangerIdx=i; }
      }
      else bestSafe=MathMax(bestSafe, sim);
   }

   double simContrast=(bestDanger-bestSafe);
   double exposure=ComputeExposureScore(symIdx);
   double raw = ScoreW_SimContrast*simContrast + ScoreW_Exposure*exposure;

   P_danger=Sigmoid(raw);
   return bestDangerIdx;
}

void UpdateMode(const int symIdx, const double P_danger, const datetime now)
{
   BrainMode m=gMode[symIdx];
   datetime last=gModeLastChange[symIdx];
   bool cooldownOk=(last==0) || ((now-last) >= ModeCooldownMinutes*60);

   if(m==MODE_NORMAL)
   {
      if(P_danger>=gT2Enter){ gMode[symIdx]=MODE_DANGER; gModeLastChange[symIdx]=now; }
      else if(P_danger>=gT1Enter){ gMode[symIdx]=MODE_CAUTION; gModeLastChange[symIdx]=now; }
   }
   else if(m==MODE_CAUTION)
   {
      if(P_danger>=gT2Enter){ gMode[symIdx]=MODE_DANGER; gModeLastChange[symIdx]=now; }
      else if(P_danger<=gT1Exit){ gMode[symIdx]=MODE_NORMAL; gModeLastChange[symIdx]=now; }
   }
   else
   {
      if(cooldownOk && P_danger<=gT2Exit)
      {
         gMode[symIdx]=MODE_CAUTION;
         gModeLastChange[symIdx]=now;
      }
   }

   if(gMode[symIdx]==MODE_DANGER) gLRScale[symIdx]=LRScale_Danger;
   else if(gMode[symIdx]==MODE_CAUTION) gLRScale[symIdx]=LRScale_Caution;
   else gLRScale[symIdx]=1.0;
}

bool GetBestAdapterBiasSmart(const int symIdx, const double &f_now[], double &outBias[], double &simBest)
{
   simBest=-1e9;
   int best=-1;

   double mini_now[];
   ArrayResize(mini_now, MINI_DIM);
   for(int k=0;k<MINI_DIM;k++) mini_now[k]=gMiniCache[symIdx][k];

   double dfp_now[];
   bool haveDFP=false;
   if(gFPPrevValid[symIdx])
   {
      double tmpCur[]; ArrayResize(tmpCur,6);
      double tmpPrev[]; ArrayResize(tmpPrev,6);
      for(int k=0;k<6;k++){ tmpCur[k]=f_now[k]; tmpPrev[k]=gFPPrev[symIdx][k]; }
      haveDFP = BuildDeltaVec(tmpCur, tmpPrev, 6, dfp_now);
   }

   double dmini_now[];
   bool haveDMini=false;
   if(UseMiniStateFingerprint && gMiniPrevValid[symIdx] && gMiniValid[symIdx])
   {
      double cur[]; ArrayResize(cur,MINI_DIM);
      double prev[]; ArrayResize(prev,MINI_DIM);
      for(int k=0;k<MINI_DIM;k++){ cur[k]=gMiniCache[symIdx][k]; prev[k]=gMiniPrev[symIdx][k]; }
      haveDMini = BuildDeltaVec(cur, prev, MINI_DIM, dmini_now);
   }

   int n=ArraySize(gProtos);
   for(int i=0;i<n;i++)
   {
      if(!gProtos[i].isDanger) continue;
      if(ArraySize(gProtos[i].features)!=ArraySize(f_now)) continue;

      double sim = CombinedSim(symIdx, f_now, mini_now, dfp_now, haveDFP, dmini_now, haveDMini, gProtos[i])
                  + RegimeContextSimilarityBonus(GetAtrRatioCached_Base(gSymbols[symIdx]), gProtos[i].atrRatio);

      double bonus=0.0;
      if(gPositionsCount[symIdx]>0)
      {
         int curLabel  = gBadLabel[symIdx];
         int curBasket = gBadBasketDir[symIdx];
         int curTrend  = gBadTrendDir[symIdx];

         if(curLabel>=0 && gProtos[i].label==curLabel) bonus += 0.05;
         if(curBasket!=0 && gProtos[i].basketDir==curBasket) bonus += 0.03;
         if(curTrend!=0  && gProtos[i].trendDir==curTrend)   bonus += 0.03;
      }

      double score = sim + bonus;
      if(score>simBest){ simBest=score; best=i; }
   }
   if(best<0) return false;

   int aN=ArraySize(gProtos[best].adapterBias);
   if(aN<=0) return false;

   ArrayResize(outBias,aN);
   for(int a=0;a<aN;a++) outBias[a]=gProtos[best].adapterBias[a];

   gProtos[best].usedCount++;
   gProtos[best].lastUsed=TimeCurrent();
   return true;
}

double ComputeAlphaMix(const int symIdx, const double simBest)
{
   double p=gPDanger[symIdx];
   double sim=MathMax(0.0, MathMin(1.0, (simBest+1.0)*0.5));
   double a=(p - gT1Enter) / MathMax(1e-6, (1.0 - gT1Enter));
   a=Clamp(a,0.0,1.0);
   return a*sim;
}

bool SaveDangerMemoryFile(const string filename)
{
   int h=FileOpen(filename, FILE_WRITE|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   FileWriteInteger(h, 5);
   int n=ArraySize(gProtos);
   FileWriteInteger(h, n);

   for(int i=0;i<n;i++)
   {
      FileWriteInteger(h, gProtos[i].label);
      FileWriteInteger(h, gProtos[i].basketDir);
      FileWriteInteger(h, gProtos[i].trendDir);
      FileWriteDouble (h, gProtos[i].atrRatio);
      FileWriteLong   (h, (long)gProtos[i].created);

      FileWriteInteger(h, gProtos[i].isDanger ? 1 : 0);

      WriteDoubleArray(h, gProtos[i].features);
      WriteDoubleArray(h, gProtos[i].stateMini);
      WriteDoubleArray(h, gProtos[i].deltaFP);
      WriteDoubleArray(h, gProtos[i].deltaMini);
      WriteDoubleArray(h, gProtos[i].adapterBias);
      WriteDoubleArray(h, gProtos[i].qSnap);
      FileWriteDouble(h, gProtos[i].painMean);
      FileWriteDouble(h, gProtos[i].painCount);
      WriteDoubleArray(h, gProtos[i].macroSig);
      WriteDoubleArray(h, gProtos[i].microSig);
      FileWriteDouble(h, gProtos[i].deepBasketRate);
      FileWriteDouble(h, gProtos[i].regimeBreakRate);
      FileWriteDouble(h, gProtos[i].counterTrendFailureRate);
      FileWriteDouble(h, gProtos[i].reversalTrapRate);
      FileWriteDouble(h, gProtos[i].recoveryFailureRate);

      FileWriteDouble(h, gProtos[i].survivalScore);
      FileWriteInteger(h, (int)gProtos[i].usedCount);
      FileWriteLong(h, (long)gProtos[i].lastUsed);
   }

   FileClose(h);
   return true;
}

bool LoadDangerMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename, FILE_READ|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   int ver=FileReadInteger(h);
   if(ver!=1 && ver!=2 && ver!=3 && ver!=4 && ver!=5){ FileClose(h); return false; }

   int n=FileReadInteger(h);
   if(n<0 || n>50000){ FileClose(h); return false; }

   ArrayResize(gProtos,0);
   ArrayResize(gProtos,n);

   for(int i=0;i<n;i++)
   {
      if(ver>=2)
      {
         gProtos[i].label     = FileReadInteger(h);
         gProtos[i].basketDir = FileReadInteger(h);
         gProtos[i].trendDir  = FileReadInteger(h);
         gProtos[i].atrRatio  = FileReadDouble(h);
         gProtos[i].created   = (datetime)FileReadLong(h);
      }
      else
      {
         int oldType=FileReadInteger(h);
         gProtos[i].label=oldType;
         gProtos[i].basketDir=0;
         gProtos[i].trendDir=0;
         gProtos[i].atrRatio=1.0;
         gProtos[i].created=0;
      }

      gProtos[i].isDanger = (FileReadInteger(h)==1);

      ReadDoubleArray(h, gProtos[i].features);

      if(ver>=3)
      {
         ReadDoubleArray(h, gProtos[i].stateMini);
      }
      else ArrayResize(gProtos[i].stateMini, 0);

      if(ver>=4)
      {
         ReadDoubleArray(h, gProtos[i].deltaFP);
         ReadDoubleArray(h, gProtos[i].deltaMini);
      }
      else
      {
         ArrayResize(gProtos[i].deltaFP, 0);
         ArrayResize(gProtos[i].deltaMini, 0);
      }

      ReadDoubleArray(h, gProtos[i].adapterBias);

      if(ver>=2)
      {
         ReadDoubleArray(h, gProtos[i].qSnap);
      }
      else ArrayResize(gProtos[i].qSnap,0);

      if(ver>=5)
      {
         gProtos[i].painMean = FileReadDouble(h);
         gProtos[i].painCount = FileReadDouble(h);
         ReadDoubleArray(h, gProtos[i].macroSig);
         ReadDoubleArray(h, gProtos[i].microSig);
         gProtos[i].deepBasketRate = FileReadDouble(h);
         gProtos[i].regimeBreakRate = FileReadDouble(h);
         gProtos[i].counterTrendFailureRate = FileReadDouble(h);
         gProtos[i].reversalTrapRate = FileReadDouble(h);
         gProtos[i].recoveryFailureRate = FileReadDouble(h);
      }
      else
      {
         gProtos[i].painMean = 0.0;
         gProtos[i].painCount = 0.0;
         ArrayResize(gProtos[i].macroSig,0);
         ArrayResize(gProtos[i].microSig,0);
         gProtos[i].deepBasketRate = 0.0;
         gProtos[i].regimeBreakRate = 0.0;
         gProtos[i].counterTrendFailureRate = 0.0;
         gProtos[i].reversalTrapRate = 0.0;
         gProtos[i].recoveryFailureRate = 0.0;
      }

      gProtos[i].survivalScore = FileReadDouble(h);
      gProtos[i].usedCount = (uint)FileReadInteger(h);
      gProtos[i].lastUsed = (datetime)FileReadLong(h);
   }

   FileClose(h);
   return true;
}

void SeedTestProtos()
{
   if(ArraySize(gProtos)>0) return;

   ProtoEntry p;
   p.label=LBL_AGAINST_UPTREND;
   p.basketDir=-1;
   p.trendDir=+1;
   p.atrRatio=1.0;
   p.created=TimeCurrent();
   p.isDanger=true;

   ArrayResize(p.features,6);
   p.features[0]=0.6; p.features[1]=0.6; p.features[2]=0.1; p.features[3]=0.2; p.features[4]=0.1; p.features[5]=0.4;
   NormalizeVec(p.features);

   ArrayResize(p.stateMini,0);
   ArrayResize(p.deltaFP,0);
   ArrayResize(p.deltaMini,0);

   ArrayResize(p.adapterBias, ActionCount);
   for(int i=0;i<ActionCount;i++) p.adapterBias[i]=0.0;
   if(ActionCount>=1) p.adapterBias[0]= +0.4;
   if(ActionCount>=2) p.adapterBias[1]= -0.2;
   if(ActionCount>=3) p.adapterBias[2]= -0.6;

   ArrayResize(p.qSnap,0);
   p.survivalScore=1.0; p.usedCount=0; p.lastUsed=0;

   int n=ArraySize(gProtos);
   ArrayResize(gProtos,n+1); gProtos[n]=p;

   p.isDanger=false;
   p.label=LBL_AGAINST_DOWNTREND;
   p.basketDir=+1;
   p.trendDir=-1;
   if(ActionCount>=1) p.adapterBias[0]=0.0;
   if(ActionCount>=2) p.adapterBias[1]=0.0;
   if(ActionCount>=3) p.adapterBias[2]=0.0;
   n=ArraySize(gProtos);
   ArrayResize(gProtos,n+1); gProtos[n]=p;
}

bool SaveQMemoryFile(const string filename)
{
   int h=FileOpen(filename, FILE_WRITE|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   FileWriteInteger(h, 2);
   int n=ArraySize(gQMem);
   FileWriteInteger(h, n);

   for(int i=0;i<n;i++)
   {
      FileWriteInteger(h, gQMem[i].regime);
      FileWriteLong   (h, (long)gQMem[i].created);
      FileWriteLong   (h, (long)gQMem[i].lastUsed);
      FileWriteInteger(h, (int)gQMem[i].usedCount);

      WriteDoubleArray(h, gQMem[i].stateKey);
      WriteDoubleArray(h, gQMem[i].qVals);

      FileWriteDouble(h, gQMem[i].conf);
      FileWriteDouble(h, gQMem[i].score);
   }

   FileClose(h);
   return true;
}

bool LoadQMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;

   int h=FileOpen(filename, FILE_READ|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   int ver=FileReadInteger(h);
   if(ver!=1){ FileClose(h); return false; }

   int n=FileReadInteger(h);
   if(n<0 || n>200000){ FileClose(h); return false; }

   ArrayResize(gQMem,0);
   ArrayResize(gQMem,n);

   for(int i=0;i<n;i++)
   {
      gQMem[i].regime   = FileReadInteger(h);
      gQMem[i].created  = (datetime)FileReadLong(h);
      gQMem[i].lastUsed = (datetime)FileReadLong(h);
      gQMem[i].usedCount= (uint)FileReadInteger(h);

      ReadDoubleArray(h, gQMem[i].stateKey);
      ReadDoubleArray(h, gQMem[i].qVals);

      gQMem[i].conf  = FileReadDouble(h);
      gQMem[i].score = FileReadDouble(h);
   }

   FileClose(h);
   PruneQMemoryIfNeeded();
   return true;
}

bool SaveDDEventMemoryFile(const string filename)
{
   int h=FileOpen(filename, FILE_WRITE|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   FileWriteInteger(h, 1);

   int n=ArraySize(gDDEvents);
   FileWriteInteger(h, n);

   for(int i=0;i<n;i++)
   {
      FileWriteLong(h, (long)gDDEvents[i].created);
      FileWriteLong(h, (long)gDDEvents[i].triggerTime);
      FileWriteString(h, gDDEvents[i].symbol);
      FileWriteInteger(h, gDDEvents[i].magic);
      FileWriteInteger(h, gDDEvents[i].regimeAtTrigger);
      FileWriteInteger(h, gDDEvents[i].basketDirAtTrigger);
      FileWriteDouble(h, gDDEvents[i].ddAtTrigger);
      FileWriteInteger(h, gDDEvents[i].hardTrigger ? 1 : 0);
      FileWriteInteger(h, gDDEvents[i].completed ? 1 : 0);

      WriteDoubleArray(h, gDDEvents[i].triggerStateKey);
      WriteDoubleArray(h, gDDEvents[i].triggerQVals);
      FileWriteDouble(h, gDDEvents[i].peakDD);
      FileWriteInteger(h, gDDEvents[i].basketDepthMax);
      FileWriteDouble(h, gDDEvents[i].timeUnderWaterNorm);
      FileWriteDouble(h, gDDEvents[i].recoveryFailureScore);
      FileWriteDouble(h, gDDEvents[i].painSeverity);
      FileWriteInteger(h, gDDEvents[i].eventType);
      WriteDoubleArray(h, gDDEvents[i].macroSig);
      WriteDoubleArray(h, gDDEvents[i].microSig);

      int preN=ArraySize(gDDEvents[i].preBaskets);
      FileWriteInteger(h, preN);
      for(int b=0;b<preN;b++)
         WriteBasketSnapshot(h, gDDEvents[i].preBaskets[b]);

      int postN=ArraySize(gDDEvents[i].postBaskets);
      FileWriteInteger(h, postN);
      for(int b=0;b<postN;b++)
         WriteBasketSnapshot(h, gDDEvents[i].postBaskets[b]);

      int tickN=ArraySize(gDDEvents[i].ticks);
      FileWriteInteger(h, tickN);
      for(int t=0;t<tickN;t++)
         WriteTickTraceItem(h, gDDEvents[i].ticks[t]);
   }

   FileClose(h);
   return true;
}

bool LoadDDEventMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;

   int h=FileOpen(filename, FILE_READ|FILE_BIN);
   if(h==INVALID_HANDLE) return false;

   int ver=FileReadInteger(h);
   if(ver!=1){ FileClose(h); return false; }

   int n=FileReadInteger(h);
   if(n<0 || n>10000){ FileClose(h); return false; }

   ArrayResize(gDDEvents, n);

   for(int i=0;i<n;i++)
   {
      gDDEvents[i].created = (datetime)FileReadLong(h);
      gDDEvents[i].triggerTime = (datetime)FileReadLong(h);
      gDDEvents[i].symbol = FileReadString(h);
      gDDEvents[i].magic = FileReadInteger(h);
      gDDEvents[i].regimeAtTrigger = FileReadInteger(h);
      gDDEvents[i].basketDirAtTrigger = FileReadInteger(h);
      gDDEvents[i].ddAtTrigger = FileReadDouble(h);
      gDDEvents[i].hardTrigger = (FileReadInteger(h)==1);
      gDDEvents[i].completed = (FileReadInteger(h)==1);

      ReadDoubleArray(h, gDDEvents[i].triggerStateKey);
      ReadDoubleArray(h, gDDEvents[i].triggerQVals);

      int preN=FileReadInteger(h);
      ArrayResize(gDDEvents[i].preBaskets, preN);
      for(int b=0;b<preN;b++)
         ReadBasketSnapshot(h, gDDEvents[i].preBaskets[b]);

      int postN=FileReadInteger(h);
      ArrayResize(gDDEvents[i].postBaskets, postN);
      for(int b=0;b<postN;b++)
         ReadBasketSnapshot(h, gDDEvents[i].postBaskets[b]);

      int tickN=FileReadInteger(h);
      ArrayResize(gDDEvents[i].ticks, tickN);
      for(int t=0;t<tickN;t++)
         ReadTickTraceItem(h, gDDEvents[i].ticks[t]);
   }

   FileClose(h);
   return true;
}

