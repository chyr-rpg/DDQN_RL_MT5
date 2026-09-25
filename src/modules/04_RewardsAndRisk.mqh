//+------------------------------------------------------------------+
//| 04_RewardsAndRisk.mqh                                           |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Reward shaping, basket-health feedback, archive-aware reward con|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

double GetEAEquity();

bool AllowLearnEntryOrAveraging(const int symIdx)
{
   if(!UseDangerBrain) return true;
   if(gMode[symIdx]==MODE_DANGER) return false;
   return true;
}
bool AllowLearnClose(const int symIdx){ return true; }

double ProfitReturnRewardForSymbol(const int symIdx,const double closedProfit)
{
   double denom = GetSymbolVirtualBudgetBase(symIdx);
   if(denom<=1e-9) denom=1000.0;

   double r = closedProfit / denom;
   r = Clamp(r, -ProfitReturnClamp, +ProfitReturnClamp);
   return r * ProfitRewardScale;
}

double RecoveryQualityRewardFromCounts(const int wins, const int losses, const int total)
{
   if(total<=0) return 0.0;
   double winRatio = (double)wins/(double)total;

   double r=0.0;
   r += RecWinRatioScale * winRatio;
   r -= RecLossCountPenalty * (double)losses;
   r -= RecTradesUsedPenalty * (double)total;

   return Clamp(r, -RecQualityClamp, +RecQualityClamp);
}

double CurrentMarginStressRatio()
{
   double budgetBase = (EquityBudget>1e-9 ? EquityBudget : gEAStartEquity);
   if(budgetBase<=1e-9) budgetBase = 1000.0;
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double freeRatio = Clamp(SafeDiv(freeMargin, budgetBase, 0.0), 0.0, 2.0);
   return Clamp(1.0 - freeRatio, 0.0, 1.0);
}

double CurrentReplayRiskBias(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   double danger = Clamp(gDangerReplaySeverityEMA[symIdx], 0.0, 5.0) / 5.0;
   double deep   = Clamp(gDeepBasketReplaySeverityEMA[symIdx], 0.0, 5.0) / 5.0;
   return Clamp(0.6*danger + 0.4*deep, 0.0, 1.0);
}

double BasketAgeDays(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   if(gFirstTradeTime[symIdx] <= 0) return 0.0;
   double ageSec = (double)(TimeCurrent() - gFirstTradeTime[symIdx]);
   if(ageSec <= 0.0) return 0.0;
   return ageSec / 86400.0;
}

double NormalizeMoneyReturnToBudget(const double money,const double clampAbs)
{
   double denom = (EquityBudget>1e-9 ? EquityBudget : gEAStartEquity);
   if(denom<=1e-9) denom=1000.0;

   double lim = MathMax(clampAbs, 1e-6);
   return Clamp(money / denom, -lim, +lim);
}

double GetSymbolVirtualBudgetBase(const int symIdx)
{
   double denom = (EquityBudget>1e-9 ? EquityBudget : gEAStartEquity);
   if(denom<=1e-9) denom=1000.0;

   if(UsePerSymbolVirtualBudget)
   {
      int n=MathMax(1, gSymbolCount);
      denom /= (double)n;
   }

   return MathMax(denom, 100.0);
}

double GetSymbolFloatingDDPct(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   string symbol = gSymbols[symIdx];
   int magic = gMagics[symIdx];
   double ddMoney = GetSymbolFloatingLossMoney(symbol, magic);
   double base = GetSymbolVirtualBudgetBase(symIdx);
   return Clamp(SafeDiv(ddMoney, base, 0.0), 0.0, 1.0);
}

double GetSymbolBasketOpenPnL(const string symbol,const int magic)
{
   double totalPnL=0.0;
   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      totalPnL += PositionGetDouble(POSITION_PROFIT);
   }
   return totalPnL;
}

void ResetDenseBasketHealthTracker(const int symIdx)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   gDenseBasketHealth[symIdx].active=false;
   gDenseBasketHealth[symIdx].lastBarTime=0;
   gDenseBasketHealth[symIdx].prevOpenReturn=0.0;
   gDenseBasketHealth[symIdx].prevDD=0.0;
   gDenseBasketHealth[symIdx].prevPositions=0;
   gDenseBasketHealth[symIdx].prevAgeDays=0.0;
}

void PendingAccrueDenseRewardForSymbol(const int symIdx,const double reward)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(MathAbs(reward) <= 1e-12) return;

   for(int i=0;i<ArraySize(gPending);i++)
   {
      if(!gPending[i].active) continue;
      if(gPending[i].symIdx!=symIdx) continue;
      gPending[i].denseRewardAccum += reward;
   }
}

double ComputeDenseBasketHealthReward(const double prevOpenReturn,
                                      const double prevDD,
                                      const int prevPositions,
                                      const double prevAgeDays,
                                      const double curOpenReturn,
                                      const double curDD,
                                      const int curPositions,
                                      const double curAgeDays,
                                      const bool extreme)
{
   double reward=0.0;
   double pnlDelta = curOpenReturn - prevOpenReturn;
   reward += DenseBasketPnLDeltaScale * pnlDelta;

   double ddDelta = Clamp(curDD - prevDD, -1.0, 1.0);
   if(ddDelta > 0.0)
      reward -= DenseBasketDDDeltaPenaltyScale * ddDelta;
   else
      reward += DenseBasketDDRecoveryScale * (-ddDelta);

   int posDelta = MathMax(curPositions - prevPositions, 0);
   if(posDelta > 0)
      reward -= DenseBasketAddPenaltyScale * (double)(posDelta * posDelta);

   double ageDelta = MathMax(curAgeDays - prevAgeDays, 0.0);
   if(ageDelta > 0.0 && pnlDelta <= 0.0)
      reward -= DenseBasketStallAgePenaltyScale * ageDelta;

   if(curPositions <= 1 && pnlDelta > 0.0 && ddDelta <= 0.0)
   {
      double progress = Clamp(pnlDelta / MathMax(DenseBasketOpenReturnClamp, 1e-6), 0.0, 1.0);
      reward += DenseBasketOneRoundProgressScale * progress;
   }

   reward = ScaleRewardByRegime(reward, extreme);
   return Clamp(reward, -DenseBasketRewardClamp, DenseBasketRewardClamp);
}

void AccrueDenseBasketHealthFeedback(const string symbol,
                                     const int symIdx,
                                     const int magic,
                                     const int positionsCount,
                                     const bool extreme)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;

   if(!UseDenseBasketHealthReward || positionsCount<=0)
   {
      ResetDenseBasketHealthTracker(symIdx);
      return;
   }

   datetime barTime=iTime(symbol, BaseTF, 1);
   if(barTime<=0) return;

   double curOpenPnL = GetSymbolBasketOpenPnL(symbol, magic);
   double curOpenReturn = NormalizeMoneyReturnToBudget(curOpenPnL, DenseBasketOpenReturnClamp);
   double curDD = GetSymbolFloatingDDPct(symIdx);
   double curAgeDays = BasketAgeDays(symIdx);

   if(!gDenseBasketHealth[symIdx].active || gDenseBasketHealth[symIdx].lastBarTime<=0)
   {
      gDenseBasketHealth[symIdx].active=true;
      gDenseBasketHealth[symIdx].lastBarTime=barTime;
      gDenseBasketHealth[symIdx].prevOpenReturn=curOpenReturn;
      gDenseBasketHealth[symIdx].prevDD=curDD;
      gDenseBasketHealth[symIdx].prevPositions=positionsCount;
      gDenseBasketHealth[symIdx].prevAgeDays=curAgeDays;
      return;
   }

   if(gDenseBasketHealth[symIdx].lastBarTime==barTime) return;

   double reward = ComputeDenseBasketHealthReward(gDenseBasketHealth[symIdx].prevOpenReturn,
                                                  gDenseBasketHealth[symIdx].prevDD,
                                                  gDenseBasketHealth[symIdx].prevPositions,
                                                  gDenseBasketHealth[symIdx].prevAgeDays,
                                                  curOpenReturn,
                                                  curDD,
                                                  positionsCount,
                                                  curAgeDays,
                                                  extreme);

   int basketDir = 0;
   if(gActiveBasketEpisodes[symIdx].active)
      basketDir = gActiveBasketEpisodes[symIdx].basketDir;

   double replayDangerRisk=0.0, replayDeepRisk=0.0, replayEfficient=0.0, replayRecentCaution=0.0;
   ComputeReplayRewardProfile(symIdx, basketDir, positionsCount, replayDangerRisk, replayDeepRisk, replayEfficient, replayRecentCaution);

   reward -= RewardV2ReplayDenseAntiPatternScale *
             (0.45 * replayDangerRisk + 0.75 * replayDeepRisk + 0.20 * MathMax(0.0, replayRecentCaution));

   if(positionsCount <= 1 && curOpenReturn >= gDenseBasketHealth[symIdx].prevOpenReturn && curDD <= gDenseBasketHealth[symIdx].prevDD)
      reward += 0.20 * RewardV2ReplayEfficientBonusScale * Clamp(replayEfficient, 0.0, 2.0);

   reward = Clamp(reward, -DenseBasketRewardClamp, DenseBasketRewardClamp);

   if(MathAbs(reward) > 1e-10)
   {
      PendingAccrueDenseRewardForSymbol(symIdx, reward);
      if(gActiveBasketEpisodes[symIdx].active)
      {
         gActiveBasketEpisodes[symIdx].rewardAccum += reward;
         gActiveBasketEpisodes[symIdx].maxDD=MathMax(gActiveBasketEpisodes[symIdx].maxDD, curDD);
         gActiveBasketEpisodes[symIdx].lastTime=TimeCurrent();
      }
   }

   gDenseBasketHealth[symIdx].active=true;
   gDenseBasketHealth[symIdx].lastBarTime=barTime;
   gDenseBasketHealth[symIdx].prevOpenReturn=curOpenReturn;
   gDenseBasketHealth[symIdx].prevDD=curDD;
   gDenseBasketHealth[symIdx].prevPositions=positionsCount;
   gDenseBasketHealth[symIdx].prevAgeDays=curAgeDays;
}

double SymbolPipSize(const string symbol)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   int digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS);
   if(digits==3 || digits==5) return point*10.0;
   return point;
}

double PipsToPriceDistance(const string symbol,const double pips)
{
   return MathMax(pips,0.0) * SymbolPipSize(symbol);
}

bool GetLatestBasketEntryInfo(const string symbol,
                              const int magic,
                              const int basketDir,
                              double &entryPrice,
                              datetime &entryTime)
{
   entryPrice=0.0;
   entryTime=0;
   if(basketDir==0) return false;

   int desiredType=(basketDir>0 ? POSITION_TYPE_BUY : POSITION_TYPE_SELL);
   bool found=false;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      if((int)PositionGetInteger(POSITION_TYPE)!=desiredType) continue;

      datetime t=(datetime)PositionGetInteger(POSITION_TIME);
      double px=PositionGetDouble(POSITION_PRICE_OPEN);

      if(!found || t>entryTime)
      {
         found=true;
         entryTime=t;
         entryPrice=px;
      }
   }

   return found;
}

double ComputeEffectiveMinAddSpacing(const string symbol,const double refPrice)
{
   double fixedDist = PipsToPriceDistance(symbol, MinAddSpacingPips);
   double atr = GetATRValueTF(symbol, BaseTF, 14, 1);
   double atrDist = MathMax(MinAddSpacingATRFrac, 0.0) * MathMax(atr, 0.0);
   double pctDist = MathMax(MinAddSpacingPricePct, 0.0) * MathAbs(refPrice);
   return MathMax(fixedDist, MathMax(atrDist, pctDist));
}

bool PassesMinAddSpacingGate(const string symbol,
                             const int symIdx,
                             const int magic,
                             const int basketDir,
                             const double candidatePrice,
                             double &requiredGap,
                             double &actualGap,
                             double &lastEntryPrice)
{
   requiredGap=0.0;
   actualGap=0.0;
   lastEntryPrice=0.0;

   if(!UseMinAddSpacingGate) return true;
   if(basketDir==0) return true;
   if(candidatePrice<=0.0) return true;

   datetime lastEntryTime=0;
   if(!GetLatestBasketEntryInfo(symbol, magic, basketDir, lastEntryPrice, lastEntryTime))
      return true;

   double refPrice = (lastEntryPrice>0.0 ? lastEntryPrice : candidatePrice);
   double minSpacing = ComputeEffectiveMinAddSpacing(symbol, refPrice);
   if(minSpacing<0.0) minSpacing=0.0;

   double baseStep=0.0;
   double adaptiveStep=baseStep;
   if(symIdx>=0 && symIdx<MAX_SYMBOLS)
   {
      baseStep=gGridStepCache[symIdx];
      adaptiveStep=gGridActiveStep[symIdx];
   }

   if(baseStep<0.0) baseStep=0.0;
   if(adaptiveStep<baseStep) adaptiveStep=baseStep;

   double extraAdaptiveGap = adaptiveStep - baseStep;
   if(extraAdaptiveGap<0.0) extraAdaptiveGap=0.0;

   requiredGap = minSpacing + extraAdaptiveGap;
   if(requiredGap<=0.0) return true;

   actualGap = MathAbs(candidatePrice - lastEntryPrice);
   return (actualGap + 1e-12 >= requiredGap);
}

bool PassesOpenIntervalGate(const string symbol,
                           const int symIdx)
{
   if(!UseMinOpenIntervalGate) return true;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return true;

   datetime now = TimeCurrent();
   datetime lastOpen = gLastTradeOpenTime[symIdx];
   if(lastOpen>0)
   {
      int minSec = MathMax(0, MinSecondsBetweenOpens);
      if(minSec>0 && (now - lastOpen) < minSec)
         return false;
   }

   int minBars = MathMax(0, MinExecBarsBetweenOpens);
   if(minBars>0)
   {
      datetime lastBar = iTime(symbol, TF_EXEC, minBars);
      if(lastBar>0 && lastOpen>=lastBar)
         return false;
   }

   return true;
}



double ArchiveEpisodeBaseWeight(const EpisodeMemory &e,
                                const string symbol,
                                const int regimeType,
                                const int patternType,
                                const int liquidityType,
                                const int basketDir,
                                const int positionsCount,
                                const int idx,
                                const int startIdx,
                                const int endIdx)
{
   if(e.symbol != symbol) return 0.0;

   double w = 0.20;
   if(e.regimeType == regimeType) w += 0.25;
   else if(MathAbs(e.regimeType - regimeType) == 1) w += 0.12;

   if(e.patternType == patternType)   w += 0.20;
   if(e.liquidityType == liquidityType) w += 0.15;
   if(basketDir != 0 && e.basketDir == basketDir) w += 0.10;

   int posGap = MathAbs(e.openPositionsMax - MathMax(1, positionsCount));
   w *= SafeDiv(1.0, 1.0 + 0.35 * (double)posGap, 1.0);

   double progress = SafeDiv((double)(idx - startIdx + 1), (double)MathMax(1, endIdx - startIdx + 1), 1.0);
   w *= (0.65 + 0.35 * Clamp(progress, 0.0, 1.0));
   return w;
}

void ComputeArchiveEpisodeProfile(const int symIdx,
                                  const int basketDir,
                                  const int positionsCount,
                                  double &badRiskOut,
                                  double &cleanQualityOut,
                                  double &deepRiskOut,
                                  double &efficiencyOut)
{
   badRiskOut=0.0;
   cleanQualityOut=0.0;
   deepRiskOut=0.0;
   efficiencyOut=0.0;

   if(!UseArchiveAwareReward) return;

   int n=ArraySize(gEpisodeMemory);
   if(n<=0) return;

   string symbol = gSymbols[symIdx];
   int regimeType=0, patternType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, -1, regimeType, patternType, liquidityType, sessionType, atrRatio);

   int scan = MathMax(50, ArchiveRewardScanLimit);
   int startIdx = MathMax(0, n - scan);

   double sumW=0.0;
   double badAcc=0.0, cleanAcc=0.0, deepAcc=0.0, effAcc=0.0;

   for(int i=startIdx;i<n;i++)
   {
      double w = ArchiveEpisodeBaseWeight(gEpisodeMemory[i], symbol, regimeType, patternType, liquidityType,
                                          basketDir, positionsCount, i, startIdx, n-1);
      if(w <= 0.10) continue;

      double bad = 0.0;
      bad += Clamp(gEpisodeMemory[i].maxDrawdownPct / MathMax(RewardV2CleanCycleMaxDD * 2.0, 0.01), 0.0, 4.0);
      bad += 0.30 * (double)MathMax(0, gEpisodeMemory[i].addCount);
      bad += 0.20 * (double)MathMax(0, gEpisodeMemory[i].openPositionsMax - 1);
      if(gEpisodeMemory[i].rewardEfficiency < 0.0)
         bad += Clamp(-gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);
      if(gEpisodeMemory[i].inefficientRecovery != 0) bad += 0.75;
      if(gEpisodeMemory[i].forcedStopLikeEvent != 0) bad += 0.75;

      double deep = Clamp(gEpisodeMemory[i].maxDrawdownPct / 0.05, 0.0, 5.0);
      deep += 0.20 * (double)MathMax(0, gEpisodeMemory[i].addCount);
      deep += 0.15 * (double)MathMax(0, gEpisodeMemory[i].openPositionsMax - 1);

      double clean = 0.0;
      if(gEpisodeMemory[i].oneRoundTrade != 0 && gEpisodeMemory[i].rewardTotal > 0.0)
         clean = Clamp(1.0 - SafeDiv(gEpisodeMemory[i].maxDrawdownPct, MathMax(RewardV2CleanCycleMaxDD, 1e-6), 1.0), 0.0, 1.0);

      double eff = Clamp(gEpisodeMemory[i].rewardEfficiency, -2.0, 2.0);

      badAcc += w * bad;
      cleanAcc += w * clean;
      deepAcc += w * deep;
      effAcc += w * eff;
      sumW += w;
   }

   if(sumW <= 1e-9) return;

   badRiskOut = Clamp(badAcc / sumW, 0.0, 4.0);
   cleanQualityOut = Clamp(cleanAcc / sumW, 0.0, 1.0);
   deepRiskOut = Clamp(deepAcc / sumW, 0.0, 5.0);
   efficiencyOut = Clamp(effAcc / sumW, -2.0, 2.0);
}

void ComputeArchivePatternAndRegimePenalty(const int symIdx,
                                           double &patternPenaltyOut,
                                           double &regimePenaltyOut)
{
   patternPenaltyOut=0.0;
   regimePenaltyOut=0.0;

   if(!UseArchiveAwareReward) return;

   string symbol = gSymbols[symIdx];
   int regimeType=0, patternType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, -1, regimeType, patternType, liquidityType, sessionType, atrRatio);

   int scan = MathMax(50, ArchiveRewardScanLimit);

   int np=ArraySize(gPatternMemory);
   if(np>0)
   {
      int sp=MathMax(0, np-scan);
      double sumW=0.0, penAcc=0.0;
      for(int i=sp;i<np;i++)
      {
         if(gPatternMemory[i].symbol != symbol) continue;

         double w=0.20;
         if(gPatternMemory[i].patternType == patternType) w += 0.35;
         if(gPatternMemory[i].volatilityClass == regimeType) w += 0.25;
         if(gPatternMemory[i].liquidityClass == liquidityType) w += 0.20;
         double progress = SafeDiv((double)(i-sp+1), (double)MathMax(1, np-sp), 1.0);
         w *= (0.65 + 0.35 * Clamp(progress, 0.0, 1.0));

         double pen = Clamp(gPatternMemory[i].ddMean / 0.05, 0.0, 4.0);
         pen += 0.20 * Clamp(gPatternMemory[i].addMean, 0.0, 6.0);
         if(gPatternMemory[i].strengthClass == 0) pen += 0.50;
         if(gPatternMemory[i].rewardEfficiencyMean < 0.0)
            pen += Clamp(-gPatternMemory[i].rewardEfficiencyMean, 0.0, 2.0);

         penAcc += w * pen;
         sumW += w;
      }
      if(sumW > 1e-9)
         patternPenaltyOut = Clamp(penAcc / sumW, 0.0, 4.0);
   }

   int nr=ArraySize(gRegimeEventMemory);
   if(nr>0)
   {
      int sr=MathMax(0, nr-scan);
      double sumW=0.0, penAcc=0.0;
      for(int i=sr;i<nr;i++)
      {
         if(gRegimeEventMemory[i].symbol != symbol) continue;

         double w=0.10;
         if(gRegimeEventMemory[i].regimeType == regimeType) w += 0.45;
         else if(MathAbs(gRegimeEventMemory[i].regimeType - regimeType) == 1) w += 0.20;
         double progress = SafeDiv((double)(i-sr+1), (double)MathMax(1, nr-sr), 1.0);
         w *= (0.65 + 0.35 * Clamp(progress, 0.0, 1.0));

         double pen = 0.0;
         if(gRegimeEventMemory[i].eventType != 0) pen += 1.00;
         pen += Clamp(gRegimeEventMemory[i].avgSpread / 25.0, 0.0, 1.5);
         if(gRegimeEventMemory[i].avgRewardEfficiency < 0.0)
            pen += Clamp(-gRegimeEventMemory[i].avgRewardEfficiency, 0.0, 2.0) * 0.75;

         penAcc += w * pen;
         sumW += w;
      }
      if(sumW > 1e-9)
         regimePenaltyOut = Clamp(penAcc / sumW, 0.0, 3.0);
   }

}

double ArchiveLiveRecencyWeight(const datetime memEndTime,
                                const int idx,
                                const int startIdx,
                                const int endIdx)
{
   double halfLife = MathMax(ArchiveLiveRecentHalfLifeDays, 0.5);
   double ageDays = MathMax(0.0, (double)(TimeCurrent() - memEndTime) / 86400.0);
   double decay = MathExp(-0.6931471805599453 * ageDays / halfLife);

   double progress = SafeDiv((double)(idx - startIdx + 1),
                             (double)MathMax(1, endIdx - startIdx + 1),
                             1.0);
   double progressW = 0.55 + 0.45 * Clamp(progress, 0.0, 1.0);
   return Clamp(decay * progressW, 0.05, 1.50);
}

double ArchiveLiveDecisionConfidence(const double &q[])
{
   int n=ArraySize(q);
   if(n<=1) return 0.0;

   int best=0;
   int second=1;
   if(q[second] > q[best]){ int t=best; best=second; second=t; }

   for(int i=2;i<n;i++)
   {
      if(q[i] > q[best])
      {
         second=best;
         best=i;
      }
      else if(i!=best && q[i] > q[second])
      {
         second=i;
      }
   }

   double gap = q[best] - q[second];
   double scale=1.0;
   for(int i=0;i<n;i++) scale += MathAbs(q[i]);
   scale /= (double)n;

   return Clamp(gap / MathMax(scale, 1e-6), 0.0, 1.0);
}


bool BuildArchiveLiveActionBias(const int symIdx,
                                const int regime,
                                const double &state[],
                                const double &qIn[],
                                double &biasOut[],
                                double &blendAlphaOut)
{
   ArrayResize(biasOut, ActionCount);
   for(int a=0;a<ActionCount;a++) biasOut[a]=0.0;
   blendAlphaOut=0.0;

   if(!UseArchiveLiveSimilarity) return false;
   if(symIdx<0 || symIdx>=gSymbolCount) return false;
   if(ActionCount < 3) return false;

   string symbol = gSymbols[symIdx];
   int regimeType=0, patternType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, regime, regimeType, patternType, liquidityType, sessionType, atrRatio);

   int scanBase = MathMax(80, ArchiveLiveScanLimit);

   double dirScore[3];
   double dirWeight[3];
   for(int k=0;k<3;k++)
   {
      dirScore[k]=0.0;
      dirWeight[k]=0.0;
   }

   double holdPenaltyAcc=0.0;
   double holdBonusAcc=0.0;
   double holdWeight=0.0;

   int ne=ArraySize(gEpisodeMemory);
   if(ne>0)
   {
      int se=MathMax(0, ne-scanBase);
      for(int i=se;i<ne;i++)
      {
         if(gEpisodeMemory[i].symbol != symbol) continue;

         double recW = ArchiveLiveRecencyWeight(gEpisodeMemory[i].endTime, i, se, ne-1);

         double ctxW = 0.20;
         if(gEpisodeMemory[i].regimeType == regimeType) ctxW += 0.24;
         else if(MathAbs(gEpisodeMemory[i].regimeType - regimeType) == 1) ctxW += 0.10;
         if(gEpisodeMemory[i].patternType == patternType) ctxW += 0.22;
         if(gEpisodeMemory[i].liquidityType == liquidityType) ctxW += 0.16;
         if(gEpisodeMemory[i].sessionType == sessionType) ctxW += 0.10;

         double risk = 0.0;
         risk += Clamp(gEpisodeMemory[i].maxDrawdownPct / MathMax(RewardV2CleanCycleMaxDD * 2.0, 0.01), 0.0, 4.0);
         risk += 0.28 * (double)MathMax(0, gEpisodeMemory[i].addCount);
         risk += 0.18 * (double)MathMax(0, gEpisodeMemory[i].openPositionsMax - 1);
         if(gEpisodeMemory[i].inefficientRecovery != 0) risk += 0.65;
         if(gEpisodeMemory[i].forcedStopLikeEvent != 0) risk += 0.75;
         if(gEpisodeMemory[i].rewardEfficiency < 0.0)
            risk += Clamp(-gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);

         double clean = 0.0;
         if(gEpisodeMemory[i].rewardTotal > 0.0)
         {
            clean += 0.35 * Clamp(gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);
            if(gEpisodeMemory[i].oneRoundTrade != 0)
               clean += Clamp(1.0 - SafeDiv(gEpisodeMemory[i].maxDrawdownPct,
                                            MathMax(RewardV2CleanCycleMaxDD, 1e-6), 1.0), 0.0, 1.0);
         }

         double signedQuality = 0.45 * clean - 0.65 * risk;
         if(gEpisodeMemory[i].rewardEfficiency > 0.0)
            signedQuality += 0.18 * Clamp(gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);

         int actionDir = (gEpisodeMemory[i].basketDir>0 ? 1 : (gEpisodeMemory[i].basketDir<0 ? 2 : 0));
         double w = ctxW * recW;

         if(actionDir>=1 && actionDir<=2)
         {
            dirScore[actionDir]  += w * signedQuality;
            dirWeight[actionDir] += w;
         }

         holdPenaltyAcc += w * MathMax(0.0, risk - 0.35 * clean);
         holdBonusAcc   += w * MathMax(0.0, clean - 0.25 * risk);
         holdWeight     += w;
      }
   }

   double patternCaution=0.0;
   double patternSupport=0.0;
   int np=ArraySize(gPatternMemory);
   if(np>0)
   {
      int sp=MathMax(0, np-scanBase);
      double sumW=0.0;
      for(int i=sp;i<np;i++)
      {
         if(gPatternMemory[i].symbol != symbol) continue;

         double recW = ArchiveLiveRecencyWeight(gPatternMemory[i].endTime, i, sp, np-1);
         double w=0.18;
         if(gPatternMemory[i].patternType == patternType) w += 0.35;
         if(gPatternMemory[i].volatilityClass == regimeType) w += 0.24;
         if(gPatternMemory[i].liquidityClass == liquidityType) w += 0.18;
         w *= recW;

         double caution = Clamp(gPatternMemory[i].ddMean / 0.05, 0.0, 4.0);
         caution += 0.18 * Clamp(gPatternMemory[i].addMean, 0.0, 6.0);
         if(gPatternMemory[i].strengthClass == 0) caution += 0.45;
         if(gPatternMemory[i].rewardEfficiencyMean < 0.0)
            caution += Clamp(-gPatternMemory[i].rewardEfficiencyMean, 0.0, 2.0);

         double support = 0.0;
         if(gPatternMemory[i].strengthClass == 2)
            support += 0.45;
         if(gPatternMemory[i].rewardEfficiencyMean > 0.0)
            support += 0.35 * Clamp(gPatternMemory[i].rewardEfficiencyMean, 0.0, 2.0);
         support -= 0.15 * Clamp(gPatternMemory[i].addMean, 0.0, 6.0);

         patternCaution += w * MathMax(0.0, caution);
         patternSupport += w * MathMax(0.0, support);
         sumW += w;
      }

      if(sumW > 1e-9)
      {
         patternCaution = Clamp(patternCaution / sumW, 0.0, 4.0);
         patternSupport = Clamp(patternSupport / sumW, 0.0, 2.0);
      }
   }

   double regimeCaution=0.0;
   int nr=ArraySize(gRegimeEventMemory);
   if(nr>0)
   {
      int sr=MathMax(0, nr-scanBase);
      double sumW=0.0;
      for(int i=sr;i<nr;i++)
      {
         if(gRegimeEventMemory[i].symbol != symbol) continue;

         double recW = ArchiveLiveRecencyWeight(gRegimeEventMemory[i].endTime, i, sr, nr-1);
         double w=0.10;
         if(gRegimeEventMemory[i].regimeType == regimeType) w += 0.45;
         else if(MathAbs(gRegimeEventMemory[i].regimeType - regimeType) == 1) w += 0.18;
         w *= recW;

         double caution = 0.0;
         if(gRegimeEventMemory[i].eventType != 0) caution += 1.00;
         caution += Clamp(gRegimeEventMemory[i].avgSpread / 25.0, 0.0, 1.5);
         caution += 0.20 * Clamp(MathAbs(gRegimeEventMemory[i].avgVol - GetAtrRatioCached_Base(symbol)), 0.0, 1.5);
         if(gRegimeEventMemory[i].avgRewardEfficiency < 0.0)
            caution += 0.75 * Clamp(-gRegimeEventMemory[i].avgRewardEfficiency, 0.0, 2.0);

         regimeCaution += w * caution;
         sumW += w;
      }

      if(sumW > 1e-9)
         regimeCaution = Clamp(regimeCaution / sumW, 0.0, 3.0);
   }

   double efficientSupport=0.0;
   double efficientCaution=0.0;
   if(UseEfficientPeriodLiveSupport)
   {
      int neff=ArraySize(gEfficientReplayBank.periods);
      if(neff>0)
      {
         int seff=MathMax(0, neff-scanBase);
         double sumW=0.0;
         for(int i=seff;i<neff;i++)
         {
            if(gEfficientReplayBank.periods[i].symIdx != symIdx) continue;

            double recW = ArchiveLiveRecencyWeight(gEfficientReplayBank.periods[i].endTime, i, seff, neff-1);
            double w=0.16;
            if(gEfficientReplayBank.periods[i].regimeType == regimeType) w += 0.28;
            else if(MathAbs(gEfficientReplayBank.periods[i].regimeType - regimeType) == 1) w += 0.12;
            if(gEfficientReplayBank.periods[i].patternType == patternType) w += 0.20;
            if(gEfficientReplayBank.periods[i].liquidityType == liquidityType) w += 0.14;
            if(gEfficientReplayBank.periods[i].sessionType == sessionType) w += 0.10;
            w *= recW;

            double support = 0.0;
            support += Clamp(gEfficientReplayBank.periods[i].rewardEfficiency, -2.0, 2.0);
            support += 0.20 * Clamp((double)gEfficientReplayBank.periods[i].oneRoundCount, 0.0, 6.0);
            support -= 0.18 * Clamp(gEfficientReplayBank.periods[i].ddMax / 0.05, 0.0, 4.0);
            support -= 0.08 * Clamp((double)gEfficientReplayBank.periods[i].addCountTotal, 0.0, 10.0);

            efficientSupport += w * MathMax(0.0, support);
            efficientCaution += w * MathMax(0.0, -support);
            sumW += w;
         }
         if(sumW > 1e-9)
         {
            efficientSupport = Clamp(efficientSupport / sumW, 0.0, 2.5);
            efficientCaution = Clamp(efficientCaution / sumW, 0.0, 2.5);
         }
      }
   }

   double holdBias = 0.0;
   if(holdWeight > 1e-9)
      holdBias = Clamp((holdPenaltyAcc - 0.60 * holdBonusAcc) / holdWeight, -2.0, 3.0);

   holdBias += EfficientPeriodLiveSupportScale * (0.60 * efficientCaution - 0.35 * efficientSupport);

   for(int a=1;a<=2;a++)
   {
      if(dirWeight[a] > 1e-9)
         biasOut[a] += Clamp(dirScore[a] / dirWeight[a], -2.5, 2.5);
   }

   double patternAdj = ArchiveLivePatternCautionScale * (patternCaution - 0.45 * patternSupport);
   double regimeAdj = ArchiveLiveRegimeCautionScale * regimeCaution;

   biasOut[0] += holdBias + patternAdj + regimeAdj;
   double efficientTradeAdj = EfficientPeriodLiveSupportScale * (0.45 * efficientSupport - 0.35 * efficientCaution);
   biasOut[1] += efficientTradeAdj - 0.50 * (patternAdj + regimeAdj);
   biasOut[2] += efficientTradeAdj - 0.50 * (patternAdj + regimeAdj);

   double mean=0.0;
   for(int a=0;a<ActionCount;a++) mean += biasOut[a];
   mean /= (double)ActionCount;
   double maxAbs=0.0;
   for(int a=0;a<ActionCount;a++)
   {
      biasOut[a] -= mean;
      maxAbs = MathMax(maxAbs, MathAbs(biasOut[a]));
   }
   if(maxAbs <= 1e-9) return false;

   for(int a=0;a<ActionCount;a++)
      biasOut[a] = 1.5 * biasOut[a] / maxAbs;

   double conf = ArchiveLiveDecisionConfidence(qIn);
   double uncertainty = Clamp(1.0 - conf, 0.0, 1.0);
   blendAlphaOut = Clamp(ArchiveLiveBlendWeight * (0.35 + 0.65 * uncertainty),
                         ArchiveLiveMinBlendWeight,
                         ArchiveLiveBlendWeight);

   return true;
}

double ComputeArchiveAddRiskScore(const int symIdx,
                                  const int basketDir,
                                  const int positionsCount)
{
   if(!UseArchiveAddRiskGate) return 0.0;
   if(symIdx<0 || symIdx>=gSymbolCount) return 0.0;
   if(basketDir==0) return 0.0;
   if(positionsCount < ArchiveAddRiskGateMinPositions) return 0.0;

   string symbol = gSymbols[symIdx];
   int regimeType=0, patternType=0, liquidityType=0, sessionType=0;
   double atrRatio=0.0;
   GetDecisionContextCached(symIdx, -1, regimeType, patternType, liquidityType, sessionType, atrRatio);
   int candidatePositions = positionsCount + 1;
   int scanBase = MathMax(80, ArchiveLiveScanLimit);

   double sumW=0.0;
   double riskAcc=0.0;
   double cleanAcc=0.0;

   int ne=ArraySize(gEpisodeMemory);
   if(ne>0)
   {
      int se=MathMax(0, ne-scanBase);
      for(int i=se;i<ne;i++)
      {
         if(gEpisodeMemory[i].symbol != symbol) continue;
         if(gEpisodeMemory[i].basketDir != basketDir) continue;

         double w=0.22;
         if(gEpisodeMemory[i].regimeType == regimeType) w += 0.24;
         if(gEpisodeMemory[i].patternType == patternType) w += 0.22;
         if(gEpisodeMemory[i].liquidityType == liquidityType) w += 0.14;
         if(gEpisodeMemory[i].sessionType == sessionType) w += 0.10;

         int posGap=MathAbs(gEpisodeMemory[i].openPositionsMax - candidatePositions);
         w *= SafeDiv(1.0, 1.0 + 0.30 * (double)posGap, 1.0);
         w *= ArchiveLiveRecencyWeight(gEpisodeMemory[i].endTime, i, se, ne-1);
         if(w<=0.05) continue;

         double risk = 0.0;
         risk += Clamp(gEpisodeMemory[i].maxDrawdownPct / 0.05, 0.0, 5.0);
         risk += 0.26 * (double)MathMax(0, gEpisodeMemory[i].addCount);
         risk += 0.18 * (double)MathMax(0, gEpisodeMemory[i].openPositionsMax - 1);
         if(gEpisodeMemory[i].inefficientRecovery != 0) risk += 0.75;
         if(gEpisodeMemory[i].forcedStopLikeEvent != 0) risk += 0.70;
         if(gEpisodeMemory[i].rewardEfficiency < 0.0)
            risk += Clamp(-gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);

         double clean = 0.0;
         if(gEpisodeMemory[i].rewardTotal > 0.0)
         {
            clean += 0.30 * Clamp(gEpisodeMemory[i].rewardEfficiency, 0.0, 2.0);
            if(gEpisodeMemory[i].oneRoundTrade != 0) clean += 0.60;
         }

         riskAcc += w * risk;
         cleanAcc += w * clean;
         sumW += w;
      }
   }

   double patternPenalty=0.0, regimePenalty=0.0;
   ComputeArchivePatternAndRegimePenalty(symIdx, patternPenalty, regimePenalty);

   double deepSeqRisk=0.0;
   int nds=ArraySize(gDeepBasketReplayBank.sequences);
   if(nds>0)
   {
      int sd=MathMax(0, nds-scanBase);
      double sumWSeq=0.0;
      for(int i=sd;i<nds;i++)
      {
         if(gDeepBasketReplayBank.sequences[i].symIdx != symIdx) continue;
         if(gDeepBasketReplayBank.sequences[i].basketDir != basketDir) continue;

         int posGap=MathAbs(gDeepBasketReplayBank.sequences[i].maxPositions - candidatePositions);
         double w=(0.25 + 0.15 / (1.0 + (double)posGap));
         w *= ArchiveLiveRecencyWeight(gDeepBasketReplayBank.sequences[i].endTime, i, sd, nds-1);

         double seqRisk = Clamp(gDeepBasketReplayBank.sequences[i].maxDD / 0.05, 0.0, 5.0);
         seqRisk += 0.25 * Clamp((double)gDeepBasketReplayBank.sequences[i].addCount, 0.0, 8.0);
         if(gDeepBasketReplayBank.sequences[i].finalReward < 0.0)
            seqRisk += Clamp(-gDeepBasketReplayBank.sequences[i].finalReward, 0.0, 2.0);

         deepSeqRisk += w * seqRisk;
         sumWSeq += w;
      }
      if(sumWSeq > 1e-9)
         deepSeqRisk = Clamp(deepSeqRisk / sumWSeq, 0.0, 5.0);
   }

   double efficientOffset=0.0;
   if(UseEfficientPeriodLiveSupport)
   {
      int neff=ArraySize(gEfficientReplayBank.periods);
      if(neff>0)
      {
         int seff=MathMax(0, neff-scanBase);
         double sumWEff=0.0;
         for(int i=seff;i<neff;i++)
         {
            if(gEfficientReplayBank.periods[i].symIdx != symIdx) continue;
            double w=0.16;
            if(gEfficientReplayBank.periods[i].regimeType == regimeType) w += 0.24;
            if(gEfficientReplayBank.periods[i].patternType == patternType) w += 0.18;
            if(gEfficientReplayBank.periods[i].liquidityType == liquidityType) w += 0.12;
            if(gEfficientReplayBank.periods[i].sessionType == sessionType) w += 0.10;
            w *= ArchiveLiveRecencyWeight(gEfficientReplayBank.periods[i].endTime, i, seff, neff-1);

            double eff = Clamp(gEfficientReplayBank.periods[i].rewardEfficiency, -2.0, 2.0);
            eff -= 0.20 * Clamp(gEfficientReplayBank.periods[i].ddMax / 0.05, 0.0, 4.0);
            eff -= 0.08 * Clamp((double)gEfficientReplayBank.periods[i].addCountTotal, 0.0, 8.0);
            efficientOffset += w * eff;
            sumWEff += w;
         }
         if(sumWEff > 1e-9)
            efficientOffset = Clamp(efficientOffset / sumWEff, -2.0, 2.0);
      }
   }

   double histRisk = 0.0;
   if(sumW > 1e-9)
      histRisk = MathMax(0.0, (riskAcc / sumW) - ArchiveAddRiskGateCleanOffset * (cleanAcc / sumW));

   double currentRisk = 0.0;
   currentRisk += 1.50 * GetSymbolFloatingDDPct(symIdx);
   currentRisk += 0.75 * CurrentReplayRiskBias(symIdx);
   currentRisk += 0.20 * (double)MathMax(0, positionsCount - 1);

   return histRisk + DeepSequenceAddRiskScale * deepSeqRisk + 0.30 * patternPenalty + 0.22 * regimePenalty + currentRisk - 0.18 * MathMax(0.0, efficientOffset);
}
double RewardV2RiskyProfitPenalty(const double profitRewardNorm,
                                  const int addCount,
                                  const int maxPositions,
                                  const double episodeMaxDD,
                                  const double archiveDeepRisk,
                                  const double archiveBadRisk)
{
   if(profitRewardNorm <= 0.0) return 0.0;

   double risk = 0.0;
   risk += 1.40 * Clamp(episodeMaxDD, 0.0, 1.0);
   risk += 0.18 * (double)MathMax(0, addCount);
   risk += 0.14 * (double)MathMax(0, maxPositions - 1);
   risk += 0.10 * Clamp(archiveDeepRisk, 0.0, 5.0);
   risk += 0.08 * Clamp(archiveBadRisk, 0.0, 4.0);
   return profitRewardNorm * risk;
}

void GetRewardEpisodeContext(const int symIdx,
                             const int positionsFallback,
                             int &episodeAdds,
                             int &episodeMaxPositions,
                             double &episodeMaxDD)
{
   episodeAdds = MathMax(positionsFallback - 1, 0);
   episodeMaxPositions = MathMax(positionsFallback, 1);
   episodeMaxDD = GetSymbolFloatingDDPct(symIdx);

   if(symIdx>=0 && symIdx<MAX_SYMBOLS && gActiveBasketEpisodes[symIdx].active)
   {
      episodeAdds = MathMax(episodeAdds, gActiveBasketEpisodes[symIdx].addCount);
      episodeMaxPositions = MathMax(episodeMaxPositions, gActiveBasketEpisodes[symIdx].maxPositions);
      episodeMaxDD = MathMax(episodeMaxDD, gActiveBasketEpisodes[symIdx].maxDD);
   }
}

double RewardV2QuadraticAddPenalty(const int addCount)
{
   if(addCount <= 0) return 0.0;
   return RewardV2AddPenaltyScale * (double)addCount
        + RewardV2AddPenaltyQuadratic * (double)(addCount * addCount);
}

double RewardV2DeepBasketPenalty(const int positionsCount)
{
   int excess = MathMax(positionsCount - RewardV2DeepBasketStartPositions, 0);
   if(excess <= 0) return 0.0;
   return RewardV2DeepBasketPenaltyScale * (double)(excess * excess);
}

double RewardV2ProfitToDDQualityForSymbol(const int symIdx, const double closedProfit, const double episodeMaxDD)
{
   if(closedProfit <= 0.0) return 0.0;
   double base = MathAbs(ProfitReturnRewardForSymbol(symIdx, closedProfit));
   double ddPenaltyDenom = 1.0 + 12.0 * Clamp(episodeMaxDD, 0.0, 1.0);
   return RewardV2ProfitToMaxDDScale * Clamp(base / ddPenaltyDenom, 0.0, RewardV2RewardClamp);
}



double ComputeCloseRewardV2(const int symIdx,
                            const BasketCloseResult &closeRes,
                            const int positionsBeforeClose,
                            const bool extreme)
{
   double reward = 0.0;

   int episodeAdds = 0;
   int episodeMaxPositions = 1;
   double episodeMaxDD = 0.0;
   GetRewardEpisodeContext(symIdx, positionsBeforeClose, episodeAdds, episodeMaxPositions, episodeMaxDD);

   int basketDir = 0;
   if(symIdx>=0 && symIdx<MAX_SYMBOLS && gActiveBasketEpisodes[symIdx].active)
      basketDir = gActiveBasketEpisodes[symIdx].basketDir;

   double currentDD = GetSymbolFloatingDDPct(symIdx);
   double replayRisk = CurrentReplayRiskBias(symIdx);
   double marginStress = CurrentMarginStressRatio();
   double ageDays = BasketAgeDays(symIdx);

   double archiveBadRisk=0.0, archiveCleanQuality=0.0, archiveDeepRisk=0.0, archiveEfficiency=0.0;
   ComputeArchiveEpisodeProfile(symIdx, basketDir, episodeMaxPositions, archiveBadRisk, archiveCleanQuality, archiveDeepRisk, archiveEfficiency);

   double archivePatternPenalty=0.0, archiveRegimePenalty=0.0;
   ComputeArchivePatternAndRegimePenalty(symIdx, archivePatternPenalty, archiveRegimePenalty);

   double replayDangerRisk=0.0, replayDeepRisk=0.0, replayEfficient=0.0, replayRecentCaution=0.0;
   ComputeReplayRewardProfile(symIdx, basketDir, episodeMaxPositions, replayDangerRisk, replayDeepRisk, replayEfficient, replayRecentCaution);

   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate=ComputeRecentDeepBasketRate(symIdx);
   double recentCalm=ComputeRecentDrawdownCalmBonus(symIdx);
   double periodicQualityTilt=Clamp((recentQuality - 0.5)*2.0,-1.0,1.0);

   double profitRewardNorm = ProfitReturnRewardForSymbol(symIdx, closeRes.closedProfit);

   reward += profitRewardNorm * RewardV2BasketProfitScale;
   reward += RewardV2RecoveryQualityScale * RecoveryQualityRewardFromCounts(closeRes.wins, closeRes.losses, closeRes.total);
   reward += RewardV2ProfitToDDQualityForSymbol(symIdx, closeRes.closedProfit, episodeMaxDD);
   reward += 0.22 * periodicQualityTilt;

   if(closeRes.closedProfit > 0.0 && episodeMaxPositions <= 1)
      reward += RewardV2OneRoundBonus;

   if(closeRes.closedProfit > 0.0 &&
      episodeMaxPositions <= 1 &&
      episodeMaxDD <= RewardV2CleanCycleMaxDD)
   {
      reward += RewardV2CleanCycleBonus;
   }

   reward -= RewardV2QuadraticAddPenalty(episodeAdds);
   reward -= RewardV2DeepBasketPenalty(episodeMaxPositions);
   reward -= RewardV2EpisodeMaxDDPenaltyScale * Clamp(episodeMaxDD, 0.0, 1.0);
   reward -= RewardV2DDPenaltyScale * currentDD;
   reward -= RewardV2DangerPenaltyScale * replayRisk;
   reward -= RewardV2MarginPenaltyScale * marginStress;
   reward -= RewardV2AgePenaltyPerDay * ageDays;

   reward -= RewardV2ArchiveCloseRiskPenaltyScale * archiveBadRisk;
   reward -= 0.60 * RewardV2ArchivePatternPenaltyScale * archivePatternPenalty;
   reward -= 0.60 * RewardV2ArchiveRegimePenaltyScale * archiveRegimePenalty;

   reward -= RewardV2ReplayCloseAntiPatternScale *
             (0.55 * replayDangerRisk + 0.95 * replayDeepRisk + RewardV2ReplayRecentCautionScale * MathMax(0.0, replayRecentCaution));

   if(closeRes.closedProfit > 0.0)
   {
      reward -= RewardV2RiskyProfitPenaltyScale *
                RewardV2RiskyProfitPenalty(MathAbs(profitRewardNorm), episodeAdds, episodeMaxPositions, episodeMaxDD,
                                           archiveDeepRisk + 0.35*replayDeepRisk, archiveBadRisk + 0.25*replayDangerRisk);

      if(episodeMaxPositions <= 1)
      {
         reward += RewardV2ArchiveCleanBonusScale * archiveCleanQuality;
         reward += RewardV2ReplayEfficientBonusScale * Clamp(replayEfficient, 0.0, 2.0);
         reward += 0.16 * recentCalm;
      }
      else
      {
         reward -= 0.35 * RewardV2ReplayCloseAntiPatternScale * Clamp(replayDeepRisk, 0.0, 3.0);
         reward -= 0.16 * recentDeepRate;
      }
   }

   if(closeRes.closedProfit <= 0.0 && episodeAdds > 0)
      reward -= RewardV2FailedBasketPenaltyScale * (1.0 + (double)episodeAdds);

   reward *= PendingCloseRewardScale;
   if(closeRes.closedProfit > 0.0 && episodeMaxPositions <= 1)
      reward += PendingGoodCloseBonus;

   reward = ScaleRewardByRegime(reward, extreme);
   return Clamp(reward, -RewardV2RewardClamp, RewardV2RewardClamp);
}




double ComputeOpenRewardV2(const int symIdx,
                           const bool isBuy,
                           const int positionsBeforeOpen,
                           const double combinedTrend,
                           const double reversalRisk,
                           const bool extreme)
{
   double reward = -RewardV2OpenBaseCost;

   int addCount = MathMax(positionsBeforeOpen, 0);
   int resultingPositions = addCount + 1;
   int basketDir = (isBuy ? 1 : -1);

   reward -= RewardV2QuadraticAddPenalty(addCount);
   reward -= 0.85 * RewardV2DeepBasketPenalty(resultingPositions);

   double replayRisk = CurrentReplayRiskBias(symIdx);
   reward -= RewardV2DangerPenaltyScale * replayRisk;

   double marginStress = CurrentMarginStressRatio();
   reward -= RewardV2MarginPenaltyScale * marginStress;

   double currentDD = GetSymbolFloatingDDPct(symIdx);
   reward -= 0.60 * RewardV2EpisodeMaxDDPenaltyScale * Clamp(currentDD, 0.0, 1.0);
   reward -= 0.50 * RewardV2DDPenaltyScale * Clamp(currentDD, 0.0, 1.0);

   double ageDays = BasketAgeDays(symIdx);
   reward -= 0.50 * RewardV2AgePenaltyPerDay * ageDays;
   if(addCount > 0)
      reward -= RewardV2AgingAddPenaltyScale * ageDays * (double)addCount;

   reward -= RewardV2ReversalPenaltyScale * Clamp(reversalRisk, 0.0, 1.0);

   double archiveBadRisk=0.0, archiveCleanQuality=0.0, archiveDeepRisk=0.0, archiveEfficiency=0.0;
   ComputeArchiveEpisodeProfile(symIdx, basketDir, resultingPositions, archiveBadRisk, archiveCleanQuality, archiveDeepRisk, archiveEfficiency);

   double archivePatternPenalty=0.0, archiveRegimePenalty=0.0;
   ComputeArchivePatternAndRegimePenalty(symIdx, archivePatternPenalty, archiveRegimePenalty);

   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate=ComputeRecentDeepBasketRate(symIdx);
   double recentCalm=ComputeRecentDrawdownCalmBonus(symIdx);
   double periodicQualityTilt=Clamp((recentQuality - 0.5)*2.0,-1.0,1.0);

   reward -= RewardV2ArchiveOpenRiskPenaltyScale * (0.70 * archiveBadRisk + 0.30 * archiveDeepRisk);
   reward -= 0.50 * RewardV2ArchivePatternPenaltyScale * archivePatternPenalty;
   reward -= 0.50 * RewardV2ArchiveRegimePenaltyScale * archiveRegimePenalty;

   double replayDangerRisk=0.0, replayDeepRisk=0.0, replayEfficient=0.0, replayRecentCaution=0.0;
   ComputeReplayRewardProfile(symIdx, basketDir, resultingPositions, replayDangerRisk, replayDeepRisk, replayEfficient, replayRecentCaution);

   reward -= RewardV2ReplayOpenAntiPatternScale *
             (0.60 * replayDangerRisk + 0.90 * replayDeepRisk + RewardV2ReplayRecentCautionScale * MathMax(0.0, replayRecentCaution));

   if(addCount <= 0)
   {
      reward += RewardV2ArchiveCleanBonusScale * archiveCleanQuality;
      reward += 0.15 * RewardV2ArchiveCleanBonusScale * Clamp(archiveEfficiency, 0.0, 1.0);
      reward += RewardV2ReplayEfficientBonusScale * Clamp(replayEfficient, 0.0, 2.0);

      reward += 0.35 * periodicQualityTilt;
      reward += 0.18 * recentCalm;
      reward -= 0.30 * recentDeepRate;
   }
   else
   {
      reward -= 0.35 * RewardV2ReplayOpenAntiPatternScale * Clamp(replayDeepRisk, 0.0, 3.0);
      reward += 0.10 * RewardV2ReplayEfficientBonusScale * Clamp(replayEfficient, 0.0, 1.5);

      reward -= 0.18 * recentDeepRate;
      reward -= 0.12 * (1.0 - recentQuality);
   }

   reward *= PendingOpenRewardScale;
   reward = ScaleRewardByRegime(reward, extreme);
   return Clamp(reward, -RewardV2RewardClamp, RewardV2RewardClamp);
}



double GetEAOpenPnL()
{
   double totalPnL = 0.0;
   int total = PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      string sym = PositionGetString(POSITION_SYMBOL);
      long   mg  = PositionGetInteger(POSITION_MAGIC);
      int idx = SymbolIndex(sym);
      if(idx<0) continue;
      if((int)mg != gMagics[idx]) continue;
      totalPnL += PositionGetDouble(POSITION_PROFIT);
   }
   return totalPnL;
}

double GetEAEquity()
{
   return gEAStartEquity + gEAClosedProfit + GetEAOpenPnL();
}

void RefreshRewardTickBaseline()
{
   datetime now=TimeCurrent();
   if(gRewardBaselineTick==now && gTickBalanceBaseline>0.0) return;

   gTickEquityBaseline=GetEAEquity();
   gTickBalanceBaseline=gEAStartEquity;
   if(gTickBalanceBaseline<=1e-9) gTickBalanceBaseline=gTickEquityBaseline;
   if(gTickBalanceBaseline<=1e-9) gTickBalanceBaseline=1000.0;
   gRewardBaselineTick=now;
}

bool CheckEquityStop()
{
   if(!UseEquityStop) return false;
   if(EquityRiskPercent <= 0.0) return false;
   if(gAccountEquityStopPeak <= 0.0) return false;

   double eq = GetWatchedAccountEquity();
   return (eq < gAccountEquityStopPeak * (1.0 - EquityRiskPercent/100.0));
}

double GetCurrentEAFloatingLossMoney()
{
   double openPnL = GetEAOpenPnL();
   if(openPnL >= 0.0) return 0.0;
   return -openPnL;
}

bool CheckEquityLossStop()
{
   if(!UseEquityLossStop) return false;
   if(EquityLossStopAmount <= 0.0) return false;

   double floatingLoss = 0.0;

   if(UseGlobalAccountWatchdog)
   {
      double totalPnL = 0.0;

      for(int i=0; i<PositionsTotal(); i++)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket<=0) continue;
         if(!PositionSelectByTicket(ticket)) continue;

         if(WatchAllAccountPositions)
         {
            totalPnL += PositionGetDouble(POSITION_PROFIT);
         }
         else
         {
            int mg = (int)PositionGetInteger(POSITION_MAGIC);
            if(mg >= WatchdogMagicMin && mg <= WatchdogMagicMax)
               totalPnL += PositionGetDouble(POSITION_PROFIT);
         }
      }

      if(totalPnL < 0.0)
         floatingLoss = -totalPnL;
   }
   else
   {
      floatingLoss = GetCurrentEAFloatingLossMoney();
   }

   return (floatingLoss >= EquityLossStopAmount);
}

void StartEquityLossStopCooldown()
{
   int sec = MathMax(0, EquityLossStopCooldownSeconds);
   if(sec <= 0)
      gEquityLossStopResumeTime = 0;
   else
      gEquityLossStopResumeTime = TimeCurrent() + sec;
}

void ResetEquityLossStopCooldownIfExpired()
{
   if(gEquityLossStopResumeTime > 0 && TimeCurrent() >= gEquityLossStopResumeTime)
      gEquityLossStopResumeTime = 0;
}

double GetWatchedAccountEquity()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double openPnL = 0.0;

   if(UseGlobalAccountWatchdog)
   {
      for(int i=0; i<PositionsTotal(); i++)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket<=0) continue;
         if(!PositionSelectByTicket(ticket)) continue;

         if(WatchAllAccountPositions)
         {
            openPnL += PositionGetDouble(POSITION_PROFIT);
         }
         else
         {
            int mg = (int)PositionGetInteger(POSITION_MAGIC);
            if(mg >= WatchdogMagicMin && mg <= WatchdogMagicMax)
               openPnL += PositionGetDouble(POSITION_PROFIT);
         }
      }
   }
   else
   {
      // fallback to this EA's own live positions only
      for(int i=0;i<PositionsTotal();i++)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket<=0) continue;
         if(!PositionSelectByTicket(ticket)) continue;

         string sym = PositionGetString(POSITION_SYMBOL);
         int idx = SymbolIndex(sym);
         if(idx<0) continue;

         int mg = (int)PositionGetInteger(POSITION_MAGIC);
         if(mg != gMagics[idx]) continue;

         openPnL += PositionGetDouble(POSITION_PROFIT);
      }
   }

   return balance + openPnL;
}


bool ProfitPauseActive()
{
   if(!UseProfitPause) return false;
   if(gProfitPauseResumeTime <= 0) return false;
   return (TimeCurrent() < gProfitPauseResumeTime);
}

void ResetProfitPauseIfExpired()
{
   if(gProfitPauseResumeTime > 0 && TimeCurrent() >= gProfitPauseResumeTime)
   {
      gProfitPauseResumeTime = 0;
      gProfitCycleClosedProfit = 0.0;
   }
}

void ResetForcedEntryState(const int symIdx)
{
   gForcedEntryActive[symIdx]   = false;
   gForcedEntryArmTime[symIdx]  = 0;
   gForcedEntryDeadline[symIdx] = 0;
}

void MarkTradeOpened(const int symIdx, const datetime when)
{
   gLastTradeOpenTime[symIdx] = (when>0 ? when : TimeCurrent());
   ResetForcedEntryState(symIdx);
   if(symIdx>=0 && symIdx<MAX_SYMBOLS)
      gDecisionSupportCache[symIdx].valid=false;
}

void UpdateForcedEntryWatchdog(const int symIdx)
{
   if(!UseForcedEntryWatchdog) return;

   datetime now = TimeCurrent();

   if(gLastTradeOpenTime[symIdx] <= 0)
      gLastTradeOpenTime[symIdx] = now;

   // if basket is already open, watchdog is not needed
   if(gPositionsCount[symIdx] > 0)
   {
      ResetForcedEntryState(symIdx);
      return;
   }

   int idleSec   = MathMax(1, ForcedEntryIdleMinutes) * 60;
   int windowSec = MathMax(1, ForcedEntryWindowMinutes) * 60;

   if(!gForcedEntryActive[symIdx])
   {
      if((now - gLastTradeOpenTime[symIdx]) >= idleSec)
      {
         gForcedEntryActive[symIdx]   = true;
         gForcedEntryArmTime[symIdx]  = now;
         gForcedEntryDeadline[symIdx] = now + windowSec;

         Print("FORCED ENTRY ARMED | sym=", gSymbols[symIdx],
               " idleMin=", ForcedEntryIdleMinutes,
               " deadline=", TimeToString(gForcedEntryDeadline[symIdx], TIME_DATE|TIME_SECONDS));
      }
   }
}

bool ForcedEntryRequiredNow(const int symIdx)
{
   if(!UseForcedEntryWatchdog) return false;
   if(!gForcedEntryActive[symIdx]) return false;
   if(gPositionsCount[symIdx] > 0) return false;
   return true;
}


