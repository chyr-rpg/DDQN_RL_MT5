//+------------------------------------------------------------------+
//| 13_StrategyManager.mqh                                          |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Main per-symbol DDQN/grid strategy orchestration and basket mana|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

void ManagePairWithDQN(string symbol,int symIdx,int &positionsCount,CArrayDouble &trades,int magic,datetime &firstTradeTime)
{
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double state[];
   datetime baseBarTimeCache = iTime(symbol, BaseTF, 1);
   double avgCacheRef = (positionsCount>0 && trades.Total()>0 ? BasketAvgPrice(trades,gTradeLots[symIdx],0.5*(bid+ask)) : 0.5*(bid+ask));
   int basketDirCacheNow = (positionsCount>0 ? BasketDir(symbol,magic) : 0);
   double lastEntryCacheRef = GetLastEntryPriceState(symbol, magic, basketDirCacheNow, avgCacheRef);
   long avgQCache = QuantizePriceByPoint(avgCacheRef, point);
   long lastEntryQCache = QuantizePriceByPoint(lastEntryCacheRef, point);

   bool usedStateCache=false;
   if(ShouldUseFastStateCache() && symIdx>=0 && symIdx<MAX_SYMBOLS)
   {
      if(gStateCache[symIdx].valid && gStateCache[symIdx].symbol==symbol && gStateCache[symIdx].baseBarTime==baseBarTimeCache &&
         gStateCache[symIdx].positionsCount==positionsCount && gStateCache[symIdx].basketDir==basketDirCacheNow &&
         gStateCache[symIdx].avgQ==avgQCache && gStateCache[symIdx].lastEntryQ==lastEntryQCache)
      {
         CloneState(gStateCache[symIdx].state, state);
         usedStateCache=true;
      }
   }

   if(!usedStateCache)
   {
      BuildState(symbol,symIdx,positionsCount,trades,state);
      if(ShouldUseFastStateCache() && symIdx>=0 && symIdx<MAX_SYMBOLS)
      {
         gStateCache[symIdx].valid=true;
         gStateCache[symIdx].symbol=symbol;
         gStateCache[symIdx].baseBarTime=baseBarTimeCache;
         gStateCache[symIdx].positionsCount=positionsCount;
         gStateCache[symIdx].basketDir=basketDirCacheNow;
         gStateCache[symIdx].avgQ=avgQCache;
         gStateCache[symIdx].lastEntryQ=lastEntryQCache;
         CloneState(state, gStateCache[symIdx].state);
      }
   }

   bool extreme=IsExtremeState(symIdx,positionsCount);
   if(SeparateExtremeStates)
   {
      int sz=ArraySize(state);
      if(sz>0) state[sz-1]=(extreme?1.0:0.0);
   }

   double atrRatio = GetAtrRatioCached_Base(symbol);
   int activeRegime = (UseRegimeBank ? RegimeIndexFromRatio(atrRatio) : 0);

   int wantIn = ArraySize(state);
   if(gDQN[symIdx][activeRegime].input_dim != wantIn)
   {
      for(int r=0;r<REGIME_COUNT;r++)
         InitOrRandomizeDQN(symIdx,r,wantIn);
   }

   double qNow[];
   DQNForward(symIdx,activeRegime,state,qNow);

   int basketDirNow = (positionsCount>0 ? BasketDir(symbol,magic) : 0);
   CaptureBasketSnapshot(symbol,symIdx,magic,activeRegime,basketDirNow,positionsCount,state,qNow);
   UpdateDDEventLifecycle(symIdx, activeRegime, basketDirNow, state, qNow);

   AccrueDenseBasketHealthFeedback(symbol, symIdx, magic, positionsCount, extreme);

   UpdateForcedEntryWatchdog(symIdx);
   bool forceEntryNow = ForcedEntryRequiredNow(symIdx);

   DecisionSupportContext liveSupport;
   BuildDecisionSupportContext(symIdx, activeRegime, state, qNow, forceEntryNow, basketDirNow, liveSupport);

   double combinedTrend=0.0;
   double reversalRisk=0.0;
   ComputeLiveTrendBiasAndReversal(symbol, symIdx, activeRegime, liveSupport, combinedTrend, reversalRisk);

   bool zscoreTradingPaused = (ZScoreBlockAllTrading && IsZScoreTradingPaused(symIdx));

   if(UseZScoreRiskGuard && ZScoreBlockAllTrading && UseZScoreCloseSmallBasket)
   {
      int closeMaxTrades = MathMax(0, ZScoreCloseBasketMaxTrades);
      bool smallBasket = (positionsCount > 0 && positionsCount <= closeMaxTrades);
      bool freshExtremeSignal = (gZScoreExtremeNow[symIdx] &&
                                 gZScoreLastClosedBarTF[symIdx] > 0 &&
                                 gZScoreCloseHandledBar[symIdx] != gZScoreLastClosedBarTF[symIdx]);

      if(smallBasket && freshExtremeSignal)
      {
         double closedProfitRisk = 0.0;
         int closedCountRisk = 0;
         if(ClosePositions(symbol, magic, closedProfitRisk, closedCountRisk))
         {
            gZScoreCloseHandledBar[symIdx] = gZScoreLastClosedBarTF[symIdx];
            CountOpenPositions();
            positionsCount = gPositionsCount[symIdx];
            firstTradeTime = gFirstTradeTime[symIdx];
            gDecisionSupportCache[symIdx].valid = false;

            if(positionsCount == 0)
            {
               firstTradeTime = 0;
               ResetDenseBasketHealthTracker(symIdx);
            }
            return;
         }

      }
   }

   int hedgeMagic = ZScoreEmergencyHedgeMagic(magic);
   bool emergencyHedgeActive = HasOpenPositionsByMagic(symbol, hedgeMagic);
   if(!emergencyHedgeActive && gZScoreEmergencyHedgeActive[symIdx])
   {
      gZScoreEmergencyHedgeActive[symIdx] = false;
      gZScoreEmergencyOriginalDir[symIdx] = 0;
      gZScoreEmergencyHedgeBar[symIdx] = 0;
   }
   else if(emergencyHedgeActive)
   {
      gZScoreEmergencyHedgeActive[symIdx] = true;
      if(gZScoreEmergencyOriginalDir[symIdx]==0 && basketDirNow!=0)
         gZScoreEmergencyOriginalDir[symIdx] = basketDirNow;
   }

   bool freshExtremeSignal = (gZScoreExtremeNow[symIdx] &&
                              gZScoreLastClosedBarTF[symIdx] > 0 &&
                              gZScoreCloseHandledBar[symIdx] != gZScoreLastClosedBarTF[symIdx]);

   if(UseZScoreRiskGuard && ZScoreBlockAllTrading && UseZScoreEmergencyHedge && freshExtremeSignal)
   {
      int closeMaxTrades = MathMax(0, ZScoreCloseBasketMaxTrades);
      bool deepBasket = (positionsCount > closeMaxTrades);
      if(deepBasket && positionsCount > 0 && basketDirNow != 0 && !emergencyHedgeActive)
      {
         if(OpenEmergencyHedgeAgainstBasket(symbol, magic, hedgeMagic, basketDirNow))
         {
            gZScoreEmergencyHedgeActive[symIdx] = true;
            gZScoreEmergencyOriginalDir[symIdx] = basketDirNow;
            gZScoreEmergencyHedgeBar[symIdx] = gZScoreLastClosedBarTF[symIdx];
            gZScoreCloseHandledBar[symIdx] = gZScoreLastClosedBarTF[symIdx];
            emergencyHedgeActive = true;
         }
      }
   }

   if(gZScoreEmergencyHedgeActive[symIdx])
   {
      emergencyHedgeActive = HasOpenPositionsByMagic(symbol, hedgeMagic);
      if(!emergencyHedgeActive)
      {
         gZScoreEmergencyHedgeActive[symIdx] = false;
         gZScoreEmergencyOriginalDir[symIdx] = 0;
         gZScoreEmergencyHedgeBar[symIdx] = 0;
      }
   }

   double hedgePnL = (gZScoreEmergencyHedgeActive[symIdx] ? SumPositionProfitByMagic(symbol, hedgeMagic) : 0.0);
   double mainPnL = SumPositionProfitByMagic(symbol, magic);
   double combinedPnL = mainPnL + hedgePnL;
   double combinedCloseTarget = (RecoveryCloseAtBreakevenOnly ? 0.0 : RecoveryCombinedCloseMoney);

   if(gZScoreEmergencyHedgeActive[symIdx] && combinedPnL >= combinedCloseTarget)
   {
      double c1=0.0,c2=0.0;
      int n1=0,n2=0;
      bool ok1=ClosePositions(symbol, magic, c1, n1);
      bool ok2=ClosePositions(symbol, hedgeMagic, c2, n2);
      if(ok1 || ok2)
      {
         gZScoreEmergencyHedgeActive[symIdx] = false;
         gZScoreEmergencyOriginalDir[symIdx] = 0;
         gZScoreEmergencyHedgeBar[symIdx] = 0;
         CountOpenPositions();
         positionsCount = gPositionsCount[symIdx];
         firstTradeTime = gFirstTradeTime[symIdx];
         gDecisionSupportCache[symIdx].valid = false;
         if(positionsCount == 0)
         {
            firstTradeTime = 0;
            ResetDenseBasketHealthTracker(symIdx);
         }
         return;
      }
   }

   bool recoveryAddsAllowed = (!gZScoreEmergencyHedgeActive[symIdx] ||
                               (UseRecoveryAfterEmergencyHedge &&
                                !zscoreTradingPaused &&
                                gZScoreQuietBars[symIdx] >= MathMax(1, RecoveryRestartQuietBars)));
   bool blockNewFirstEntries = (zscoreTradingPaused || gZScoreEmergencyHedgeActive[symIdx]);

   //==========================================================
   // 1) NO OPEN BASKET -> DQN decides whether to open first leg
   //==========================================================
   if(positionsCount==0)
   {
      if(blockNewFirstEntries)
         return;

      if(IsReentryCooldownActive(symbol, symIdx) &&
         !(forceEntryNow && ForceEntryBypassesReentryCooldown))
      {
         return;
      }

      int action = (forceEntryNow
                    ? DQNSelectForcedEntryAction(symIdx,activeRegime,state)
                    : DQNSelectAction(symIdx,activeRegime,state));

      ENUM_ORDER_TYPE orderType=ORDER_TYPE_BUY;
      bool shouldTrade=false;

      if(action==1){ orderType=ORDER_TYPE_BUY;  shouldTrade=true; }
      else if(action==2){ orderType=ORDER_TYPE_SELL; shouldTrade=true; }

      if(zscoreTradingPaused)
         shouldTrade=false;

      // Normal mode respects danger policy.
      // Forced-entry mode can bypass first-entry blocking if configured.
      if(!forceEntryNow || !ForcedEntryIgnoreDanger)
      {
         if(UseDangerBrain && DangerActionPolicy==0)
         {
            BrainMode m=gMode[symIdx];
            if(m==MODE_DANGER && DangerBlocksNewEntries)
               shouldTrade=false;
         }

         if(UseDangerBrain && DangerActionPolicy!=0)
         {
            if(!AllowActionByPolicy(symIdx, positionsCount, action, orderType))
               shouldTrade=false;
         }
      }

      if(shouldTrade)
      {
         if(!PassesOpenIntervalGate(symbol, symIdx))
         {
            if(VerboseLogging)
               Print("ENTRY BLOCKED | open interval gate | symbol=", symbol,
                     " secSinceLast=", (int)(TimeCurrent()-gLastTradeOpenTime[symIdx]));
            shouldTrade=false;
         }
      }

      if(shouldTrade)
      {
         double baseVol0=NormalizeVolume(symbol, Lots*GetEquityBudgetScale());
         double vol0=baseVol0;
         bool deepBasketStrongFirstTradeArmed=ShouldUseDeepBasketStrongFirstTrade(symIdx);

         if(deepBasketStrongFirstTradeArmed)
         {
            vol0=NormalizeVolume(symbol, baseVol0*DeepBasketStrongFirstTradeMult);

            if(VerboseLogging)
               Print("DEEP BASKET STRONG FIRST TRADE USED | symbol=", symbol,
                     " closedCount=", gDeepBasketStrongEntryClosedCount[symIdx],
                     " baseVol=", DoubleToString(baseVol0,2),
                     " boostedVol=", DoubleToString(vol0,2),
                     " mult=", DoubleToString(DeepBasketStrongFirstTradeMult,2));
         }

         if(!blockNewFirstEntries && OpenPosition(symbol,orderType,vol0,0,0,magic))
         {
            if(deepBasketStrongFirstTradeArmed)
               ClearDeepBasketStrongFirstTradeArm(symIdx);

            int positionsAfter=0;
            int basketDirAfter=0;
            double entryPrice0=0.0;
            double entryVol0=vol0;
            double basketAvg0=0.0;
            datetime entryTime0=TimeCurrent();

            if(BuildPostTradeSnapshot(symbol,
                                      symIdx,
                                      magic,
                                      orderType,
                                      vol0,
                                      positionsAfter,
                                      basketDirAfter,
                                      entryPrice0,
                                      entryVol0,
                                      basketAvg0,
                                      entryTime0))
            {
               positionsCount = positionsAfter;
               firstTradeTime = gFirstTradeTime[symIdx];
               if(firstTradeTime==0) firstTradeTime=entryTime0;

               MarkTradeOpened(symIdx, entryTime0);

               gBadHaveBasketFP[symIdx]=false;
               gBadHaveMiniBasket[symIdx]=false;

               PendingAdd(symIdx,
                          activeRegime,
                          action,
                          basketDirAfter,
                          positionsAfter,
                          state,
                          entryPrice0,
                          entryVol0,
                          positionsAfter,
                          basketAvg0,
                          entryTime0);

               if(isTraining && AllowLearnEntryOrAveraging(symIdx))
               {
                  double reward=ComputeOpenRewardV2(symIdx,
                                                    (orderType==ORDER_TYPE_BUY),
                                                    0,
                                                    combinedTrend,
                                                    reversalRisk,
                                                    extreme);
                  double nextStateAfterOpen[];
                  BuildPostOpenNextState(symbol, symIdx, positionsAfter, gTrades[symIdx], nextStateAfterOpen);
                  SubmitTransitionWithNextState(symIdx,activeRegime,state,action,reward,false,nextStateAfterOpen);
               }
            }
            else
            {
               positionsCount = gPositionsCount[symIdx];
               firstTradeTime = gFirstTradeTime[symIdx];
               if(firstTradeTime==0) firstTradeTime=TimeCurrent();
            }
         }

         Print("NO BASKET | action=", action,
               " epsilon=", DoubleToString(currentEpsilon,4),
               " danger=", DoubleToString(gPDanger[symIdx],4),
               " mode=", (int)gMode[symIdx],
               " forceEntry=", (forceEntryNow ? "YES" : "NO"),
               " lastOpenAgoSec=", (int)(TimeCurrent()-gLastTradeOpenTime[symIdx]));
      }
   }
   //==========================================================
   // 2) OPEN BASKET -> ORIGINAL GRID ADD LOGIC ONLY
   //==========================================================
   else
   {
      if(positionsCount < MaxTrades)
      {
         int basketDirLive = BasketDir(symbol,magic);
         if(basketDirLive!=0)
         {
            ENUM_POSITION_TYPE pt = (basketDirLive>0 ? POSITION_TYPE_BUY : POSITION_TYPE_SELL);

            double buyLevel=0.0, sellLevel=0.0;
            CalculateLevelsSimple(symbol,symIdx,positionsCount,trades,magic,buyLevel,sellLevel);

            if(pt==POSITION_TYPE_BUY)
            {
               double candidateBuyPrice=SymbolInfoDouble(symbol,SYMBOL_ASK);
               if(candidateBuyPrice<=0.0) candidateBuyPrice=ask;

               if(bid<=buyLevel)
               {
                  if(!PassesOpenIntervalGate(symbol, symIdx))
                  {
                     if(VerboseLogging)
                        Print("ADD BLOCKED | open interval gate | symbol=", symbol,
                              " secSinceLast=", (int)(TimeCurrent()-gLastTradeOpenTime[symIdx]),
                              " candidate=", DoubleToString(candidateBuyPrice,_Digits));
                  }
                  else
                  {
                     double requiredAddGap=0.0, actualAddGap=0.0, lastAddPrice=0.0;
                     if(!PassesMinAddSpacingGate(symbol, symIdx, magic, basketDirLive, candidateBuyPrice,
                                                 requiredAddGap, actualAddGap, lastAddPrice))
                     {
                        if(VerboseLogging)
                           Print("ADD BLOCKED | spacing gate | symbol=", symbol,
                                 " candidate=", DoubleToString(candidateBuyPrice,_Digits),
                                 " buyLevel=", DoubleToString(buyLevel,_Digits),
                                 " lastEntry=", DoubleToString(lastAddPrice,_Digits),
                                 " actualGap=", DoubleToString(actualAddGap,_Digits),
                                 " requiredGap=", DoubleToString(requiredAddGap,_Digits));
                     }
                     else
                     {
                        int preOpenCount = positionsCount;
                        double smartSpacingMult=1.0;
                        double smartLotScale=1.0;
                        double smartAddRisk=0.0;
                        bool smartAddOk=EvaluateSmartAddGate(symbol,
                                                             symIdx,
                                                             activeRegime,
                                                             state,
                                                             basketDirLive,
                                                             preOpenCount,
                                                             combinedTrend,
                                                             reversalRisk,
                                                             extreme,
                                                             smartSpacingMult,
                                                             smartLotScale,
                                                             smartAddRisk);

                        double basketAvgLive=BasketAvgPrice(trades,gTradeLots[symIdx],candidateBuyPrice);
                        double baseStep=MathMax(0.0,basketAvgLive-buyLevel);
                        double gatedBuyLevel=(smartSpacingMult>1.0 ? basketAvgLive-baseStep*smartSpacingMult : buyLevel);

                        if(!smartAddOk)
                        {
                           if(VerboseLogging)
                              Print("ADD BLOCKED | smart gate | symbol=", symbol,
                                    " risk=", DoubleToString(smartAddRisk,4),
                                    " buyLevel=", DoubleToString(buyLevel,_Digits));
                        }
                        else if(bid>gatedBuyLevel)
                        {
                           if(VerboseLogging)
                              Print("ADD DELAYED | widened buy level | symbol=", symbol,
                                    " risk=", DoubleToString(smartAddRisk,4),
                                    " bid=", DoubleToString(bid,_Digits),
                                    " gatedBuyLevel=", DoubleToString(gatedBuyLevel,_Digits));
                        }
                        else
                        {
                           double vol=NormalizeVolume(symbol,CalculateLot(symbol,preOpenCount)*smartLotScale);

                           if(recoveryAddsAllowed && OpenPosition(symbol,ORDER_TYPE_BUY,vol,0,0,magic))
                           {
                              int positionsAfter=0;
                              int basketDirAfter=0;
                              double entryPriceBuy=0.0;
                              double entryVolBuy=vol;
                              double basketAvgBuy=0.0;
                              datetime entryTimeBuy=TimeCurrent();

                              if(BuildPostTradeSnapshot(symbol,
                                                        symIdx,
                                                        magic,
                                                        ORDER_TYPE_BUY,
                                                        vol,
                                                        positionsAfter,
                                                        basketDirAfter,
                                                        entryPriceBuy,
                                                        entryVolBuy,
                                                        basketAvgBuy,
                                                        entryTimeBuy))
                              {
                                 positionsCount = positionsAfter;
                                 if(firstTradeTime==0)
                                    firstTradeTime = gFirstTradeTime[symIdx];

                                 MarkTradeOpened(symIdx, entryTimeBuy);

                                 int learnAction = 1;

                                 PendingAdd(symIdx,
                                            activeRegime,
                                            learnAction,
                                            basketDirAfter,
                                            positionsAfter,
                                            state,
                                            entryPriceBuy,
                                            entryVolBuy,
                                            positionsAfter,
                                            basketAvgBuy,
                                            entryTimeBuy);

                                 if(isTraining && AllowLearnEntryOrAveraging(symIdx))
                                 {
                                    double reward=ComputeOpenRewardV2(symIdx,true,preOpenCount,combinedTrend,reversalRisk,extreme);
                                    double nextStateAfterOpen[];
                                    BuildPostOpenNextState(symbol, symIdx, positionsAfter, gTrades[symIdx], nextStateAfterOpen);
                                    SubmitTransitionWithNextState(symIdx,activeRegime,state,learnAction,reward,false,nextStateAfterOpen);
                                 }
                              }
                              else
                              {
                                 positionsCount = gPositionsCount[symIdx];
                              }
                           }
                        }
                     }
                  }
               }
            }
            else
            {
               double candidateSellPrice=SymbolInfoDouble(symbol,SYMBOL_BID);
               if(candidateSellPrice<=0.0) candidateSellPrice=bid;

               if(ask>=sellLevel)
               {
                  if(!PassesOpenIntervalGate(symbol, symIdx))
                  {
                     if(VerboseLogging)
                        Print("ADD BLOCKED | open interval gate | symbol=", symbol,
                              " secSinceLast=", (int)(TimeCurrent()-gLastTradeOpenTime[symIdx]),
                              " candidate=", DoubleToString(candidateSellPrice,_Digits));
                  }
                  else
                  {
                     double requiredAddGap=0.0, actualAddGap=0.0, lastAddPrice=0.0;
                     if(!PassesMinAddSpacingGate(symbol, symIdx, magic, basketDirLive, candidateSellPrice,
                                                 requiredAddGap, actualAddGap, lastAddPrice))
                     {
                        if(VerboseLogging)
                           Print("ADD BLOCKED | spacing gate | symbol=", symbol,
                                 " candidate=", DoubleToString(candidateSellPrice,_Digits),
                                 " sellLevel=", DoubleToString(sellLevel,_Digits),
                                 " lastEntry=", DoubleToString(lastAddPrice,_Digits),
                                 " actualGap=", DoubleToString(actualAddGap,_Digits),
                                 " requiredGap=", DoubleToString(requiredAddGap,_Digits));
                     }
                     else
                     {
                        int preOpenCount = positionsCount;
                        double smartSpacingMult=1.0;
                        double smartLotScale=1.0;
                        double smartAddRisk=0.0;
                        bool smartAddOk=EvaluateSmartAddGate(symbol,
                                                             symIdx,
                                                             activeRegime,
                                                             state,
                                                             basketDirLive,
                                                             preOpenCount,
                                                             combinedTrend,
                                                             reversalRisk,
                                                             extreme,
                                                             smartSpacingMult,
                                                             smartLotScale,
                                                             smartAddRisk);

                        double basketAvgLive=BasketAvgPrice(trades,gTradeLots[symIdx],candidateSellPrice);
                        double baseStep=MathMax(0.0,sellLevel-basketAvgLive);
                        double gatedSellLevel=(smartSpacingMult>1.0 ? basketAvgLive+baseStep*smartSpacingMult : sellLevel);

                        if(!smartAddOk)
                        {
                           if(VerboseLogging)
                              Print("ADD BLOCKED | smart gate | symbol=", symbol,
                                    " risk=", DoubleToString(smartAddRisk,4),
                                    " sellLevel=", DoubleToString(sellLevel,_Digits));
                        }
                        else if(ask<gatedSellLevel)
                        {
                           if(VerboseLogging)
                              Print("ADD DELAYED | widened sell level | symbol=", symbol,
                                    " risk=", DoubleToString(smartAddRisk,4),
                                    " ask=", DoubleToString(ask,_Digits),
                                    " gatedSellLevel=", DoubleToString(gatedSellLevel,_Digits));
                        }
                        else
                        {
                           double vol=NormalizeVolume(symbol,CalculateLot(symbol,preOpenCount)*smartLotScale);

                           if(recoveryAddsAllowed && OpenPosition(symbol,ORDER_TYPE_SELL,vol,0,0,magic))
                           {
                              int positionsAfter=0;
                              int basketDirAfter=0;
                              double entryPriceSell=0.0;
                              double entryVolSell=vol;
                              double basketAvgSell=0.0;
                              datetime entryTimeSell=TimeCurrent();

                              if(BuildPostTradeSnapshot(symbol,
                                                        symIdx,
                                                        magic,
                                                        ORDER_TYPE_SELL,
                                                        vol,
                                                        positionsAfter,
                                                        basketDirAfter,
                                                        entryPriceSell,
                                                        entryVolSell,
                                                        basketAvgSell,
                                                        entryTimeSell))
                              {
                                 positionsCount = positionsAfter;
                                 if(firstTradeTime==0)
                                    firstTradeTime = gFirstTradeTime[symIdx];

                                 MarkTradeOpened(symIdx, entryTimeSell);

                                 int learnAction = 2;

                                 PendingAdd(symIdx,
                                            activeRegime,
                                            learnAction,
                                            basketDirAfter,
                                            positionsAfter,
                                            state,
                                            entryPriceSell,
                                            entryVolSell,
                                            positionsAfter,
                                            basketAvgSell,
                                            entryTimeSell);

                                 if(isTraining && AllowLearnEntryOrAveraging(symIdx))
                                 {
                                    double reward=ComputeOpenRewardV2(symIdx,false,preOpenCount,combinedTrend,reversalRisk,extreme);
                                    double nextStateAfterOpen[];
                                    BuildPostOpenNextState(symbol, symIdx, positionsAfter, gTrades[symIdx], nextStateAfterOpen);
                                    SubmitTransitionWithNextState(symIdx,activeRegime,state,learnAction,reward,false,nextStateAfterOpen);
                                 }
                              }
                              else
                              {
                                 positionsCount = gPositionsCount[symIdx];
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //==========================================================
   // 3) HYBRID BASKET TP
   //==========================================================
   if(positionsCount>0 && !gZScoreEmergencyHedgeActive[symIdx])
   {
      int basketDirLive=BasketDir(symbol,magic);
      if(basketDirLive!=0)
      {
         double avgTP=0.0;
         double targetPrice=0.0;
         bool shouldClose=BasketTPHit(symbol,symIdx,magic,avgTP,targetPrice);

         if(shouldClose)
         {
            int positionsBeforeClose=positionsCount;

            BasketCloseResult closeRes;
            PositionCloseItem closedItems[];
            if(ClosePositionsDetailed(symbol,magic,closeRes,closedItems))
            {
               CountOpenPositions();
               positionsCount = gPositionsCount[symIdx];
               firstTradeTime = gFirstTradeTime[symIdx];
               gDecisionSupportCache[symIdx].valid=false;

               if(closeRes.allClosed)
               {
                  firstTradeTime=0;
                  ResetDenseBasketHealthTracker(symIdx);
               }

               if(isTraining && AllowLearnClose(symIdx))
               {
                  double reward = ComputeCloseRewardV2(symIdx, closeRes, positionsBeforeClose, extreme);

                  double nextState[];
                  BuildCloseNextState(symbol,symIdx,positionsCount,nextState);

                  double resolvedRewardSum=0.0;
                  int resolvedTransitions=PendingResolveForSymbolClose(symIdx, reward, closeRes.allClosed, nextState, closedItems, resolvedRewardSum);
                  if(resolvedTransitions>0)
                  {
                     episodeCount += resolvedTransitions;
                     totalReward += resolvedRewardSum;
                  }
               }
            }
         }
      }
   }

   if(ShouldTrainReplayBatchNow(symIdx, symbol))
      TrainReplayBatch();
}

