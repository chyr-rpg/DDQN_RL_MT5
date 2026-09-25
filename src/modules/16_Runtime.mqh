//+------------------------------------------------------------------+
//| 16_Runtime.mqh                                                  |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| MT5 lifecycle handlers (OnInit/OnDeinit/OnTimer/OnTick) and DQN |
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

int OnInit()
{
   for(int i=0;i<MAX_SYMBOLS;i++)
   {
      gZScoreLastClosedBarTF[i]=0;
      gZScoreLastValue[i]=0.0;
      gZScoreLastAbs[i]=0.0;
      gZScoreExtremeNow[i]=false;
      gZScorePauseTrading[i]=false;
      gZScoreQuietBars[i]=0;
      gZScoreLastExtremeBarTime[i]=0;
   }

   InitPerfCaches();
   trade.SetDeviationInPoints(Slippage);
   MathSrand((int)GetTickCount());

   bool loadedPersistentMemory = false;

   gT1Enter=T1_CautionEnter;
   gT1Exit =T1_CautionExit;
   gT2Enter=T2_DangerEnter;
   gT2Exit =T2_DangerExit;

   gSymbolCount=1;
   gSymbols[0]=_Symbol;
   gMagics[0]=InpMagic;

   PendingInit();
   ArrayResize(gReplay,0);
   ArrayResize(gQMem,0);
   ArrayResize(gBasketHistory,0);
   ArrayResize(gTickTrace,0);
   ArrayResize(gDDEvents,0);

   gDDEventActive=false;
   gDDEventTriggerTime=0;
   gDDEventTriggerDD=0.0;
   gDDEventHard=false;
   gDDEventStartBasketIndex=-1;
   gDDEventStartTickIndex=-1;
   gDDEventPostBasketCount=0;

   for(int i=0;i<gSymbolCount;i++)
   {
      gGridLastBar[i]=0;
      gGridStepCache[i]=0.0;
      gGridActiveChannel[i]=0;
      gGridActiveStep[i]=0.0;

      gExtremeDDStart[i]=0;
      gExtremeDDArmed[i]=false;

      gPositionsCount[i]=0;
      gTrades[i].Clear();
      gTradeLots[i].Clear();
      gFirstTradeTime[i]=0;

      gLastBarTime[i]=0;
      gLastReplayDecisionBarTime[i]=0;
      gReplayDecisionBarCounter[i]=0;
      ResetDenseBasketHealthTracker(i);
      gLoadedDQNFastModeMeta[i]=false;
      gLoadedDQNMetaNote[i]="";
      gDecisionCtxBarTime[i]=0;
      gDecisionCtxHourBucket[i]=0;
      gDecisionCtxRegimeType[i]=0;
      gDecisionCtxPatternType[i]=0;
      gDecisionCtxLiquidityType[i]=0;
      gDecisionCtxSessionType[i]=0;
      gDecisionCtxAtrRatio[i]=0.0;
      gDecisionCtxValid[i]=false;
      gFastQMemLastRefreshTime[i]=0;
      gFastQMemLastDecisionBarTime[i]=0;
      gFastQMemDecisionBarCounter[i]=0;
      gFastQMemCachedRegime[i]=-1000;
      gFastQMemCachedValid[i]=false;
      gFastQMemCachedConf[i]=0.0;
      for(int a=0;a<3;a++)
         gFastQMemCachedQ[i][a]=0.0;

      ResetDecisionSupportContext(gDecisionSupportCache[i]);

      gD1LastBarTime[i]=0;
      gD1DistCache[i]=0.0;
      gD1SlopeCache[i]=0.0;
      gD1SideDurCache[i]=0.0;

      gATRLastBarTime[i]=0;
      gATRratioCache[i]=1.0;

      hRSI_Base[i]  = iRSI(gSymbols[i], BaseTF, RSI_Period, PRICE_CLOSE);
      hCCI_Base[i]  = iCCI(gSymbols[i], BaseTF, CCI_Period, PRICE_TYPICAL);
      hMACD_Base[i] = iMACD(gSymbols[i], BaseTF, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
      hEMA_Base[i]  = iMA(gSymbols[i], BaseTF, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
      hRVI_Base[i]  = iRVI(gSymbols[i], BaseTF, RVI_Period);
      hATRfast_Base[i] = iATR(gSymbols[i], BaseTF, ATR_FastPeriod);
      hATRslow_Base[i] = iATR(gSymbols[i], BaseTF, ATR_SlowPeriod);

      gIndLastBarTime[i]=0;
      gRSI_Base[i]=50.0;
      gCCI_Base[i]=0.0;
      gMACD_BaseMain[i]=0.0;
      gEMA_BaseVal[i]=0.0;
      gRVI_BaseMain[i]=0.0;
      gATRfast_BaseVal[i]=0.0;
      gATRslow_BaseVal[i]=0.0;

      hRSI_H1[i]=INVALID_HANDLE; hMACD_H1[i]=INVALID_HANDLE; hEMA_H1[i]=INVALID_HANDLE; hRVI_H1[i]=INVALID_HANDLE;
      hATRfast_H1[i]=INVALID_HANDLE; hATRslow_H1[i]=INVALID_HANDLE;
      gH1LastBar[i]=0; gRSI_H1v[i]=50.0; gMACD_H1v[i]=0.0; gEMA_H1v[i]=0.0; gRVI_H1v[i]=0.0; gATRratio_H1[i]=1.0;
      
      gLastTradeOpenTime[i]=TimeCurrent();
      gForcedEntryActive[i]=false;
      gForcedEntryArmTime[i]=0;
      gForcedEntryDeadline[i]=0;
       
      if(UseH1Features)
      {
         hRSI_H1[i]     = iRSI(gSymbols[i], H1_TF, RSI_Period, PRICE_CLOSE);
         hMACD_H1[i]    = iMACD(gSymbols[i], H1_TF, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
         hEMA_H1[i]     = iMA(gSymbols[i], H1_TF, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
         hRVI_H1[i]     = iRVI(gSymbols[i], H1_TF, RVI_Period);
         hATRfast_H1[i] = iATR(gSymbols[i], H1_TF, ATR_FastPeriod);
         hATRslow_H1[i] = iATR(gSymbols[i], H1_TF, ATR_SlowPeriod);
      }

      hRSI_H4[i]=INVALID_HANDLE; hMACD_H4[i]=INVALID_HANDLE; hEMA_H4[i]=INVALID_HANDLE; hRVI_H4[i]=INVALID_HANDLE;
      hATRfast_H4[i]=INVALID_HANDLE; hATRslow_H4[i]=INVALID_HANDLE;
      gH4LastBar[i]=0; gRSI_H4v[i]=50.0; gMACD_H4v[i]=0.0; gEMA_H4v[i]=0.0; gRVI_H4v[i]=0.0; gATRratio_H4[i]=1.0;

      if(UseH4Features)
      {
         hRSI_H4[i]     = iRSI(gSymbols[i], H4_TF, RSI_Period, PRICE_CLOSE);
         hMACD_H4[i]    = iMACD(gSymbols[i], H4_TF, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
         hEMA_H4[i]     = iMA(gSymbols[i], H4_TF, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
         hRVI_H4[i]     = iRVI(gSymbols[i], H4_TF, RVI_Period);
         hATRfast_H4[i] = iATR(gSymbols[i], H4_TF, ATR_FastPeriod);
         hATRslow_H4[i] = iATR(gSymbols[i], H4_TF, ATR_SlowPeriod);
      }

      double tmpState[];
      BuildState(gSymbols[i], i, 0, gTrades[i], tmpState);
      int inDim = ArraySize(tmpState);
      if(inDim<=0) inDim=8;

      bool unifiedLoaded = LoadUnifiedPersistenceForSymbol(i,inDim);
      if(unifiedLoaded)
         loadedPersistentMemory = true;
      else
      {
         // Unified persistence is the only DQN persistence path now.
         // If no persisted model exists yet, initialize a unified model
         // in regime slot 0 and mirror it into the compatibility regime slots.
         InitOrRandomizeDQN(i,0,inDim);
         for(int r=1;r<REGIME_COUNT;r++)
         {
            gDQN[i][r] = gDQN[i][0];
            gTargetDQN[i][r] = gTargetDQN[i][0];
         }
      }

      gMode[i]=MODE_NORMAL;
      gModeLastChange[i]=0;
      gPDanger[i]=0.0;
      gLRScale[i]=1.0;

      gFPLastBar[i]=0;
      for(int k=0;k<6;k++) { gFPCache[i][k]=0.0; gFPPrev[i][k]=0.0; }
      gFPPrevValid[i]=false;

      gMiniLastBar[i]=0;
      gMiniValid[i]=false;
      gMiniPrevValid[i]=false;
      for(int k=0;k<MINI_DIM;k++)
      {
         gMiniCache[i][k]=0.0;
         gMiniPrev[i][k]=0.0;
      }

      gBadDDStart[i]=0;
      gBadDDConfirmed[i]=false;
      gBadLabel[i]=-1;
      gBadBasketDir[i]=0;
      gBadTrendDir[i]=0;
      gBadAtrRatio[i]=1.0;

      gBadHaveBasketFP[i]=false;
      gBadHaveQStart[i]=false;
      gBadHavePreFP[i]=false;
      gBadHaveEndFP[i]=false;

      gBadHaveMiniStart[i]=false;
      gBadHaveMiniBasket[i]=false;
      gBadHaveMiniPre[i]=false;
      gBadHaveMiniEnd[i]=false;

      for(int k=0;k<6;k++)
      {
         gBadFPStart[i][k]=0.0;
         gBadFPBasketOpen[i][k]=0.0;
         gBadFPPre[i][k]=0.0;
         gBadFPEnd[i][k]=0.0;
      }
      for(int a=0;a<8;a++) gBadQStart[i][a]=0.0;

      for(int k=0;k<MINI_DIM;k++)
      {
         gBadMiniStart[i][k]=0.0;
         gBadMiniBasketOpen[i][k]=0.0;
         gBadMiniPre[i][k]=0.0;
         gBadMiniEnd[i][k]=0.0;
      }

      gLastForcedStopLossMoney[i]=0.0;
      gLastForcedStopTime[i]=0;
      gLastForcedStopPendingLearn[i]=false;

      gForcedStopHaveFP[i]=false;
      gForcedStopHaveMini[i]=false;
      gForcedStopHaveQ[i]=false;

      gForcedStopBasketDir[i]=0;
      gForcedStopTrendDir[i]=0;
      gForcedStopAtrRatio[i]=1.0;
      gForcedStopLabel[i]=-1;

      for(int k=0;k<6;k++)
         gForcedStopFP[i][k]=0.0;

      for(int k=0;k<MINI_DIM;k++)
         gForcedStopMini[i][k]=0.0;

      for(int a=0;a<8;a++)
         gForcedStopQ[i][a]=0.0;

      gCachedSymbolPnL[i]=0.0;
      gHasPositionTypeCache[i]=false;
      gBasketDirCache[i]=0;
      gBasketAvgPriceCache[i]=0.0;
      gBasketAvgValid[i]=false;
   }

   if(UseDangerBrain)
   {
      string memFile = MQLInfoString(MQL_PROGRAM_NAME) + "_DangerMem_" + _Symbol + ".dat";
      bool loaded=false;
      if(SaveDangerMemory) loaded = LoadDangerMemoryFile(memFile);
      if(!loaded) SeedTestProtos();
      else loadedPersistentMemory = true;

      PruneProtosIfNeeded();
      EventSetTimer(3);
   }

   if(UseQMemory && SaveQMemory)
   {
      string qmemFile = MQLInfoString(MQL_PROGRAM_NAME) + "_QMem_" + _Symbol + ".dat";
      if(LoadQMemoryFile(qmemFile))
         loadedPersistentMemory = true;
   }

   if(UseDDEventMemory && SaveDDEventMemory)
   {
      string ddFile = MQLInfoString(MQL_PROGRAM_NAME) + "_DDEvents_" + _Symbol + ".dat";
      if(LoadDDEventMemoryFile(ddFile))
         loadedPersistentMemory = true;
   }

   gEAStartEquity = (EquityBudget>1e-9 ? EquityBudget : AccountInfoDouble(ACCOUNT_BALANCE));
   if(gEAStartEquity<=1e-9) gEAStartEquity=1000.0;

   gEAClosedProfit=0.0;
   gEquityLossStopResumeTime=0;
   gProfitPauseResumeTime=0;
   gProfitCycleClosedProfit=0.0;
   
   // virtual-budget model baseline stays unchanged
   maxEquity=gEAStartEquity;
   gTickEquityBaseline=gEAStartEquity;
   gTickBalanceBaseline=gEAStartEquity;
   gRewardBaselineTick=0;
   
   // separate real account/watchdog peak for equity stop only
   gAccountEquityStopPeak = GetWatchedAccountEquity();
   if(gAccountEquityStopPeak <= 1e-9)
      gAccountEquityStopPeak = AccountInfoDouble(ACCOUNT_BALANCE);
   if(gAccountEquityStopPeak <= 1e-9)
      gAccountEquityStopPeak = 1000.0;

   isTraining=TrainingMode;
   tickCounter=0;
   totalReward=0.0;
   episodeCount=0;
   gTargetSyncCounter=0;

   if(isTraining)
   {
      if(ResumeLowEpsilon && loadedPersistentMemory)
         currentEpsilon = MathMax(MinExplorationRate, ExplorationRate * LoadedEpsilonFactor);
      else
         currentEpsilon = ExplorationRate;
   }
   else
   {
      currentEpsilon = MinExplorationRate;
   }

   for(int i=0;i<MAX_SYMBOLS;i++)
   {
      gActiveBasketEpisodes[i].active=false;
      gActiveBasketEpisodes[i].episodeId=0;
      gActiveBasketEpisodes[i].symIdx=i;
      ArrayResize(gActiveBasketEpisodes[i].replayItemIndexes,0);

      gActiveEfficientPeriods[i].active=false;
      gActiveEfficientPeriods[i].periodId=0;
      gActiveEfficientPeriods[i].symIdx=i;
      ArrayResize(gActiveEfficientPeriods[i].replayItemIndexes,0);
   }

   SyncAllTargetNetworks();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{

   if(UseDangerBrain)
   {
      EventKillTimer();
      if(PruneOnDeinit) PruneProtosIfNeeded();
      if(SaveDangerMemory)
      {
         string memFile = MQLInfoString(MQL_PROGRAM_NAME) + "_DangerMem_" + _Symbol + ".dat";
         SaveDangerMemoryFile(memFile);
      }
   }

   if(UseQMemory && SaveQMemory)
   {
      string qmemFile = MQLInfoString(MQL_PROGRAM_NAME) + "_QMem_" + _Symbol + ".dat";
      SaveQMemoryFile(qmemFile);
   }

   if(UseDDEventMemory && SaveDDEventMemory)
   {
      if(gDDEventActive)
         FinalizeActiveDDEvent();

      string ddFile = MQLInfoString(MQL_PROGRAM_NAME) + "_DDEvents_" + _Symbol + ".dat";
      SaveDDEventMemoryFile(ddFile);
   }

   if(UseDQN && SaveQTable)
   {
      for(int i=0;i<gSymbolCount;i++)
         SaveUnifiedPersistenceForSymbol(i);
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      if(hRSI_Base[i]!=INVALID_HANDLE)  IndicatorRelease(hRSI_Base[i]);
      if(hCCI_Base[i]!=INVALID_HANDLE)  IndicatorRelease(hCCI_Base[i]);
      if(hMACD_Base[i]!=INVALID_HANDLE) IndicatorRelease(hMACD_Base[i]);
      if(hEMA_Base[i]!=INVALID_HANDLE)  IndicatorRelease(hEMA_Base[i]);
      if(hRVI_Base[i]!=INVALID_HANDLE)  IndicatorRelease(hRVI_Base[i]);
      if(hATRfast_Base[i]!=INVALID_HANDLE) IndicatorRelease(hATRfast_Base[i]);
      if(hATRslow_Base[i]!=INVALID_HANDLE) IndicatorRelease(hATRslow_Base[i]);

      if(hRSI_H1[i]!=INVALID_HANDLE)     IndicatorRelease(hRSI_H1[i]);
      if(hMACD_H1[i]!=INVALID_HANDLE)    IndicatorRelease(hMACD_H1[i]);
      if(hEMA_H1[i]!=INVALID_HANDLE)     IndicatorRelease(hEMA_H1[i]);
      if(hRVI_H1[i]!=INVALID_HANDLE)     IndicatorRelease(hRVI_H1[i]);
      if(hATRfast_H1[i]!=INVALID_HANDLE) IndicatorRelease(hATRfast_H1[i]);
      if(hATRslow_H1[i]!=INVALID_HANDLE) IndicatorRelease(hATRslow_H1[i]);

      if(hRSI_H4[i]!=INVALID_HANDLE)     IndicatorRelease(hRSI_H4[i]);
      if(hMACD_H4[i]!=INVALID_HANDLE)    IndicatorRelease(hMACD_H4[i]);
      if(hEMA_H4[i]!=INVALID_HANDLE)     IndicatorRelease(hEMA_H4[i]);
      if(hRVI_H4[i]!=INVALID_HANDLE)     IndicatorRelease(hRVI_H4[i]);
      if(hATRfast_H4[i]!=INVALID_HANDLE) IndicatorRelease(hATRfast_H4[i]);
      if(hATRslow_H4[i]!=INVALID_HANDLE) IndicatorRelease(hATRslow_H4[i]);
   }

   if(DrawZonesOnChart)
   {
      int total=ArraySize(zones);
      for(int i=0;i<total;i++)
      {
         ObjectDelete(0,zones[i].name);
         ObjectDelete(0,zones[i].name+"_lbl");
      }
   }
   
   ObjectDelete(0,gGridPanelName);
}

void OnTimer()
{
   if(!UseDangerBrain) return;

   datetime now=TimeCurrent();
   for(int i=0;i<gSymbolCount;i++)
   {
      string sym=gSymbols[i];
      datetime bt=iTime(sym, FingerprintTF, 0);
      if(bt==0) continue;
      if(bt==gFPLastBar[i]) continue;

      gFPLastBar[i]=bt;

      double f_now[];
      if(!ComputeFingerprint(sym, FingerprintTF, FingerprintBars, f_now)) continue;

      for(int k=0;k<6;k++) gFPPrev[i][k]=gFPCache[i][k];
      gFPPrevValid[i]=true;

      for(int k=0;k<6;k++) gFPCache[i][k]=f_now[k];

      double pd=0.0;
      int bestDangerIdx = DangerPredict(f_now, i, pd);
      gPDanger[i]=pd;

      UpdateMode(i, pd, now);

      if(bestDangerIdx>=0)
      {
         double exposure=ComputeExposureScore(i);
         if(pd>gT2Enter && exposure<0.05)
            ProtoScoreBump(bestDangerIdx, -0.02);
      }
   }
}

void OnTick()
{
   double equity=GetEAEquity();
   if(equity>maxEquity) maxEquity=equity;

   double watchedEq = GetWatchedAccountEquity();
   if(watchedEq > gAccountEquityStopPeak)
      gAccountEquityStopPeak = watchedEq;

   RefreshRewardTickBaseline();
   ResetProfitPauseIfExpired();
   ResetEquityLossStopCooldownIfExpired();
   
   if(HandleEquityLossStop())
   {
      StartEquityLossStopCooldown();
      CountOpenPositions();
      return;
   }

   if(CheckProfitPauseTrigger())
   {
      CloseAllPositions();
      CountOpenPositions();
      StartProfitPause();
      return;
   }

   if(ProfitPauseActive())
      return;

   if(UseEquityStop && CheckEquityStop())
   {
      CloseAllPositions();
      CountOpenPositions();

      for(int i=0;i<gSymbolCount;i++)
      {
         gFirstTradeTime[i]=0;
         gLastTradeOpenTime[i]=TimeCurrent();
         ResetForcedEntryState(i);
      }

      // reset only the real account-equity stop peak
      gAccountEquityStopPeak = GetWatchedAccountEquity();
      if(gAccountEquityStopPeak <= 1e-9)
         gAccountEquityStopPeak = AccountInfoDouble(ACCOUNT_BALANCE);

      return;
   }

   CountOpenPositions();

   if(UseDangerBrain && gFPLastBar[0]==0)
      OnTimer();

   datetime now = TimeCurrent();

   static datetime s_lastPendingExpireCheck = 0;
   bool runPendingExpire = false;
   if(UsePendingTransitions && ArraySize(gPending) > 0)
   {
      if(now != s_lastPendingExpireCheck)
      {
         s_lastPendingExpireCheck = now;
         runPendingExpire = true;
      }
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      string sym = gSymbols[i];

      UpdateBaseIndicatorsIfNewBar(i);
      UpdateH1IfNewBar(i);
      UpdateH4IfNewBar(i);
      UpdateMultiChannelGridStats(i);
      GetAtrRatioCached_Base(sym);
      UpdateZScoreRiskGuardForSymbol(sym, i);

      datetime lastClosed = iTime(sym, BaseTF, 1);
      bool isNewClosedBar = (lastClosed > 0 && lastClosed != gLastBarTime[i]);

      if(isNewClosedBar)
      {
         gLastBarTime[i] = lastClosed;

         DetectZones(sym);
         UpdateZones(sym);
         UpdateMiniStateCache(i);
         if(!MQLInfoInteger(MQL_TESTER)) UpdateStructureVisualization(sym);
         MaybePrintReplayDiagnostics(i);
      }
   }

   if(runPendingExpire)
      PendingExpireOld(now);

   if(CheckCCIExit())
      return;

   TrailingStop();
   CheckTakeProfit();

   if(!UseDQN)
      return;

   tickCounter++;
   if(isTraining && TrainingFreq>0 && (tickCounter % TrainingFreq)==0)
   {
      if(currentEpsilon > MinExplorationRate)
         currentEpsilon *= ExplorationDecay;

      if(currentEpsilon < MinExplorationRate)
         currentEpsilon = MinExplorationRate;
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      ManagePairWithDQN(gSymbols[i], i, gPositionsCount[i], gTrades[i], gMagics[i], gFirstTradeTime[i]);
      if(!MQLInfoInteger(MQL_TESTER)) UpdateGridStatusPanel(i);
   }
}

//+------------------------------------------------------------------+
void InitOrRandomizeDQN(int symIdx,int regime,int inputDim)
{
   if(inputDim <= 0) inputDim = 8;

   if(UseBranchScaffold)
      InitBranchLayoutForStateDim(inputDim);

   DQNNetwork net = gDQN[symIdx][regime];

   net.input_dim  = inputDim;
   net.hidden_dim = (HiddenSize>0 ? HiddenSize : 24);
   net.hidden_dim2= (HiddenSize2>0 ? HiddenSize2 : 0);
   net.output_dim = (ActionCount>0 ? ActionCount : 3);

   net.basket_h1 = (gBranchLayout.basketCount>0 ? BranchHidden1Size(gBranchLayout.basketCount) : 0);
   net.basket_h2 = (gBranchLayout.basketCount>0 ? BranchHidden2Size(gBranchLayout.basketCount, net.basket_h1) : 0);

   net.indicator_h1 = (gBranchLayout.indicatorCount>0 ? BranchHidden1Size(gBranchLayout.indicatorCount) : 0);
   net.indicator_h2 = (gBranchLayout.indicatorCount>0 ? BranchHidden2Size(gBranchLayout.indicatorCount, net.indicator_h1) : 0);

   net.volatility_h1 = (gBranchLayout.volatilityCount>0 ? BranchHidden1Size(gBranchLayout.volatilityCount) : 0);
   net.volatility_h2 = (gBranchLayout.volatilityCount>0 ? BranchHidden2Size(gBranchLayout.volatilityCount, net.volatility_h1) : 0);

   net.structure_h1 = (gBranchLayout.structureCount>0 ? BranchHidden1Size(gBranchLayout.structureCount) : 0);
   net.structure_h2 = (gBranchLayout.structureCount>0 ? BranchHidden2Size(gBranchLayout.structureCount, net.structure_h1) : 0);

   net.zone_h1 = (gBranchLayout.zoneCandleCount>0 ? BranchHidden1Size(gBranchLayout.zoneCandleCount) : 0);
   net.zone_h2 = (gBranchLayout.zoneCandleCount>0 ? BranchHidden2Size(gBranchLayout.zoneCandleCount, net.zone_h1) : 0);

   net.fusion_dim = net.basket_h2 + net.indicator_h2 + net.volatility_h2 + net.structure_h2 + net.zone_h2;
   if(net.fusion_dim <= 0) net.fusion_dim = net.input_dim;

   InitBranchEncoder(gBranchLayout.basketCount, net.basket_h1, net.basket_h2, net.basket_W1, net.basket_b1, net.basket_W2, net.basket_b2);
   InitBranchEncoder(gBranchLayout.indicatorCount, net.indicator_h1, net.indicator_h2, net.indicator_W1, net.indicator_b1, net.indicator_W2, net.indicator_b2);
   InitBranchEncoder(gBranchLayout.volatilityCount, net.volatility_h1, net.volatility_h2, net.volatility_W1, net.volatility_b1, net.volatility_W2, net.volatility_b2);
   InitBranchEncoder(gBranchLayout.structureCount, net.structure_h1, net.structure_h2, net.structure_W1, net.structure_b1, net.structure_W2, net.structure_b2);
   InitBranchEncoder(gBranchLayout.zoneCandleCount, net.zone_h1, net.zone_h2, net.zone_W1, net.zone_b1, net.zone_W2, net.zone_b2);

   int fusionDim = net.fusion_dim;
   int hidDim1 = net.hidden_dim;
   int hidDim2 = net.hidden_dim2;
   int headDim = (hidDim2>0 ? hidDim2 : hidDim1);
   int outDim = net.output_dim;

   ArrayResize(net.W1, hidDim1*fusionDim);
   ArrayResize(net.b1, hidDim1);
   ArrayResize(net.W2, MathMax(0,hidDim2*hidDim1));
   ArrayResize(net.b2, MathMax(0,hidDim2));

   ArrayResize(net.WV, headDim);
   ArrayResize(net.bV, 1);

   ArrayResize(net.WA, outDim*headDim);
   ArrayResize(net.bA, outDim);

   ArrayResize(net.feat_mean, net.input_dim);
   ArrayResize(net.feat_std,  net.input_dim);

   double scale1 = 1.0 / MathSqrt((double)MathMax(fusionDim,1));
   double scale2 = 1.0 / MathSqrt((double)MathMax(hidDim1,1));
   double scaleV = 1.0 / MathSqrt((double)MathMax(headDim,1));
   double scaleA = 1.0 / MathSqrt((double)MathMax(headDim,1));

   for(int i=0;i<ArraySize(net.W1);i++)
   {
      double r = (double)MathRand()/32767.0;
      net.W1[i] = (r*2.0-1.0)*scale1;
   }
   for(int i=0;i<ArraySize(net.b1);i++) net.b1[i]=0.0;

   for(int i=0;i<ArraySize(net.W2);i++)
   {
      double r = (double)MathRand()/32767.0;
      net.W2[i] = (r*2.0-1.0)*scale2;
   }
   for(int i=0;i<ArraySize(net.b2);i++) net.b2[i]=0.0;

   for(int i=0;i<ArraySize(net.WV);i++)
   {
      double r = (double)MathRand()/32767.0;
      net.WV[i] = (r*2.0-1.0)*scaleV;
   }
   for(int i=0;i<ArraySize(net.bV);i++) net.bV[i]=0.0;

   for(int i=0;i<ArraySize(net.WA);i++)
   {
      double r = (double)MathRand()/32767.0;
      net.WA[i] = (r*2.0-1.0)*scaleA;
   }
   for(int i=0;i<ArraySize(net.bA);i++) net.bA[i]=0.0;

   for(int i=0;i<net.input_dim;i++)
   {
      net.feat_mean[i]=0.0;
      net.feat_std[i]=1.0;
   }

   gDQN[symIdx][regime] = net;
   SyncTargetNetFor(symIdx, regime);
}
