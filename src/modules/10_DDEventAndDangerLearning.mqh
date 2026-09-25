//+------------------------------------------------------------------+
//| 10_DDEventAndDangerLearning.mqh                                 |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Basket/tick history, drawdown-event lifecycle, DD-event queries,|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

void DDEventTrimIfNeeded()
{
   while(ArraySize(gDDEvents) > MaxDDEventsStored)
      ArrayRemove(gDDEvents,0,1);
}

void BasketHistoryPush(const BasketSnapshot &snap)
{
   int n=ArraySize(gBasketHistory);
   ArrayResize(gBasketHistory,n+1);
   gBasketHistory[n]=snap;

   if(ArraySize(gBasketHistory)>BasketHistoryCapacity)
      ArrayRemove(gBasketHistory,0,1);
}

void TickTracePush(const TickTraceItem &t)
{
   int n=ArraySize(gTickTrace);
   ArrayResize(gTickTrace,n+1);
   gTickTrace[n]=t;

   if(ArraySize(gTickTrace)>TickTraceCapacity)
      ArrayRemove(gTickTrace,0,1);
}

void CaptureBasketSnapshot(const string symbol,
                           const int symIdx,
                           const int magic,
                           const int regime,
                           const int basketDir,
                           const int positionsCount,
                           const double &state[],
                           const double &qVals[])
{
   if(!UseDDEventMemory) return;

   BasketSnapshot s;
   s.timeStamp=TimeCurrent();
   s.symbol=symbol;
   s.magic=magic;
   s.regime=regime;
   s.basketDir=basketDir;
   s.positionsCount=positionsCount;
   s.equity=GetEAEquity();
   s.openPnL=CalculatePositionsPnL(symbol,magic);

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   s.midPrice=mid;
   s.avgPrice=(positionsCount>0 && gTrades[symIdx].Total()>0 ? BasketAvgPrice(gTrades[symIdx],gTradeLots[symIdx],mid) : mid);
   s.entryPrice=s.avgPrice;
   s.gridStep=(gGridActiveStep[symIdx]>0.0 ? gGridActiveStep[symIdx] : GetGridStepCached(symIdx));
   s.danger=(UseDangerBrain ? gPDanger[symIdx] : 0.0);

   ArrayResize(s.stateKey,0);
   BuildQMemoryKey(symIdx,state,s.stateKey);

   ArrayResize(s.qVals,ArraySize(qVals));
   for(int i=0;i<ArraySize(qVals);i++) s.qVals[i]=qVals[i];

   BasketHistoryPush(s);
   
   if(gDDEventActive)
      AppendLatestSnapshotToActiveDDEvent();   
}

void AppendLatestSnapshotToActiveDDEvent()
{
   if(!UseDDEventMemory) return;
   if(!gDDEventActive) return;

   int evN=ArraySize(gDDEvents);
   int histN=ArraySize(gBasketHistory);
   if(evN<=0 || histN<=0) return;

   BasketSnapshot lastSnap = gBasketHistory[histN-1];

   int n=ArraySize(gDDEvents[evN-1].postBaskets);
   ArrayResize(gDDEvents[evN-1].postBaskets,n+1);
   gDDEvents[evN-1].postBaskets[n]=lastSnap;

   gDDEventPostBasketCount++;
}

void CaptureTickTrace(const int symIdx)
{
   if(!UseDDEventMemory) return;
   if(!gDDEventActive) return;

   string sym=gSymbols[symIdx];

   if(DDEventTraceOnBars)
   {
      ENUM_TIMEFRAMES tf=DDEventTraceTFForSymbol(sym);
      datetime barTime=iTime(sym, tf, 0);
      if(barTime<=0) return;
      if(gDDEventLastTraceBarTime[symIdx]==barTime) return;
      gDDEventLastTraceBarTime[symIdx]=barTime;
   }

   TickTraceItem t;
   t.timeStamp=TimeCurrent();

   double bid=SymbolInfoDouble(sym,SYMBOL_BID);
   double ask=SymbolInfoDouble(sym,SYMBOL_ASK);
   double point=SymbolInfoDouble(sym,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   t.bid=bid;
   t.ask=ask;
   t.mid=0.5*(bid+ask);
   t.spreadPoints=(ask-bid)/point;
   t.equity=GetEAEquity();
   t.danger=(UseDangerBrain ? gPDanger[symIdx] : 0.0);

   TickTracePush(t);
}

void StartDDEvent(const int symIdx,
                  const int regime,
                  const int basketDir,
                  const double ddMoney,
                  const bool hardTrigger,
                  const double &state[],
                  const double &qVals[])
{
   if(!UseDDEventMemory) return;
   if(gDDEventActive) return;

   gDDEventActive=true;
   gDDEventTriggerTime=TimeCurrent();
   gDDEventTriggerDD=ddMoney;
   gDDEventHard=hardTrigger;
   gDDEventStartBasketIndex=MathMax(0, ArraySize(gBasketHistory)-BasketsBeforeDD);
   gDDEventStartTickIndex=ArraySize(gTickTrace);
   gDDEventPostBasketCount=0;
   gDDEventLastTraceBarTime[symIdx]=0;

   DDEventRecord ev;
   ev.created=TimeCurrent();
   ev.triggerTime=gDDEventTriggerTime;
   ev.symbol=gSymbols[symIdx];
   ev.magic=gMagics[symIdx];
   ev.regimeAtTrigger=regime;
   ev.basketDirAtTrigger=basketDir;
   ev.ddAtTrigger=ddMoney;
   ev.hardTrigger=hardTrigger;
   ev.completed=false;

   ArrayResize(ev.triggerStateKey,0);
   BuildQMemoryKey(symIdx,state,ev.triggerStateKey);

   ArrayResize(ev.triggerQVals,ArraySize(qVals));
   for(int i=0;i<ArraySize(qVals);i++) ev.triggerQVals[i]=qVals[i];

   ArrayResize(ev.preBaskets,0);
   int fromIdx=gDDEventStartBasketIndex;
   int toIdx=ArraySize(gBasketHistory)-1;
   for(int i=fromIdx;i<=toIdx;i++)
   {
      int n=ArraySize(ev.preBaskets);
      ArrayResize(ev.preBaskets,n+1);
      ev.preBaskets[n]=gBasketHistory[i];
   }

   ArrayResize(ev.postBaskets,0);
   ArrayResize(ev.ticks,0);

   int m=ArraySize(gDDEvents);
   ArrayResize(gDDEvents,m+1);
   gDDEvents[m]=ev;

   DDEventTrimIfNeeded();
}



void FinalizeActiveDDEvent()
{
   if(!UseDDEventMemory) return;
   if(!gDDEventActive) return;

   int n=ArraySize(gDDEvents);
   if(n<=0)
   {
      gDDEventActive=false;
      gDDEventTriggerTime=0;
      gDDEventTriggerDD=0.0;
      gDDEventHard=false;
      gDDEventStartBasketIndex=-1;
      gDDEventStartTickIndex=-1;
      gDDEventPostBasketCount=0;
      return;
   }

   int tickFrom=gDDEventStartTickIndex;
   if(tickFrom<0) tickFrom=0;
   if(tickFrom>ArraySize(gTickTrace)) tickFrom=ArraySize(gTickTrace);

   int tickN=ArraySize(gTickTrace)-tickFrom;
   if(tickN<0) tickN=0;
   if(tickN>MaxTicksPerEvent) tickN=MaxTicksPerEvent;

   ArrayResize(gDDEvents[n-1].ticks,tickN);
   for(int i=0;i<tickN;i++)
      gDDEvents[n-1].ticks[i]=gTickTrace[tickFrom+i];

   gDDEvents[n-1].completed=true;

   gDDEventActive=false;
   gDDEventTriggerTime=0;
   gDDEventTriggerDD=0.0;
   gDDEventHard=false;
   gDDEventStartBasketIndex=-1;
   gDDEventStartTickIndex=-1;
   gDDEventPostBasketCount=0;
   for(int s=0;s<MAX_SYMBOLS;s++) gDDEventLastTraceBarTime[s]=0;
}

void UpdateDDEventLifecycle(const int symIdx,
                            const int regime,
                            const int basketDir,
                            const double &state[],
                            const double &qVals[])
{
   if(!UseDDEventMemory) return;

   double eq=GetEAEquity();
   double ddMoney=(maxEquity>eq ? (maxEquity-eq) : 0.0);

   if(!gDDEventActive)
   {
      if(ddMoney >= HardDDTriggerMoney)
      {
         StartDDEvent(symIdx, regime, basketDir, ddMoney, true, state, qVals);
      }
      else if(ddMoney >= SoftDDTriggerMoney)
      {
         StartDDEvent(symIdx, regime, basketDir, ddMoney, false, state, qVals);
      }
   }
   else
   {
      CaptureTickTrace(symIdx);

      if(gDDEventPostBasketCount >= BasketsAfterDD)
         FinalizeActiveDDEvent();
   }
}

bool QueryDDEventBiasWithKey(const int symIdx,
                             const int regime,
                             const int action,
                             const int basketDir,
                             const double &stateKey[],
                             double &biasOut[])
{
   ArrayResize(biasOut, ActionCount);
   for(int a=0;a<ActionCount;a++) biasOut[a]=0.0;

   if(!UseDDEventBias) return false;
   if(ArraySize(gDDEvents)<=0) return false;
   if(ArraySize(stateKey)<=0) return false;

   double sumW=0.0;
   int start=MathMax(0, ArraySize(gDDEvents)-DDEventMaxEventsScan);

   for(int i=start;i<ArraySize(gDDEvents);i++)
   {
      if(gDDEvents[i].symbol != gSymbols[symIdx]) continue;
      if(gDDEvents[i].regimeAtTrigger!=regime && UseRegimeBank) continue;
      if(ArraySize(gDDEvents[i].triggerStateKey)!=ArraySize(stateKey)) continue;

      double sim=CosSim(stateKey,gDDEvents[i].triggerStateKey);
      if(sim < DDEventMinSim) continue;

      int triggerAction = (gDDEvents[i].basketDirAtTrigger>0 ? 1 : (gDDEvents[i].basketDirAtTrigger<0 ? 2 : 0));

      double risk = Clamp(gDDEvents[i].ddAtTrigger / MathMax(HardDDTriggerMoney, 1.0), 0.0, 2.0);
      if(gDDEvents[i].hardTrigger) risk += 0.50;
      risk = Clamp(risk, 0.0, 2.5);

      double w = sim * (1.0 + 0.25 * risk);
      if(gDDEvents[i].hardTrigger) w *= 1.15;

      biasOut[0] += w * (DDEventHoldBias + 0.35 * risk);

      if(triggerAction>=1 && triggerAction<=2)
      {
         double dirPenalty = DDEventPrePenalty + DDEventExpandPenalty * risk;
         biasOut[triggerAction] -= w * dirPenalty;
         if(action==triggerAction)
            biasOut[action] -= 0.35 * w * dirPenalty;
      }

      if(ArraySize(gDDEvents[i].postBaskets)>0)
      {
         int sameDirCount=0;
         int oppDirCount=0;
         for(int b=0;b<ArraySize(gDDEvents[i].postBaskets);b++)
         {
            int dir=gDDEvents[i].postBaskets[b].basketDir;
            if(dir==gDDEvents[i].basketDirAtTrigger) sameDirCount++;
            else if(dir==-gDDEvents[i].basketDirAtTrigger) oppDirCount++;
         }

         if(triggerAction>=1 && triggerAction<=2)
         {
            if(sameDirCount>oppDirCount)
            {
               biasOut[0] += w * 0.25 * DDEventHoldBias;
               biasOut[triggerAction] -= w * DDEventSameDirPenalty;
            }
            else if(!DDEventUseCautionOnly && oppDirCount>sameDirCount)
            {
               int altAction = (triggerAction==1 ? 2 : 1);
               biasOut[altAction] += w * DDEventRecoveryBoost;
            }
         }
      }

      sumW += w;
   }

   if(sumW<=1e-12) return false;

   for(int a=0;a<ActionCount;a++)
      biasOut[a] /= sumW;

   return true;
}
int DQNSelectAction(int symIdx,int regime,double &state[])
{
   double eps=currentEpsilon;
   if(UseDangerBrain && gMode[symIdx]==MODE_DANGER) eps *= 0.5;

   if((double)MathRand()/32767.0 < eps)
      return (MathRand() % ActionCount);

   double qBase[];
   DQNForwardInference(symIdx,regime,state,qBase);

   double qAdj[];
   ArrayResize(qAdj, ArraySize(qBase));
   for(int a=0;a<ArraySize(qBase);a++) qAdj[a]=qBase[a];

   double totalDelta[];
   InitActionBias(totalDelta);

   int basketDir=0;
   if(gPositionsCount[symIdx]>0)
      basketDir=BasketDir(gSymbols[symIdx], gMagics[symIdx]);

   DecisionSupportContext support;
   BuildDecisionSupportContext(symIdx, regime, state, qBase, false, basketDir, support);
   ApplyDecisionSupportDelta(support,totalDelta);

   if(UseQualityAdaptiveEntry && gPositionsCount[symIdx]<=0)
   {
      double qualityDelta[];
      BuildAdaptiveEntryQualityDelta(symIdx, regime, qBase, support, qualityDelta);
      AccumulateActionBias(totalDelta,qualityDelta);
   }

   ApplyBudgetedSubordinateBias(symIdx, false, qBase, totalDelta, qAdj);

   string symbol=gSymbols[symIdx];
   ApplyFlexibleDirectionalDiscipline(symbol, symIdx, regime, support, false, qAdj);
   ApplyConfirmedSetupAccelerator(symbol, symIdx, regime, support, qAdj);

   int best=0;
   double maxQ=qAdj[0];
   for(int a=1;a<ActionCount;a++)
      if(qAdj[a]>maxQ){ maxQ=qAdj[a]; best=a; }

   return best;
}


int TrendDirFromH1(const string symbol, const int symIdx)
{
   if(!UseH1Features) return 0;

   double atrS = gATRslow_BaseVal[symIdx];
   if(atrS<=1e-12) atrS=1e-12;

   double close1 = iClose(symbol, H1_TF, 1);
   double ema1   = gEMA_H1v[symIdx];

   double z = (close1 - ema1) / atrS;
   if(z >  0.15) return +1;
   if(z < -0.15) return -1;
   return 0;
}

int BasketDir(const string symbol, const int magic)
{
   int idx=SymbolIndex(symbol);
   if(idx>=0 && magic==gMagics[idx])
   {
      if(gPositionsCount[idx] <= 0) return 0;
      if(gHasPositionTypeCache[idx]) return gBasketDirCache[idx];
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

void BuildBadEpisodeBiasSmart(const int label, double &bias[], const double &qSnap[], const int qN)
{
   ArrayResize(bias, ActionCount);
   for(int a=0;a<ActionCount;a++) bias[a]=0.0;

   if(ActionCount>=1) bias[0] += BadBias_HoldBoost;

   if(label==LBL_AGAINST_UPTREND)
   {
      if(ActionCount>=2) bias[1] += BadBias_WithTrendBoost;
      if(ActionCount>=3) bias[2] -= BadBias_AgainstPenalty;
   }
   else
   {
      if(ActionCount>=2) bias[1] -= BadBias_AgainstPenalty;
      if(ActionCount>=3) bias[2] += BadBias_WithTrendBoost;
   }

   if(qN>0 && qN==ActionCount)
   {
      int best=0;
      double mx=qSnap[0];
      for(int a=1;a<ActionCount;a++) if(qSnap[a]>mx){ mx=qSnap[a]; best=a; }
      bias[best] -= BadBias_RepeatPenalty;
   }
}
void MergeProto(ProtoEntry &p,
                const double &f[], const double &miniMaybe[], const bool haveMini,
                const double &dfpMaybe[], const bool haveDFP,
                const double &dminiMaybe[], const bool haveDMini,
                const double &bias[], const double fAlpha, const double bAlpha)
{
   int nF=ArraySize(p.features);
   for(int i=0;i<nF;i++)
      p.features[i] = (1.0-fAlpha)*p.features[i] + fAlpha*f[i];
   NormalizeVec(p.features);

   if(haveMini && UseMiniStateFingerprint && ArraySize(p.stateMini)==MINI_DIM && ArraySize(miniMaybe)==MINI_DIM)
   {
      for(int k=0;k<MINI_DIM;k++)
         p.stateMini[k] = (1.0-fAlpha)*p.stateMini[k] + fAlpha*miniMaybe[k];
      NormalizeVecN(p.stateMini);
   }

   if(haveDFP && UseDeltaSimilarity && ArraySize(p.deltaFP)==6 && ArraySize(dfpMaybe)==6)
   {
      for(int k=0;k<6;k++)
         p.deltaFP[k] = (1.0-fAlpha)*p.deltaFP[k] + fAlpha*dfpMaybe[k];
      NormalizeVec(p.deltaFP);
   }

   if(haveDMini && UseDeltaSimilarity && ArraySize(p.deltaMini)==MINI_DIM && ArraySize(dminiMaybe)==MINI_DIM)
   {
      for(int k=0;k<MINI_DIM;k++)
         p.deltaMini[k] = (1.0-fAlpha)*p.deltaMini[k] + fAlpha*dminiMaybe[k];
      NormalizeVecN(p.deltaMini);
   }

   int nB=MathMin(ArraySize(p.adapterBias), ArraySize(bias));
   for(int a=0;a<nB;a++)
      p.adapterBias[a] = (1.0-bAlpha)*p.adapterBias[a] + bAlpha*bias[a];

   p.survivalScore += 1.0;
   p.lastUsed = TimeCurrent();
}

int FindMergeCandidate(const bool wantDanger,
                       const int label, const int basketDir, const int trendDir,
                       const int symIdx,
                       const double &f_now[], const double &mini_now[], const bool haveMini,
                       const double &dfp_now[], const bool haveDFP,
                       const double &dmini_now[], const bool haveDMini,
                       double &bestSimOut)
{
   bestSimOut=-1e9;
   int best=-1;

   int n=ArraySize(gProtos);
   for(int i=0;i<n;i++)
   {
      if(gProtos[i].isDanger != wantDanger) continue;
      if(wantDanger && gProtos[i].label != label) continue;
      if(ArraySize(gProtos[i].features)!=ArraySize(f_now)) continue;

      double sim = CombinedSim(symIdx, f_now, mini_now, dfp_now, haveDFP, dmini_now, haveDMini, gProtos[i]);

      if(basketDir!=0 && gProtos[i].basketDir!=0 && gProtos[i].basketDir!=basketDir) continue;
      if(trendDir!=0  && gProtos[i].trendDir!=0  && gProtos[i].trendDir !=trendDir)  continue;

      if(sim>bestSimOut){ bestSimOut=sim; best=i; }
   }
   return best;
}

int LearnBadEpisodeSmart(const int symIdx, const bool missedDangerAtConfirm)
{
   if(gBadLabel[symIdx] < 0) return -1;

   double f_now[]; ArrayResize(f_now,6);
   if(gBadHaveBasketFP[symIdx])       for(int k=0;k<6;k++) f_now[k]=gBadFPBasketOpen[symIdx][k];
   else if(gBadHavePreFP[symIdx])     for(int k=0;k<6;k++) f_now[k]=gBadFPPre[symIdx][k];
   else                               for(int k=0;k<6;k++) f_now[k]=gBadFPStart[symIdx][k];
   NormalizeVec(f_now);

   double mini_now[]; ArrayResize(mini_now, MINI_DIM);
   bool haveMini=false;
   if(UseMiniStateFingerprint)
   {
      if(gBadHaveMiniBasket[symIdx]) { for(int k=0;k<MINI_DIM;k++) mini_now[k]=gBadMiniBasketOpen[symIdx][k]; haveMini=true; }
      else if(gBadHaveMiniPre[symIdx]) { for(int k=0;k<MINI_DIM;k++) mini_now[k]=gBadMiniPre[symIdx][k]; haveMini=true; }
      else if(gBadHaveMiniStart[symIdx]) { for(int k=0;k<MINI_DIM;k++) mini_now[k]=gBadMiniStart[symIdx][k]; haveMini=true; }
      if(haveMini) NormalizeVecN(mini_now);
   }

   double dfp_now[]; bool haveDFP=false;
   if(UseDeltaSimilarity && gBadHavePreFP[symIdx])
   {
      double cur[]; ArrayResize(cur,6);
      double prev[]; ArrayResize(prev,6);
      for(int k=0;k<6;k++){ cur[k]=gBadFPStart[symIdx][k]; prev[k]=gBadFPPre[symIdx][k]; }
      haveDFP = BuildDeltaVec(cur, prev, 6, dfp_now);
   }

   double dmini_now[]; bool haveDMini=false;
   if(UseDeltaSimilarity && UseMiniStateFingerprint && gBadHaveMiniPre[symIdx] && gBadHaveMiniStart[symIdx])
   {
      double cur[]; ArrayResize(cur,MINI_DIM);
      double prev[]; ArrayResize(prev,MINI_DIM);
      for(int k=0;k<MINI_DIM;k++){ cur[k]=gBadMiniStart[symIdx][k]; prev[k]=gBadMiniPre[symIdx][k]; }
      haveDMini = BuildDeltaVec(cur, prev, MINI_DIM, dmini_now);
   }

   double qSnap[]; ArrayResize(qSnap, ActionCount);
   for(int a=0;a<ActionCount;a++) qSnap[a]=0.0;
   bool haveQ=(gBadHaveQStart[symIdx] && ActionCount<=8);
   if(haveQ) for(int a=0;a<ActionCount;a++) qSnap[a]=gBadQStart[symIdx][a];

   double bias[];
   if(haveQ) BuildBadEpisodeBiasSmart(gBadLabel[symIdx], bias, qSnap, ActionCount);
   else { double emptyQ[]; ArrayResize(emptyQ,0); BuildBadEpisodeBiasSmart(gBadLabel[symIdx], bias, emptyQ, 0); }

   double bestSim=-1e9;
   int best=FindMergeCandidate(true, gBadLabel[symIdx], gBadBasketDir[symIdx], gBadTrendDir[symIdx], symIdx,
                               f_now, mini_now, haveMini, dfp_now, haveDFP, dmini_now, haveDMini, bestSim);

   if(best>=0 && bestSim>=BadProtoMergeSim)
   {
      if(UseDeltaSimilarity)
      {
         if(ArraySize(gProtos[best].deltaFP)==0 && haveDFP){ ArrayResize(gProtos[best].deltaFP,6); for(int k=0;k<6;k++) gProtos[best].deltaFP[k]=dfp_now[k]; }
         if(ArraySize(gProtos[best].deltaMini)==0 && haveDMini){ ArrayResize(gProtos[best].deltaMini,MINI_DIM); for(int k=0;k<MINI_DIM;k++) gProtos[best].deltaMini[k]=dmini_now[k]; }
      }

      MergeProto(gProtos[best], f_now, mini_now, haveMini, dfp_now, haveDFP, dmini_now, haveDMini,
                 bias, BadProtoFeatureEMA, BadProtoBiasEMA);

      ProtoScoreBump(best, missedDangerAtConfirm ? 2.0 : 1.0);
      return best;
   }

   ProtoEntry p;
   p.label     = gBadLabel[symIdx];
   p.basketDir = gBadBasketDir[symIdx];
   p.trendDir  = gBadTrendDir[symIdx];
   p.atrRatio  = gBadAtrRatio[symIdx];
   p.created   = TimeCurrent();
   p.isDanger=true;

   ArrayResize(p.features,6);
   for(int k=0;k<6;k++) p.features[k]=f_now[k];
   NormalizeVec(p.features);

   ArrayResize(p.stateMini, (haveMini ? MINI_DIM : 0));
   if(haveMini) for(int k=0;k<MINI_DIM;k++) p.stateMini[k]=mini_now[k];

   ArrayResize(p.deltaFP, (haveDFP ? 6 : 0));
   if(haveDFP) for(int k=0;k<6;k++) p.deltaFP[k]=dfp_now[k];

   ArrayResize(p.deltaMini, (haveDMini ? MINI_DIM : 0));
   if(haveDMini) for(int k=0;k<MINI_DIM;k++) p.deltaMini[k]=dmini_now[k];

   ArrayResize(p.adapterBias, ActionCount);
   for(int a=0;a<ActionCount;a++) p.adapterBias[a]=bias[a];

   ArrayResize(p.qSnap, (haveQ ? ActionCount : 0));
   if(haveQ) for(int a=0;a<ActionCount;a++) p.qSnap[a]=qSnap[a];

   p.survivalScore=1.0 + (missedDangerAtConfirm ? 2.0 : 1.0);
   p.usedCount=0;
   p.lastUsed=TimeCurrent();

   int n=ArraySize(gProtos);
   ArrayResize(gProtos,n+1);
   gProtos[n]=p;

   PruneProtosIfNeeded();
   return n;
}

