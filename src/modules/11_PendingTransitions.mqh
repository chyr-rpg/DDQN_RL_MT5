//+------------------------------------------------------------------+
//| 11_PendingTransitions.mqh                                       |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Delayed/pending transition bookkeeping, close matching, post-tra|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

void PendingInit()
{
   ArrayResize(gPending, MaxPendingTransitions);
   for(int i=0;i<ArraySize(gPending);i++)
   {
      gPending[i].active=false;
      gPending[i].symIdx=-1;
      gPending[i].regime=0;
      gPending[i].action=0;
      gPending[i].created=0;
      gPending[i].basketDir=0;
      gPending[i].positionsAtOpen=0;
      gPending[i].episodeId=0;
      ArrayResize(gPending[i].state,0);
   }
}

int PendingFindFreeSlot()
{
   for(int i=0;i<ArraySize(gPending);i++)
      if(!gPending[i].active) return i;

   int oldest=0;
   datetime t=LONG_MAX;
   for(int i=0;i<ArraySize(gPending);i++)
   {
      if(gPending[i].created < t)
      {
         t=gPending[i].created;
         oldest=i;
      }
   }
   return oldest;
}


double GetLiveBasketAvgPrice(const string symbol,const int magic,const double fallback)
{
   int idx=SymbolIndex(symbol);
   if(idx>=0 && magic==gMagics[idx])
   {
      if(gBasketAvgValid[idx])
         return gBasketAvgPriceCache[idx];

      if(gTrades[idx].Total()>0 && gTradeLots[idx].Total()==gTrades[idx].Total())
         return BasketAvgPrice(gTrades[idx],gTradeLots[idx],fallback);
   }

   double weighted=0.0;
   double totalLots=0.0;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      double price=PositionGetDouble(POSITION_PRICE_OPEN);
      double vol=PositionGetDouble(POSITION_VOLUME);
      weighted += price*vol;
      totalLots += vol;
   }

   if(totalLots<=0.0) return fallback;
   return weighted/totalLots;
}

bool FindLatestOpenedPositionMeta(const string symbol,
                                  const int magic,
                                  const ENUM_ORDER_TYPE orderType,
                                  const double expectedVolume,
                                  double &entryPrice,
                                  double &entryVolume,
                                  datetime &entryTime)
{
   int desiredType=(orderType==ORDER_TYPE_BUY ? POSITION_TYPE_BUY : POSITION_TYPE_SELL);

   bool found=false;
   datetime bestTime=0;
   double bestVolDiff=DBL_MAX;
   double bestPrice=0.0;
   double bestVolume=0.0;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      if((int)PositionGetInteger(POSITION_TYPE)!=desiredType) continue;

      datetime t=(datetime)PositionGetInteger(POSITION_TIME);
      double vol=PositionGetDouble(POSITION_VOLUME);
      double px =PositionGetDouble(POSITION_PRICE_OPEN);
      double diff=MathAbs(vol-expectedVolume);

      if(!found || t>bestTime || (t==bestTime && diff<bestVolDiff))
      {
         found=true;
         bestTime=t;
         bestVolDiff=diff;
         bestPrice=px;
         bestVolume=vol;
      }
   }

   if(!found) return false;

   entryPrice=bestPrice;
   entryVolume=bestVolume;
   entryTime=bestTime;
   return true;
}

void PendingAdd(const int symIdx,
                const int regime,
                const int action,
                const int basketDir,
                const int positionsAtOpen,
                const double &state[],
                const double entryPrice,
                const double entryVolume,
                const int legIndex,
                const double basketAvgAtEntry,
                const datetime entryTime)
{
   if(!UsePendingTransitions) return;

   int idx=PendingFindFreeSlot();
   if(idx<0) return;

   gPending[idx].active=true;
   gPending[idx].symIdx=symIdx;
   gPending[idx].regime=regime;
   gPending[idx].action=action;
   gPending[idx].created=TimeCurrent();
   gPending[idx].basketDir=basketDir;
   gPending[idx].positionsAtOpen=positionsAtOpen;
   gPending[idx].legIndex=legIndex;
   gPending[idx].entryTime=entryTime;
   gPending[idx].closeTime=0;
   gPending[idx].entryPrice=entryPrice;
   gPending[idx].closePrice=0.0;
   gPending[idx].entryVolume=entryVolume;
   gPending[idx].basketAvgAtEntry=basketAvgAtEntry;
   gPending[idx].individualPnL=0.0;
   gPending[idx].denseRewardAccum=0.0;
   gPending[idx].episodeId=EnsureActiveBasketEpisode(symIdx, basketDir, MathMax(positionsAtOpen,1), entryTime);

   ArrayResize(gPending[idx].state,ArraySize(state));
   for(int i=0;i<ArraySize(state);i++) gPending[idx].state[i]=state[i];
}

bool BuildPostTradeSnapshot(const string symbol,
                            const int symIdx,
                            const int magic,
                            const ENUM_ORDER_TYPE expectedOrderType,
                            const double expectedVolume,
                            int &positionsAfter,
                            int &basketDirAfter,
                            double &entryPrice,
                            double &entryVolume,
                            double &basketAvgAfter,
                            datetime &entryTime)
{
   CountOpenPositions();

   positionsAfter = gPositionsCount[symIdx];
   basketDirAfter = BasketDir(symbol,magic);

   entryPrice  = 0.0;
   entryVolume = expectedVolume;
   entryTime   = TimeCurrent();

   bool found = FindLatestOpenedPositionMeta(symbol,
                                             magic,
                                             expectedOrderType,
                                             expectedVolume,
                                             entryPrice,
                                             entryVolume,
                                             entryTime);

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   if(!found)
   {
      entryPrice = (expectedOrderType==ORDER_TYPE_BUY ? ask : bid);
      entryVolume = expectedVolume;
      entryTime = TimeCurrent();
   }

   basketAvgAfter = GetLiveBasketAvgPrice(symbol,magic,(entryPrice>0.0 ? entryPrice : mid));

   return (positionsAfter > 0 && basketDirAfter != 0);
}

void BuildCloseNextState(const string symbol,
                         const int symIdx,
                         const int positionsAfterClose,
                         double &nextState[])
{
   CArrayDouble emptyTrades;
   BuildState(symbol,symIdx,positionsAfterClose,emptyTrades,nextState);

   bool extremeAfter=IsExtremeState(symIdx,positionsAfterClose);
   if(SeparateExtremeStates)
   {
      int sz=ArraySize(nextState);
      if(sz>0) nextState[sz-1]=(extremeAfter?1.0:0.0);
   }
}

void ArchiveClosedLegLearning(const PendingTransition &pt,const double resolvedReward)
{
   int n=ArraySize(gClosedLegHistory);
   ArrayResize(gClosedLegHistory,n+1);

   gClosedLegHistory[n].entryTime=pt.entryTime;
   gClosedLegHistory[n].closeTime=pt.closeTime;
   gClosedLegHistory[n].symbol=gSymbols[pt.symIdx];
   gClosedLegHistory[n].regime=pt.regime;
   gClosedLegHistory[n].action=pt.action;
   gClosedLegHistory[n].basketDir=pt.basketDir;
   gClosedLegHistory[n].legIndex=pt.legIndex;
   gClosedLegHistory[n].entryPrice=pt.entryPrice;
   gClosedLegHistory[n].closePrice=pt.closePrice;
   gClosedLegHistory[n].entryVolume=pt.entryVolume;
   gClosedLegHistory[n].basketAvgAtEntry=pt.basketAvgAtEntry;
   gClosedLegHistory[n].individualPnL=pt.individualPnL;
   gClosedLegHistory[n].resolvedReward=resolvedReward;

   if(MaxClosedLegHistory>0 && ArraySize(gClosedLegHistory)>MaxClosedLegHistory)
   {
      int keep=MaxClosedLegHistory;
      int drop=ArraySize(gClosedLegHistory)-keep;
      for(int i=0;i<keep;i++)
         gClosedLegHistory[i]=gClosedLegHistory[i+drop];
      ArrayResize(gClosedLegHistory,keep);
   }
}

double ComputePerPositionCloseAdjustment(const PendingTransition &pt)
{
   if(!UsePerPositionCloseLearning) return 0.0;

   double point=SymbolInfoDouble(gSymbols[pt.symIdx],SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double vol=MathMax(pt.entryVolume,1e-6);
   double pnlPerLot=pt.individualPnL/vol;
   double distPoints=0.0;
   if(pt.basketAvgAtEntry>0.0)
      distPoints=MathAbs(pt.entryPrice-pt.basketAvgAtEntry)/point;

   double holdHours=0.0;
   if(pt.closeTime>pt.entryTime)
      holdHours=(double)(pt.closeTime-pt.entryTime)/3600.0;

   double pnlAdj = Clamp(pnlPerLot * LegPnLPerLotScale, -1.5, 1.5);
   double distAdj= Clamp(distPoints * LegDistancePointsScale * (pt.individualPnL>=0.0 ? 1.0 : -1.0), -0.5, 0.5);
   double legAdj = Clamp((double)MathMax(0,pt.legIndex-1) * LegIndexRewardScale * (pt.individualPnL>=0.0 ? 1.0 : -1.0), -0.5, 0.5);
   double holdAdj= Clamp(holdHours * LegHoldHoursPenaltyScale, 0.0, 0.5);

   return pnlAdj + distAdj + legAdj - holdAdj;
}

int PendingFindBestMatch(const int symIdx,const PositionCloseItem &item,const bool &matched[])
{
   double point=SymbolInfoDouble(gSymbols[symIdx],SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   int best=-1;
   double bestScore=DBL_MAX;

   for(int i=0;i<ArraySize(gPending);i++)
   {
      if(matched[i]) continue;
      if(!gPending[i].active) continue;
      if(gPending[i].symIdx!=symIdx) continue;
      if(gPending[i].basketDir!=item.basketDir) continue;

      double score=0.0;
      score += MathAbs(gPending[i].entryVolume-item.volume);
      score += MathAbs(gPending[i].entryPrice-item.entryPrice)/point;
      score += 0.25*MathAbs((double)(gPending[i].legIndex-item.legIndex));
      score += 0.001*MathAbs((double)(gPending[i].entryTime-item.entryTime));

      if(score<bestScore)
      {
         bestScore=score;
         best=i;
      }
   }

   return best;
}


double PendingResolveIndexCore(const int idx,
                               const double reward,
                               const bool done,
                               const double &nextState[])
{
   if(idx<0 || idx>=ArraySize(gPending)) return 0.0;
   if(!gPending[idx].active) return 0.0;

   int symIdx=gPending[idx].symIdx;
   int regime=gPending[idx].regime;
   int action=gPending[idx].action;

   double resolvedReward=reward + gPending[idx].denseRewardAccum + ComputePerPositionCloseAdjustment(gPending[idx]);

   ReplayPush(symIdx, regime, gPending[idx].state, action, resolvedReward, nextState, done);
   UpdateQMemoryResolved(symIdx, regime, gPending[idx].state, action, resolvedReward);

   if(!UseReplayBuffer)
   {
      double st[];
      ArrayResize(st,ArraySize(gPending[idx].state));
      for(int i=0;i<ArraySize(gPending[idx].state);i++) st[i]=gPending[idx].state[i];

      double ns[];
      ArrayResize(ns,ArraySize(nextState));
      for(int i=0;i<ArraySize(nextState);i++) ns[i]=nextState[i];

      DQNUpdate(symIdx, regime, st, action, resolvedReward, ns, done);
   }

   ArchiveClosedLegLearning(gPending[idx],resolvedReward);

   if(gPending[idx].episodeId>0 && (done || gPositionsCount[symIdx]<=1))
      FinalizeActiveBasketEpisode(symIdx, resolvedReward);

   gPending[idx].active=false;
   gPending[idx].episodeId=0;
   gPending[idx].denseRewardAccum=0.0;
   ArrayResize(gPending[idx].state,0);
   return resolvedReward;
}

double PendingResolveIndex(const int idx,
                           const double reward,
                           const bool done,
                           const double &nextState[])
{
   return PendingResolveIndexCore(idx,reward,done,nextState);
}

double PendingResolveIndexDetailed(const int idx,
                                   const double reward,
                                   const bool done,
                                   const double &nextState[],
                                   const PositionCloseItem &legItem)
{
   if(idx<0 || idx>=ArraySize(gPending)) return 0.0;
   if(!gPending[idx].active) return 0.0;

   gPending[idx].closeTime=legItem.closeTime;
   gPending[idx].closePrice=legItem.closePrice;
   gPending[idx].individualPnL=legItem.profit;

   return PendingResolveIndexCore(idx,reward,done,nextState);
}

int PendingResolveForSymbolClose(const int symIdx,
                                 const double reward,
                                 const bool done,
                                 const double &nextState[],
                                 PositionCloseItem &closedItems[],
                                 double &resolvedRewardSum)
{
   int resolved=0;
   resolvedRewardSum=0.0;

   bool matched[];
   ArrayResize(matched,ArraySize(gPending));
   for(int i=0;i<ArraySize(matched);i++) matched[i]=false;

   for(int j=0;j<ArraySize(closedItems);j++)
   {
      int idx=PendingFindBestMatch(symIdx,closedItems[j],matched);
      if(idx<0) continue;

      matched[idx]=true;
      double rr=PendingResolveIndexDetailed(idx, reward, done, nextState, closedItems[j]);
      resolvedRewardSum += rr;
      resolved++;
   }

   return resolved;
}

void PendingExpireOld(const datetime now)
{
   if(!UsePendingTransitions) return;
   int maxSec=MathMax(60, PendingExpireMinutes*60);

   for(int i=0;i<ArraySize(gPending);i++)
   {
      if(!gPending[i].active) continue;
      if((now - gPending[i].created) < maxSec) continue;

      double nextState[];
      ArrayResize(nextState, ArraySize(gPending[i].state));
      for(int k=0;k<ArraySize(nextState);k++) nextState[k]=gPending[i].state[k];

      PendingResolveIndex(i, 0.0, false, nextState);
   }
}

