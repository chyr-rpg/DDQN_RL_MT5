//+------------------------------------------------------------------+
//| 12_TradeExecution.mqh                                           |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Order opening/closing, basket P&L, lot sizing, TP/trailing logic|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

ENUM_POSITION_TYPE GetPositionType(string symbol,int magic)
{
   int idx=SymbolIndex(symbol);
   if(idx>=0 && magic==gMagics[idx] && gHasPositionTypeCache[idx])
      return (gBasketDirCache[idx] > 0 ? POSITION_TYPE_BUY : POSITION_TYPE_SELL);

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL)==symbol &&
         (int)PositionGetInteger(POSITION_MAGIC)==magic)
      {
         ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         return pt;
      }
   }

   return POSITION_TYPE_BUY;
}

bool OpenPosition(string symbol,ENUM_ORDER_TYPE orderType,double volume,int stopLossPoints,int takeProfitPoints,int magic)
{
   trade.SetExpertMagicNumber(magic);
   trade.SetDeviationInPoints(Slippage);

   double vol=NormalizeVolume(symbol,volume);
   if(vol<=0.0) return false;

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double price=0.0, sl=0.0, tp=0.0;

   if(orderType==ORDER_TYPE_BUY)
   {
      price=SymbolInfoDouble(symbol,SYMBOL_ASK);
      if(stopLossPoints>0) sl=price - stopLossPoints*point;
      if(takeProfitPoints>0) tp=price + takeProfitPoints*point;
   }
   else
   {
      price=SymbolInfoDouble(symbol,SYMBOL_BID);
      if(stopLossPoints>0) sl=price + stopLossPoints*point;
      if(takeProfitPoints>0) tp=price - takeProfitPoints*point;
   }

   return trade.PositionOpen(symbol,orderType,vol,price,sl,tp);
}


void SortPositionCloseItems(PositionCloseItem &items[])
{
   int n=ArraySize(items);
   for(int i=0;i<n-1;i++)
   {
      for(int j=i+1;j<n;j++)
      {
         if(items[j].entryTime < items[i].entryTime)
         {
            PositionCloseItem tmp=items[i];
            items[i]=items[j];
            items[j]=tmp;
         }
      }
   }

   for(int k=0;k<n;k++)
      items[k].legIndex=k+1;
}

int CollectBasketPositionSnapshots(const string symbol,const int magic,PositionCloseItem &items[])
{
   ArrayResize(items,0);

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      int n=ArraySize(items);
      ArrayResize(items,n+1);

      items[n].ticket=ticket;
      items[n].basketDir=((int)PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY ? +1 : -1);
      items[n].legIndex=0;
      items[n].entryTime=(datetime)PositionGetInteger(POSITION_TIME);
      items[n].closeTime=0;
      items[n].entryPrice=PositionGetDouble(POSITION_PRICE_OPEN);
      items[n].closePrice=0.0;
      items[n].volume=PositionGetDouble(POSITION_VOLUME);
      items[n].profit=PositionGetDouble(POSITION_PROFIT);
   }

   SortPositionCloseItems(items);
   return ArraySize(items);
}

bool ClosePositionsDetailed(const string symbol,
                            const int magic,
                            BasketCloseResult &result,
                            PositionCloseItem &closedItems[])
{
   ArrayResize(closedItems,0);

   result.attempted=0;
   result.closed=0;
   result.attemptedVolume=0.0;
   result.closedVolume=0.0;
   result.attemptedProfit=0.0;
   result.closedProfit=0.0;
   result.wins=0;
   result.losses=0;
   result.total=0;
   result.allClosed=false;

   PositionCloseItem openItems[];
   int count=CollectBasketPositionSnapshots(symbol,magic,openItems);
   if(count<=0) return false;

   result.attempted=count;
   result.total=count;

   for(int i=0;i<count;i++)
   {
      result.attemptedVolume += openItems[i].volume;
      result.attemptedProfit += openItems[i].profit;

      if(openItems[i].profit>0.0) result.wins++;
      else                        result.losses++;
   }

   for(int i=0;i<count;i++)
   {
      double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
      double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
      openItems[i].closeTime=TimeCurrent();
      openItems[i].closePrice=(openItems[i].basketDir>0 ? bid : ask);

      if(trade.PositionClose(openItems[i].ticket))
      {
         int n=ArraySize(closedItems);
         ArrayResize(closedItems,n+1);
         closedItems[n]=openItems[i];

         result.closed++;
         result.closedVolume += openItems[i].volume;
         result.closedProfit += openItems[i].profit;
      }
   }

   result.allClosed=(result.closed==result.attempted && result.attempted>0);

   if(result.closed>0)
   {
      gEAClosedProfit += result.closedProfit;
      gProfitCycleClosedProfit += result.closedProfit;

      if(result.allClosed)
         ArmDeepBasketStrongFirstTrade(symbol, result.closed);
   }

   return (result.closed>0);
}

bool ClosePositions(string symbol,int magic,double &closedProfitOut,int &closedCountOut)
{
   BasketCloseResult closeRes;
   PositionCloseItem closedItems[];
   bool success=ClosePositionsDetailed(symbol,magic,closeRes,closedItems);

   closedProfitOut=closeRes.closedProfit;
   closedCountOut=closeRes.closed;

   if(success && closeRes.allClosed && closeRes.closed>0)
   {
      MarkBasketClosedForCooldown(symbol);
   }

   return success;
}

void CloseAllPositions()
{
   if(UseGlobalAccountWatchdog)
   {
      for(int i=PositionsTotal()-1; i>=0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket<=0) continue;
         if(!PositionSelectByTicket(ticket)) continue;

         bool shouldClose = false;

         if(WatchAllAccountPositions)
         {
            shouldClose = true;
         }
         else
         {
            int mg = (int)PositionGetInteger(POSITION_MAGIC);
            if(mg >= WatchdogMagicMin && mg <= WatchdogMagicMax)
               shouldClose = true;
         }

         if(shouldClose)
            trade.PositionClose(ticket);
      }

      for(int j=0; j<gSymbolCount; j++)
         gFirstTradeTime[j]=0;

      return;
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      double cp=0.0;
      int cc=0;
      ClosePositions(gSymbols[i],gMagics[i],cp,cc);
      gFirstTradeTime[i]=0;
   }
}

double CalculatePositionsPnL(string symbol,int magic)
{
   int idx=SymbolIndex(symbol);
   if(idx>=0 && magic==gMagics[idx])
      return gCachedSymbolPnL[idx];

   double total=0.0;
   int n=PositionsTotal();
   for(int i=0;i<n;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL)==symbol &&
         (int)PositionGetInteger(POSITION_MAGIC)==magic)
      {
         total += PositionGetDouble(POSITION_PROFIT);
      }
   }
   return total;
}

double CalculateLot(const string symbol,const int positionsCount)
{
   double scale=GetEquityBudgetScale();
   double lot = Lots * scale * MathPow(LotExponent, positionsCount);
   return NormalizeVolume(symbol, lot);
}

void CalculateLevelsSimple(const string symbol,
                           const int symIdx,
                           const int positionsCount,
                           CArrayDouble &trades,
                           const int magic,
                           double &buyLevel,
                           double &sellLevel)
{
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   double baseStep=GetGridStepCached(symIdx);
   if(baseStep<=0.0)
   {
      double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
      if(point<=0.0) point=0.00001;
      baseStep=(double)DefaultPips*point;
      if(baseStep<=0.0) baseStep=point*10.0;
   }

   double step=ComputeAdaptiveGridStep(symbol,symIdx,positionsCount,baseStep);

   gGridActiveChannel[symIdx]=0;
   gGridActiveStep[symIdx]=step;

   if(positionsCount > 0 && trades.Total() > 0)
   {
      double avg=(gBasketAvgValid[symIdx]
                  ? gBasketAvgPriceCache[symIdx]
                  : BasketAvgPrice(trades,gTradeLots[symIdx],mid));

      ENUM_POSITION_TYPE pt=GetPositionType(symbol, magic);

      if(pt==POSITION_TYPE_BUY)
      {
         buyLevel  = avg - step;
         sellLevel = 0.0;
      }
      else
      {
         sellLevel = avg + step;
         buyLevel  = 0.0;
      }
   }
   else
   {
      buyLevel  = bid - step;
      sellLevel = ask + step;
   }
}

bool CheckCCIExit()
{
   if(!UseCCI) return false;
   bool acted=false;

   for(int i=0;i<gSymbolCount;i++)
   {
      if(gPositionsCount[i]<=0) continue;

      string sym=gSymbols[i];
      int magic=gMagics[i];
      double cciVal=gCCI_Base[i];

      ENUM_POSITION_TYPE pt=GetPositionType(sym,magic);
      if((pt==POSITION_TYPE_BUY && cciVal < -CCI_Level) ||
         (pt==POSITION_TYPE_SELL && cciVal >  CCI_Level))
      {
         double cp=0.0; int cc=0;
         ClosePositions(sym,magic,cp,cc);
         acted=true;
      }
   }
   return acted;
}

double BasketBaseLotsEquivalent(const int positionsCount)
{
   return MathMax((double)MathMax(1,positionsCount) * Lots, 1e-6);
}

double BasketTotalLots(const int symIdx)
{
   double total=0.0;
   for(int i=0;i<gTradeLots[symIdx].Total();i++)
      total += gTradeLots[symIdx].At(i);
   return MathMax(total, 1e-6);
}

double ComputeHybridTakeProfitDistance(const int symIdx,const int positionsCount,const double channelStep)
{
   double point=SymbolInfoDouble(gSymbols[symIdx],SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double distChannel=MathMax(point, ChannelCloseAlpha * MathMax(channelStep, point));
   if(!UseHybridBasketTP)
      return distChannel;

   double totalLots=BasketTotalLots(symIdx);
   double legacyLots=BasketBaseLotsEquivalent(positionsCount);

   double distLegacy=(double)LegacyTakeProfitPts * point * (legacyLots / totalLots);
   if(distLegacy<=0.0)
      distLegacy=distChannel;

   double depthAdj=1.0 / MathMax(1.0, 1.0 + BasketTPDepthFactor * MathMax(0, positionsCount-1));
   double distChannelTight=MathMax(point * MathMax(1.0,(double)LegacyTakeProfitPts*BasketTPMinFactor), distChannel * depthAdj);

   double hybrid=MathMin(distChannelTight, distLegacy);
   hybrid=Clamp(hybrid, point, distChannel);

   return hybrid;
}

bool BasketTPHit(const string symbol,const int symIdx,const int magic,double &avgOut,double &targetOut)
{
   avgOut=0.0;
   targetOut=0.0;

   if(gPositionsCount[symIdx] <= 0) return false;
   if(gTrades[symIdx].Total() <= 0) return false;

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double mid=0.5*(bid+ask);

   double avg=(gBasketAvgValid[symIdx]
               ? gBasketAvgPriceCache[symIdx]
               : BasketAvgPrice(gTrades[symIdx],gTradeLots[symIdx],mid));

   int basketDir=BasketDir(symbol,magic);
   if(basketDir==0) return false;

   double channelStep=GetCloseChannelStep(symIdx,gPositionsCount[symIdx]);
   if(channelStep<=0.0) channelStep=GetGridStepCached(symIdx);

   double dist=ComputeHybridTakeProfitDistance(symIdx,gPositionsCount[symIdx],channelStep);

   double targetPrice=(basketDir>0) ? (avg + dist) : (avg - dist);

   avgOut=avg;
   targetOut=targetPrice;

   return (basketDir>0) ? (bid >= targetPrice) : (ask <= targetPrice);
}

void CheckPairTakeProfit(string symbol,int symIdx,CArrayDouble &trades,int magic)
{
   double avg=0.0,target=0.0;
   if(BasketTPHit(symbol,symIdx,magic,avg,target))
   {
      double cp=0.0;
      int cc=0;
      ClosePositions(symbol,magic,cp,cc);
   }
}

void CheckTakeProfit()
{
   for(int i=0;i<gSymbolCount;i++)
      if(gPositionsCount[i]>0)
         CheckPairTakeProfit(gSymbols[i],i,gTrades[i],gMagics[i]);
}

void TrailingStopForPair(string symbol,int symIdx,CArrayDouble &trades,int magic)
{
   if(gPositionsCount[symIdx] <= 0) return;
   if(trades.Total()<=0) return;

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double mid=0.5*(bid+ask);
   double avg=(gBasketAvgValid[symIdx]
               ? gBasketAvgPriceCache[symIdx]
               : BasketAvgPrice(trades,gTradeLots[symIdx],mid));

   int basketDir = BasketDir(symbol,magic);
   if(basketDir==0) return;

   if(basketDir>0)
   {
      if((bid-avg) <= TrailStart*point)
         return;
   }
   else
   {
      if((avg-ask) <= TrailStart*point)
         return;
   }

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;

      double sl=PositionGetDouble(POSITION_SL);

      if(basketDir>0)
      {
         double newSL=bid - TrailStop*point;
         if(newSL>sl || sl==0.0)
            trade.PositionModify(ticket,newSL,0.0);
      }
      else
      {
         double newSL=ask + TrailStop*point;
         if(newSL<sl || sl==0.0)
            trade.PositionModify(ticket,newSL,0.0);
      }
   }
}

void TrailingStop()
{
   if(!UseTrailingStop) return;
   for(int i=0;i<gSymbolCount;i++)
      TrailingStopForPair(gSymbols[i],i,gTrades[i],gMagics[i]);
}

bool AllowActionByPolicy(const int symIdx, const int positionsCount, const int action, const ENUM_ORDER_TYPE orderType)
{
   if(!UseDangerBrain) return true;

   BrainMode m=gMode[symIdx];

   if(DangerActionPolicy==0) return true;

   int trend = TrendDirFromH1(gSymbols[symIdx], symIdx);

   if(m==MODE_DANGER)
   {
      if(DangerActionPolicy==1)
      {
         return (action==0);
      }
      if(DangerActionPolicy==2)
      {
         if(positionsCount>0) return false;
         if(trend==0) return false;
         if(orderType==ORDER_TYPE_BUY && trend<0) return false;
         if(orderType==ORDER_TYPE_SELL && trend>0) return false;
         return true;
      }
   }

   if(m==MODE_CAUTION)
   {
      if(trend==0) return (action==0);
      if(orderType==ORDER_TYPE_BUY && trend<0) return false;
      if(orderType==ORDER_TYPE_SELL && trend>0) return false;
      return true;
   }

   return true;
}

//  Next part starts at: 

bool ComputeExactStatArbZScoreClosedBar(const string symbol,
                                        const string symbol2,
                                        const ENUM_TIMEFRAMES tf,
                                        const int period,
                                        const datetime barTime,
                                        double &zScore,
                                        int &colorCode)
{
   zScore = 0.0;
   colorCode = 0;
   if(period <= 1 || StringLen(symbol2) <= 0 || barTime <= 0)
      return false;

   double spreads[];
   ArrayResize(spreads, period);

   for(int j=0; j<period; ++j)
   {
      datetime tj = iTime(symbol, tf, j + 1);
      if(tj <= 0) return false;

      double close1 = iClose(symbol, tf, j + 1);
      double close2[];
      if(CopyClose(symbol2, tf, tj, 1, close2) <= 0)
         return false;

      double price2 = close2[0];
      if(close1 <= 0.0 || price2 <= 0.0)
         spreads[j] = 0.0;
      else
         spreads[j] = MathLog(close1) - MathLog(price2);

      if(tj == barTime)
      {
         // nothing; current closed bar should be j==0 in normal use
      }
   }

   double sum = 0.0;
   for(int j=0; j<period; ++j) sum += spreads[j];
   double mean = sum / (double)period;

   double variance_sum = 0.0;
   for(int j=0; j<period; ++j)
      variance_sum += MathPow(spreads[j] - mean, 2.0);

   double std_dev = MathSqrt(variance_sum / (double)period);
   if(std_dev > 0.0000001)
      zScore = (spreads[0] - mean) / std_dev;
   else
      zScore = 0.0;

   if(zScore >= ZScoreExtremeThreshold) colorCode = 2;
   else if(zScore <= -ZScoreExtremeThreshold) colorCode = 1;
   else colorCode = 0;

   return true;
}


int ZScoreEmergencyHedgeMagic(const int baseMagic)
{
   return baseMagic + ZScoreEmergencyHedgeMagicOffset;
}

int CountPositionsByMagicSimple(const string symbol,const int magic)
{
   int count=0;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0 || !PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      count++;
   }
   return count;
}

double SumPositionLotsByMagic(const string symbol,const int magic)
{
   double lots=0.0;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0 || !PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      lots += PositionGetDouble(POSITION_VOLUME);
   }
   return lots;
}

double SumPositionProfitByMagic(const string symbol,const int magic)
{
   double pnl=0.0;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0 || !PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC)!=magic) continue;
      pnl += PositionGetDouble(POSITION_PROFIT);
   }
   return pnl;
}

bool HasOpenPositionsByMagic(const string symbol,const int magic)
{
   return (CountPositionsByMagicSimple(symbol,magic) > 0);
}

bool OpenEmergencyHedgeAgainstBasket(const string symbol,const int mainMagic,const int hedgeMagic,const int basketDir)
{
   if(basketDir==0) return false;
   double totalLots = SumPositionLotsByMagic(symbol, mainMagic);
   if(totalLots <= 0.0) return false;

   double hedgeLots = NormalizeVolume(symbol, totalLots * MathMax(0.0, ZScoreEmergencyHedgeLotFactor));
   if(hedgeLots <= 0.0) return false;

   ENUM_ORDER_TYPE hedgeType = (basketDir > 0 ? ORDER_TYPE_SELL : ORDER_TYPE_BUY);
   return OpenPosition(symbol, hedgeType, hedgeLots, 0, 0, hedgeMagic);
}

void UpdateZScoreRiskGuardForSymbol(const string symbol, const int symIdx)
{
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS)
      return;

   if(!UseZScoreRiskGuard)
   {
      gZScoreLastValue[symIdx] = 0.0;
      gZScoreLastAbs[symIdx] = 0.0;
      gZScoreExtremeNow[symIdx] = false;
      gZScorePauseTrading[symIdx] = false;
      gZScoreQuietBars[symIdx] = 0;
      gZScoreLastValueA[symIdx] = 0.0;
      gZScoreLastAbsA[symIdx] = 0.0;
      gZScoreExtremeA[symIdx] = false;
      gZScoreLastValueB[symIdx] = 0.0;
      gZScoreLastAbsB[symIdx] = 0.0;
      gZScoreExtremeB[symIdx] = false;
      gZScoreEmergencyHedgeActive[symIdx] = false;
      gZScoreEmergencyOriginalDir[symIdx] = 0;
      gZScoreEmergencyHedgeBar[symIdx] = 0;
      return;
   }

   if(StringLen(ZScoreSymbol2) <= 0)
      return;

   datetime closedBarTime = iTime(symbol, ZScoreTF, 1);
   if(closedBarTime <= 0 || closedBarTime == gZScoreLastClosedBarTF[symIdx])
      return;

   gZScoreLastClosedBarTF[symIdx] = closedBarTime;

   double zA = 0.0;
   int colorA = 0;
   bool okA = ComputeExactStatArbZScoreClosedBar(symbol, ZScoreSymbol2, ZScoreTF, ZScorePeriod, closedBarTime, zA, colorA);
   if(!okA)
      return;

   gZScoreLastValueA[symIdx] = zA;
   gZScoreLastAbsA[symIdx] = MathAbs(zA);
   gZScoreExtremeA[symIdx] = (colorA == 1 || colorA == 2);

   double zB = 0.0;
   int colorB = 0;
   bool okB = false;
   if(UseSecondZScoreMarket && StringLen(ZScoreSymbol2_B) > 0)
   {
      okB = ComputeExactStatArbZScoreClosedBar(symbol, ZScoreSymbol2_B, ZScoreTF, ZScorePeriod_B, closedBarTime, zB, colorB);
      if(okB)
      {
         gZScoreLastValueB[symIdx] = zB;
         gZScoreLastAbsB[symIdx] = MathAbs(zB);
         gZScoreExtremeB[symIdx] = (colorB == 1 || colorB == 2);
      }
      else
      {
         gZScoreLastValueB[symIdx] = 0.0;
         gZScoreLastAbsB[symIdx] = 0.0;
         gZScoreExtremeB[symIdx] = false;
      }
   }
   else
   {
      gZScoreLastValueB[symIdx] = 0.0;
      gZScoreLastAbsB[symIdx] = 0.0;
      gZScoreExtremeB[symIdx] = false;
   }

   bool combinedExtreme = gZScoreExtremeA[symIdx];
   if(UseSecondZScoreMarket && StringLen(ZScoreSymbol2_B) > 0)
   {
      if(UseZScoreOrRule)
         combinedExtreme = (gZScoreExtremeA[symIdx] || gZScoreExtremeB[symIdx]);
      else
         combinedExtreme = (gZScoreExtremeA[symIdx] && gZScoreExtremeB[symIdx]);
   }

   // Effective z-score exposed to DDQN/state = the stronger of the active monitors by absolute value
   if(gZScoreLastAbsB[symIdx] > gZScoreLastAbsA[symIdx])
   {
      gZScoreLastValue[symIdx] = gZScoreLastValueB[symIdx];
      gZScoreLastAbs[symIdx]   = gZScoreLastAbsB[symIdx];
   }
   else
   {
      gZScoreLastValue[symIdx] = gZScoreLastValueA[symIdx];
      gZScoreLastAbs[symIdx]   = gZScoreLastAbsA[symIdx];
   }

   gZScoreExtremeNow[symIdx] = combinedExtreme;

   if(gZScoreExtremeNow[symIdx])
   {
      gZScorePauseTrading[symIdx] = true;
      gZScoreQuietBars[symIdx] = 0;
      gZScoreLastExtremeBarTime[symIdx] = closedBarTime;
   }
   else if(gZScorePauseTrading[symIdx])
   {
      gZScoreQuietBars[symIdx]++;
      if(gZScoreQuietBars[symIdx] >= MathMax(1, ZScoreResumeQuietBars))
      {
         gZScorePauseTrading[symIdx] = false;
         gZScoreQuietBars[symIdx] = MathMax(1, ZScoreResumeQuietBars);
      }
   }
}

bool IsZScoreTradingPaused(const int symIdx)
{
   if(!UseZScoreRiskGuard) return false;
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return false;
   return gZScorePauseTrading[symIdx];
}

