//+------------------------------------------------------------------+
//| 06_GridRegimeIndicators.mqh                                     |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Position counting, basket/grid calculations, ATR regime handling|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

void CountOpenPositions()
{
   for(int i=0;i<gSymbolCount;i++)
   {
      gPositionsCount[i]=0;
      gTrades[i].Clear();
      gTradeLots[i].Clear();

      gCachedSymbolPnL[i]=0.0;
      gHasPositionTypeCache[i]=false;
      gBasketDirCache[i]=0;
      gBasketAvgPriceCache[i]=0.0;
      gBasketAvgValid[i]=false;
   }

   int total=PositionsTotal();
   for(int i=0;i<total;i++)
   {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym=PositionGetString(POSITION_SYMBOL);
      long mg=PositionGetInteger(POSITION_MAGIC);
      double price=PositionGetDouble(POSITION_PRICE_OPEN);
      double vol=PositionGetDouble(POSITION_VOLUME);
      double pnl=PositionGetDouble(POSITION_PROFIT);

      int idx=SymbolIndex(sym);
      if(idx<0) continue;
      if((int)mg!=gMagics[idx]) continue;

      gPositionsCount[idx]++;
      gTrades[idx].Add(price);
      gTradeLots[idx].Add(vol);
      gCachedSymbolPnL[idx] += pnl;

      if(!gHasPositionTypeCache[idx])
      {
         ENUM_POSITION_TYPE pt=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         gHasPositionTypeCache[idx]=true;
         gBasketDirCache[idx]=(pt==POSITION_TYPE_BUY ? +1 : -1);
      }
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      if(gTrades[i].Total()>0 && gTradeLots[i].Total()==gTrades[i].Total())
      {
         double weighted=0.0;
         double totalLots=0.0;

         for(int j=0;j<gTrades[i].Total();j++)
         {
            double vol=gTradeLots[i].At(j);
            weighted += gTrades[i].At(j)*vol;
            totalLots += vol;
         }

         if(totalLots>0.0)
         {
            gBasketAvgPriceCache[i]=weighted/totalLots;
            gBasketAvgValid[i]=true;
         }
      }
   }
}

double BasketAvgPrice(CArrayDouble &trades,CArrayDouble &tradeLots,double fallback)
{
   if(trades.Total()<=0 || tradeLots.Total()!=trades.Total()) return fallback;

   double weighted=0.0;
   double totalLots=0.0;
   for(int i=0;i<trades.Total();i++)
   {
      double vol=tradeLots.At(i);
      weighted += trades.At(i)*vol;
      totalLots += vol;
   }

   if(totalLots<=0.0) return fallback;
   return weighted / totalLots;
}

double GetEquityBudgetScale()
{
   if(EquityBudget<=0.0) return 1.0;
   double bal=AccountInfoDouble(ACCOUNT_BALANCE);
   if(bal<=1e-9) return 1.0;
   double scale=EquityBudget/bal;
   if(scale>1.0) scale=1.0;
   if(scale<0.01) scale=0.01;
   return scale;
}

double GetAtrRatioCached_Base(const string symbol)
{
   int idx=SymbolIndex(symbol);
   if(idx<0) return 1.0;

   datetime bt=iTime(symbol,BaseTF,0);
   if(bt!=gATRLastBarTime[idx])
   {
      double f=0.0,s=0.0;
      if(Copy1(hATRfast_Base[idx],0,f)) gATRfast_BaseVal[idx]=f;
      if(Copy1(hATRslow_Base[idx],0,s)) gATRslow_BaseVal[idx]=s;

      double ratio=1.0;
      if(s>0.0) ratio=Clamp(f/s,0.1,5.0);

      gATRratioCache[idx]=ratio;
      gATRLastBarTime[idx]=bt;
   }
   return gATRratioCache[idx];
}

int RegimeIndexFromRatio(const double atrRatio)
{
   if(!UseRegimeBank) return 0;
   if(atrRatio < RegimeLowThresh)  return 0;
   if(atrRatio > RegimeHighThresh) return 2;
   return 1;
}

bool ShouldTrainAllRegimesNow()
{
   if(!TrainAllRegimes) return false;
   if(!TrainAllRegimesWarmupOnly) return true;
   return (ArraySize(gReplay) < MathMax(100, RegimeWarmupReplayCount));
}

double RegimeContextSimilarityBonus(const double currentAtrRatio,const double protoAtrRatio)
{
   if(!UseRegimeBank) return 0.0;
   int cur = RegimeIndexFromRatio(currentAtrRatio);
   int prv = RegimeIndexFromRatio(protoAtrRatio);
   if(cur==prv) return DangerRegimeMatchBonus;
   if(MathAbs(cur-prv)>=2) return -DangerRegimeMismatchPenalty;
   return 0.0;
}

void ComputeSoftRegimeInferenceWeights(const double atrRatio,double &wOut[])
{
   ArrayResize(wOut, REGIME_COUNT);
   for(int r=0;r<REGIME_COUNT;r++) wOut[r]=0.0;
   if(!UseRegimeBank)
   {
      wOut[0]=1.0;
      return;
   }

   double span = MathMax(0.05, RegimeHighThresh - RegimeLowThresh);
   double width = MathMax(0.05, RegimeInferenceBlendWidth * MathMax(1.0, span));
   double centers[REGIME_COUNT];
   centers[0] = RegimeLowThresh - 0.50 * span;
   centers[1] = 0.50 * (RegimeLowThresh + RegimeHighThresh);
   centers[2] = RegimeHighThresh + 0.50 * span;

   double sumW=0.0;
   for(int r=0;r<REGIME_COUNT;r++)
   {
      double d = (atrRatio - centers[r]) / width;
      wOut[r] = MathExp(-0.5 * d * d);
      sumW += wOut[r];
   }
   if(sumW <= 1e-12)
   {
      int hard = RegimeIndexFromRatio(atrRatio);
      for(int r=0;r<REGIME_COUNT;r++) wOut[r] = (r==hard ? 1.0 : 0.0);
      return;
   }
   for(int r=0;r<REGIME_COUNT;r++) wOut[r] /= sumW;
}

void DQNForwardInference(int symIdx,int regime,const double &state[],double &qOut[])
{
   if(!UseRegimeBank || !UseSoftRegimeInference)
   {
      DQNForward(symIdx, regime, state, qOut);
      return;
   }

   double atrRatio = GetAtrRatioCached_Base(gSymbols[symIdx]);
   double rw[];
   ComputeSoftRegimeInferenceWeights(atrRatio, rw);

   ArrayResize(qOut, ActionCount);
   for(int a=0;a<ActionCount;a++) qOut[a]=0.0;

   for(int r=0;r<REGIME_COUNT;r++)
   {
      if(r>=ArraySize(rw) || rw[r] <= 1e-12) continue;
      double qTmp[];
      DQNForward(symIdx, r, state, qTmp);
      if(ArraySize(qTmp) != ActionCount) continue;
      for(int a=0;a<ActionCount;a++) qOut[a] += rw[r] * qTmp[a];
   }
}


double SmoothRange(const double prev,const double cur)
{
   double a=Clamp(GridRangeSmooth,0.0,1.0);
   if(a<=0.0) return cur;
   if(prev<=0.0) return cur;
   return (1.0-a)*prev + a*cur;
}

double GetFallbackGridStep(const string symbol)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double step=(double)DefaultPips*point;
   if(step<=0.0) step=point*10.0;
   return step;
}


double ChannelWidth_BaseTF(const string symbol,const int barsBack,const int shift=1)
{
   int lookback=MathMax(2,barsBack);
   int startShift=MathMax(1,shift);
   int bars=iBars(symbol,BaseTF);
   if(bars<=startShift+lookback) return 0.0;

   double hi=-DBL_MAX, lo=DBL_MAX;
   for(int i=startShift; i<startShift+lookback; i++)
   {
      double h=iHigh(symbol,BaseTF,i);
      double l=iLow(symbol,BaseTF,i);
      if(h>hi) hi=h;
      if(l<lo) lo=l;
   }

   if(hi<=lo) return 0.0;
   return (hi-lo);
}
void ComputeRollingChannelWidthStats(const string symbol,
                                     const int widthLookback,
                                     const int statsBars,
                                     double &currentWidth,
                                     double &meanWidth,
                                     double &sigmaWidth)
{
   currentWidth=0.0;
   meanWidth=0.0;
   sigmaWidth=0.0;

   int lookback=MathMax(2,widthLookback);
   int samples=MathMax(5,statsBars);
   int bars=iBars(symbol,BaseTF);
   if(bars<=lookback+samples+2)
      return;

   double sum=0.0, sumSq=0.0;
   int used=0;
   for(int shift=1; shift<=samples; shift++)
   {
      double w=ChannelWidth_BaseTF(symbol,lookback,shift);
      if(w<=0.0) continue;

      if(used==0)
         currentWidth=w;

      sum   += w;
      sumSq += w*w;
      used++;
   }

   if(used<=0)
      return;

   meanWidth=sum/(double)used;
   if(currentWidth<=0.0)
      currentWidth=meanWidth;

   if(used<2)
      return;

   double var=(sumSq/(double)used) - meanWidth*meanWidth;
   if(var>0.0)
      sigmaWidth=MathSqrt(var);
}

double ComputeAdaptiveBaseGridStep(const string symbol,const double factor)
{
   if(factor<=0.0) return 0.0;

   double currentWidth=0.0, meanWidth=0.0, sigmaWidth=0.0;
   ComputeRollingChannelWidthStats(symbol,Depth,GridWidthStatsBars,currentWidth,meanWidth,sigmaWidth);

   double baseWidth=currentWidth;
   if(baseWidth<=0.0)
      baseWidth=ChannelWidth_BaseTF(symbol,Depth,1);

   if(baseWidth<=0.0)
      return 0.0;

   if(meanWidth>0.0 && currentWidth>0.0 && currentWidth<meanWidth)
   {
      double deficit=(meanWidth-currentWidth)/MathMax(meanWidth,1e-10);
      if(deficit<0.0) deficit=0.0;

      double weightRoll=deficit;
      if(weightRoll>GridRollingMeanMaxWeight)
         weightRoll=GridRollingMeanMaxWeight;

      baseWidth=currentWidth*(1.0-weightRoll) + meanWidth*weightRoll;
   }

   return baseWidth/factor;
}

double ComputeAdaptiveGridStep(const string symbol,
                               const int symIdx,
                               const int positionsCount,
                               const double baseStep)
{
   if(baseStep<=0.0) return baseStep;
   if(!UseStdevGridExpansion) return baseStep;

   int nextPos=((positionsCount>0) ? (positionsCount+1) : 1);
   int startPos=MathMax(4,GridStdevStartPosition);
   if(nextPos<startPos) return baseStep;

   int maxPos=MathMax(startPos,GridStdevMaxPosition);
   if(nextPos>maxPos) nextPos=maxPos;

   int depthStep=nextPos-startPos+1;
   if(depthStep<=0) return baseStep;

   double sigmaStep=gGridWidthSigmaCache[symIdx];
   if(sigmaStep<=0.0) return baseStep;

   double sigmaMult=GridStdevCoeff*(double)depthStep;
   if(sigmaMult<0.0) sigmaMult=0.0;
   if(sigmaMult>GridStdevMaxSigmaMult) sigmaMult=GridStdevMaxSigmaMult;

   double adaptive=baseStep + sigmaStep*sigmaMult;
   if(adaptive<=0.0) adaptive=baseStep;

   return adaptive;
}


void UpdateMultiChannelGridStats(const int symIdx)
{
   string sym=gSymbols[symIdx];
   datetime baseBar=iTime(sym,BaseTF,0);
   if(baseBar==0) return;

   if(baseBar!=gGridLastBar[symIdx])
   {
      double step=0.0;
      double sigmaStep=0.0;

      if(UseDynamicPips)
      {
         step=ComputeAdaptiveBaseGridStep(sym,PipsFactor);

         double curWidth=0.0, meanWidth=0.0, sigmaWidth=0.0;
         ComputeRollingChannelWidthStats(sym,Depth,GridWidthStatsBars,curWidth,meanWidth,sigmaWidth);
         if(PipsFactor>0.0 && sigmaWidth>0.0)
            sigmaStep=sigmaWidth/PipsFactor;
      }

      if(step<=0.0)
         step=GetFallbackGridStep(sym);

      double smoothed=SmoothRange(gGridStepCache[symIdx],step);
      if(smoothed<=0.0) smoothed=step;

      gGridStepCache[symIdx]=smoothed;
      gGridWidthSigmaCache[symIdx]=sigmaStep;
      gGridActiveStep[symIdx]=smoothed;
      gGridActiveChannel[symIdx]=0;   // single-step / BaseTF mode
      gGridLastBar[symIdx]=baseBar;
   }
}


double GetGridStepCached(const int symIdx)
{
   UpdateMultiChannelGridStats(symIdx);

   double step=gGridStepCache[symIdx];
   if(step<=0.0)
      step=GetFallbackGridStep(gSymbols[symIdx]);

   gGridActiveStep[symIdx]=step;
   gGridActiveChannel[symIdx]=0;
   return step;
}

double GetCloseChannelStep(const int symIdx,const int positionsCount)
{
   double step=GetGridStepCached(symIdx);
   gGridActiveStep[symIdx]=step;
   gGridActiveChannel[symIdx]=0;
   return step;
}


bool IsPersistentExtremeDD(const int symIdx)
{
   if(ExtremeDDMoneyTrigger<=0.0 || ExtremeDDHoursTrigger<=0)
      return false;

   double eq=GetEAEquity();
   double ddMoney=(maxEquity>eq ? (maxEquity-eq) : 0.0);

   if(ddMoney>=ExtremeDDMoneyTrigger)
   {
      if(!gExtremeDDArmed[symIdx])
      {
         gExtremeDDArmed[symIdx]=true;
         gExtremeDDStart[symIdx]=TimeCurrent();
      }

      int secNeed=ExtremeDDHoursTrigger*3600;
      if(gExtremeDDStart[symIdx]>0 && (TimeCurrent()-gExtremeDDStart[symIdx])>=secNeed)
         return true;
   }
   else
   {
      gExtremeDDArmed[symIdx]=false;
      gExtremeDDStart[symIdx]=0;
   }

   return false;
}

void UpdateGridStatusPanel(const int symIdx)
{
   if(!ShowGridStatusPanel) return;

   string sym=gSymbols[symIdx];
   double point=SymbolInfoDouble(sym,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double stepPts=GetGridStepCached(symIdx)/point;
   double closePts=GetCloseChannelStep(symIdx,MathMax(1,gPositionsCount[symIdx]))/point;

   string txt;
   txt =
      "Leg count: " + IntegerToString(gPositionsCount[symIdx]) + "\n" +
      "Grid mode: SINGLE-STEP\n" +
      "Grid TF: " + EnumToString(BaseTF) + "\n" +
      "Add step: " + DoubleToString(stepPts,1) + " pts\n" +
      "Close ref: " + DoubleToString(closePts,1) + " pts";

   if(ProfitPauseActive())
      txt += "\nPAUSED until: " + TimeToString(gProfitPauseResumeTime, TIME_DATE|TIME_MINUTES);
   else if(UseProfitPause)
      txt += "\nProfit cycle: " + DoubleToString(gProfitCycleClosedProfit,2) +
             " / " + DoubleToString(ProfitPauseTargetAmount,2);

   if(UseReplayDiagnostics && ReplayDiagnosticsInStatusPanel)
      txt += "\n" + BuildReplayDiagnosticsText(symIdx);

   if(ObjectFind(0,gGridPanelName)<0)
   {
      ObjectCreate(0,gGridPanelName,OBJ_LABEL,0,0,0);
      ObjectSetInteger(0,gGridPanelName,OBJPROP_CORNER,GridPanelCorner);
      ObjectSetInteger(0,gGridPanelName,OBJPROP_XDISTANCE,GridPanelX);
      ObjectSetInteger(0,gGridPanelName,OBJPROP_YDISTANCE,GridPanelY);
      ObjectSetInteger(0,gGridPanelName,OBJPROP_FONTSIZE,10);
      ObjectSetString(0,gGridPanelName,OBJPROP_FONT,"Arial");
      ObjectSetInteger(0,gGridPanelName,OBJPROP_COLOR,clrWhite);
   }

   ObjectSetString(0,gGridPanelName,OBJPROP_TEXT,txt);
}


void GetZoneFeatures(string symbol,double lastClose,
                     double &zoneTypeNorm,double &zoneLocNorm,double &zoneStatusNorm)
{
   zoneTypeNorm   = 0.5;
   zoneLocNorm    = 0.5;
   zoneStatusNorm = 0.5;

   int total = ArraySize(zones);
   if(total<=0) return;

   int bestIdx=-1;
   double bestDist=DBL_MAX;
   datetime now=TimeCurrent();

   for(int i=0;i<total;i++)
   {
      if(zones[i].symbol!=symbol) continue;
      if(now>zones[i].endTime) continue;

      double centre=0.5*(zones[i].high+zones[i].low);
      double dist=MathAbs(lastClose-centre);
      if(dist<bestDist){bestDist=dist; bestIdx=i;}
   }
   if(bestIdx<0) return;

   SDZone z=zones[bestIdx];
   double h=z.high, l=z.low;
   double height=h-l;
   if(height<=0.0) return;

   zoneTypeNorm = (z.isDemand ? 0.0 : 1.0);

   double loc = (lastClose - l)/height;
   zoneLocNorm = Clamp(loc,0.0,1.0);

   if(z.broken) zoneStatusNorm=1.0;
   else if(z.tested) zoneStatusNorm=0.5;
   else zoneStatusNorm=0.0;
}

void ComputeD1TrendFeaturesRaw(string symbol,double &d1DistNorm,double &d1SlopeNorm,double &d1SideDurNorm)
{
   d1DistNorm=0.0; d1SlopeNorm=0.0; d1SideDurNorm=0.0;

   int barsD1=iBars(symbol,PERIOD_D1);
   int maxNeed=MathMax(D1_SlopeLookbackDays,D1_SideLookbackDays)+1;
   if(barsD1<=maxNeed || D1_EMA_Period<=1) return;

   int emaHandle=iMA(symbol,PERIOD_D1,D1_EMA_Period,0,MODE_EMA,PRICE_CLOSE);
   if(emaHandle==INVALID_HANDLE) return;

   double emaBuf[];
   ArraySetAsSeries(emaBuf,true);
   int copied=CopyBuffer(emaHandle,0,0,maxNeed,emaBuf);
   IndicatorRelease(emaHandle);
   if(copied<maxNeed) return;

   double emaCurr=emaBuf[0];

   double atrD1=0.0;
   int atrHandle=iATR(symbol,PERIOD_D1,ATR_SlowPeriod);
   if(atrHandle!=INVALID_HANDLE)
   {
      double buf[1];
      if(CopyBuffer(atrHandle,0,0,1,buf)>0) atrD1=buf[0];
      IndicatorRelease(atrHandle);
   }

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   if(atrD1<=0.0) atrD1=point*10000.0;

   double closeCurr=iClose(symbol,PERIOD_D1,0);
   d1DistNorm=Clamp((closeCurr-emaCurr)/atrD1,-5.0,5.0);

   int lookback=MathMin(D1_SlopeLookbackDays,barsD1-1);
   if(lookback>0 && lookback<copied)
   {
      double emaOld=emaBuf[lookback];
      d1SlopeNorm=Clamp((emaCurr-emaOld)/(atrD1*lookback),-2.0,2.0);
   }

   int maxSide=MathMin(D1_SideLookbackDays,barsD1-1);
   int sideCurr=(closeCurr>=emaCurr?1:-1);
   int count=0;
   for(int i=0;i<maxSide;i++)
   {
      double c=iClose(symbol,PERIOD_D1,i);
      double e=(i<copied?emaBuf[i]:emaCurr);
      int side=(c>=e?1:-1);
      if(side==sideCurr) count++;
      else break;
   }

   if(D1_MaxSideDurationDays>0.0)
      d1SideDurNorm=Clamp((double)count/D1_MaxSideDurationDays,0.0,1.0);
}

void GetD1TrendFeatures(string symbol,double &d1DistNorm,double &d1SlopeNorm,double &d1SideDurNorm)
{
   int idx=SymbolIndex(symbol);
   if(idx<0){ d1DistNorm=0.0; d1SlopeNorm=0.0; d1SideDurNorm=0.0; return; }

   datetime bt=iTime(symbol,PERIOD_D1,0);
   if(bt!=gD1LastBarTime[idx])
   {
      ComputeD1TrendFeaturesRaw(symbol,gD1DistCache[idx],gD1SlopeCache[idx],gD1SideDurCache[idx]);
      gD1LastBarTime[idx]=bt;
   }
   d1DistNorm=gD1DistCache[idx];
   d1SlopeNorm=gD1SlopeCache[idx];
   d1SideDurNorm=gD1SideDurCache[idx];
}

void UpdateBaseIndicatorsIfNewBar(const int symIdx)
{
   string sym=gSymbols[symIdx];
   datetime bt=iTime(sym,BaseTF,0);
   if(bt==0) return;
   if(bt==gIndLastBarTime[symIdx]) return;

   double v;
   if(Copy1(hRSI_Base[symIdx],0,v))  gRSI_Base[symIdx]=v;
   if(Copy1(hCCI_Base[symIdx],0,v))  gCCI_Base[symIdx]=v;
   if(Copy1(hMACD_Base[symIdx],0,v)) gMACD_BaseMain[symIdx]=v;
   if(Copy1(hEMA_Base[symIdx],0,v))  gEMA_BaseVal[symIdx]=v;
   if(Copy1(hRVI_Base[symIdx],0,v))  gRVI_BaseMain[symIdx]=v;
   if(Copy1(hATRfast_Base[symIdx],0,v)) gATRfast_BaseVal[symIdx]=v;
   if(Copy1(hATRslow_Base[symIdx],0,v)) gATRslow_BaseVal[symIdx]=v;

   gIndLastBarTime[symIdx]=bt;
}

void UpdateH1IfNewBar(const int symIdx)
{
   if(!UseH1Features) return;

   string sym=gSymbols[symIdx];
   datetime bt=iTime(sym,H1_TF,0);
   if(bt==0) return;
   if(bt==gH1LastBar[symIdx]) return;

   double v;
   if(Copy1(hRSI_H1[symIdx],0,v))  gRSI_H1v[symIdx]=v;
   if(Copy1(hMACD_H1[symIdx],0,v)) gMACD_H1v[symIdx]=v;
   if(Copy1(hEMA_H1[symIdx],0,v))  gEMA_H1v[symIdx]=v;
   if(Copy1(hRVI_H1[symIdx],0,v))  gRVI_H1v[symIdx]=v;

   double aF=0.0,aS=0.0;
   if(Copy1(hATRfast_H1[symIdx],0,aF) && Copy1(hATRslow_H1[symIdx],0,aS) && aS>0.0)
      gATRratio_H1[symIdx]=Clamp(aF/aS,0.1,5.0);

   gH1LastBar[symIdx]=bt;
}

void UpdateH4IfNewBar(const int symIdx)
{
   if(!UseH4Features) return;

   string sym=gSymbols[symIdx];
   datetime bt=iTime(sym,H4_TF,0);
   if(bt==0) return;
   if(bt==gH4LastBar[symIdx]) return;

   double v;
   if(Copy1(hRSI_H4[symIdx],0,v))  gRSI_H4v[symIdx]=v;
   if(Copy1(hMACD_H4[symIdx],0,v)) gMACD_H4v[symIdx]=v;
   if(Copy1(hEMA_H4[symIdx],0,v))  gEMA_H4v[symIdx]=v;
   if(Copy1(hRVI_H4[symIdx],0,v))  gRVI_H4v[symIdx]=v;

   double aF=0.0,aS=0.0;
   if(Copy1(hATRfast_H4[symIdx],0,aF) && Copy1(hATRslow_H4[symIdx],0,aS) && aS>0.0)
      gATRratio_H4[symIdx]=Clamp(aF/aS,0.1,5.0);

   gH4LastBar[symIdx]=bt;
}


struct DDQNBranchLayout
{
   int basketStart;
   int basketCount;
   int indicatorStart;
   int indicatorCount;
   int volatilityStart;
   int volatilityCount;
   int structureStart;
   int structureCount;
   int zoneCandleStart;
   int zoneCandleCount;
   int totalCount;
};

DDQNBranchLayout gBranchLayout;

struct SlowFeatureCacheEntry
{
   bool valid;
   string symbol;
   datetime execBar;
   datetime midBar;
   datetime longBar;
   long avgQ;
   long lastEntryQ;
   double feats[];
};

SlowFeatureCacheEntry gIndicatorCache[MAX_SYMBOLS];
SlowFeatureCacheEntry gVolatilityCache[MAX_SYMBOLS];
SlowFeatureCacheEntry gStructureCache[MAX_SYMBOLS];
SlowFeatureCacheEntry gZoneCandleCache[MAX_SYMBOLS];

int gDecisionBarsSinceTrain = 0;

