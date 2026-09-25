//+------------------------------------------------------------------+
//| 03_CoreUtilities.mqh                                            |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| General symbol, array, normalization, similarity, serialization |
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

int SymbolIndex(const string symbol)
{
   for(int i=0;i<gSymbolCount;i++)
      if(gSymbols[i]==symbol) return i;
   return -1;
}

datetime gLastBasketCloseTime[MAX_SYMBOLS];
datetime gLastBasketCloseExecBar[MAX_SYMBOLS];

void MarkBasketClosedForCooldown(const string symbol)
{
   int symIdx = SymbolIndex(symbol);
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return;
   gLastBasketCloseTime[symIdx] = TimeCurrent();
   datetime t = iTime(symbol, TF_EXEC, 1);
   if(t <= 0) t = TimeCurrent();
   gLastBasketCloseExecBar[symIdx] = t;
}

bool IsReentryCooldownActive(const string symbol,const int symIdx)
{
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return false;
   if(ReentryCooldownSeconds <= 0 && ReentryCooldownExecBars <= 0) return false;

   bool blocked = false;

   if(ReentryCooldownSeconds > 0 && gLastBasketCloseTime[symIdx] > 0)
   {
      if((TimeCurrent() - gLastBasketCloseTime[symIdx]) < ReentryCooldownSeconds)
         blocked = true;
   }

   if(ReentryCooldownExecBars > 0 && gLastBasketCloseExecBar[symIdx] > 0)
   {
      datetime curExecBar = iTime(symbol, TF_EXEC, 1);
      if(curExecBar <= 0) curExecBar = TimeCurrent();
      int barsPassed = iBarShift(symbol, TF_EXEC, gLastBasketCloseExecBar[symIdx], false) - iBarShift(symbol, TF_EXEC, curExecBar, false);
      if(barsPassed < 0) barsPassed = 0;
      if(barsPassed < ReentryCooldownExecBars)
         blocked = true;
   }

   return blocked;
}

void ClearDeepBasketStrongFirstTradeArm(const int symIdx)
{
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return;
   gDeepBasketStrongEntryArmed[symIdx] = false;
   gDeepBasketStrongEntryArmTime[symIdx] = 0;
   gDeepBasketStrongEntryClosedCount[symIdx] = 0;
}

void ArmDeepBasketStrongFirstTrade(const string symbol,const int closedCount)
{
   if(!UseDeepBasketStrongFirstTrade) return;
   if(closedCount < DeepBasketStrongFirstTradeMinClosed) return;

   int symIdx = SymbolIndex(symbol);
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return;

   gDeepBasketStrongEntryArmed[symIdx] = true;
   gDeepBasketStrongEntryArmTime[symIdx] = TimeCurrent();
   gDeepBasketStrongEntryClosedCount[symIdx] = closedCount;

   if(VerboseLogging)
      Print("DEEP BASKET STRONG FIRST TRADE ARMED | symbol=", symbol,
            " closedCount=", closedCount,
            " mult=", DoubleToString(DeepBasketStrongFirstTradeMult, 2));
}

bool ShouldUseDeepBasketStrongFirstTrade(const int symIdx)
{
   if(!UseDeepBasketStrongFirstTrade) return false;
   if(symIdx < 0 || symIdx >= MAX_SYMBOLS) return false;
   if(!gDeepBasketStrongEntryArmed[symIdx]) return false;
   if(gPositionsCount[symIdx] > 0) return false;
   return true;
}


double Clamp(const double v, const double lo, const double hi)
{
   if(v < lo) return lo;
   if(v > hi) return hi;
   return v;
}

bool Copy1(const int handle, const int buffer, double &outVal)
{
   if(handle == INVALID_HANDLE) return false;
   double tmp[1];
   if(CopyBuffer(handle, buffer, 0, 1, tmp) <= 0) return false;
   outVal = tmp[0];
   return true;
}

void CloneState(const double &src[], double &dst[])
{
   int n=ArraySize(src);
   ArrayResize(dst,n);
   for(int i=0;i<n;i++) dst[i]=src[i];
}

void BuildPostOpenNextState(const string symbol,
                            const int symIdx,
                            const int positionsAfterOpen,
                            CArrayDouble &trades,
                            double &nextState[])
{
   BuildState(symbol, symIdx, positionsAfterOpen, trades, nextState);

   bool extremeAfter=IsExtremeState(symIdx,positionsAfterOpen);
   if(SeparateExtremeStates)
   {
      int sz=ArraySize(nextState);
      if(sz>0) nextState[sz-1]=(extremeAfter?1.0:0.0);
   }
}

void SubmitTransitionWithNextState(const int symIdx,
                                   const int regime,
                                   const double &state[],
                                   const int action,
                                   const double reward,
                                   const bool done,
                                   const double &nextState[])
{
   double stateCopy[];
   double nextCopy[];
   CloneState(state,stateCopy);
   CloneState(nextState,nextCopy);

   if(UseRealPostActionTransitions)
      ReplayPush(symIdx,regime,state,action,reward,nextState,done);
   else
      ReplayPush(symIdx,regime,state,action,reward,state,done);

   if(!UseReplayBuffer)
   {
      if(UseRealPostActionTransitions)
         DQNUpdate(symIdx,regime,stateCopy,action,reward,nextCopy,done);
      else
         DQNUpdate(symIdx,regime,stateCopy,action,reward,stateCopy,done);
   }
}




void WriteDoubleArray(const int h, const double &arr[])
{
   int sz=ArraySize(arr);
   FileWriteInteger(h, sz);
   for(int i=0;i<sz;i++) FileWriteDouble(h, arr[i]);
}

void ReadDoubleArray(const int h, double &arr[])
{
   int sz=FileReadInteger(h);
   ArrayResize(arr, sz);
   for(int i=0;i<sz;i++) arr[i]=FileReadDouble(h);
}

void WriteBasketSnapshot(const int h, const BasketSnapshot &snap)
{
   FileWriteLong(h, (long)snap.timeStamp);
   FileWriteString(h, snap.symbol);
   FileWriteInteger(h, snap.magic);
   FileWriteInteger(h, snap.regime);
   FileWriteInteger(h, snap.basketDir);
   FileWriteInteger(h, snap.positionsCount);
   FileWriteDouble(h, snap.equity);
   FileWriteDouble(h, snap.openPnL);
   FileWriteDouble(h, snap.avgPrice);
   FileWriteDouble(h, snap.midPrice);
   FileWriteDouble(h, snap.entryPrice);
   FileWriteDouble(h, snap.gridStep);
   FileWriteDouble(h, snap.danger);
   WriteDoubleArray(h, snap.stateKey);
   WriteDoubleArray(h, snap.qVals);
}

void ReadBasketSnapshot(const int h, BasketSnapshot &snap)
{
   snap.timeStamp=(datetime)FileReadLong(h);
   snap.symbol=FileReadString(h);
   snap.magic=FileReadInteger(h);
   snap.regime=FileReadInteger(h);
   snap.basketDir=FileReadInteger(h);
   snap.positionsCount=FileReadInteger(h);
   snap.equity=FileReadDouble(h);
   snap.openPnL=FileReadDouble(h);
   snap.avgPrice=FileReadDouble(h);
   snap.midPrice=FileReadDouble(h);
   snap.entryPrice=FileReadDouble(h);
   snap.gridStep=FileReadDouble(h);
   snap.danger=FileReadDouble(h);
   ReadDoubleArray(h, snap.stateKey);
   ReadDoubleArray(h, snap.qVals);
}

void WriteTickTraceItem(const int h, const TickTraceItem &tick)
{
   FileWriteLong(h, (long)tick.timeStamp);
   FileWriteDouble(h, tick.bid);
   FileWriteDouble(h, tick.ask);
   FileWriteDouble(h, tick.mid);
   FileWriteDouble(h, tick.spreadPoints);
   FileWriteDouble(h, tick.equity);
   FileWriteDouble(h, tick.danger);
}

void ReadTickTraceItem(const int h, TickTraceItem &tick)
{
   tick.timeStamp=(datetime)FileReadLong(h);
   tick.bid=FileReadDouble(h);
   tick.ask=FileReadDouble(h);
   tick.mid=FileReadDouble(h);
   tick.spreadPoints=FileReadDouble(h);
   tick.equity=FileReadDouble(h);
   tick.danger=FileReadDouble(h);
}

double SafeDiv(const double num, const double den, const double fallback=0.0)
{
   if(MathAbs(den) <= 1e-12) return fallback;
   return num / den;
}

void Push(double &arr[], const double v)
{
   int n = ArraySize(arr);
   ArrayResize(arr, n+1);
   arr[n] = v;
}

double NormalizeVolume(const string symbol, double vol)
{
   double vmin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vmax  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double vstep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(vstep <= 0.0) vstep = 0.01;
   if(vmin  <= 0.0) vmin  = vstep;
   if(vmax  <= 0.0) vmax  = vol;

   if(vol < vmin) vol = vmin;
   if(vol > vmax) vol = vmax;

   double steps = MathFloor(vol / vstep + 1e-9);
   double out   = steps * vstep;

   if(out < vmin) out = vmin;
   if(out > vmax) out = vmax;

   int digits = 0;
   if(vstep < 1.0)
   {
      double lg = -MathLog10(vstep);
      if(lg < 0.0) lg = 0.0;
      digits = (int)MathCeil(lg);
      if(digits > 8) digits = 8;
   }
   return NormalizeDouble(out, digits);
}

double Sigmoid(const double x)
{
   double z=MathMax(-60.0, MathMin(60.0, x));
   return 1.0/(1.0+MathExp(-z));
}

void NormalizeVec(double &v[])
{
   double norm=0.0;
   for(int i=0;i<ArraySize(v);i++) norm += v[i]*v[i];
   norm=MathSqrt(norm);
   if(norm<=1e-12) return;
   for(int i=0;i<ArraySize(v);i++) v[i]/=norm;
}
void NormalizeVecN(double &v[]) { NormalizeVec(v); }

void NormalizeVecSlice(double &v[], const int start, const int count)
{
   int n=ArraySize(v);
   int c=MathMax(count,0);
   if(c<=0 || start<0 || start>=n) return;
   int end=MathMin(start+c,n);
   double norm=0.0;
   for(int i=start;i<end;i++) norm += v[i]*v[i];
   norm=MathSqrt(norm);
   if(norm<=1e-12) return;
   for(int i=start;i<end;i++) v[i]/=norm;
}

double CosSim(const double &a[], const double &b[])
{
   int n=ArraySize(a);
   if(n<=0 || ArraySize(b)!=n) return -1.0;
   double dot=0, aa=0, bb=0;
   for(int i=0;i<n;i++){ dot+=a[i]*b[i]; aa+=a[i]*a[i]; bb+=b[i]*b[i]; }
   if(aa<=1e-12 || bb<=1e-12) return -1.0;
   return dot/(MathSqrt(aa)*MathSqrt(bb));
}

bool BuildDeltaVec(const double &cur[], const double &prev[], int n, double &out[])
{
   ArrayResize(out, n);
   double s2=0.0;
   for(int i=0;i<n;i++)
   {
      out[i]=cur[i]-prev[i];
      s2 += out[i]*out[i];
   }
   if(s2<=1e-12) return false;
   NormalizeVec(out);
   return true;
}

double ProtoAgeFactor(const ProtoEntry &p)
{
   if(ProtoHalfLifeDays<=0.0) return 1.0;
   datetime now=TimeCurrent();
   double ageSec=(double)(now - p.created);
   if(ageSec<=0.0) return 1.0;

   double half=ProtoHalfLifeDays*86400.0;
   if(half<=1.0) return 1.0;

   double k = 0.6931471805599453;
   double f = MathExp(-k * (ageSec/half));
   return Clamp(f, 0.05, 1.0);
}

void ProtoScoreBump(int idx, double delta)
{
   if(idx<0 || idx>=ArraySize(gProtos)) return;
   gProtos[idx].survivalScore = MathMax(0.0, gProtos[idx].survivalScore + delta);
}

void PruneProtosIfNeeded()
{
   int n=ArraySize(gProtos);
   if(n<=MaxProtosStored) return;

   while(ArraySize(gProtos) > MaxProtosStored)
   {
      int worst=-1;
      double worstEff=DBL_MAX;
      datetime worstTime=TimeCurrent();

      int m=ArraySize(gProtos);
      for(int i=0;i<m;i++)
      {
         double eff = gProtos[i].survivalScore * ProtoAgeFactor(gProtos[i]);
         datetime t = gProtos[i].created;

         if(gProtos[i].survivalScore >= PruneMinKeepScore)
            eff += 1e6;

         if(eff < worstEff || (MathAbs(eff-worstEff)<1e-9 && t < worstTime))
         {
            worstEff=eff;
            worstTime=t;
            worst=i;
         }
      }

      if(worst<0) worst=0;
      ArrayRemove(gProtos, worst, 1);
   }
}

double QMemAgeFactor(const QMemEntry &e)
{
   if(ProtoHalfLifeDays<=0.0) return 1.0;

   datetime now=TimeCurrent();
   double ageSec=(double)(now - e.created);
   if(ageSec<=0.0) return 1.0;

   double half=ProtoHalfLifeDays*86400.0;
   if(half<=1.0) return 1.0;

   double k=0.6931471805599453;
   double f=MathExp(-k*(ageSec/half));
   return Clamp(f,0.05,1.0);
}

double QMemSimilarity(const double &a[], const double &b[])
{
   return CosSim(a,b);
}

void PruneQMemoryIfNeeded()
{
   if(ArraySize(gQMem)<=MaxQMemEntries) return;

   while(ArraySize(gQMem) > MaxQMemEntries)
   {
      int worst=-1;
      double worstEff=DBL_MAX;
      datetime worstTime=TimeCurrent();

      int n=ArraySize(gQMem);
      for(int i=0;i<n;i++)
      {
         double eff = gQMem[i].score * gQMem[i].conf * QMemAgeFactor(gQMem[i]);
         datetime t = gQMem[i].created;

         if(gQMem[i].score >= QMemPruneMinScore)
            eff += 1e6;

         if(eff < worstEff || (MathAbs(eff-worstEff)<1e-9 && t < worstTime))
         {
            worstEff=eff;
            worstTime=t;
            worst=i;
         }
      }

      if(worst<0) worst=0;
      ArrayRemove(gQMem,worst,1);
   }
}

bool ComputeFingerprint(const string sym, ENUM_TIMEFRAMES tf, int bars, double &out[])
{
   if(bars<20) return false;

   MqlRates rates[];
   int got=CopyRates(sym, tf, 0, bars+2, rates);
   if(got < bars+2) return false;

   double rets[];
   ArrayResize(rets,bars);

   double mean=0.0;
   for(int i=0;i<bars;i++)
   {
      double c0=rates[i].close;
      double c1=rates[i+1].close;
      rets[i]=SafeDiv(c0-c1, c1, 0.0);
      mean += rets[i];
   }
   mean /= bars;

   double var=0.0;
   for(int i=0;i<bars;i++){ double d=rets[i]-mean; var+=d*d; }
   var /= MathMax(1,bars-1);
   double std=MathSqrt(var);

   int fast=10, slow=50;
   double emaF=rates[bars].close, emaS=rates[bars].close;
   double kf=2.0/(fast+1.0), ks=2.0/(slow+1.0);
   for(int i=bars-1;i>=0;i--)
   {
      emaF = rates[i].close*kf + emaF*(1.0-kf);
      emaS = rates[i].close*ks + emaS*(1.0-ks);
   }
   double slope = SafeDiv(emaF-emaS, rates[0].close, 0.0);

   int pos=0, neg=0;
   for(int i=0;i<bars;i++){ if(rets[i]>0) pos++; else if(rets[i]<0) neg++; }
   int dom=(pos>=neg?1:-1);

   int same=0;
   for(int i=0;i<bars;i++){ if(dom==1 && rets[i]>0) same++; if(dom==-1 && rets[i]<0) same++; }
   double monot = SafeDiv(same, bars, 0.0);

   double sameMag=0, oppMag=0;
   for(int i=0;i<bars;i++)
   {
      double r=rets[i];
      if(dom==1){ if(r>0) sameMag+=r; else oppMag+=-r; }
      else      { if(r<0) sameMag+=-r; else oppMag+=r;  }
   }
   double pullback = SafeDiv(oppMag, sameMag+1e-12, 0.0);

   int aFast=14, aSlow=100;
   int nF=MathMin(aFast,bars), nS=MathMin(aSlow,bars);
   double trF=0, trS=0;
   for(int i=0;i<nF;i++) trF += (rates[i].high-rates[i].low);
   for(int i=0;i<nS;i++) trS += (rates[i].high-rates[i].low);
   trF = SafeDiv(trF,nF,0.0);
   trS = SafeDiv(trS,nS,0.0);
   double atrRatio = SafeDiv(trF,trS,1.0);

   double maxAbs=0.0;
   for(int i=0;i<bars;i++) maxAbs=MathMax(maxAbs, MathAbs(rets[i]));
   double impulse=SafeDiv(maxAbs, std+1e-12, 0.0);

   ArrayResize(out,6);
   out[0]=slope;
   out[1]=monot;
   out[2]=pullback;
   out[3]=atrRatio;
   out[4]=impulse;
   out[5]=std;

   NormalizeVec(out);
   return true;
}

// forward
