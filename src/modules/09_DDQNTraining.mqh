//+------------------------------------------------------------------+
//| 09_DDQNTraining.mqh                                             |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Online/target network synchronization, DQN persistence, Double-D|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

void CloneDQN(const DQNNetwork &src, DQNNetwork &dst)
{
   dst.input_dim  = src.input_dim;
   dst.hidden_dim = src.hidden_dim;
   dst.hidden_dim2= src.hidden_dim2;
   dst.output_dim = src.output_dim;
   dst.fusion_dim = src.fusion_dim;

   dst.basket_h1    = src.basket_h1;
   dst.basket_h2    = src.basket_h2;
   dst.indicator_h1 = src.indicator_h1;
   dst.indicator_h2 = src.indicator_h2;
   dst.volatility_h1= src.volatility_h1;
   dst.volatility_h2= src.volatility_h2;
   dst.structure_h1 = src.structure_h1;
   dst.structure_h2 = src.structure_h2;
   dst.zone_h1      = src.zone_h1;
   dst.zone_h2      = src.zone_h2;

   #define CLONE_ARR(name) ArrayResize(dst.name, ArraySize(src.name)); for(int i=0;i<ArraySize(src.name);i++) dst.name[i]=src.name[i];
   CLONE_ARR(basket_W1); CLONE_ARR(basket_b1); CLONE_ARR(basket_W2); CLONE_ARR(basket_b2);
   CLONE_ARR(indicator_W1); CLONE_ARR(indicator_b1); CLONE_ARR(indicator_W2); CLONE_ARR(indicator_b2);
   CLONE_ARR(volatility_W1); CLONE_ARR(volatility_b1); CLONE_ARR(volatility_W2); CLONE_ARR(volatility_b2);
   CLONE_ARR(structure_W1); CLONE_ARR(structure_b1); CLONE_ARR(structure_W2); CLONE_ARR(structure_b2);
   CLONE_ARR(zone_W1); CLONE_ARR(zone_b1); CLONE_ARR(zone_W2); CLONE_ARR(zone_b2);
   CLONE_ARR(W1); CLONE_ARR(b1); CLONE_ARR(W2); CLONE_ARR(b2); CLONE_ARR(WV); CLONE_ARR(bV); CLONE_ARR(WA); CLONE_ARR(bA);
   CLONE_ARR(feat_mean); CLONE_ARR(feat_std);
   #undef CLONE_ARR
}

void SyncTargetNetFor(const int symIdx,const int regime)
{
   CloneDQN(gDQN[symIdx][regime], gTargetDQN[symIdx][regime]);
}

void SoftUpdateTargetNetFor(const int symIdx,const int regime,const double tau)
{
   SoftUpdateDQN(gDQN[symIdx][regime], gTargetDQN[symIdx][regime], tau);
}

void SyncAllTargetNetworks()
{
   for(int i=0;i<gSymbolCount;i++)
      for(int r=0;r<REGIME_COUNT;r++)
         SyncTargetNetFor(i,r);
}

void SoftUpdateDQN(const DQNNetwork &online, DQNNetwork &target, const double tau)
{
   double a=tau;
   if(a<0.0) a=0.0;
   if(a>1.0) a=1.0;

   target.input_dim  = online.input_dim;
   target.hidden_dim = online.hidden_dim;
   target.hidden_dim2= online.hidden_dim2;
   target.output_dim = online.output_dim;
   target.fusion_dim = online.fusion_dim;

   target.basket_h1    = online.basket_h1;
   target.basket_h2    = online.basket_h2;
   target.indicator_h1 = online.indicator_h1;
   target.indicator_h2 = online.indicator_h2;
   target.volatility_h1= online.volatility_h1;
   target.volatility_h2= online.volatility_h2;
   target.structure_h1 = online.structure_h1;
   target.structure_h2 = online.structure_h2;
   target.zone_h1      = online.zone_h1;
   target.zone_h2      = online.zone_h2;

   #define SOFT_ARR(name) ArrayResize(target.name, ArraySize(online.name)); for(int i=0;i<ArraySize(online.name);i++) target.name[i]=(1.0-a)*target.name[i] + a*online.name[i];
   SOFT_ARR(basket_W1); SOFT_ARR(basket_b1); SOFT_ARR(basket_W2); SOFT_ARR(basket_b2);
   SOFT_ARR(indicator_W1); SOFT_ARR(indicator_b1); SOFT_ARR(indicator_W2); SOFT_ARR(indicator_b2);
   SOFT_ARR(volatility_W1); SOFT_ARR(volatility_b1); SOFT_ARR(volatility_W2); SOFT_ARR(volatility_b2);
   SOFT_ARR(structure_W1); SOFT_ARR(structure_b1); SOFT_ARR(structure_W2); SOFT_ARR(structure_b2);
   SOFT_ARR(zone_W1); SOFT_ARR(zone_b1); SOFT_ARR(zone_W2); SOFT_ARR(zone_b2);
   SOFT_ARR(W1); SOFT_ARR(b1); SOFT_ARR(W2); SOFT_ARR(b2); SOFT_ARR(WV); SOFT_ARR(bV); SOFT_ARR(WA); SOFT_ARR(bA); SOFT_ARR(feat_mean); SOFT_ARR(feat_std);
   #undef SOFT_ARR
}

bool SaveDQNForSymbol(int symIdx,int regime,string filename)
{
   int handle=FileOpen(filename, FILE_WRITE|FILE_BIN);
   if(handle==INVALID_HANDLE) return false;

   int version=7;
   DQNNetwork net=gDQN[symIdx][regime];
   string metaNote=BuildCurrentDQNMetadataNote();

   FileWriteInteger(handle,version);
   FileWriteInteger(handle,net.input_dim);
   FileWriteInteger(handle,net.hidden_dim);
   FileWriteInteger(handle,net.hidden_dim2);
   FileWriteInteger(handle,net.output_dim);
   FileWriteInteger(handle,net.fusion_dim);

   FileWriteInteger(handle,net.basket_h1);    FileWriteInteger(handle,net.basket_h2);
   FileWriteInteger(handle,net.indicator_h1); FileWriteInteger(handle,net.indicator_h2);
   FileWriteInteger(handle,net.volatility_h1);FileWriteInteger(handle,net.volatility_h2);
   FileWriteInteger(handle,net.structure_h1); FileWriteInteger(handle,net.structure_h2);
   FileWriteInteger(handle,net.zone_h1);      FileWriteInteger(handle,net.zone_h2);

   WriteDoubleArray(handle, net.basket_W1);    WriteDoubleArray(handle, net.basket_b1);
   WriteDoubleArray(handle, net.basket_W2);    WriteDoubleArray(handle, net.basket_b2);
   WriteDoubleArray(handle, net.indicator_W1); WriteDoubleArray(handle, net.indicator_b1);
   WriteDoubleArray(handle, net.indicator_W2); WriteDoubleArray(handle, net.indicator_b2);
   WriteDoubleArray(handle, net.volatility_W1);WriteDoubleArray(handle, net.volatility_b1);
   WriteDoubleArray(handle, net.volatility_W2);WriteDoubleArray(handle, net.volatility_b2);
   WriteDoubleArray(handle, net.structure_W1); WriteDoubleArray(handle, net.structure_b1);
   WriteDoubleArray(handle, net.structure_W2); WriteDoubleArray(handle, net.structure_b2);
   WriteDoubleArray(handle, net.zone_W1);      WriteDoubleArray(handle, net.zone_b1);
   WriteDoubleArray(handle, net.zone_W2);      WriteDoubleArray(handle, net.zone_b2);

   WriteDoubleArray(handle, net.W1);
   WriteDoubleArray(handle, net.b1);
   WriteDoubleArray(handle, net.W2);
   WriteDoubleArray(handle, net.b2);
   WriteDoubleArray(handle, net.WV);
   WriteDoubleArray(handle, net.bV);
   WriteDoubleArray(handle, net.WA);
   WriteDoubleArray(handle, net.bA);
   WriteDoubleArray(handle, net.feat_mean);
   WriteDoubleArray(handle, net.feat_std);

   FileWriteInteger(handle, (FastTrainingMode ? 1 : 0));
   FileWriteInteger(handle, (FastTrainingSkipIndicatorBranch ? 1 : 0));
   FileWriteInteger(handle, (FastTrainingSkipVolatilityBranch ? 1 : 0));
   FileWriteInteger(handle, (FastTrainingSkipStructureBranch ? 1 : 0));
   FileWriteInteger(handle, (FastTrainingSkipZoneCandleBranch ? 1 : 0));
   FileWriteInteger(handle, MathMax(1, TrainEveryNDecisionBars));
   FileWriteString(handle, metaNote);

   FileClose(handle);
   return true;
}

bool LoadDQNForSymbol(int symIdx,int regime,string filename)
{
   if(!FileIsExist(filename)) return false;

   int handle=FileOpen(filename, FILE_READ|FILE_BIN);
   if(handle==INVALID_HANDLE) return false;

   int version=FileReadInteger(handle);
   if(version<1)
   {
      FileClose(handle);
      return false;
   }

   DQNNetwork net=gDQN[symIdx][regime];
   net.input_dim  = FileReadInteger(handle);
   net.hidden_dim = FileReadInteger(handle);
   if(version>=7)
   {
      net.hidden_dim2= FileReadInteger(handle);
      net.output_dim = FileReadInteger(handle);
   }
   else
   {
      net.hidden_dim2= 0;
      net.output_dim = FileReadInteger(handle);
   }

   if(version>=5)
   {
      net.fusion_dim = FileReadInteger(handle);

      net.basket_h1    = FileReadInteger(handle); net.basket_h2    = FileReadInteger(handle);
      net.indicator_h1 = FileReadInteger(handle); net.indicator_h2 = FileReadInteger(handle);
      net.volatility_h1= FileReadInteger(handle); net.volatility_h2= FileReadInteger(handle);
      net.structure_h1 = FileReadInteger(handle); net.structure_h2 = FileReadInteger(handle);
      net.zone_h1      = FileReadInteger(handle); net.zone_h2      = FileReadInteger(handle);

      ReadDoubleArray(handle, net.basket_W1);    ReadDoubleArray(handle, net.basket_b1);
      ReadDoubleArray(handle, net.basket_W2);    ReadDoubleArray(handle, net.basket_b2);
      ReadDoubleArray(handle, net.indicator_W1); ReadDoubleArray(handle, net.indicator_b1);
      ReadDoubleArray(handle, net.indicator_W2); ReadDoubleArray(handle, net.indicator_b2);
      ReadDoubleArray(handle, net.volatility_W1);ReadDoubleArray(handle, net.volatility_b1);
      ReadDoubleArray(handle, net.volatility_W2);ReadDoubleArray(handle, net.volatility_b2);
      ReadDoubleArray(handle, net.structure_W1); ReadDoubleArray(handle, net.structure_b1);
      ReadDoubleArray(handle, net.structure_W2); ReadDoubleArray(handle, net.structure_b2);
      ReadDoubleArray(handle, net.zone_W1);      ReadDoubleArray(handle, net.zone_b1);
      ReadDoubleArray(handle, net.zone_W2);      ReadDoubleArray(handle, net.zone_b2);

      ReadDoubleArray(handle, net.W1);
      ReadDoubleArray(handle, net.b1);
      if(version>=7)
      {
         ReadDoubleArray(handle, net.W2);
         ReadDoubleArray(handle, net.b2);
      }
      else
      {
         ArrayResize(net.W2,0);
         ArrayResize(net.b2,0);
      }
      ReadDoubleArray(handle, net.WV);
      ReadDoubleArray(handle, net.bV);
      ReadDoubleArray(handle, net.WA);
      ReadDoubleArray(handle, net.bA);
      ReadDoubleArray(handle, net.feat_mean);
      ReadDoubleArray(handle, net.feat_std);

      gLoadedDQNFastModeMeta[symIdx]=false;
      gLoadedDQNMetaNote[symIdx]="legacy_v5_model; no fast-training metadata saved";

      if(version>=6)
      {
         gLoadedDQNFastModeMeta[symIdx] = (FileReadInteger(handle)!=0);
         int skipIndicator = FileReadInteger(handle);
         int skipVolatility= FileReadInteger(handle);
         int skipStructure = FileReadInteger(handle);
         int skipZone      = FileReadInteger(handle);
         int replayEveryN  = FileReadInteger(handle);
         gLoadedDQNMetaNote[symIdx]=FileReadString(handle);
         if(StringLen(gLoadedDQNMetaNote[symIdx])<=0)
            gLoadedDQNMetaNote[symIdx]=StringFormat("fast_training=%d; skip_indicator=%d; skip_volatility=%d; skip_structure=%d; skip_zone_candle=%d; replay_every_n_decision_bars=%d",
                                                    (gLoadedDQNFastModeMeta[symIdx] ? 1 : 0),
                                                    skipIndicator,
                                                    skipVolatility,
                                                    skipStructure,
                                                    skipZone,
                                                    replayEveryN);
      }

      gDQN[symIdx][regime] = net;
      FileClose(handle);
      SyncTargetNetFor(symIdx, regime);
      return true;
   }

   FileClose(handle);
   return false;
}

void DuelingComposeQ(const double valueScalar,
                     double &adv[],
                     double &qOut[])
{
   int outDim=ArraySize(adv);
   ArrayResize(qOut,outDim);

   if(outDim<=0) return;

   double meanAdv=0.0;
   for(int o=0;o<outDim;o++)
      meanAdv += adv[o];
   meanAdv /= (double)outDim;

   for(int o=0;o<outDim;o++)
      qOut[o] = valueScalar + (adv[o] - meanAdv);
}

int ArgMaxQ(double &q[])
{
   int n=ArraySize(q);
   if(n<=0) return 0;

   int best=0;
   double bestVal=q[0];

   for(int i=1;i<n;i++)
   {
      if(q[i]>bestVal)
      {
         bestVal=q[i];
         best=i;
      }
   }
   return best;
}

double HuberLossGrad(const double error,const double delta)
{
   double a=MathAbs(error);
   if(a<=delta)
      return error;
   return (error>0.0 ? delta : -delta);
}

double ClipScalar(const double v,const double clipAbs)
{
   if(clipAbs<=0.0) return v;
   if(v> clipAbs) return clipAbs;
   if(v<-clipAbs) return -clipAbs;
   return v;
}

void ClipArrayInPlace(double &arr[],const double clipAbs)
{
   if(clipAbs<=0.0) return;
   int n=ArraySize(arr);
   for(int i=0;i<n;i++)
      arr[i]=ClipScalar(arr[i],clipAbs);
}

double ComputeBootstrappedQTarget(int symIdx,int regime,double reward,double &nextState[],bool done)
{
   if(done) return reward;

   int outDim=gDQN[symIdx][regime].output_dim;
   if(outDim<=0) return reward;

   if(UseDoubleDQN)
   {
      double qNextOnline[];
      DQNForward(symIdx,regime,nextState,qNextOnline);

      int bestNextAction=ArgMaxQ(qNextOnline);
      if(bestNextAction<0) bestNextAction=0;
      if(bestNextAction>=outDim) bestNextAction=outDim-1;

      double qNextTarget[];
      if(UseTargetNet) DQNForwardTarget(symIdx,regime,nextState,qNextTarget);
      else             DQNForward(symIdx,regime,nextState,qNextTarget);

      return reward + DQNGamma * qNextTarget[bestNextAction];
   }
   else
   {
      double qNext[];
      if(UseTargetNet) DQNForwardTarget(symIdx,regime,nextState,qNext);
      else             DQNForward(symIdx,regime,nextState,qNext);

      double maxNext=qNext[0];
      for(int o=1;o<outDim;o++)
         if(qNext[o]>maxNext) maxNext=qNext[o];

      return reward + DQNGamma * maxNext;
   }
}

double QValueForAction(int symIdx,int regime,double &state[],int action)
{
   double q[];
   DQNForwardInference(symIdx,regime,state,q);

   int outDim=ArraySize(q);
   if(outDim<=0) return 0.0;
   if(action<0) action=0;
   if(action>=outDim) action=outDim-1;

   return q[action];
}

double ComputeReplayPriorityFromTransition(int symIdx,
                                           int regime,
                                           double &state[],
                                           int action,
                                           double reward,
                                           double &nextState[],
                                           bool done)
{
   double target = ComputeBootstrappedQTarget(symIdx,regime,reward,nextState,done);
   double qNow   = QValueForAction(symIdx,regime,state,action);
   double tdErr  = target - qNow;
   return MathAbs(tdErr) + ReplayPriorityEps;
}

void DQNForwardNet(const DQNNetwork &net,const double &state[],double &qOut[])
{
   int inDim=net.input_dim;
   if(UseBranchScaffold && gBranchLayout.totalCount != inDim)
      InitBranchLayoutForStateDim(inDim);
   int fusionDim=net.fusion_dim;
   int hidDim1=net.hidden_dim;
   int hidDim2=net.hidden_dim2;
   bool useSecondTrunk = (hidDim2>0 && ArraySize(net.W2)==hidDim2*hidDim1 && ArraySize(net.b2)==hidDim2);
   int headDim=(useSecondTrunk ? hidDim2 : hidDim1);
   int outDim=net.output_dim;

   double x[];
   ArrayResize(x,inDim);
   int sz=ArraySize(state);

   for(int i=0;i<inDim;i++)
   {
      double v=(i<sz ? state[i] : 0.0);
      if(UseInputNorm && i < ArraySize(net.feat_std) && net.feat_std[i] > 0.0)
         v=(v - net.feat_mean[i]) / net.feat_std[i];
      x[i]=v;
   }

   double basket_z1[],basket_a1[],basket_z2[],basket_a2[];
   double indicator_z1[],indicator_a1[],indicator_z2[],indicator_a2[];
   double volatility_z1[],volatility_a1[],volatility_z2[],volatility_a2[];
   double structure_z1[],structure_a1[],structure_z2[],structure_a2[];
   double zone_z1[],zone_a1[],zone_z2[],zone_a2[];

   ForwardBranchEncoder(x, gBranchLayout.basketStart, gBranchLayout.basketCount,
                        net.basket_W1, net.basket_b1, net.basket_W2, net.basket_b2,
                        basket_z1, basket_a1, basket_z2, basket_a2);
   ForwardBranchEncoder(x, gBranchLayout.indicatorStart, gBranchLayout.indicatorCount,
                        net.indicator_W1, net.indicator_b1, net.indicator_W2, net.indicator_b2,
                        indicator_z1, indicator_a1, indicator_z2, indicator_a2);
   ForwardBranchEncoder(x, gBranchLayout.volatilityStart, gBranchLayout.volatilityCount,
                        net.volatility_W1, net.volatility_b1, net.volatility_W2, net.volatility_b2,
                        volatility_z1, volatility_a1, volatility_z2, volatility_a2);
   ForwardBranchEncoder(x, gBranchLayout.structureStart, gBranchLayout.structureCount,
                        net.structure_W1, net.structure_b1, net.structure_W2, net.structure_b2,
                        structure_z1, structure_a1, structure_z2, structure_a2);
   ForwardBranchEncoder(x, gBranchLayout.zoneCandleStart, gBranchLayout.zoneCandleCount,
                        net.zone_W1, net.zone_b1, net.zone_W2, net.zone_b2,
                        zone_z1, zone_a1, zone_z2, zone_a2);

   double fusion[];
   ArrayResize(fusion, fusionDim);
   int cursor=0;
   for(int i=0;i<ArraySize(basket_a2);i++) fusion[cursor++]=basket_a2[i];
   for(int i=0;i<ArraySize(indicator_a2);i++) fusion[cursor++]=indicator_a2[i];
   for(int i=0;i<ArraySize(volatility_a2);i++) fusion[cursor++]=volatility_a2[i];
   for(int i=0;i<ArraySize(structure_a2);i++) fusion[cursor++]=structure_a2[i];
   for(int i=0;i<ArraySize(zone_a2);i++) fusion[cursor++]=zone_a2[i];
   while(cursor<fusionDim) fusion[cursor++]=0.0;

   double h1[];
   ArrayResize(h1,hidDim1);
   for(int hh=0;hh<hidDim1;hh++)
   {
      double sum=net.b1[hh];
      for(int i=0;i<fusionDim;i++)
         sum += net.W1[W1Index(fusionDim,hh,i)] * fusion[i];
      h1[hh]=SiLU(sum);
   }

   double h2[];
   if(useSecondTrunk)
   {
      ArrayResize(h2,hidDim2);
      for(int hh=0;hh<hidDim2;hh++)
      {
         double sum=net.b2[hh];
         for(int i=0;i<hidDim1;i++)
            sum += net.W2[DenseIndex(hidDim1,hh,i)] * h1[i];
         h2[hh]=SiLU(sum);
      }
   }

   double valueScalar = net.bV[0];
   for(int hh=0;hh<headDim;hh++)
      valueScalar += net.WV[WVIndex(headDim,0,hh)] * (useSecondTrunk ? h2[hh] : h1[hh]);

   double adv[];
   ArrayResize(adv,outDim);

   for(int o=0;o<outDim;o++)
   {
      double sum=net.bA[o];
      for(int hh=0;hh<headDim;hh++)
         sum += net.WA[WAIndex(headDim,o,hh)] * (useSecondTrunk ? h2[hh] : h1[hh]);
      adv[o]=sum;
   }

   DuelingComposeQ(valueScalar,adv,qOut);
}


void DQNForward(int symIdx,int regime,const double &state[],double &qOut[])
{
   DQNForwardNet(gDQN[symIdx][regime],state,qOut);
}

void DQNForwardTarget(int symIdx,int regime,const double &state[],double &qOut[])
{
   DQNForwardNet(gTargetDQN[symIdx][regime],state,qOut);
}

void DQNUpdateSingle(int symIdx,int regime,double &state[],int action,double reward,double &nextState[],bool done)
{
   DQNNetwork net = gDQN[symIdx][regime];
   int inDim = net.input_dim;
   if(UseBranchScaffold && gBranchLayout.totalCount != inDim)
      InitBranchLayoutForStateDim(inDim);

   int fusionDim = net.fusion_dim;
   int hidDim1   = net.hidden_dim;
   int hidDim2   = net.hidden_dim2;
   bool useSecondTrunk = (hidDim2>0 && ArraySize(net.W2)==hidDim2*hidDim1 && ArraySize(net.b2)==hidDim2);
   int headDim   = (useSecondTrunk ? hidDim2 : hidDim1);
   int outDim    = net.output_dim;

   if(action<0 || action>=outDim) return;

   double x[];
   ArrayResize(x,inDim);
   int sz=ArraySize(state);

   for(int i=0;i<inDim;i++)
   {
      double v=(i<sz ? state[i] : 0.0);
      if(UseInputNorm && i < ArraySize(net.feat_std) && net.feat_std[i] > 0.0)
         v=(v - net.feat_mean[i]) / net.feat_std[i];
      x[i]=v;
   }

   double basket_z1[],basket_a1[],basket_z2[],basket_a2[];
   double indicator_z1[],indicator_a1[],indicator_z2[],indicator_a2[];
   double volatility_z1[],volatility_a1[],volatility_z2[],volatility_a2[];
   double structure_z1[],structure_a1[],structure_z2[],structure_a2[];
   double zone_z1[],zone_a1[],zone_z2[],zone_a2[];

   ForwardBranchEncoder(x, gBranchLayout.basketStart, gBranchLayout.basketCount,
                        net.basket_W1, net.basket_b1, net.basket_W2, net.basket_b2,
                        basket_z1, basket_a1, basket_z2, basket_a2);
   ForwardBranchEncoder(x, gBranchLayout.indicatorStart, gBranchLayout.indicatorCount,
                        net.indicator_W1, net.indicator_b1, net.indicator_W2, net.indicator_b2,
                        indicator_z1, indicator_a1, indicator_z2, indicator_a2);
   ForwardBranchEncoder(x, gBranchLayout.volatilityStart, gBranchLayout.volatilityCount,
                        net.volatility_W1, net.volatility_b1, net.volatility_W2, net.volatility_b2,
                        volatility_z1, volatility_a1, volatility_z2, volatility_a2);
   ForwardBranchEncoder(x, gBranchLayout.structureStart, gBranchLayout.structureCount,
                        net.structure_W1, net.structure_b1, net.structure_W2, net.structure_b2,
                        structure_z1, structure_a1, structure_z2, structure_a2);
   ForwardBranchEncoder(x, gBranchLayout.zoneCandleStart, gBranchLayout.zoneCandleCount,
                        net.zone_W1, net.zone_b1, net.zone_W2, net.zone_b2,
                        zone_z1, zone_a1, zone_z2, zone_a2);

   double fusion[];
   ArrayResize(fusion,fusionDim);
   int basketStart=0;
   int indicatorStart=ArraySize(basket_a2);
   int volatilityStart=indicatorStart + ArraySize(indicator_a2);
   int structureStart=volatilityStart + ArraySize(volatility_a2);
   int zoneStart=structureStart + ArraySize(structure_a2);

   int cursor=0;
   for(int i=0;i<ArraySize(basket_a2);i++) fusion[cursor++]=basket_a2[i];
   for(int i=0;i<ArraySize(indicator_a2);i++) fusion[cursor++]=indicator_a2[i];
   for(int i=0;i<ArraySize(volatility_a2);i++) fusion[cursor++]=volatility_a2[i];
   for(int i=0;i<ArraySize(structure_a2);i++) fusion[cursor++]=structure_a2[i];
   for(int i=0;i<ArraySize(zone_a2);i++) fusion[cursor++]=zone_a2[i];
   while(cursor<fusionDim) fusion[cursor++]=0.0;

   double h1[];
   double zTrunk1[];
   ArrayResize(h1,hidDim1);
   ArrayResize(zTrunk1,hidDim1);
   for(int hh=0;hh<hidDim1;hh++)
   {
      double sum=net.b1[hh];
      for(int i=0;i<fusionDim;i++)
         sum += net.W1[W1Index(fusionDim,hh,i)] * fusion[i];
      zTrunk1[hh]=sum;
      h1[hh]=SiLU(sum);
   }

   double h2[];
   double zTrunk2[];
   double finalH[];
   double finalZ[];
   if(useSecondTrunk)
   {
      ArrayResize(h2,hidDim2);
      ArrayResize(zTrunk2,hidDim2);
      for(int hh=0;hh<hidDim2;hh++)
      {
         double sum=net.b2[hh];
         for(int i=0;i<hidDim1;i++)
            sum += net.W2[DenseIndex(hidDim1,hh,i)] * h1[i];
         zTrunk2[hh]=sum;
         h2[hh]=SiLU(sum);
      }
      ArrayResize(finalH,hidDim2);
      ArrayResize(finalZ,hidDim2);
      for(int i=0;i<hidDim2;i++){ finalH[i]=h2[i]; finalZ[i]=zTrunk2[i]; }
   }
   else
   {
      ArrayResize(finalH,hidDim1);
      ArrayResize(finalZ,hidDim1);
      for(int i=0;i<hidDim1;i++){ finalH[i]=h1[i]; finalZ[i]=zTrunk1[i]; }
   }

   double valueScalar = net.bV[0];
   for(int hh=0;hh<headDim;hh++)
      valueScalar += net.WV[WVIndex(headDim,0,hh)] * finalH[hh];

   double adv[];
   ArrayResize(adv,outDim);
   for(int o=0;o<outDim;o++)
   {
      double sum=net.bA[o];
      for(int hh=0;hh<headDim;hh++)
         sum += net.WA[WAIndex(headDim,o,hh)] * finalH[hh];
      adv[o]=sum;
   }

   double q[];
   DuelingComposeQ(valueScalar,adv,q);

   double target = ComputeBootstrappedQTarget(symIdx,regime,reward,nextState,done);
   double tdError = target - q[action];

   double dLoss_dQaction;
   if(UseHuberLoss)
      dLoss_dQaction = -HuberLossGrad(tdError,HuberDelta);
   else
      dLoss_dQaction = -tdError;

   double lr = DQNLearningRate;
   if(UseDangerBrain) lr *= gLRScale[symIdx];

   double dLoss_dV = dLoss_dQaction;

   double dLoss_dAdv[];
   ArrayResize(dLoss_dAdv,outDim);
   double invOut = 1.0 / (double)outDim;

   for(int o=0;o<outDim;o++)
   {
      if(o==action) dLoss_dAdv[o] = dLoss_dQaction * (1.0 - invOut);
      else          dLoss_dAdv[o] = dLoss_dQaction * (0.0 - invOut);
   }

   if(UseGradientClipping)
   {
      dLoss_dV = ClipScalar(dLoss_dV,GradientClipValue);
      ClipArrayInPlace(dLoss_dAdv,GradientClipValue);
   }

   double oldWV[];
   ArrayResize(oldWV,headDim);
   for(int hh=0;hh<headDim;hh++) oldWV[hh] = net.WV[WVIndex(headDim,0,hh)];

   double oldWA[];
   ArrayResize(oldWA,outDim*headDim);
   for(int i=0;i<ArraySize(oldWA);i++) oldWA[i] = net.WA[i];

   for(int hh=0;hh<headDim;hh++)
   {
      int idxWV = WVIndex(headDim,0,hh);
      double grad = dLoss_dV * finalH[hh];
      if(UseGradientClipping) grad = ClipScalar(grad,GradientClipValue);
      net.WV[idxWV] -= lr * grad;
   }
   {
      double gradb = dLoss_dV;
      if(UseGradientClipping) gradb = ClipScalar(gradb,GradientClipValue);
      net.bV[0] -= lr * gradb;
   }

   for(int o=0;o<outDim;o++)
   {
      for(int hh=0;hh<headDim;hh++)
      {
         int idxWA = WAIndex(headDim,o,hh);
         double grad = dLoss_dAdv[o] * finalH[hh];
         if(UseGradientClipping) grad = ClipScalar(grad,GradientClipValue);
         net.WA[idxWA] -= lr * grad;
      }
      double gradb = dLoss_dAdv[o];
      if(UseGradientClipping) gradb = ClipScalar(gradb,GradientClipValue);
      net.bA[o] -= lr * gradb;
   }

   double dHead[];
   ArrayResize(dHead,headDim);
   for(int hh=0;hh<headDim;hh++)
   {
      double grad = oldWV[hh] * dLoss_dV;
      for(int o=0;o<outDim;o++)
         grad += oldWA[WAIndex(headDim,o,hh)] * dLoss_dAdv[o];
      grad *= SiLUDerivativeFromPreAct(finalZ[hh]);
      dHead[hh] = grad;
   }
   if(UseGradientClipping) ClipArrayInPlace(dHead,GradientClipValue);

   double dTrunk1[];
   if(useSecondTrunk)
   {
      double oldW2[];
      ArrayResize(oldW2,ArraySize(net.W2));
      for(int i=0;i<ArraySize(net.W2);i++) oldW2[i]=net.W2[i];

      for(int r=0;r<hidDim2;r++)
      {
         for(int c=0;c<hidDim1;c++)
         {
            int idx=DenseIndex(hidDim1,r,c);
            double grad=dHead[r]*h1[c];
            if(UseGradientClipping) grad=ClipScalar(grad,GradientClipValue);
            net.W2[idx] -= lr*grad;
         }
         double gradb=dHead[r];
         if(UseGradientClipping) gradb=ClipScalar(gradb,GradientClipValue);
         net.b2[r] -= lr*gradb;
      }

      ArrayResize(dTrunk1,hidDim1);
      for(int c=0;c<hidDim1;c++)
      {
         double s=0.0;
         for(int r=0;r<hidDim2;r++)
            s += oldW2[DenseIndex(hidDim1,r,c)] * dHead[r];
         dTrunk1[c]=s * SiLUDerivativeFromPreAct(zTrunk1[c]);
      }
   }
   else
   {
      ArrayResize(dTrunk1,hidDim1);
      for(int i=0;i<hidDim1;i++) dTrunk1[i]=dHead[i];
   }
   if(UseGradientClipping) ClipArrayInPlace(dTrunk1,GradientClipValue);

   double oldW1[];
   ArrayResize(oldW1,ArraySize(net.W1));
   for(int i=0;i<ArraySize(net.W1);i++) oldW1[i]=net.W1[i];

   for(int hh=0;hh<hidDim1;hh++)
   {
      for(int i=0;i<fusionDim;i++)
      {
         int idx = W1Index(fusionDim,hh,i);
         double grad = dTrunk1[hh] * fusion[i];
         if(UseGradientClipping) grad = ClipScalar(grad,GradientClipValue);
         net.W1[idx] -= lr * grad;
      }
      double gradb = dTrunk1[hh];
      if(UseGradientClipping) gradb = ClipScalar(gradb,GradientClipValue);
      net.b1[hh] -= lr * gradb;
   }

   double dFusion[];
   ArrayResize(dFusion,fusionDim);
   for(int i=0;i<fusionDim;i++)
   {
      double s=0.0;
      for(int hh=0;hh<hidDim1;hh++)
         s += oldW1[W1Index(fusionDim,hh,i)] * dTrunk1[hh];
      dFusion[i]=s;
   }
   if(UseGradientClipping) ClipArrayInPlace(dFusion,GradientClipValue);

   if(ArraySize(basket_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(basket_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[basketStart+i];
      BackpropBranchEncoder(x, gBranchLayout.basketStart, gBranchLayout.basketCount,
                            net.basket_W1, net.basket_b1, net.basket_W2, net.basket_b2,
                            basket_z1, basket_a1, basket_z2, g, lr);
   }
   if(ArraySize(indicator_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(indicator_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[indicatorStart+i];
      BackpropBranchEncoder(x, gBranchLayout.indicatorStart, gBranchLayout.indicatorCount,
                            net.indicator_W1, net.indicator_b1, net.indicator_W2, net.indicator_b2,
                            indicator_z1, indicator_a1, indicator_z2, g, lr);
   }
   if(ArraySize(volatility_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(volatility_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[volatilityStart+i];
      BackpropBranchEncoder(x, gBranchLayout.volatilityStart, gBranchLayout.volatilityCount,
                            net.volatility_W1, net.volatility_b1, net.volatility_W2, net.volatility_b2,
                            volatility_z1, volatility_a1, volatility_z2, g, lr);
   }
   if(ArraySize(structure_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(structure_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[structureStart+i];
      BackpropBranchEncoder(x, gBranchLayout.structureStart, gBranchLayout.structureCount,
                            net.structure_W1, net.structure_b1, net.structure_W2, net.structure_b2,
                            structure_z1, structure_a1, structure_z2, g, lr);
   }
   if(ArraySize(zone_a2)>0)
   {
      double g[]; ArrayResize(g,ArraySize(zone_a2));
      for(int i=0;i<ArraySize(g);i++) g[i]=dFusion[zoneStart+i];
      BackpropBranchEncoder(x, gBranchLayout.zoneCandleStart, gBranchLayout.zoneCandleCount,
                            net.zone_W1, net.zone_b1, net.zone_W2, net.zone_b2,
                            zone_z1, zone_a1, zone_z2, g, lr);
   }

   gDQN[symIdx][regime] = net;
}


void DQNUpdate(int symIdx,int activeRegime,double &state[],int action,double reward,double &nextState[],bool done)
{
   DQNUpdateSingle(symIdx,activeRegime,state,action,reward,nextState,done);

   if(ShouldTrainAllRegimesNow())
   {
      for(int r=0;r<REGIME_COUNT;r++)
      {
         if(r==activeRegime) continue;
         DQNUpdateSingle(symIdx,r,state,action,reward,nextState,done);
      }
   }

   gTargetSyncCounter++;
   if(UseTargetNet && TargetSyncFreq>0 && gTargetSyncCounter>=TargetSyncFreq)
   {
      SyncAllTargetNetworks();
      gTargetSyncCounter=0;
   }
}


void RefreshReplayBankPriority(const int src,const int idx,const ReplayItem &it,const double pr)
{
   double blended = MathMax(0.01, pr + ReplayPriorityReward(it.reward));
   if(src==REPLAY_SRC_MAIN)
   {
      if(idx>=0 && idx<ArraySize(gReplay))
         gReplay[idx].priority = MathMax(0.01, 0.5*gReplay[idx].priority + 0.5*blended);
      return;
   }
   if(src==REPLAY_SRC_RECENT)
   {
      if(idx>=0 && idx<ArraySize(gRecentReplayBank.items))
         gRecentReplayBank.items[idx].priority = MathMax(0.01, 0.5*gRecentReplayBank.items[idx].priority + 0.5*blended);
      return;
   }
   if(src==REPLAY_SRC_DANGER)
   {
      if(idx>=0 && idx<ArraySize(gDangerReplayBank.items))
         gDangerReplayBank.items[idx].priority = MathMax(0.01, (1.0-DangerBankPriorityAlpha)*gDangerReplayBank.items[idx].priority + DangerBankPriorityAlpha*(blended*(1.0+0.25*MathMax(0,it.dangerClass))));
      return;
   }
   if(src==REPLAY_SRC_DEEP)
   {
      if(idx>=0 && idx<ArraySize(gDeepBasketReplayBank.items))
      {
         double w = DeepBasketSampleWeight(it);
         gDeepBasketReplayBank.items[idx].priority = MathMax(0.01, (1.0-DeepBankPriorityAlpha)*gDeepBasketReplayBank.items[idx].priority + DeepBankPriorityAlpha*(blended * MathMax(1.0, w)));
      }
      return;
   }
   if(src==REPLAY_SRC_EFFICIENT)
   {
      if(idx>=0 && idx<ArraySize(gEfficientReplayBank.items))
      {
         double effBoost = (it.addDepthClass==0 && it.reward>0.0 ? 1.10 : 1.0);
         gEfficientReplayBank.items[idx].priority = MathMax(0.01, (1.0-EfficientBankPriorityAlpha)*gEfficientReplayBank.items[idx].priority + EfficientBankPriorityAlpha*(blended*effBoost));
      }
      return;
   }
}

void TrainReplayBatch()
{
   if(!UseReplayBuffer) return;

   int n=ArraySize(gReplay);
   if(n<ReplayWarmup || ReplayBatchSize<=0) return;

   int iters=MathMax(1,ReplayTrainIters);
   int batchN=MathMax(1,ReplayBatchSize);

   for(int t=0;t<iters;t++)
   {
      for(int b=0;b<batchN;b++)
      {
         int idx=-1;
         int sampleSrc=REPLAY_SRC_MAIN;
         ReplayItem it;
         if(!SampleReplayItemMixed(it, sampleSrc, idx)) return;
         if(sampleSrc==REPLAY_SRC_MAIN && (idx<0 || idx>=ArraySize(gReplay))) continue;
         if(it.regime<0 || it.regime>=REGIME_COUNT) continue;
         if(it.symIdx<0 || it.symIdx>=gSymbolCount) continue;

         DQNUpdateSingle(it.symIdx,it.regime,it.state,it.action,it.reward,it.nextState,it.done);

         double pr = ComputeReplayPriorityFromTransition(it.symIdx,
                                                         it.regime,
                                                         it.state,
                                                         it.action,
                                                         it.reward,
                                                         it.nextState,
                                                         it.done);

         RefreshReplayBankPriority(sampleSrc, idx, it, pr);

         if(UseTargetNet && UseSoftTargetUpdate)
            SoftUpdateTargetNetFor(it.symIdx,it.regime,SoftTargetTau);
      }
   }

   if(gReplayPendingTrainCount>0)
      gReplayPendingTrainCount--;

   if(UseTargetNet && !UseSoftTargetUpdate)
   {
      gTargetSyncCounter++;
      if(TargetSyncFreq>0 && gTargetSyncCounter>=TargetSyncFreq)
      {
         SyncAllTargetNetworks();
         gTargetSyncCounter=0;
      }
   }
}

