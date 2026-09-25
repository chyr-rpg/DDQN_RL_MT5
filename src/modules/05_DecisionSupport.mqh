//+------------------------------------------------------------------+
//| 05_DecisionSupport.mqh                                          |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Q-memory voting, DD-event caution, danger vetoes, archive/pain p|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

int ArgMaxActionFromQ(const double &q[])
{
   int n=ArraySize(q);
   if(n<=0) return 0;

   int best=0;
   double bestV=q[0];
   for(int a=1;a<n;a++)
      if(q[a]>bestV){ bestV=q[a]; best=a; }

   return best;
}

int ArgMaxDirectionalActionFromQ(const double &q[])
{
   int n=ArraySize(q);
   if(n>=3) return (q[1] >= q[2] ? 1 : 2);
   if(n>=2) return 1;
   return 0;
}

double ComputeDDQNActionGap(const double &q[], const bool forcedDir=false)
{
   int n=ArraySize(q);
   if(n<=1) return 0.0;

   if(forcedDir && n>=3)
      return MathMax(0.0, MathAbs(q[1]-q[2]));

   double best=-1e100, second=-1e100;
   for(int a=0;a<n;a++)
   {
      double v=q[a];
      if(v>best)
      {
         second=best;
         best=v;
      }
      else if(v>second)
      {
         second=v;
      }
   }

   if(second<=-1e99) second=best;
   return MathMax(0.0, best-second);
}

double ComputeDrawdownWorseningScore(const int symIdx)
{
   double curDD = GetSymbolFloatingDDPct(symIdx);
   double prevDD = curDD;

   if(symIdx>=0 && symIdx<MAX_SYMBOLS && gDenseBasketHealth[symIdx].active)
      prevDD = gDenseBasketHealth[symIdx].prevDD;

   double worsen = MathMax(0.0, curDD - prevDD);
   return Clamp(SafeDiv(worsen, 0.02, 0.0), 0.0, 1.0);
}

void InitActionBias(double &bias[])
{
   ArrayResize(bias, ActionCount);
   for(int a=0;a<ActionCount;a++) bias[a]=0.0;
}

void AccumulateActionBias(double &dst[], const double &src[])
{
   int n=MathMin(ArraySize(dst), ArraySize(src));
   for(int a=0;a<n;a++)
      dst[a] += src[a];
}

void BuildQMemoryVoteBias(const int symIdx,
                          const double &qBase[],
                          const double &qMem[],
                          const double memConf,
                          const bool forcedDir,
                          double &outBias[])
{
   InitActionBias(outBias);
   if(ArraySize(qBase)<=0 || ArraySize(qMem)<=0) return;

   double conf = Clamp(memConf, 0.0, 1.0);
   if(conf<=1e-12) return;

   double base = conf * QMemBlendWeight;
   double ddPct = GetSymbolFloatingDDPct(symIdx);
   double worsen = ComputeDrawdownWorseningScore(symIdx);
   double pDanger = Clamp(gPDanger[symIdx], 0.0, 1.0);

   int ddqnBest = ArgMaxActionFromQ(qBase);
   int qmemBest = ArgMaxActionFromQ(qMem);
   int ddqnDir  = ArgMaxDirectionalActionFromQ(qBase);
   int qmemDir  = ArgMaxDirectionalActionFromQ(qMem);

   double qMemGap = ComputeDDQNActionGap(qMem, forcedDir);
   double intensity = 0.60 + 0.40 * Clamp(SafeDiv(qMemGap, 1.0, 0.0), 0.0, 1.0);

   if(!forcedDir && ActionCount>=3)
   {
      if(qmemBest==ddqnBest)
      {
         double support = base * QMemConfirmScale * intensity * (0.75 + 0.25 * (1.0 - ddPct));
         outBias[ddqnBest] += support;
         if(ddqnBest==0)
            outBias[0] += 0.15 * support;
      }
      else
      {
         double caution = base * QMemConflictToHoldScale * intensity * (0.70 + 0.80*ddPct + 0.40*worsen + 0.35*pDanger);
         outBias[0] += caution;
         if(ddqnBest>=1 && ddqnBest<=2)
            outBias[ddqnBest] -= caution * QMemConflictDirPenaltyScale;
      }
   }
   else
   {
      if(qmemDir==ddqnDir)
      {
         double support = base * QMemConfirmScale * 0.60 * intensity;
         if(ddqnDir>=1 && ddqnDir<=2)
            outBias[ddqnDir] += support;
      }
      else
      {
         double caution = base * 0.60 * intensity * (0.60 + ddPct + 0.40*worsen + 0.35*pDanger);
         if(ddqnDir>=1 && ddqnDir<=2)
            outBias[ddqnDir] -= caution * QMemConflictDirPenaltyScale;

         if(qmemDir>=1 && qmemDir<=2 && qmemDir!=ddqnDir)
            outBias[qmemDir] += 0.15 * caution;
      }
   }
}

void BuildDDEventCautionBias(const int symIdx,
                             const double &qBase[],
                             const double &ddBiasIn[],
                             const bool forcedDir,
                             double &outBias[])
{
   InitActionBias(outBias);
   if(ArraySize(ddBiasIn)<=0 || ArraySize(qBase)<=0) return;

   double ddPct = GetSymbolFloatingDDPct(symIdx);
   double worsen = ComputeDrawdownWorseningScore(symIdx);
   double pDanger = Clamp(gPDanger[symIdx], 0.0, 1.0);

   double boost = DDEventBlendWeight * (1.0 + DDEventDrawdownBoostScale * (0.80*ddPct + 0.70*worsen + 0.30*pDanger));

   int ddqnBest = ArgMaxActionFromQ(qBase);
   int ddqnDir  = ArgMaxDirectionalActionFromQ(qBase);

   if(!forcedDir && ActionCount>=3)
   {
      outBias[0] += MathMax(0.0, ddBiasIn[0]) * boost;
      if(ddqnBest>=1 && ddqnBest<=2)
      {
         double dirPenalty = MathMax(0.0, -MathMin(0.0, ddBiasIn[ddqnBest])) * boost;
         outBias[ddqnBest] -= dirPenalty;
      }
   }
   else
   {
      if(ddqnDir>=1 && ddqnDir<=2)
      {
         double dirPenalty = MathMax(0.0, -MathMin(0.0, ddBiasIn[ddqnDir])) * boost;
         outBias[ddqnDir] -= dirPenalty;

         if(!DDEventUseCautionOnly)
         {
            int alt=(ddqnDir==1 ? 2 : 1);
            if(alt>=1 && alt<=2 && alt<ArraySize(ddBiasIn))
               outBias[alt] += 0.15 * MathMax(0.0, ddBiasIn[alt]) * boost;
         }
      }
   }
}

void BuildDangerVetoBias(const int symIdx,
                         const double &qBase[],
                         const double &dangerBiasIn[],
                         const double simBest,
                         const bool forcedDir,
                         double &outBias[])
{
   InitActionBias(outBias);
   if(ArraySize(dangerBiasIn)<=0 || ArraySize(qBase)<=0) return;

   double alpha = ComputeAlphaMix(symIdx, simBest);
   if(alpha<=1e-12) return;

   double ddPct = GetSymbolFloatingDDPct(symIdx);
   double worsen = ComputeDrawdownWorseningScore(symIdx);
   double pDanger = Clamp(gPDanger[symIdx], 0.0, 1.0);

   double severity = alpha * (0.75 + 0.65*pDanger + 0.75*ddPct + 0.60*worsen);
   if(gMode[symIdx]==MODE_DANGER)  severity *= 1.15;
   if(gMode[symIdx]==MODE_CAUTION) severity *= 0.90;

   int ddqnBest = ArgMaxActionFromQ(qBase);
   int ddqnDir  = ArgMaxDirectionalActionFromQ(qBase);

   if(!forcedDir && ActionCount>=3)
   {
      double holdVeto = DangerHoldVetoScale * severity * (0.40 + MathMax(0.0, dangerBiasIn[0]));
      outBias[0] += holdVeto;

      if(ddqnBest>=1 && ddqnBest<=2)
      {
         double dirPenalty = DangerHoldVetoScale * severity * (0.30 + MathMax(0.0, -MathMin(0.0, dangerBiasIn[ddqnBest])));
         outBias[ddqnBest] -= dirPenalty;

         if(gMode[symIdx]==MODE_CAUTION && dangerBiasIn[ddqnBest] > 0.0)
            outBias[ddqnBest] += DangerDirectionalLeakScale * severity * 0.25 * dangerBiasIn[ddqnBest];
      }
   }
   else
   {
      if(ddqnDir>=1 && ddqnDir<=2)
      {
         double dirPenalty = DangerHoldVetoScale * severity * (0.25 + MathMax(0.0, -MathMin(0.0, dangerBiasIn[ddqnDir])));
         outBias[ddqnDir] -= dirPenalty;

         int alt=(ddqnDir==1 ? 2 : 1);
         if(alt>=1 && alt<=2 && alt<ArraySize(dangerBiasIn) && dangerBiasIn[alt] > 0.0)
            outBias[alt] += DangerDirectionalLeakScale * severity * dangerBiasIn[alt];
      }
   }
}

void ApplyBudgetedSubordinateBias(const int symIdx,
                                  const bool forcedDir,
                                  const double &qBase[],
                                  const double &delta[],
                                  double &qOut[])
{
   int n=ArraySize(qBase);
   ArrayResize(qOut,n);
   for(int a=0;a<n;a++) qOut[a]=qBase[a];

   double gap = ComputeDDQNActionGap(qBase, forcedDir);
   double budget = Clamp(SubordinateBiasCapFrac, 0.0, 1.0) * gap;
   if(budget<=1e-12) return;

   double maxAbs=0.0;
   int start = (forcedDir && n>=3 ? 1 : 0);
   int end   = (forcedDir && n>=3 ? 3 : n);

   for(int a=start;a<end;a++)
      maxAbs = MathMax(maxAbs, MathAbs(delta[a]));

   if(maxAbs<=1e-12) return;

   double scale = MathMin(1.0, SafeDiv(budget, maxAbs, 1.0));
   for(int a=0;a<n;a++)
      qOut[a] += scale * delta[a];
}



double MaxAbsArrayValue(const double &arr[])
{
   double out=0.0;
   int n=ArraySize(arr);
   for(int i=0;i<n;i++)
      out=MathMax(out,MathAbs(arr[i]));
   return out;
}

void ResetDecisionSupportContext(DecisionSupportContext &ctx)
{
   ctx.valid=false;
   ctx.forcedEntry=false;
   ctx.barTime=0;
   ctx.refreshTime=0;
   ctx.regime=0;
   ctx.positionsCount=0;
   ctx.basketDir=0;
   ctx.brainMode=0;
   for(int a=0;a<3;a++) ctx.actionDelta[a]=0.0;
   ctx.qMemoryAgreement=0.0;
   ctx.qMemoryConflict=0.0;
   ctx.archiveCaution=0.0;
   ctx.ddEventRisk=0.0;
   ctx.dangerProbability=0.0;
   ctx.oneRoundPrior=0.0;
   ctx.addRiskPrior=0.0;
   ctx.supportConfidence=0.0;
   ctx.painRecurrenceRisk=0.0;
   ctx.macroReversalTrapPrior=0.0;
   ctx.counterTrendFailurePrior=0.0;
   ctx.regimeBreakPainPrior=0.0;
   ctx.recoveryFalseStartRisk=0.0;
   ctx.deepBasketPainPrior=0.0;
   ctx.painMemoryAgreement=0.0;
   ctx.macroMicroConflict=0.0;
   ctx.lateTrendFadePenalty=0.0;
   ctx.painConfidence=0.0;
   ctx.trendPersistenceProb=0.0;
   ctx.trendReversalProb=0.0;
   ctx.spikeRiskProb=0.0;
   ctx.expectedBasketDepth=0.0;
   ctx.trendContinuationQuality=0.0;
   ctx.breakoutReclaimQuality=0.0;
   ctx.reversalTransitionQuality=0.0;
   ctx.modeDominanceScore=0.0;
   ctx.modeConflictScore=0.0;
}

void CopyDecisionSupportContext(const DecisionSupportContext &src, DecisionSupportContext &dst)
{
   dst.valid=src.valid;
   dst.forcedEntry=src.forcedEntry;
   dst.barTime=src.barTime;
   dst.refreshTime=src.refreshTime;
   dst.regime=src.regime;
   dst.positionsCount=src.positionsCount;
   dst.basketDir=src.basketDir;
   dst.brainMode=src.brainMode;
   for(int a=0;a<3;a++) dst.actionDelta[a]=src.actionDelta[a];
   dst.qMemoryAgreement=src.qMemoryAgreement;
   dst.qMemoryConflict=src.qMemoryConflict;
   dst.archiveCaution=src.archiveCaution;
   dst.ddEventRisk=src.ddEventRisk;
   dst.dangerProbability=src.dangerProbability;
   dst.oneRoundPrior=src.oneRoundPrior;
   dst.addRiskPrior=src.addRiskPrior;
   dst.supportConfidence=src.supportConfidence;
   dst.painRecurrenceRisk=src.painRecurrenceRisk;
   dst.macroReversalTrapPrior=src.macroReversalTrapPrior;
   dst.counterTrendFailurePrior=src.counterTrendFailurePrior;
   dst.regimeBreakPainPrior=src.regimeBreakPainPrior;
   dst.recoveryFalseStartRisk=src.recoveryFalseStartRisk;
   dst.deepBasketPainPrior=src.deepBasketPainPrior;
   dst.painMemoryAgreement=src.painMemoryAgreement;
   dst.macroMicroConflict=src.macroMicroConflict;
   dst.lateTrendFadePenalty=src.lateTrendFadePenalty;
   dst.painConfidence=src.painConfidence;
   dst.trendPersistenceProb=src.trendPersistenceProb;
   dst.trendReversalProb=src.trendReversalProb;
   dst.spikeRiskProb=src.spikeRiskProb;
   dst.expectedBasketDepth=src.expectedBasketDepth;
   dst.trendContinuationQuality=src.trendContinuationQuality;
   dst.breakoutReclaimQuality=src.breakoutReclaimQuality;
   dst.reversalTransitionQuality=src.reversalTransitionQuality;
   dst.modeDominanceScore=src.modeDominanceScore;
   dst.modeConflictScore=src.modeConflictScore;
}

datetime DecisionSupportBarTime(const string symbol)
{
   datetime barTime=iTime(symbol,BaseTF,1);
   if(barTime<=0) barTime=iTime(symbol,BaseTF,0);
   if(barTime<=0) barTime=TimeCurrent();
   return barTime;
}

void ResetDecisionSupportCacheAll()
{
   for(int i=0;i<MAX_SYMBOLS;i++)
      ResetDecisionSupportContext(gDecisionSupportCache[i]);
}

void ApplyDecisionSupportDelta(const DecisionSupportContext &ctx,double &totalDelta[])
{
   if(ArraySize(totalDelta)<ActionCount)
      ArrayResize(totalDelta,ActionCount);
   int n=MathMin(ActionCount,3);
   for(int a=0;a<n;a++)
      totalDelta[a]+=ctx.actionDelta[a];
}

bool BuildDecisionSupportContext(const int symIdx,
                                 const int regime,
                                 const double &state[],
                                 const double &qBase[],
                                 const bool forcedEntry,
                                 const int basketDirHint,
                                 DecisionSupportContext &ctx)
{
   ResetDecisionSupportContext(ctx);
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return false;

   string symbol=gSymbols[symIdx];
   datetime barTime=DecisionSupportBarTime(symbol);
   int positionsCount=gPositionsCount[symIdx];
   int basketDir=basketDirHint;
   if(positionsCount>0 && basketDir==0)
      basketDir=BasketDir(symbol,gMagics[symIdx]);
   int brainMode=(int)gMode[symIdx];

   bool cacheOk=(UseDecisionSupportCache &&
                 gDecisionSupportCache[symIdx].valid &&
                 gDecisionSupportCache[symIdx].forcedEntry==forcedEntry &&
                 gDecisionSupportCache[symIdx].barTime==barTime &&
                 gDecisionSupportCache[symIdx].regime==regime &&
                 gDecisionSupportCache[symIdx].positionsCount==positionsCount &&
                 gDecisionSupportCache[symIdx].basketDir==basketDir &&
                 gDecisionSupportCache[symIdx].brainMode==brainMode);

   if(cacheOk)
   {
      CopyDecisionSupportContext(gDecisionSupportCache[symIdx],ctx);
      return true;
   }

   ctx.valid=true;
   ctx.forcedEntry=forcedEntry;
   ctx.barTime=barTime;
   ctx.refreshTime=TimeCurrent();
   ctx.regime=regime;
   ctx.positionsCount=positionsCount;
   ctx.basketDir=basketDir;
   ctx.brainMode=brainMode;
   ctx.dangerProbability=Clamp(gPDanger[symIdx],0.0,1.0);

   double totalDelta[];
   InitActionBias(totalDelta);

   double supportAcc=0.0;
   double supportW=0.0;

   double stateKey[];
   bool haveStateKey=BuildQMemoryKey(symIdx,state,stateKey);

   if(UseQMemory && haveStateKey)
   {
      double qMem[];
      double memConf=0.0;
      if(GetQMemoryForDecision(symIdx,regime,symbol,stateKey,qMem,memConf))
      {
         double qBias[];
         BuildQMemoryVoteBias(symIdx,qBase,qMem,memConf,forcedEntry,qBias);
         AccumulateActionBias(totalDelta,qBias);

         ctx.qMemoryAgreement=ArchiveLiveDecisionConfidence(qMem);
         ctx.qMemoryConflict=1.0-ctx.qMemoryAgreement;
         supportAcc += Clamp(0.60*memConf + 0.40*ctx.qMemoryAgreement,0.0,1.0);
         supportW   += 1.0;
      }
   }

   if(UseDDEventBias && haveStateKey)
   {
      double ddBiasRaw[];
      if(QueryDDEventBiasWithKey(symIdx, regime, -1, basketDir, stateKey, ddBiasRaw))
      {
         double ddBias[];
         BuildDDEventCautionBias(symIdx,qBase,ddBiasRaw,forcedEntry,ddBias);
         AccumulateActionBias(totalDelta,ddBias);

         double holdCaution=(ArraySize(ddBiasRaw)>0 ? MathMax(0.0,ddBiasRaw[0]) : 0.0);
         double longPenalty=(ArraySize(ddBiasRaw)>1 ? MathMax(0.0,-ddBiasRaw[1]) : 0.0);
         double shortPenalty=(ArraySize(ddBiasRaw)>2 ? MathMax(0.0,-ddBiasRaw[2]) : 0.0);
         ctx.ddEventRisk=Clamp(0.50*holdCaution + 0.25*longPenalty + 0.25*shortPenalty,0.0,1.0);

         supportAcc += Clamp(MaxAbsArrayValue(ddBiasRaw),0.0,1.0);
         supportW   += 1.0;
      }
   }

   if(UseArchiveLiveSimilarity)
   {
      double archiveBias[];
      double archiveAlpha=0.0;
      if(BuildArchiveLiveActionBias(symIdx, regime, state, qBase, archiveBias, archiveAlpha))
      {
         double scaledArchive[];
         InitActionBias(scaledArchive);
         int nA=MathMin(ArraySize(scaledArchive),ArraySize(archiveBias));
         for(int a=0;a<nA;a++)
            scaledArchive[a]=archiveAlpha*archiveBias[a];
         AccumulateActionBias(totalDelta,scaledArchive);

         double holdBias=(ArraySize(archiveBias)>0 ? archiveBias[0] : 0.0);
         double dirBias=0.0;
         if(ArraySize(archiveBias)>2)
            dirBias=MathMax(archiveBias[1],archiveBias[2]);

         ctx.archiveCaution=Clamp(MathMax(0.0,archiveAlpha*holdBias),0.0,1.0);
         ctx.oneRoundPrior =Clamp(MathMax(0.0,archiveAlpha*dirBias),0.0,1.0);

         supportAcc += Clamp(archiveAlpha,0.0,1.0);
         supportW   += 1.0;
      }
   }

   if(UseDangerBrain && gMode[symIdx]!=MODE_NORMAL)
   {
      double f_now[];
      ArrayResize(f_now,6);
      for(int k=0;k<6;k++) f_now[k]=gFPCache[symIdx][k];

      double rawBias[];
      double simBest=-1e9;
      if(GetBestAdapterBiasSmart(symIdx,f_now,rawBias,simBest))
      {
         double dangerBias[];
         BuildDangerVetoBias(symIdx,qBase,rawBias,simBest,forcedEntry,dangerBias);
         AccumulateActionBias(totalDelta,dangerBias);

         double dangerConf=Clamp((simBest+1.0)*0.5,0.0,1.0);
         ctx.dangerProbability=Clamp(MathMax(ctx.dangerProbability,dangerConf),0.0,1.0);

         supportAcc += dangerConf;
         supportW   += 1.0;
      }
   }

   if(haveStateKey)
   {
      ComputePainMemorySupport(symIdx,
                               regime,
                               symbol,
                               stateKey,
                               ctx.painRecurrenceRisk,
                               ctx.macroReversalTrapPrior,
                               ctx.counterTrendFailurePrior,
                               ctx.regimeBreakPainPrior,
                               ctx.recoveryFalseStartRisk,
                               ctx.deepBasketPainPrior,
                               ctx.painMemoryAgreement,
                               ctx.macroMicroConflict,
                               ctx.lateTrendFadePenalty,
                               ctx.painConfidence);

      supportAcc += Clamp(0.45*ctx.painConfidence +
                          0.30*ctx.painRecurrenceRisk +
                          0.25*ctx.painMemoryAgreement,0.0,1.0);
      supportW   += 1.0;
   }

   ctx.addRiskPrior=Clamp(0.22*ctx.archiveCaution +
                          0.18*ctx.ddEventRisk +
                          0.18*ctx.dangerProbability +
                          0.14*CurrentReplayRiskBias(symIdx) +
                          0.10*ctx.painRecurrenceRisk +
                          0.08*ctx.deepBasketPainPrior +
                          0.05*ctx.macroReversalTrapPrior +
                          0.05*ctx.regimeBreakPainPrior,0.0,1.0);
   ctx.oneRoundPrior=Clamp(0.82*ctx.oneRoundPrior +
                           0.10*(1.0-ctx.deepBasketPainPrior) +
                           0.08*(1.0-ctx.macroReversalTrapPrior),0.0,1.0);
   ctx.supportConfidence=(supportW>0.0 ? Clamp(supportAcc/supportW,0.0,1.0) : 0.0);

   for(int a=0;a<3;a++)
      ctx.actionDelta[a]=(a<ArraySize(totalDelta) ? totalDelta[a] : 0.0);

   ctx.trendPersistenceProb = ComputeOptionBTrendPersistenceProb(symIdx,state,qBase,ctx);
   ctx.trendReversalProb    = ComputeOptionBTrendReversalProb(symIdx,state,qBase,ctx);
   ctx.spikeRiskProb        = ComputeOptionBSpikeRiskProb(symIdx,state,qBase,ctx);
   ctx.expectedBasketDepth  = ComputeOptionBExpectedBasketDepth(symIdx,state,qBase,ctx,
                                                                ctx.trendPersistenceProb,
                                                                ctx.trendReversalProb,
                                                                ctx.spikeRiskProb);
   ComputeStrategyModeScores(symIdx,state,ctx,
                             ctx.trendContinuationQuality,
                             ctx.breakoutReclaimQuality,
                             ctx.reversalTransitionQuality,
                             ctx.modeDominanceScore,
                             ctx.modeConflictScore);

   if(UseDecisionSupportCache)
      CopyDecisionSupportContext(ctx,gDecisionSupportCache[symIdx]);

   return true;
}


double ComputeRecentDeepBasketRate(const int symIdx)
{
   int n=ArraySize(gEpisodeMemory);
   if(n<=0) return 0.0;

   int used=0;
   double acc=0.0, wsum=0.0;
   for(int i=n-1; i>=0 && used<12; --i)
   {
      if(gEpisodeMemory[i].symIdx!=symIdx) continue;
      double w=1.0/(1.0 + 0.18*(double)used);
      double deep=(gEpisodeMemory[i].openPositionsMax >= DeepBasketAddThreshold ? 1.0 : 0.0);
      if(gEpisodeMemory[i].maxDrawdownPct >= DangerReplayDDThresholdPct) deep=MathMax(deep,0.80);
      if(gEpisodeMemory[i].inefficientRecovery>0) deep=MathMax(deep,0.65);
      acc += w * Clamp(deep,0.0,1.0);
      wsum += w;
      used++;
   }
   if(wsum<=0.0) return 0.0;
   return Clamp(acc/wsum,0.0,1.0);
}

double ComputeRecentDrawdownCalmBonus(const int symIdx)
{
   int n=ArraySize(gEpisodeMemory);
   if(n<=0) return 0.5;

   int used=0;
   double acc=0.0, wsum=0.0;
   for(int i=n-1; i>=0 && used<12; --i)
   {
      if(gEpisodeMemory[i].symIdx!=symIdx) continue;
      double w=1.0/(1.0 + 0.16*(double)used);
      double calm=1.0 - SafeDiv(gEpisodeMemory[i].maxDrawdownPct, MathMax(RewardV2CleanCycleMaxDD,1e-6), 1.0);
      if(gEpisodeMemory[i].openPositionsMax >= DeepBasketAddThreshold) calm -= 0.35;
      if(gEpisodeMemory[i].forcedStopLikeEvent>0) calm -= 0.25;
      acc += w * Clamp(calm,-1.0,1.0);
      wsum += w;
      used++;
   }
   if(wsum<=0.0) return 0.5;
   return Clamp(0.5 + 0.5*(acc/wsum),0.0,1.0);
}

double ComputeRecentTradingQualityScore(const int symIdx)
{
   int n=ArraySize(gEpisodeMemory);
   if(n<=0) return 0.5;

   int used=0;
   double acc=0.0, wsum=0.0;
   for(int i=n-1; i>=0 && used<12; --i)
   {
      if(gEpisodeMemory[i].symIdx!=symIdx) continue;
      double w=1.0/(1.0 + 0.15*(double)used);

      double score=0.0;
      if(gEpisodeMemory[i].pnlFinal > 0.0) score += 0.22;
      else if(gEpisodeMemory[i].pnlFinal < 0.0) score -= 0.18;

      score += 0.24 * Clamp(gEpisodeMemory[i].rewardEfficiency,-1.0,1.5);
      score += 0.18 * (gEpisodeMemory[i].oneRoundTrade>0 ? 1.0 : 0.0);
      score += 0.14 * Clamp(1.0 - SafeDiv(gEpisodeMemory[i].maxDrawdownPct, MathMax(RewardV2CleanCycleMaxDD,1e-6), 1.0), -1.0, 1.0);

      if(gEpisodeMemory[i].openPositionsMax >= DeepBasketAddThreshold) score -= 0.28;
      if(gEpisodeMemory[i].inefficientRecovery > 0) score -= 0.20;
      if(gEpisodeMemory[i].forcedStopLikeEvent > 0) score -= 0.25;
      if(gEpisodeMemory[i].maxDrawdownPct >= DangerReplayDDThresholdPct) score -= 0.22;

      acc += w * Clamp(score,-1.0,1.0);
      wsum += w;
      used++;
   }

   if(wsum<=0.0) return 0.5;
   return Clamp(0.5 + 0.5*(acc/wsum),0.0,1.0);
}


double ComputeCounterTrendTrapRiskScore(const string symbol,const int symIdx)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double dirE=0.0,strE=0.0,persE=0.0,accE=0.0,overE=0.0,mrE=0.0,contE=0.0,revE=0.0,volE=0.0;
   double dirM=0.0,strM=0.0,persM=0.0,accM=0.0,overM=0.0,mrM=0.0,contM=0.0,revM=0.0,volM=0.0;
   double dirL=0.0,strL=0.0,persL=0.0,accL=0.0,overL=0.0,mrL=0.0,contL=0.0,revL=0.0,volL=0.0;

   ComputeTrendFactorTF(symbol, TF_EXEC, point, dirE,strE,persE,accE,overE,mrE,contE,revE,volE);
   ComputeTrendFactorTF(symbol, TF_MID,  point, dirM,strM,persM,accM,overM,mrM,contM,revM,volM);
   ComputeTrendFactorTF(symbol, TF_LONG, point, dirL,strL,persL,accL,overL,mrL,contL,revL,volL);

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double setupConsensus = 1.0 - Clamp((MathAbs(dirE-dirM) + MathAbs(dirM-dirL) + MathAbs(dirE-dirL))/6.0, 0.0, 1.0);
   double bridgeAgreement = 1.0 - Clamp((MathAbs(dirM-dirL) + MathAbs(dirL-macroBias))/4.0, 0.0, 1.0);
   double continuation = Clamp(0.42*setupConsensus*(0.50*contE + 0.30*contM + 0.20*contL) + 0.38*macroContinuation + 0.20*bridgeAgreement, 0.0, 1.0);
   double reversal = Clamp(0.30*(0.45*revE + 0.35*revM + 0.20*revL) + 0.30*macroTransition + 0.20*MathAbs(macroReclaim) + 0.20*(1.0-macroContinuation), 0.0, 1.0);
   double maturity = Clamp(0.40*(0.55*overE + 0.30*overM + 0.15*overL) + 0.40*macroMaturity + 0.20*lateTrendTrap, 0.0, 1.0);

   return Clamp(0.34*continuation +
                0.16*(1.0-reversal) +
                0.12*maturity +
                0.12*(1.0-bridgeAgreement) +
                0.14*macroContinuation +
                0.07*MathAbs(macroReclaim) +
                0.05*ComputeRecentDeepBasketRate(symIdx), 0.0, 1.0);
}

double ComputeReversalConfirmationScore(const string symbol)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double dirE=0.0,strE=0.0,persE=0.0,accE=0.0,overE=0.0,mrE=0.0,contE=0.0,revE=0.0,volE=0.0;
   double dirM=0.0,strM=0.0,persM=0.0,accM=0.0,overM=0.0,mrM=0.0,contM=0.0,revM=0.0,volM=0.0;
   double dirL=0.0,strL=0.0,persL=0.0,accL=0.0,overL=0.0,mrL=0.0,contL=0.0,revL=0.0,volL=0.0;

   ComputeTrendFactorTF(symbol, TF_EXEC, point, dirE,strE,persE,accE,overE,mrE,contE,revE,volE);
   ComputeTrendFactorTF(symbol, TF_MID,  point, dirM,strM,persM,accM,overM,mrM,contM,revM,volM);
   ComputeTrendFactorTF(symbol, TF_LONG, point, dirL,strL,persL,accL,overL,mrL,contL,revL,volL);

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double setupConsensus = 1.0 - Clamp((MathAbs(dirE-dirM) + MathAbs(dirM-dirL) + MathAbs(dirE-dirL))/6.0, 0.0, 1.0);
   double bridgeAgreement = 1.0 - Clamp((MathAbs(dirM-dirL) + MathAbs(dirL-macroBias))/4.0, 0.0, 1.0);

   return Clamp(0.24*(0.45*revE + 0.35*revM + 0.20*revL) +
                0.24*macroTransition +
                0.18*MathAbs(macroReclaim) +
                0.12*bridgeAgreement +
                0.12*(1.0-macroContinuation) +
                0.10*(1.0-lateTrendTrap), 0.0, 1.0);
}

double ComputeRegimeBreakScore(const string symbol,const int symIdx)
{
   double volExec = Clamp(0.55*RealizedVolLevelTF(symbol, TF_EXEC, 8) + 0.45*RealizedVolLevelTF(symbol, TF_EXEC, 24), 0.0, 1.0);
   double expExec = Clamp(0.45*RealizedVolDeltaTF(symbol, TF_EXEC, 8) + 0.25*RealizedVolGammaTF(symbol, TF_EXEC, 8) + 0.30*RangeExpansionTF(symbol, TF_EXEC, 5, 20), 0.0, 1.0);
   double expMid  = Clamp(0.45*RealizedVolDeltaTF(symbol, TF_MID, 8)  + 0.25*RealizedVolGammaTF(symbol, TF_MID, 8)  + 0.30*RangeExpansionTF(symbol, TF_MID, 5, 20), 0.0, 1.0);
   double wickInst = Clamp(0.50*WickInstabilityTF(symbol,TF_EXEC,1) + 0.30*WickInstabilityTF(symbol,TF_MID,1) + 0.20*WickInstabilityTF(symbol,TF_LONG,1), 0.0, 1.0);
   double spread = ComputeSpreadPressureScore(symbol,symIdx);
   return Clamp(0.40*expExec + 0.20*MathMax(0.0, expExec-volExec) + 0.15*MathMax(0.0, expMid-volExec) + 0.15*wickInst + 0.10*spread, 0.0, 1.0);
}

enum PainEventType
{
   PAIN_EVENT_GENERIC = 0,
   PAIN_EVENT_COUNTERTREND_TRAP = 1,
   PAIN_EVENT_LATE_TREND_FADE = 2,
   PAIN_EVENT_REGIME_BREAK = 3,
   PAIN_EVENT_FALSE_REVERSAL = 4,
   PAIN_EVENT_RECOVERY_FALSE_START = 5
};

void BuildMacroPainSignature(const string symbol,const int symIdx,double &sig[])
{
   ArrayResize(sig,6);
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);
   double h4Vol=Clamp(VolumeRatioTF(symbol, PERIOD_H4, 10),0.0,2.0)/2.0;

   sig[0]=Clamp(0.5 + 0.5*macroBias,0.0,1.0);
   sig[1]=Clamp(macroContinuation,0.0,1.0);
   sig[2]=Clamp(macroMaturity,0.0,1.0);
   sig[3]=Clamp(MathAbs(macroReclaim),0.0,1.0);
   sig[4]=Clamp(macroTransition,0.0,1.0);
   sig[5]=Clamp(0.70*regimeBreak + 0.30*h4Vol,0.0,1.0);
   NormalizeVecN(sig);
}

void BuildMicroPainSignature(const string symbol,const int symIdx,double &sig[])
{
   ArrayResize(sig,6);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double reversal=ComputeReversalConfirmationScore(symbol);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);
   double spread=ComputeSpreadPressureScore(symbol,symIdx);
   double wick=Clamp(0.55*WickInstabilityTF(symbol,TF_EXEC,1)+0.45*WickInstabilityTF(symbol,TF_MID,1),0.0,1.0);
   double volBurst=Clamp(0.65*VolumeRatioTF(symbol,TF_EXEC,10)+0.35*VolumeRatioTF(symbol,TF_MID,10),0.0,2.0)/2.0;

   sig[0]=Clamp(trapRisk,0.0,1.0);
   sig[1]=Clamp(reversal,0.0,1.0);
   sig[2]=Clamp(regimeBreak,0.0,1.0);
   sig[3]=Clamp(spread,0.0,1.0);
   sig[4]=Clamp(wick,0.0,1.0);
   sig[5]=Clamp(volBurst,0.0,1.0);
   NormalizeVecN(sig);
}
void ComputePainMemorySupport(const int symIdx,
                              const int regime,
                              const string symbol,
                              const double &stateKey[],
                              double &painRecurrence,
                              double &macroTrapPrior,
                              double &counterTrendPrior,
                              double &regimeBreakPainPrior,
                              double &recoveryFalseStartRisk,
                              double &deepBasketPainPrior,
                              double &painAgreement,
                              double &macroMicroConflict,
                              double &lateTrendFadePenalty,
                              double &painConfidence)
{
   painRecurrence=0.0;
   macroTrapPrior=0.0;
   counterTrendPrior=0.0;
   regimeBreakPainPrior=0.0;
   recoveryFalseStartRisk=0.0;
   deepBasketPainPrior=0.0;
   painAgreement=0.0;
   macroMicroConflict=0.0;
   lateTrendFadePenalty=0.0;
   painConfidence=0.0;

   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;

   double curMacro[]; BuildMacroPainSignature(symbol,symIdx,curMacro);
   double curMicro[]; BuildMicroPainSignature(symbol,symIdx,curMicro);

   double macroSum=0.0, microSum=0.0, wsum=0.0;
   int start=MathMax(0,ArraySize(gDDEvents)-40);
   for(int i=start;i<ArraySize(gDDEvents);i++)
   {
      if(gDDEvents[i].symbol!=symbol) continue;
      if(UseRegimeBank && gDDEvents[i].regimeAtTrigger!=regime) continue;

      double simKey=0.0;
      if(ArraySize(stateKey)>0 && ArraySize(gDDEvents[i].triggerStateKey)==ArraySize(stateKey))
         simKey=Clamp(CosSim(stateKey,gDDEvents[i].triggerStateKey),0.0,1.0);
      double simMacro=(ArraySize(gDDEvents[i].macroSig)==ArraySize(curMacro) ? Clamp(CosSim(curMacro,gDDEvents[i].macroSig),0.0,1.0) : 0.0);
      double simMicro=(ArraySize(gDDEvents[i].microSig)==ArraySize(curMicro) ? Clamp(CosSim(curMicro,gDDEvents[i].microSig),0.0,1.0) : 0.0);
      double pain=Clamp(gDDEvents[i].painSeverity,0.0,1.0);
      double w=(0.30*simKey + 0.35*simMacro + 0.35*simMicro) * (1.0 + 0.90*pain);
      if(w<=1e-8) continue;

      painRecurrence += w * pain;
      deepBasketPainPrior += w * Clamp((double)gDDEvents[i].basketDepthMax / MathMax(3.0,(double)DeepBasketAddThreshold),0.0,1.0);
      double evMacroRegime=(ArraySize(gDDEvents[i].macroSig)>5 ? Clamp(gDDEvents[i].macroSig[5],0.0,1.0) : 0.0);
      regimeBreakPainPrior += w * (gDDEvents[i].eventType==PAIN_EVENT_REGIME_BREAK ? 1.0 : evMacroRegime);
      double evMicroTrap=(ArraySize(gDDEvents[i].microSig)>0 ? Clamp(gDDEvents[i].microSig[0],0.0,1.0) : 0.0);
      counterTrendPrior += w * (gDDEvents[i].eventType==PAIN_EVENT_COUNTERTREND_TRAP ? 1.0 : evMicroTrap);
      recoveryFalseStartRisk += w * (gDDEvents[i].eventType==PAIN_EVENT_RECOVERY_FALSE_START ? 1.0 : Clamp(gDDEvents[i].recoveryFailureScore,0.0,1.0));
      double evMacroMaturity=(ArraySize(gDDEvents[i].macroSig)>2 ? Clamp(gDDEvents[i].macroSig[2],0.0,1.0) : 0.0);
      lateTrendFadePenalty += w * (gDDEvents[i].eventType==PAIN_EVENT_LATE_TREND_FADE ? 1.0 : evMacroMaturity);
      double evMacroTransition=(ArraySize(gDDEvents[i].macroSig)>4 ? Clamp(gDDEvents[i].macroSig[4],0.0,1.0) : 0.0);
      double evMacroReclaim=(ArraySize(gDDEvents[i].macroSig)>3 ? Clamp(gDDEvents[i].macroSig[3],0.0,1.0) : 0.0);
      macroTrapPrior += w * Clamp(0.50*simMacro + 0.25*evMacroTransition + 0.25*evMacroReclaim,0.0,1.0);
      macroSum += w*simMacro;
      microSum += w*simMicro;
      wsum += w;
   }

   for(int i=0;i<ArraySize(gProtos);i++)
   {
      if(!gProtos[i].isDanger) continue;
      double simMacro=(ArraySize(gProtos[i].macroSig)==ArraySize(curMacro) ? Clamp(CosSim(curMacro,gProtos[i].macroSig),0.0,1.0) : 0.0);
      double simMicro=(ArraySize(gProtos[i].microSig)==ArraySize(curMicro) ? Clamp(CosSim(curMicro,gProtos[i].microSig),0.0,1.0) : 0.0);
      double pain=Clamp(gProtos[i].painMean,0.0,1.0);
      double w=(0.55*simMacro + 0.45*simMicro) * (1.0 + 0.75*pain);
      if(w<=1e-8) continue;

      painRecurrence += w * pain;
      deepBasketPainPrior += w * Clamp(gProtos[i].deepBasketRate,0.0,1.0);
      regimeBreakPainPrior += w * Clamp(gProtos[i].regimeBreakRate,0.0,1.0);
      counterTrendPrior += w * Clamp(gProtos[i].counterTrendFailureRate,0.0,1.0);
      recoveryFalseStartRisk += w * Clamp(gProtos[i].recoveryFailureRate,0.0,1.0);
      lateTrendFadePenalty += w * Clamp(gProtos[i].reversalTrapRate,0.0,1.0);
      macroTrapPrior += w * Clamp(0.55*simMacro + 0.45*gProtos[i].counterTrendFailureRate,0.0,1.0);
      macroSum += w*simMacro;
      microSum += w*simMicro;
      wsum += w;
   }

   if(wsum>1e-8)
   {
      painRecurrence=Clamp(painRecurrence/wsum,0.0,1.0);
      macroTrapPrior=Clamp(macroTrapPrior/wsum,0.0,1.0);
      counterTrendPrior=Clamp(counterTrendPrior/wsum,0.0,1.0);
      regimeBreakPainPrior=Clamp(regimeBreakPainPrior/wsum,0.0,1.0);
      recoveryFalseStartRisk=Clamp(recoveryFalseStartRisk/wsum,0.0,1.0);
      deepBasketPainPrior=Clamp(deepBasketPainPrior/wsum,0.0,1.0);
      lateTrendFadePenalty=Clamp(lateTrendFadePenalty/wsum,0.0,1.0);
      painAgreement=Clamp(0.5*(macroSum/wsum) + 0.5*(microSum/wsum),0.0,1.0);
      macroMicroConflict=Clamp(MathAbs((macroSum/wsum) - (microSum/wsum)),0.0,1.0);
      painConfidence=Clamp(0.55*painAgreement + 0.45*Clamp(MathMin(wsum,3.0)/3.0,0.0,1.0),0.0,1.0);
   }
}

double ComputeOneRoundOpportunityScore(const string symbol,
                                       const int symIdx,
                                       const double &qBase[],
                                       const DecisionSupportContext &support)
{
   double qGap=Clamp(SafeDiv(ComputeDDQNActionGap(qBase,false),0.25,0.0),0.0,1.0);
   double spreadInv=1.0 - ComputeSpreadPressureScore(symbol,symIdx);
   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double ddCalm=ComputeRecentDrawdownCalmBonus(symIdx);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double reversalConfirm=ComputeReversalConfirmationScore(symbol);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   int dirBest=ArgMaxDirectionalActionFromQ(qBase);
   double macroFit=0.5;
   if(dirBest==1)      macroFit=Clamp(0.5 + 0.5*macroBias, 0.0, 1.0);
   else if(dirBest==2) macroFit=Clamp(0.5 - 0.5*macroBias, 0.0, 1.0);

   double persistenceProb=Clamp(support.trendPersistenceProb,0.0,1.0);
   double reversalProb=Clamp(support.trendReversalProb,0.0,1.0);
   double spikeRiskProb=Clamp(support.spikeRiskProb,0.0,1.0);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);
   double trendMode=Clamp(support.trendContinuationQuality,0.0,1.0);
   double reclaimMode=Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double transitionMode=Clamp(support.reversalTransitionQuality,0.0,1.0);
   double modeDominance=Clamp(support.modeDominanceScore,0.0,1.0);
   double modeConflict=Clamp(support.modeConflictScore,0.0,1.0);
   double directionalBehaviorFit=Clamp(macroFit*persistenceProb + (1.0-macroFit)*reversalProb,0.0,1.0);

   double score=
      0.15*support.oneRoundPrior +
      0.10*support.supportConfidence +
      0.07*support.qMemoryAgreement +
      0.10*qGap +
      0.07*spreadInv +
      0.08*recentQuality +
      0.07*ddCalm +
      0.06*(1.0 - support.dangerProbability) +
      0.06*reversalConfirm +
      0.04*(1.0-trapRisk) +
      0.04*(1.0-regimeBreak) +
      0.04*macroFit +
      0.03*(1.0-macroMaturity) +
      0.02*(1.0-lateTrendTrap) +
      0.03*MathAbs(macroReclaim) +
      0.09*directionalBehaviorFit +
      0.05*(1.0-spikeRiskProb) +
      0.05*(1.0-expectedDepth) +
      0.06*trendMode +
      0.05*reclaimMode +
      0.03*modeDominance -
      0.06*transitionMode -
      0.03*modeConflict;

   return Clamp(score,0.0,1.0);
}

double ComputeDeepBasketRiskScore(const string symbol,
                                  const int symIdx,
                                  const DecisionSupportContext &support)
{
   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate=ComputeRecentDeepBasketRate(symIdx);
   double ddWorsen=Clamp(ComputeDrawdownWorseningScore(symIdx),0.0,1.0);
   double spread=ComputeSpreadPressureScore(symbol,symIdx);
   double replayRisk=Clamp(CurrentReplayRiskBias(symIdx),0.0,1.0);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double persistenceProb=Clamp(support.trendPersistenceProb,0.0,1.0);
   double reversalProb=Clamp(support.trendReversalProb,0.0,1.0);
   double spikeRiskProb=Clamp(support.spikeRiskProb,0.0,1.0);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);
   double trendMode=Clamp(support.trendContinuationQuality,0.0,1.0);
   double reclaimMode=Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double transitionMode=Clamp(support.reversalTransitionQuality,0.0,1.0);
   double modeConflict=Clamp(support.modeConflictScore,0.0,1.0);

   double score=
      0.16*support.addRiskPrior +
      0.14*support.dangerProbability +
      0.10*support.archiveCaution +
      0.09*replayRisk +
      0.08*recentDeepRate +
      0.06*ddWorsen +
      0.04*spread +
      0.03*(1.0 - recentQuality) +
      0.05*trapRisk +
      0.04*regimeBreak +
      0.03*macroContinuation +
      0.02*macroTransition +
      0.01*macroMaturity +
      0.01*MathAbs(macroReclaim) +
      0.07*spikeRiskProb +
      0.10*expectedDepth +
      0.04*MathMax(0.0,persistenceProb-reversalProb) +
      0.06*transitionMode +
      0.04*modeConflict -
      0.04*trendMode -
      0.03*reclaimMode;

   return Clamp(score,0.0,1.0);
}


void BuildAdaptiveEntryQualityDelta(const int symIdx,
                                    const int regime,
                                    const double &qBase[],
                                    const DecisionSupportContext &support,
                                    double &biasOut[])
{
   InitActionBias(biasOut);
   if(!UseQualityAdaptiveEntry) return;
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(gPositionsCount[symIdx] > 0) return;

   string symbol=gSymbols[symIdx];
   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double oneRoundScore=ComputeOneRoundOpportunityScore(symbol,symIdx,qBase,support);
   double deepRiskScore=ComputeDeepBasketRiskScore(symbol,symIdx,support);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double reversalConfirm=ComputeReversalConfirmationScore(symbol);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;
   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double qualityEdge=Clamp(oneRoundScore - deepRiskScore,-1.0,1.0);
   double periodicTilt=Clamp((recentQuality - 0.5)*2.0,-1.0,1.0);
   double persistenceProb=Clamp(support.trendPersistenceProb,0.0,1.0);
   double reversalProb=Clamp(support.trendReversalProb,0.0,1.0);
   double spikeRiskProb=Clamp(support.spikeRiskProb,0.0,1.0);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);
   double trendMode=Clamp(support.trendContinuationQuality,0.0,1.0);
   double reclaimMode=Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double transitionMode=Clamp(support.reversalTransitionQuality,0.0,1.0);
   double modeDominance=Clamp(support.modeDominanceScore,0.0,1.0);
   double modeConflict=Clamp(support.modeConflictScore,0.0,1.0);

   int dirBest=ArgMaxDirectionalActionFromQ(qBase);
   double dirMacroTrap=ComputeDirectionalMacroTrapRisk(dirBest, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);
   double trapPenalty=Clamp(0.28*trapRisk + 0.12*regimeBreak + 0.16*dirMacroTrap + 0.08*lateTrendTrap + 0.08*macroContinuation - 0.18*reversalConfirm + 0.08*spikeRiskProb + 0.12*expectedDepth, 0.0, 1.0);

   double macroAlignBonus=0.0;
   if(dirBest==1)      macroAlignBonus = MathMax(0.0, 0.65*macroBias + 0.18*macroReclaim);
   else if(dirBest==2) macroAlignBonus = MathMax(0.0, -0.65*macroBias - 0.18*macroReclaim);
   macroAlignBonus = Clamp(macroAlignBonus, 0.0, 1.0);

   double directionalBehaviorFit=Clamp(macroAlignBonus*persistenceProb + (1.0-macroAlignBonus)*reversalProb,0.0,1.0);

   double continuationTilt = Clamp(0.60*trendMode + 0.25*modeDominance + 0.15*macroAlignBonus, 0.0, 1.0);
   double reclaimTilt      = Clamp(0.65*reclaimMode + 0.20*MathAbs(macroReclaim) + 0.15*modeDominance, 0.0, 1.0);
   double transitionOffense= Clamp(0.45*reversalConfirm + 0.20*reversalProb + 0.15*MathAbs(macroReclaim) + 0.10*(1.0-trapRisk) + 0.10*(1.0-expectedDepth), 0.0, 1.0);
   double transitionCaution= Clamp(0.60*transitionMode + 0.20*modeConflict + 0.10*regimeBreak + 0.10*spikeRiskProb, 0.0, 1.0);

   double dirBoost=MathMax(0.0, 0.16*qualityEdge + 0.06*periodicTilt + 0.06*reversalConfirm + 0.06*macroAlignBonus + 0.07*directionalBehaviorFit - 0.10*trapPenalty);
   double holdBoost=MathMax(0.0, 0.18*(-qualityEdge) + 0.08*MathMax(0.0,-periodicTilt) + 0.08*deepRiskScore + 0.09*trapPenalty + 0.05*dirMacroTrap + 0.05*spikeRiskProb + 0.07*expectedDepth);
   double holdPenalty=MathMax(0.0, 0.10*qualityEdge + 0.04*MathMax(0.0,periodicTilt) + 0.04*macroAlignBonus + 0.04*directionalBehaviorFit);

   if(dirBest==1 || dirBest==2)
   {
      double sameDirBoost = dirBoost + 0.10*continuationTilt + 0.08*reclaimTilt + 0.04*modeDominance;
      double transitionDirBoost = dirBoost + 0.07*transitionMode*transitionOffense;
      double transitionHoldBoost = holdBoost + 0.14*transitionCaution + 0.10*transitionMode*(1.0-transitionOffense);

      if(dirBest==1)
      {
         if(macroBias >= 0.0)
            biasOut[1] += sameDirBoost;
         else
            biasOut[1] += transitionDirBoost;

         biasOut[2] -= 0.30*dirBoost + 0.08*continuationTilt;
         biasOut[0] += (macroBias >= 0.0 ? holdBoost : transitionHoldBoost) - holdPenalty;
      }
      else
      {
         if(macroBias <= 0.0)
            biasOut[2] += sameDirBoost;
         else
            biasOut[2] += transitionDirBoost;

         biasOut[1] -= 0.30*dirBoost + 0.08*continuationTilt;
         biasOut[0] += (macroBias <= 0.0 ? holdBoost : transitionHoldBoost) - holdPenalty;
      }
   }
   else
   {
      biasOut[0] += holdBoost + 0.08*transitionCaution - 0.03*modeDominance;
   }
}



double ComputeSpreadPressureScore(const string symbol,const int symIdx)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);
   double spreadPts=MathMax(0.0,(ask-bid)/point);

   double atr=gATRslow_BaseVal[symIdx];
   if(atr<=1e-12) atr=gATRfast_BaseVal[symIdx];
   double atrPts=(atr>0.0 ? atr/point : 0.0);
   if(atrPts<=1e-6) return Clamp(spreadPts/10.0,0.0,1.0);

   return Clamp(12.0*SafeDiv(spreadPts,atrPts,0.0),0.0,1.0);
}

double ComputeSmartAddRiskScore(const string symbol,
                                const int symIdx,
                                const int regime,
                                const int basketDir,
                                const int positionsCount,
                                const double combinedTrend,
                                const double reversalRisk,
                                const bool extreme,
                                const DecisionSupportContext &support)
{
   double ddPct=GetSymbolFloatingDDPct(symIdx);
   double depth=Clamp(SafeDiv((double)MathMax(0,positionsCount-1),(double)MathMax(1,MaxTrades-1),0.0),0.0,1.0);
   double age=Clamp(SafeDiv(BasketAgeDays(symIdx),MathMax(AddGateAgeSoftDays,0.25),0.0),0.0,1.0);
   double danger=MathMax(Clamp(gPDanger[symIdx],0.0,1.0),support.dangerProbability);
   double addRisk=MathMax(support.addRiskPrior,CurrentReplayRiskBias(symIdx));
   double spread=ComputeSpreadPressureScore(symbol,symIdx);
   double recentQuality=ComputeRecentTradingQualityScore(symIdx);
   double recentDeepRate=ComputeRecentDeepBasketRate(symIdx);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);
   double spikeRiskProb=Clamp(support.spikeRiskProb,0.0,1.0);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);

   double antiTrend=0.0;
   if(basketDir>0)
   {
      if(combinedTrend<0.0) antiTrend=1.0;
      else if(combinedTrend==0.0) antiTrend=0.25;
   }
   else if(basketDir<0)
   {
      if(combinedTrend>0.0) antiTrend=1.0;
      else if(combinedTrend==0.0) antiTrend=0.25;
   }

   double risk=0.20*ddPct +
               0.17*depth +
               0.11*age +
               0.18*danger +
               0.13*addRisk +
               0.10*antiTrend +
               0.04*Clamp(reversalRisk,0.0,1.0) +
               0.02*spread +
               0.03*(1.0-recentQuality) +
               0.02*recentDeepRate +
               0.04*trapRisk +
               0.02*regimeBreak +
               0.03*spikeRiskProb +
               0.04*expectedDepth;

   if(extreme) risk += 0.10;

   double sameBias=0.0;
   double oppBias=0.0;
   if(basketDir>0)
   {
      sameBias=support.actionDelta[1];
      oppBias =support.actionDelta[2];
   }
   else if(basketDir<0)
   {
      sameBias=support.actionDelta[2];
      oppBias =support.actionDelta[1];
   }

   if(support.actionDelta[0] > sameBias + 0.18) risk += 0.08;
   if(oppBias > sameBias + 0.20)                risk += 0.12;

   return Clamp(risk,0.0,1.5);
}

bool EvaluateSmartAddGate(const string symbol,
                          const int symIdx,
                          const int regime,
                          const double &state[],
                          const int basketDir,
                          const int positionsCount,
                          const double combinedTrend,
                          const double reversalRisk,
                          const bool extreme,
                          double &spacingMultOut,
                          double &lotScaleOut,
                          double &riskScoreOut)
{
   spacingMultOut=1.0;
   lotScaleOut=1.0;
   riskScoreOut=0.0;

   if(!UseSmartAddGate) return true;
   if(positionsCount<=0 || basketDir==0) return true;

   double qBase[];
   DQNForwardInference(symIdx,regime,state,qBase);

   DecisionSupportContext support;
   BuildDecisionSupportContext(symIdx,regime,state,qBase,false,basketDir,support);

   riskScoreOut=ComputeSmartAddRiskScore(symbol,
                                         symIdx,
                                         regime,
                                         basketDir,
                                         positionsCount,
                                         combinedTrend,
                                         reversalRisk,
                                         extreme,
                                         support);

   if(riskScoreOut >= AddGateRiskBlockThreshold)
      return false;

   if(riskScoreOut > AddGateRiskWidenThreshold)
   {
      double frac=Clamp(SafeDiv(riskScoreOut-AddGateRiskWidenThreshold,
                                MathMax(AddGateRiskBlockThreshold-AddGateRiskWidenThreshold,1e-6),0.0),0.0,1.0);
      spacingMultOut=1.0 + frac*(MathMax(AddGateSpacingMaxMult,1.0)-1.0);
      lotScaleOut   =1.0 - frac*(1.0-Clamp(AddGateLotMinScale,0.05,1.0));
   }

   return true;
}


bool GetLatestEpisodeForSymbol(const int symIdx, EpisodeMemory &epOut)
{
   int n=ArraySize(gEpisodeMemory);
   for(int i=n-1;i>=0;--i)
   {
      if(gEpisodeMemory[i].symIdx!=symIdx) continue;
      epOut=gEpisodeMemory[i];
      return true;
   }
   return false;
}

int DominantStrategyMode(const DecisionSupportContext &support)
{
   double tc=Clamp(support.trendContinuationQuality,0.0,1.0);
   double br=Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double rt=Clamp(support.reversalTransitionQuality,0.0,1.0);
   if(tc<=1e-9 && br<=1e-9 && rt<=1e-9) return 0;
   if(tc>=br && tc>=rt) return 1;
   if(br>=tc && br>=rt) return 2;
   return 3;
}

double ComputeRegimeShiftAlert(const int symIdx,
                               const int regime,
                               const DecisionSupportContext &support,
                               const double macroBias)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return 0.0;
   if(gSignalTrackBarTime[symIdx]<=0) return 0.0;

   double regimeShift=0.0;
   int prevReg=gSignalTrackRegime[symIdx];
   if(prevReg>=0)
   {
      int d=MathAbs(prevReg-regime);
      if(d>=2) regimeShift=1.0;
      else if(d==1) regimeShift=0.60;
   }

   int curMode=DominantStrategyMode(support);
   double modeShift=(gSignalTrackMode[symIdx]>0 && curMode>0 && gSignalTrackMode[symIdx]!=curMode ? 1.0 : 0.0);
   double transitionRise=MathMax(0.0, Clamp(support.reversalTransitionQuality,0.0,1.0) - Clamp(gSignalTrackTransition[symIdx],0.0,1.0));

   double prevMacro=gSignalTrackMacroBias[symIdx];
   double macroFlip=0.0;
   if(MathAbs(prevMacro)>=0.12 && MathAbs(macroBias)>=0.12 && ((prevMacro>0.0 && macroBias<0.0) || (prevMacro<0.0 && macroBias>0.0)))
      macroFlip=1.0;

   return Clamp(0.34*regimeShift +
                0.24*modeShift +
                0.20*transitionRise +
                0.12*macroFlip +
                0.10*Clamp(support.modeConflictScore,0.0,1.0),0.0,1.0);
}

int ExtractDirectionalCandidate(const double &qVals[])
{
   if(ArraySize(qVals)<3) return 0;
   int dir=(qVals[1] >= qVals[2] ? 1 : 2);
   double dirQ=qVals[dir];
   if(dirQ < qVals[0] - 0.16) return 0;
   return dir;
}

void UpdateDirectionalSignalTracking(const int symIdx,
                                     const int regime,
                                     const DecisionSupportContext &support,
                                     const double &qAdj[],
                                     const double macroBias)
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;

   datetime barTime=support.barTime;
   int curMode=DominantStrategyMode(support);
   int candidate=ExtractDirectionalCandidate(qAdj);

   if(gSignalTrackBarTime[symIdx]==barTime)
   {
      if(candidate!=0)
         gSignalTrackDirCandidate[symIdx]=candidate;
      gSignalTrackRegime[symIdx]=regime;
      gSignalTrackMode[symIdx]=curMode;
      gSignalTrackMacroBias[symIdx]=macroBias;
      gSignalTrackTransition[symIdx]=Clamp(support.reversalTransitionQuality,0.0,1.0);
      return;
   }

   int persist=0;
   bool sameContext=(candidate!=0 &&
                     gSignalTrackDirCandidate[symIdx]==candidate &&
                     gSignalTrackMode[symIdx]==curMode &&
                     MathAbs(gSignalTrackRegime[symIdx]-regime)==0 &&
                     !((gSignalTrackMacroBias[symIdx]>0.12 && macroBias<-0.12) || (gSignalTrackMacroBias[symIdx]<-0.12 && macroBias>0.12)) &&
                     MathAbs(Clamp(gSignalTrackTransition[symIdx],0.0,1.0)-Clamp(support.reversalTransitionQuality,0.0,1.0))<0.24);

   if(candidate==0)
      persist=0;
   else if(sameContext)
      persist=MathMin(gSignalTrackPersistCount[symIdx]+1,8);
   else
      persist=1;

   gSignalTrackBarTime[symIdx]=barTime;
   gSignalTrackRegime[symIdx]=regime;
   gSignalTrackDirCandidate[symIdx]=candidate;
   gSignalTrackPersistCount[symIdx]=persist;
   gSignalTrackMode[symIdx]=curMode;
   gSignalTrackMacroBias[symIdx]=macroBias;
   gSignalTrackTransition[symIdx]=Clamp(support.reversalTransitionQuality,0.0,1.0);
}

double ComputeRecentWrongThesisPenalty(const string symbol,
                                      const int symIdx,
                                      const int action,
                                      const DecisionSupportContext &support,
                                      const double macroBias,
                                      const double macroReclaim,
                                      const double macroTransition)
{
   if(action<1 || action>2) return 0.0;

   EpisodeMemory ep;
   if(!GetLatestEpisodeForSymbol(symIdx,ep)) return 0.0;
   if(ep.endTime<=0) return 0.0;

   int recentSec=MathMax(PeriodSeconds(TF_EXEC)*18, 1800);
   if(recentSec<=0) recentSec=3600;
   if((TimeCurrent()-ep.endTime) > recentSec) return 0.0;

   double ugly=0.0;
   if(ep.openPositionsMax>=2) ugly += 0.30;
   ugly += 0.30*Clamp(SafeDiv(ep.maxDrawdownPct,MathMax(RewardV2CleanCycleMaxDD,1e-6),0.0),0.0,1.0);
   if(ep.inefficientRecovery>0) ugly += 0.18;
   if(ep.forcedStopLikeEvent>0) ugly += 0.12;
   if(ep.pnlFinal<=0.0) ugly += 0.10;
   ugly += 0.12*Clamp(1.0-ep.rewardEfficiency,0.0,1.0);
   ugly=Clamp(ugly,0.0,1.0);
   if(ugly<0.20) return 0.0;

   int dir=(action==1 ? 1 : -1);
   double sameRecent = (ep.basketDir!=0 && dir==ep.basketDir ? 1.0 : 0.0);
   double oppositeRecent = (ep.basketDir!=0 && dir!=ep.basketDir ? 1.0 : 0.0);
   double macroAgainst = MathMax(0.0,-dir*macroBias);
   double reclaimAgainst = MathMax(0.0,-dir*macroReclaim);
   double transitionAmb = Clamp(0.60*macroTransition + 0.40*support.modeConflictScore,0.0,1.0);

   return Clamp(ugly*(0.35*sameRecent +
                      0.20*oppositeRecent +
                      0.20*macroAgainst +
                      0.10*reclaimAgainst +
                      0.10*transitionAmb +
                      0.05*Clamp(support.expectedBasketDepth,0.0,1.0)),0.0,1.0);
}

int ComputeDirectionalRegimeState(const string symbol,
                                  const int symIdx,
                                  const int regime,
                                  const DecisionSupportContext &support,
                                  double &macroBiasOut,
                                  double &macroReclaimOut,
                                  double &regimeShiftOut,
                                  double &clarityOut,
                                  bool &neutralSafeOut)
{
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);
   macroBiasOut=macroBias;
   macroReclaimOut=macroReclaim;

   regimeShiftOut=ComputeRegimeShiftAlert(symIdx,regime,support,macroBias);

   double breakoutSigned = Clamp((macroBias + 0.65*macroReclaim),-1.0,1.0) * Clamp(support.breakoutReclaimQuality,0.0,1.0);
   double directionEvidence = Clamp(0.34*macroBias +
                                    0.16*macroReclaim +
                                    0.16*(Clamp(support.trendPersistenceProb,0.0,1.0)-Clamp(support.trendReversalProb,0.0,1.0)) +
                                    0.14*(Clamp(support.trendContinuationQuality,0.0,1.0)-Clamp(support.reversalTransitionQuality,0.0,1.0)) +
                                    0.12*breakoutSigned +
                                    0.08*Clamp(support.modeDominanceScore*(Clamp(support.trendContinuationQuality,0.0,1.0)-Clamp(support.reversalTransitionQuality,0.0,1.0)),-1.0,1.0), -1.0, 1.0);

   clarityOut = Clamp(MathAbs(directionEvidence) *
                      (1.0-0.55*Clamp(support.modeConflictScore,0.0,1.0)) *
                      (1.0-0.45*Clamp(regimeShiftOut,0.0,1.0)) *
                      (1.0-0.35*Clamp(support.reversalTransitionQuality,0.0,1.0)), 0.0, 1.0);

   double neutralStableScore = Clamp(0.24*(1.0-Clamp(support.spikeRiskProb,0.0,1.0)) +
                                     0.20*(1.0-Clamp(support.expectedBasketDepth,0.0,1.0)) +
                                     0.16*Clamp(support.oneRoundPrior,0.0,1.0) +
                                     0.12*Clamp(support.supportConfidence,0.0,1.0) +
                                     0.12*(1.0-Clamp(support.modeConflictScore,0.0,1.0)) +
                                     0.08*(1.0-Clamp(support.painRecurrenceRisk,0.0,1.0)) +
                                     0.08*(1.0-Clamp(regimeShiftOut,0.0,1.0)), 0.0, 1.0);
   neutralSafeOut = (neutralStableScore >= 0.52 &&
                     support.reversalTransitionQuality < 0.70 &&
                     support.spikeRiskProb < 0.72 &&
                     support.expectedBasketDepth < 0.72);

   if(clarityOut >= 0.22 && directionEvidence > 0.14)
      return DIR_STATE_UP;
   if(clarityOut >= 0.22 && directionEvidence < -0.14)
      return DIR_STATE_DOWN;
   return DIR_STATE_NEUTRAL;
}

double ComputeFlexibleDirectionalDisciplinePenalty(const string symbol,
                                                   const int symIdx,
                                                   const int regime,
                                                   const int action,
                                                   const DecisionSupportContext &support,
                                                   const bool forcedEntry)
{
   if(action<1 || action>2) return 0.0;

   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   double macroBias=0.0,macroContinuation=0.0,macroMaturity=0.0,macroReclaim=0.0,macroTransition=0.0,lateTrendTrap=0.0;
   ComputeMacroBiasContext(symbol, point, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);

   double reversalConfirm=ComputeReversalConfirmationScore(symbol);
   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double regimeBreak=ComputeRegimeBreakScore(symbol,symIdx);
   double macroTrap=ComputeDirectionalMacroTrapRisk(action, macroBias, macroContinuation, macroMaturity, macroReclaim, macroTransition, lateTrendTrap);
   double wrongThesis=ComputeRecentWrongThesisPenalty(symbol,symIdx,action,support,macroBias,macroReclaim,macroTransition);

   double macroBiasDir=0.0,macroReclaimDir=0.0,regimeShiftAlert=0.0,clarity=0.0;
   bool neutralSafe=false;
   int dirState=ComputeDirectionalRegimeState(symbol,symIdx,regime,support,macroBiasDir,macroReclaimDir,regimeShiftAlert,clarity,neutralSafe);

   int dir=(action==1 ? 1 : -1);
   double alignBias=dir*macroBias;
   double alignReclaim=dir*macroReclaim;
   double continuationDominant=(support.trendContinuationQuality > support.breakoutReclaimQuality + 0.04 &&
                                support.trendContinuationQuality > support.reversalTransitionQuality + 0.04 ? 1.0 : 0.0);
   double reclaimDominant=(support.breakoutReclaimQuality > support.trendContinuationQuality + 0.04 &&
                           support.breakoutReclaimQuality > support.reversalTransitionQuality + 0.04 ? 1.0 : 0.0);

   double penalty=0.0;

   if(dirState==DIR_STATE_UP || dirState==DIR_STATE_DOWN)
   {
      bool againstClear=((dirState==DIR_STATE_UP && action==2) || (dirState==DIR_STATE_DOWN && action==1));
      if(againstClear)
      {
         double counterConfirm = Clamp(0.42*reversalConfirm +
                                       0.20*Clamp(support.trendReversalProb,0.0,1.0) +
                                       0.14*MathMax(0.0,alignReclaim) +
                                       0.12*(1.0- Clamp(support.expectedBasketDepth,0.0,1.0)) +
                                       0.12*Clamp(support.reversalTransitionQuality,0.0,1.0),0.0,1.0);
         penalty += 0.58 + 0.28*MathMax(0.0,0.62-counterConfirm) + 0.18*macroTrap + 0.12*regimeShiftAlert;
      }
      else
      {
         penalty += 0.10*MathMax(0.0,-alignBias);
         penalty += 0.08*MathMax(0.0,-alignReclaim);
         penalty += 0.08*regimeShiftAlert;
         if(reclaimDominant>0.5 && alignReclaim< -0.05) penalty += 0.14;
         if(continuationDominant>0.5 && alignBias< -0.05) penalty += 0.12;
      }
   }
   else
   {
      if(!neutralSafe)
      {
         penalty += 0.20 + 0.14*Clamp(support.modeConflictScore,0.0,1.0) +
                    0.12*Clamp(support.spikeRiskProb,0.0,1.0) +
                    0.12*Clamp(support.expectedBasketDepth,0.0,1.0) +
                    0.10*regimeShiftAlert;
      }
      else
      {
         penalty += 0.10*regimeBreak + 0.10*trapRisk + 0.08*wrongThesis;
         penalty += 0.12*MathMax(0.0,-alignBias-0.10);
      }
   }

   penalty += 0.16*wrongThesis;
   penalty += 0.10*trapRisk;
   penalty += 0.08*macroTrap;

   if(forcedEntry)
      penalty *= 0.78;

   return Clamp(penalty,0.0,1.25);
}

void ApplyFlexibleDirectionalDiscipline(const string symbol,
                                        const int symIdx,
                                        const int regime,
                                        const DecisionSupportContext &support,
                                        const bool forcedEntry,
                                        double &qAdj[])
{
   if(ArraySize(qAdj)<3) return;

   for(int action=1; action<=2 && action<ArraySize(qAdj); ++action)
   {
      double penalty=ComputeFlexibleDirectionalDisciplinePenalty(symbol,symIdx,regime,action,support,forcedEntry);
      if(penalty<=0.0) continue;
      qAdj[action] -= penalty;
      if(!forcedEntry)
         qAdj[0] += 0.22*MathMin(penalty,1.10);
   }
}

void ApplyConfirmedSetupAccelerator(const string symbol,
                                    const int symIdx,
                                    const int regime,
                                    const DecisionSupportContext &support,
                                    double &qAdj[])
{
   if(symIdx<0 || symIdx>=MAX_SYMBOLS) return;
   if(gPositionsCount[symIdx]>0) return;
   if(ArraySize(qAdj)<3) return;

   double macroBias=0.0,macroReclaim=0.0,regimeShiftAlert=0.0,clarity=0.0;
   bool neutralSafe=false;
   int dirState=ComputeDirectionalRegimeState(symbol,symIdx,regime,support,macroBias,macroReclaim,regimeShiftAlert,clarity,neutralSafe);

   UpdateDirectionalSignalTracking(symIdx,regime,support,qAdj,macroBias);

   int dir=ExtractDirectionalCandidate(qAdj);
   if(dir<1 || dir>2) return;
   if(gSignalTrackPersistCount[symIdx] < 2) return;

   bool modeOkay=(DominantStrategyMode(support)==1 || DominantStrategyMode(support)==2 || (dirState==DIR_STATE_NEUTRAL && neutralSafe));
   if(!modeOkay) return;

   double trapRisk=ComputeCounterTrendTrapRiskScore(symbol,symIdx);
   double expectedDepth=Clamp(support.expectedBasketDepth,0.0,1.0);
   double oneRound=Clamp(support.oneRoundPrior,0.0,1.0);
   double supportConf=Clamp(support.supportConfidence,0.0,1.0);
   double qDir=qAdj[dir];
   double qHold=qAdj[0];
   double dirEdge=Clamp((qDir-qHold+0.10)/0.28,0.0,1.0);

   double alignScore=0.5;
   if(dirState==DIR_STATE_UP)
      alignScore=(dir==1 ? Clamp(0.65 + 0.30*clarity + 0.10*MathMax(0.0,macroReclaim),0.0,1.0) : 0.0);
   else if(dirState==DIR_STATE_DOWN)
      alignScore=(dir==2 ? Clamp(0.65 + 0.30*clarity + 0.10*MathMax(0.0,-macroReclaim),0.0,1.0) : 0.0);
   else
      alignScore=(neutralSafe ? 0.58 : 0.0);

   double setupStrength=Clamp(0.20*Clamp(support.trendContinuationQuality,0.0,1.0) +
                              0.16*Clamp(support.breakoutReclaimQuality,0.0,1.0) +
                              0.16*Clamp(support.modeDominanceScore,0.0,1.0) +
                              0.12*alignScore +
                              0.12*oneRound +
                              0.10*supportConf +
                              0.08*dirEdge +
                              0.06*Clamp((double)gSignalTrackPersistCount[symIdx]/3.0,0.0,1.0),0.0,1.0);

   double caution=Clamp(0.24*Clamp(support.reversalTransitionQuality,0.0,1.0) +
                        0.18*Clamp(support.modeConflictScore,0.0,1.0) +
                        0.16*Clamp(support.spikeRiskProb,0.0,1.0) +
                        0.14*expectedDepth +
                        0.14*trapRisk +
                        0.14*Clamp(regimeShiftAlert,0.0,1.0),0.0,1.0);

   if(setupStrength < 0.54 || caution > 0.56) return;

   double boost=Clamp(0.08 + 0.18*(setupStrength-caution),0.0,0.20);
   qAdj[dir] += boost;
   qAdj[0]   -= 0.35*boost;
}

void ComputeLiveTrendBiasAndReversal(const string symbol,
                                     const int symIdx,
                                     const int regime,
                                     const DecisionSupportContext &support,
                                     double &combinedTrend,
                                     double &reversalRisk)
{
   double macroBias=0.0,macroReclaim=0.0,regimeShiftAlert=0.0,clarity=0.0;
   bool neutralSafe=false;
   int dirState=ComputeDirectionalRegimeState(symbol,symIdx,regime,support,macroBias,macroReclaim,regimeShiftAlert,clarity,neutralSafe);

   double stateBias=(dirState==DIR_STATE_UP ? 1.0 : (dirState==DIR_STATE_DOWN ? -1.0 : 0.0));
   combinedTrend = Clamp(0.42*macroBias +
                         0.16*macroReclaim +
                         0.14*(Clamp(support.trendPersistenceProb,0.0,1.0)-Clamp(support.trendReversalProb,0.0,1.0)) +
                         0.10*(Clamp(support.trendContinuationQuality,0.0,1.0)-Clamp(support.breakoutReclaimQuality,0.0,1.0)) +
                         0.10*stateBias*clarity -
                         0.10*Clamp(support.modeConflictScore,0.0,1.0), -1.0, 1.0);

   reversalRisk = Clamp(0.26*Clamp(support.reversalTransitionQuality,0.0,1.0) +
                        0.16*Clamp(support.modeConflictScore,0.0,1.0) +
                        0.14*Clamp(support.trendReversalProb,0.0,1.0) +
                        0.12*Clamp(support.spikeRiskProb,0.0,1.0) +
                        0.12*Clamp(support.macroMicroConflict,0.0,1.0) +
                        0.10*Clamp(support.painRecurrenceRisk,0.0,1.0) +
                        0.10*Clamp(regimeShiftAlert,0.0,1.0), 0.0, 1.0);
}


int DQNSelectForcedEntryAction(const int symIdx,const int regime,double &state[])
{
   double qBase[];
   DQNForwardInference(symIdx,regime,state,qBase);

   double qAdj[];
   ArrayResize(qAdj, ArraySize(qBase));
   for(int a=0;a<ArraySize(qBase);a++) qAdj[a]=qBase[a];

   double totalDelta[];
   InitActionBias(totalDelta);

   DecisionSupportContext support;
   BuildDecisionSupportContext(symIdx, regime, state, qBase, true, 0, support);
   ApplyDecisionSupportDelta(support,totalDelta);

   ApplyBudgetedSubordinateBias(symIdx, true, qBase, totalDelta, qAdj);

   string symbol=gSymbols[symIdx];
   ApplyFlexibleDirectionalDiscipline(symbol, symIdx, regime, support, true, qAdj);

   if(ActionCount >= 3)
      return (qAdj[1] >= qAdj[2] ? 1 : 2);

   if(ActionCount >= 2)
      return 1;

   return 0;
}


bool CheckProfitPauseTrigger()
{
   if(!UseProfitPause) return false;
   if(ProfitPauseTargetAmount <= 0.0) return false;
   return (gProfitCycleClosedProfit >= ProfitPauseTargetAmount);
}

void StartProfitPause()
{
   int sec = MathMax(0, ProfitPauseDurationHours) * 3600;
   if(sec <= 0)
      gProfitPauseResumeTime = 0;
   else
      gProfitPauseResumeTime = TimeCurrent() + sec;
}

double GetSymbolFloatingLossMoney(const string symbol,const int magic)
{
   double pnl = CalculatePositionsPnL(symbol,magic);
   if(pnl >= 0.0) return 0.0;
   return -pnl;
}

bool HandleEquityLossStop()
{
   if(!CheckEquityLossStop())
      return false;

   if(EquityLossStopClosePositions)
   {
      CloseAllPositions();
      CountOpenPositions();
   }

   for(int i=0;i<gSymbolCount;i++)
   {
      gFirstTradeTime[i]=0;
      gLastTradeOpenTime[i]=TimeCurrent();
      ResetForcedEntryState(i);
   }
   // reset peak reference so old DD does not keep blocking
   maxEquity = GetEAEquity();

   // keep reward baseline aligned too
   gTickEquityBaseline = maxEquity;
   gRewardBaselineTick = 0;

   return true;
}

