//+------------------------------------------------------------------+
//| 15_Persistence.mqh                                              |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Unified DQN, archive, replay and memory persistence file builder|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

string BuildUnifiedDQNFile(const string symbol)
{
   return MQLInfoString(MQL_PROGRAM_NAME) + "_DQN_MAIN_" + symbol + ".dat";
}
string BuildRegimeDQNFile(const string symbol,const int regime)
{
   return MQLInfoString(MQL_PROGRAM_NAME) + "_DQN_MAIN_" + symbol + "_R" + IntegerToString(regime) + ".dat";
}
string BuildArchiveEpisodesFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_ARCHIVE_Episodes_" + symbol + ".dat"; }
string BuildArchivePatternsFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_ARCHIVE_Patterns_" + symbol + ".dat"; }
string BuildArchiveRegimesFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_ARCHIVE_Regimes_" + symbol + ".dat"; }
string BuildReplayMainFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_Main_" + symbol + ".dat"; }
string BuildReplayDangerFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_Danger_" + symbol + ".dat"; }
string BuildReplayDeepFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_DeepBasket_" + symbol + ".dat"; }
string BuildReplayEfficientFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_Efficient_" + symbol + ".dat"; }
string BuildReplayRecentFile(const string symbol){ return MQLInfoString(MQL_PROGRAM_NAME) + "_REPLAY_Recent_" + symbol + ".dat"; }

void WriteIntArraySimple(const int h,const int &arr[])
{
   int n=ArraySize(arr); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) FileWriteInteger(h,arr[i]);
}
void ReadIntArraySimple(const int h,int &arr[])
{
   int n=FileReadInteger(h); if(n<0) n=0;
   ArrayResize(arr,n);
   for(int i=0;i<n;i++) arr[i]=FileReadInteger(h);
}
void WriteLongArraySimple(const int h,const long &arr[])
{
   int n=ArraySize(arr); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) FileWriteLong(h,arr[i]);
}
void ReadLongArraySimple(const int h,long &arr[])
{
   int n=FileReadInteger(h); if(n<0) n=0;
   ArrayResize(arr,n);
   for(int i=0;i<n;i++) arr[i]=FileReadLong(h);
}
void WriteBarSpanRef(const int h,const BarSpanRef &r)
{
   FileWriteInteger(h,(int)r.tf);
   FileWriteLong(h,(long)r.startTime);
   FileWriteLong(h,(long)r.endTime);
   FileWriteInteger(h,r.startIndex);
   FileWriteInteger(h,r.endIndex);
}
void ReadBarSpanRef(const int h,BarSpanRef &r)
{
   r.tf=(ENUM_TIMEFRAMES)FileReadInteger(h);
   r.startTime=(datetime)FileReadLong(h);
   r.endTime=(datetime)FileReadLong(h);
   r.startIndex=FileReadInteger(h);
   r.endIndex=FileReadInteger(h);
}
void WriteReplayItemEx(const int h,const ReplayItem &it)
{
   FileWriteInteger(h,it.symIdx); FileWriteInteger(h,it.regime); FileWriteInteger(h,it.action);
   FileWriteDouble(h,it.reward); FileWriteInteger(h,(int)it.done);
   WriteDoubleArray(h,it.state); WriteDoubleArray(h,it.nextState);
   FileWriteDouble(h,it.priority);
   FileWriteInteger(h,it.replayBankType);
   FileWriteLong(h,it.episodeId); FileWriteLong(h,it.patternId); FileWriteLong(h,it.regimeId); FileWriteLong(h,it.periodId);
   FileWriteInteger(h,it.basketStateClass); FileWriteInteger(h,it.addDepthClass); FileWriteInteger(h,it.dangerClass);
   FileWriteInteger(h,it.zoneContextClass); FileWriteInteger(h,it.structureContextClass); FileWriteInteger(h,it.liquidityClass);
   FileWriteLong(h,(long)it.eventTime);
   FileWriteDouble(h,it.painSeverity);
   FileWriteDouble(h,it.recurrenceScore);
   FileWriteDouble(h,it.regimeBreakScore);
   WriteDoubleArray(h,it.macroSig);
   WriteDoubleArray(h,it.microSig);
}
void ReadReplayItemEx(const int h,ReplayItem &it)
{
   it.symIdx=FileReadInteger(h); it.regime=FileReadInteger(h); it.action=FileReadInteger(h);
   it.reward=FileReadDouble(h); it.done=(bool)FileReadInteger(h);
   ReadDoubleArray(h,it.state); ReadDoubleArray(h,it.nextState);
   it.priority=FileReadDouble(h);
   it.replayBankType=FileReadInteger(h);
   it.episodeId=FileReadLong(h); it.patternId=FileReadLong(h); it.regimeId=FileReadLong(h); it.periodId=FileReadLong(h);
   it.basketStateClass=FileReadInteger(h); it.addDepthClass=FileReadInteger(h); it.dangerClass=FileReadInteger(h);
   it.zoneContextClass=FileReadInteger(h); it.structureContextClass=FileReadInteger(h); it.liquidityClass=FileReadInteger(h);
   it.eventTime=(datetime)FileReadLong(h);
   if(!FileIsEnding(h))
   {
      it.painSeverity=FileReadDouble(h);
      it.recurrenceScore=FileReadDouble(h);
      it.regimeBreakScore=FileReadDouble(h);
      ReadDoubleArray(h,it.macroSig);
      ReadDoubleArray(h,it.microSig);
   }
   else
   {
      it.painSeverity=0.0; it.recurrenceScore=0.0; it.regimeBreakScore=0.0;
      ArrayResize(it.macroSig,0); ArrayResize(it.microSig,0);
   }
}
bool SaveReplayBankStoreFile(const string filename,const ReplayBankStore &bank)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,2); FileWriteInteger(h,bank.count); FileWriteInteger(h,bank.maxCount);
   int n=ArraySize(bank.items); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) WriteReplayItemEx(h,bank.items[i]);
   FileClose(h); return true;
}
bool LoadReplayBankStoreFile(const string filename,ReplayBankStore &bank)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   bank.count=FileReadInteger(h); bank.maxCount=FileReadInteger(h);
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(bank.items,n);
   for(int i=0;i<n;i++) ReadReplayItemEx(h,bank.items[i]);
   FileClose(h); return true;
}

void BuildMainReplayStoreForSymbol(const int symIdx,ReplayBankStore &bankOut)
{
   ArrayResize(bankOut.items,0);
   bankOut.count=0;
   bankOut.maxCount=ReplayCapacity;
   int n=ArraySize(gReplay);
   for(int i=0;i<n;i++)
   {
      if(gReplay[i].symIdx != symIdx) continue;
      int k=ArraySize(bankOut.items);
      ArrayResize(bankOut.items,k+1);
      bankOut.items[k]=gReplay[i];
   }
   bankOut.count=ArraySize(bankOut.items);
}

bool SaveMainReplayForSymbol(const int symIdx,const string filename)
{
   ReplayBankStore bank;
   BuildMainReplayStoreForSymbol(symIdx, bank);
   return SaveReplayBankStoreFile(filename, bank);
}

bool AppendMainReplayForSymbol(const string filename,const int symIdx)
{
   ReplayBankStore bank;
   ArrayResize(bank.items,0);
   bank.count=0; bank.maxCount=ReplayCapacity;
   if(!LoadReplayBankStoreFile(filename, bank)) return false;

   int n=ArraySize(bank.items);
   for(int i=0;i<n;i++)
   {
      if(bank.items[i].symIdx != symIdx) continue;
      int k=ArraySize(gReplay);
      ArrayResize(gReplay, k+1);
      gReplay[k]=bank.items[i];
   }
   if(ArraySize(gReplay) > ReplayCapacity)
      ArrayRemove(gReplay,0,ArraySize(gReplay)-ReplayCapacity);
   return (n>0);
}
bool SaveDeepBasketReplayStoreFile(const string filename,const DeepBasketReplayStore &bank)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1);
   FileWriteInteger(h,bank.itemCount); FileWriteInteger(h,bank.seqCount); FileWriteInteger(h,bank.maxItems); FileWriteInteger(h,bank.maxSeqs);
   int n=ArraySize(bank.items); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) WriteReplayItemEx(h,bank.items[i]);
   int s=ArraySize(bank.sequences); FileWriteInteger(h,s);
   for(int i=0;i<s;i++){
      FileWriteLong(h,bank.sequences[i].episodeId); FileWriteInteger(h,bank.sequences[i].symIdx);
      FileWriteLong(h,(long)bank.sequences[i].startTime); FileWriteLong(h,(long)bank.sequences[i].endTime);
      FileWriteInteger(h,bank.sequences[i].basketDir); FileWriteInteger(h,bank.sequences[i].addCount); FileWriteInteger(h,bank.sequences[i].maxPositions);
      FileWriteDouble(h,bank.sequences[i].maxDD); FileWriteDouble(h,bank.sequences[i].finalReward);
      WriteIntArraySimple(h,bank.sequences[i].replayItemIndexes);
   }
   FileClose(h); return true;
}
bool LoadDeepBasketReplayStoreFile(const string filename,DeepBasketReplayStore &bank)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   bank.itemCount=FileReadInteger(h); bank.seqCount=FileReadInteger(h); bank.maxItems=FileReadInteger(h); bank.maxSeqs=FileReadInteger(h);
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(bank.items,n);
   for(int i=0;i<n;i++) ReadReplayItemEx(h,bank.items[i]);
   int s=FileReadInteger(h); if(s<0) s=0; ArrayResize(bank.sequences,s);
   for(int i=0;i<s;i++){
      bank.sequences[i].episodeId=FileReadLong(h); bank.sequences[i].symIdx=FileReadInteger(h);
      bank.sequences[i].startTime=(datetime)FileReadLong(h); bank.sequences[i].endTime=(datetime)FileReadLong(h);
      bank.sequences[i].basketDir=FileReadInteger(h); bank.sequences[i].addCount=FileReadInteger(h); bank.sequences[i].maxPositions=FileReadInteger(h);
      bank.sequences[i].maxDD=FileReadDouble(h); bank.sequences[i].finalReward=FileReadDouble(h);
      ReadIntArraySimple(h,bank.sequences[i].replayItemIndexes);
   }
   FileClose(h); return true;
}
bool SaveEfficientReplayStoreFile(const string filename,const EfficientReplayStore &bank)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1);
   FileWriteInteger(h,bank.itemCount); FileWriteInteger(h,bank.periodCount); FileWriteInteger(h,bank.maxItems); FileWriteInteger(h,bank.maxPeriods);
   int n=ArraySize(bank.items); FileWriteInteger(h,n);
   for(int i=0;i<n;i++) WriteReplayItemEx(h,bank.items[i]);
   int s=ArraySize(bank.periods); FileWriteInteger(h,s);
   for(int i=0;i<s;i++){
      FileWriteLong(h,bank.periods[i].periodId); FileWriteInteger(h,bank.periods[i].symIdx);
      FileWriteLong(h,(long)bank.periods[i].startTime); FileWriteLong(h,(long)bank.periods[i].endTime);
      FileWriteDouble(h,bank.periods[i].rewardTotal); FileWriteDouble(h,bank.periods[i].rewardEfficiency); FileWriteDouble(h,bank.periods[i].ddMax);
      FileWriteInteger(h,bank.periods[i].oneRoundCount); FileWriteInteger(h,bank.periods[i].addCountTotal);
      FileWriteInteger(h,bank.periods[i].sessionType); FileWriteInteger(h,bank.periods[i].liquidityType); FileWriteInteger(h,bank.periods[i].regimeType); FileWriteInteger(h,bank.periods[i].patternType);
      WriteIntArraySimple(h,bank.periods[i].replayItemIndexes);
   }
   FileClose(h); return true;
}
bool LoadEfficientReplayStoreFile(const string filename,EfficientReplayStore &bank)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   bank.itemCount=FileReadInteger(h); bank.periodCount=FileReadInteger(h); bank.maxItems=FileReadInteger(h); bank.maxPeriods=FileReadInteger(h);
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(bank.items,n);
   for(int i=0;i<n;i++) ReadReplayItemEx(h,bank.items[i]);
   int s=FileReadInteger(h); if(s<0) s=0; ArrayResize(bank.periods,s);
   for(int i=0;i<s;i++){
      bank.periods[i].periodId=FileReadLong(h); bank.periods[i].symIdx=FileReadInteger(h);
      bank.periods[i].startTime=(datetime)FileReadLong(h); bank.periods[i].endTime=(datetime)FileReadLong(h);
      bank.periods[i].rewardTotal=FileReadDouble(h); bank.periods[i].rewardEfficiency=FileReadDouble(h); bank.periods[i].ddMax=FileReadDouble(h);
      bank.periods[i].oneRoundCount=FileReadInteger(h); bank.periods[i].addCountTotal=FileReadInteger(h);
      bank.periods[i].sessionType=FileReadInteger(h); bank.periods[i].liquidityType=FileReadInteger(h); bank.periods[i].regimeType=FileReadInteger(h); bank.periods[i].patternType=FileReadInteger(h);
      ReadIntArraySimple(h,bank.periods[i].replayItemIndexes);
   }
   FileClose(h); return true;
}
bool SaveEpisodeMemoryFile(const string filename)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1); int n=ArraySize(gEpisodeMemory); FileWriteInteger(h,n);
   for(int i=0;i<n;i++){
      EpisodeMemory e = gEpisodeMemory[i];
      FileWriteLong(h,e.episodeId); FileWriteString(h,e.symbol); FileWriteInteger(h,e.symIdx);
      FileWriteLong(h,(long)e.startTime); FileWriteLong(h,(long)e.endTime);
      FileWriteInteger(h,e.basketDir); FileWriteInteger(h,e.openPositionsMax); FileWriteInteger(h,e.addCount); FileWriteInteger(h,e.actionsCount);
      FileWriteDouble(h,e.entryPriceFirst); FileWriteDouble(h,e.avgEntryAtWorst); FileWriteDouble(h,e.closePriceFinal);
      FileWriteDouble(h,e.pnlFinal); FileWriteDouble(h,e.rewardTotal); FileWriteDouble(h,e.rewardEfficiency); FileWriteDouble(h,e.maxDrawdownPct); FileWriteDouble(h,e.maxDangerScore); FileWriteDouble(h,e.maxMarginStress);
      FileWriteInteger(h,e.oneRoundTrade); FileWriteInteger(h,e.forcedStopLikeEvent); FileWriteInteger(h,e.inefficientRecovery);
      FileWriteInteger(h,e.sessionType); FileWriteInteger(h,e.liquidityType); FileWriteInteger(h,e.regimeType); FileWriteInteger(h,e.patternType);
      WriteBarSpanRef(h,e.execSpan); WriteBarSpanRef(h,e.midSpan); WriteBarSpanRef(h,e.longSpan); WriteBarSpanRef(h,e.structExtSpan);
   }
   FileClose(h); return true;
}
bool LoadEpisodeMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(gEpisodeMemory,n);
   for(int i=0;i<n;i++){
      gEpisodeMemory[i].episodeId=FileReadLong(h); gEpisodeMemory[i].symbol=FileReadString(h); gEpisodeMemory[i].symIdx=FileReadInteger(h);
      gEpisodeMemory[i].startTime=(datetime)FileReadLong(h); gEpisodeMemory[i].endTime=(datetime)FileReadLong(h);
      gEpisodeMemory[i].basketDir=FileReadInteger(h); gEpisodeMemory[i].openPositionsMax=FileReadInteger(h); gEpisodeMemory[i].addCount=FileReadInteger(h); gEpisodeMemory[i].actionsCount=FileReadInteger(h);
      gEpisodeMemory[i].entryPriceFirst=FileReadDouble(h); gEpisodeMemory[i].avgEntryAtWorst=FileReadDouble(h); gEpisodeMemory[i].closePriceFinal=FileReadDouble(h);
      gEpisodeMemory[i].pnlFinal=FileReadDouble(h); gEpisodeMemory[i].rewardTotal=FileReadDouble(h); gEpisodeMemory[i].rewardEfficiency=FileReadDouble(h); gEpisodeMemory[i].maxDrawdownPct=FileReadDouble(h); gEpisodeMemory[i].maxDangerScore=FileReadDouble(h); gEpisodeMemory[i].maxMarginStress=FileReadDouble(h);
      gEpisodeMemory[i].oneRoundTrade=FileReadInteger(h); gEpisodeMemory[i].forcedStopLikeEvent=FileReadInteger(h); gEpisodeMemory[i].inefficientRecovery=FileReadInteger(h);
      gEpisodeMemory[i].sessionType=FileReadInteger(h); gEpisodeMemory[i].liquidityType=FileReadInteger(h); gEpisodeMemory[i].regimeType=FileReadInteger(h); gEpisodeMemory[i].patternType=FileReadInteger(h);
      ReadBarSpanRef(h,gEpisodeMemory[i].execSpan); ReadBarSpanRef(h,gEpisodeMemory[i].midSpan); ReadBarSpanRef(h,gEpisodeMemory[i].longSpan); ReadBarSpanRef(h,gEpisodeMemory[i].structExtSpan);
   }
   FileClose(h); return true;
}
bool SavePatternMemoryFile(const string filename)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1); int n=ArraySize(gPatternMemory); FileWriteInteger(h,n);
   for(int i=0;i<n;i++){
      FileWriteLong(h,gPatternMemory[i].patternId); FileWriteString(h,gPatternMemory[i].symbol); FileWriteInteger(h,gPatternMemory[i].symIdx);
      FileWriteInteger(h,gPatternMemory[i].patternType); FileWriteInteger(h,gPatternMemory[i].strengthClass); FileWriteInteger(h,gPatternMemory[i].volatilityClass); FileWriteInteger(h,gPatternMemory[i].liquidityClass);
      FileWriteLong(h,(long)gPatternMemory[i].startTime); FileWriteLong(h,(long)gPatternMemory[i].endTime);
      FileWriteDouble(h,gPatternMemory[i].rewardEfficiencyMean); FileWriteDouble(h,gPatternMemory[i].ddMean); FileWriteDouble(h,gPatternMemory[i].addMean);
      WriteBarSpanRef(h,gPatternMemory[i].execSpan); WriteBarSpanRef(h,gPatternMemory[i].midSpan); WriteBarSpanRef(h,gPatternMemory[i].longSpan);
      WriteLongArraySimple(h,gPatternMemory[i].linkedEpisodeIds);
   }
   FileClose(h); return true;
}
bool LoadPatternMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(gPatternMemory,n);
   for(int i=0;i<n;i++){
      gPatternMemory[i].patternId=FileReadLong(h); gPatternMemory[i].symbol=FileReadString(h); gPatternMemory[i].symIdx=FileReadInteger(h);
      gPatternMemory[i].patternType=FileReadInteger(h); gPatternMemory[i].strengthClass=FileReadInteger(h); gPatternMemory[i].volatilityClass=FileReadInteger(h); gPatternMemory[i].liquidityClass=FileReadInteger(h);
      gPatternMemory[i].startTime=(datetime)FileReadLong(h); gPatternMemory[i].endTime=(datetime)FileReadLong(h);
      gPatternMemory[i].rewardEfficiencyMean=FileReadDouble(h); gPatternMemory[i].ddMean=FileReadDouble(h); gPatternMemory[i].addMean=FileReadDouble(h);
      ReadBarSpanRef(h,gPatternMemory[i].execSpan); ReadBarSpanRef(h,gPatternMemory[i].midSpan); ReadBarSpanRef(h,gPatternMemory[i].longSpan);
      ReadLongArraySimple(h,gPatternMemory[i].linkedEpisodeIds);
   }
   FileClose(h); return true;
}
bool SaveRegimeEventMemoryFile(const string filename)
{
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return false;
   FileWriteInteger(h,1); int n=ArraySize(gRegimeEventMemory); FileWriteInteger(h,n);
   for(int i=0;i<n;i++){
      FileWriteLong(h,gRegimeEventMemory[i].regimeId); FileWriteString(h,gRegimeEventMemory[i].symbol); FileWriteInteger(h,gRegimeEventMemory[i].symIdx);
      FileWriteInteger(h,gRegimeEventMemory[i].regimeType); FileWriteInteger(h,gRegimeEventMemory[i].eventType);
      FileWriteLong(h,(long)gRegimeEventMemory[i].startTime); FileWriteLong(h,(long)gRegimeEventMemory[i].endTime);
      FileWriteDouble(h,gRegimeEventMemory[i].avgVol); FileWriteDouble(h,gRegimeEventMemory[i].avgSpread); FileWriteDouble(h,gRegimeEventMemory[i].avgADX); FileWriteDouble(h,gRegimeEventMemory[i].avgRewardEfficiency);
      WriteBarSpanRef(h,gRegimeEventMemory[i].execSpan); WriteBarSpanRef(h,gRegimeEventMemory[i].midSpan); WriteBarSpanRef(h,gRegimeEventMemory[i].longSpan);
      WriteLongArraySimple(h,gRegimeEventMemory[i].linkedEpisodeIds); WriteLongArraySimple(h,gRegimeEventMemory[i].linkedPatternIds);
   }
   FileClose(h); return true;
}
bool LoadRegimeEventMemoryFile(const string filename)
{
   if(!FileIsExist(filename)) return false;
   int h=FileOpen(filename,FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return false;
   int ver=FileReadInteger(h); if(ver!=1 && ver!=2){ FileClose(h); return false; }
   int n=FileReadInteger(h); if(n<0) n=0; ArrayResize(gRegimeEventMemory,n);
   for(int i=0;i<n;i++){
      gRegimeEventMemory[i].regimeId=FileReadLong(h); gRegimeEventMemory[i].symbol=FileReadString(h); gRegimeEventMemory[i].symIdx=FileReadInteger(h);
      gRegimeEventMemory[i].regimeType=FileReadInteger(h); gRegimeEventMemory[i].eventType=FileReadInteger(h);
      gRegimeEventMemory[i].startTime=(datetime)FileReadLong(h); gRegimeEventMemory[i].endTime=(datetime)FileReadLong(h);
      gRegimeEventMemory[i].avgVol=FileReadDouble(h); gRegimeEventMemory[i].avgSpread=FileReadDouble(h); gRegimeEventMemory[i].avgADX=FileReadDouble(h); gRegimeEventMemory[i].avgRewardEfficiency=FileReadDouble(h);
      ReadBarSpanRef(h,gRegimeEventMemory[i].execSpan); ReadBarSpanRef(h,gRegimeEventMemory[i].midSpan); ReadBarSpanRef(h,gRegimeEventMemory[i].longSpan);
      ReadLongArraySimple(h,gRegimeEventMemory[i].linkedEpisodeIds); ReadLongArraySimple(h,gRegimeEventMemory[i].linkedPatternIds);
   }
   FileClose(h); return true;
}
bool SaveUnifiedPersistenceForSymbol(const int symIdx)
{
   bool ok=true;
   // Unified persistence is always the active DQN persistence path.
   if(UseRegimeBank)
   {
      for(int r=0;r<REGIME_COUNT;r++)
         ok = SaveDQNForSymbol(symIdx,r,BuildRegimeDQNFile(gSymbols[symIdx], r)) && ok;
      ok = SaveDQNForSymbol(symIdx,0,BuildUnifiedDQNFile(gSymbols[symIdx])) && ok;
   }
   else
      ok = SaveDQNForSymbol(symIdx,0,BuildUnifiedDQNFile(gSymbols[symIdx])) && ok;
   if(SaveArchiveMemory)
   {
      ok = SaveEpisodeMemoryFile(BuildArchiveEpisodesFile(gSymbols[symIdx])) && ok;
      ok = SavePatternMemoryFile(BuildArchivePatternsFile(gSymbols[symIdx])) && ok;
      ok = SaveRegimeEventMemoryFile(BuildArchiveRegimesFile(gSymbols[symIdx])) && ok;
   }
   if(SaveReplayBanks)
   {
      if(SaveMainReplayBank)
         ok = SaveMainReplayForSymbol(symIdx, BuildReplayMainFile(gSymbols[symIdx])) && ok;
      ok = SaveReplayBankStoreFile(BuildReplayDangerFile(gSymbols[symIdx]), gDangerReplayBank) && ok;
      ok = SaveDeepBasketReplayStoreFile(BuildReplayDeepFile(gSymbols[symIdx]), gDeepBasketReplayBank) && ok;
      ok = SaveEfficientReplayStoreFile(BuildReplayEfficientFile(gSymbols[symIdx]), gEfficientReplayBank) && ok;
      if(SaveRecentReplayBank)
         ok = SaveReplayBankStoreFile(BuildReplayRecentFile(gSymbols[symIdx]), gRecentReplayBank) && ok;
   }
   return ok;
}
bool LoadUnifiedPersistenceForSymbol(const int symIdx,int inDim)
{
   bool loaded=false;
   bool anyDqnLoaded=false;

   if(UseRegimeBank)
   {
      bool haveRegime0=false;
      for(int r=0;r<REGIME_COUNT;r++)
      {
         string rf=BuildRegimeDQNFile(gSymbols[symIdx], r);
         bool okR=LoadDQNForSymbol(symIdx,r,rf);
         if(!okR && r==0)
            okR=LoadDQNForSymbol(symIdx,0,BuildUnifiedDQNFile(gSymbols[symIdx]));
         if(okR && gDQN[symIdx][r].input_dim==inDim)
         {
            anyDqnLoaded=true;
            loaded=true;
            if(r==0) haveRegime0=true;
         }
         else
         {
            if(r==0)
            {
               InitOrRandomizeDQN(symIdx,0,inDim);
            }
            else if(haveRegime0)
            {
               gDQN[symIdx][r]=gDQN[symIdx][0];
               gTargetDQN[symIdx][r]=gTargetDQN[symIdx][0];
            }
            else
            {
               InitOrRandomizeDQN(symIdx,r,inDim);
            }
         }
      }
      if(anyDqnLoaded)
      {
         if(StringLen(gLoadedDQNMetaNote[symIdx])>0)
            Print("Loaded DQN metadata [", gSymbols[symIdx], "]: ", gLoadedDQNMetaNote[symIdx]);
         if(gLoadedDQNFastModeMeta[symIdx] && !FastTrainingMode)
            Print("Loaded DQN note [", gSymbols[symIdx], "]: model was last saved from fast-training mode; full-feature inference is load-compatible, but a full-feature fine-tune is recommended before live deployment.");
      }
   }
   else
   {
      string fname=BuildUnifiedDQNFile(gSymbols[symIdx]);
      bool dqnLoaded=LoadDQNForSymbol(symIdx,0,fname);
      if(!dqnLoaded || gDQN[symIdx][0].input_dim!=inDim)
         InitOrRandomizeDQN(symIdx,0,inDim);
      else
      {
         loaded=true;
         anyDqnLoaded=true;
         if(StringLen(gLoadedDQNMetaNote[symIdx])>0)
            Print("Loaded DQN metadata [", gSymbols[symIdx], "]: ", gLoadedDQNMetaNote[symIdx]);
         if(gLoadedDQNFastModeMeta[symIdx] && !FastTrainingMode)
            Print("Loaded DQN note [", gSymbols[symIdx], "]: model was last saved from fast-training mode; full-feature inference is load-compatible, but a full-feature fine-tune is recommended before live deployment.");
      }
      for(int r=1;r<REGIME_COUNT;r++)
      {
         gDQN[symIdx][r]=gDQN[symIdx][0];
         gTargetDQN[symIdx][r]=gTargetDQN[symIdx][0];
      }
   }

   if(SaveArchiveMemory)
   {
      loaded = LoadEpisodeMemoryFile(BuildArchiveEpisodesFile(gSymbols[symIdx])) || loaded;
      loaded = LoadPatternMemoryFile(BuildArchivePatternsFile(gSymbols[symIdx])) || loaded;
      loaded = LoadRegimeEventMemoryFile(BuildArchiveRegimesFile(gSymbols[symIdx])) || loaded;
   }
   if(SaveReplayBanks)
   {
      if(SaveMainReplayBank)
         loaded = AppendMainReplayForSymbol(BuildReplayMainFile(gSymbols[symIdx]), symIdx) || loaded;
      loaded = LoadReplayBankStoreFile(BuildReplayDangerFile(gSymbols[symIdx]), gDangerReplayBank) || loaded;
      loaded = LoadDeepBasketReplayStoreFile(BuildReplayDeepFile(gSymbols[symIdx]), gDeepBasketReplayBank) || loaded;
      loaded = LoadEfficientReplayStoreFile(BuildReplayEfficientFile(gSymbols[symIdx]), gEfficientReplayBank) || loaded;
      if(SaveRecentReplayBank)
         loaded = LoadReplayBankStoreFile(BuildReplayRecentFile(gSymbols[symIdx]), gRecentReplayBank) || loaded;
   }
   return loaded;
}

