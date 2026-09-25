//+------------------------------------------------------------------+
//| 14_ZonesAndVisualization.mqh                                    |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| Supply/demand zone detection plus structure and chart visualizat|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

void DetectZones(string symbol)
{
   int bars = iBars(symbol, BaseTF);
   if(bars <= 0) return;

   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point<=0.0) point=0.00001;

   int minBase = MathMax(1, MinZoneBaseBars);
   int maxBase = MathMax(minBase, MaxZoneBaseBars);
   if(bars < maxBase + 2) return;

   int breakoutShift = 1;
   datetime lastClosedTime = iTime(symbol, BaseTF, breakoutShift);

   int added=0;
   int maxAdd=MathMax(1, MaxZonesPerBar);

   for(int baseLen=minBase; baseLen<=maxBase && added<maxAdd; baseLen++)
   {
      int startIndex = breakoutShift + baseLen;
      if(startIndex>=bars) continue;

      double highPrice=iHigh(symbol,BaseTF,startIndex);
      double lowPrice =iLow (symbol,BaseTF,startIndex);

      bool consolidated=true;
      for(int iBar=startIndex-1; iBar>breakoutShift; iBar--)
      {
         double h=iHigh(symbol,BaseTF,iBar);
         double l=iLow (symbol,BaseTF,iBar);
         if(h>highPrice) highPrice=h;
         if(l<lowPrice)  lowPrice=l;

         if((highPrice-lowPrice) > ZoneMaxHeightPoints*point)
         { consolidated=false; break; }
      }
      if(!consolidated) continue;

      double closePrice=iClose(symbol,BaseTF,breakoutShift);
      double breakoutLow=iLow(symbol,BaseTF,breakoutShift);
      double breakoutHigh=iHigh(symbol,BaseTF,breakoutShift);

      bool isDemand=(closePrice>highPrice && breakoutLow>=lowPrice);
      bool isSupply=(closePrice<lowPrice && breakoutHigh<=highPrice);
      if(!isDemand && !isSupply) continue;

      bool overlaps=false, duplicate=false;
      int totalZones=ArraySize(zones);
      for(int j=0;j<totalZones;j++)
      {
         if(zones[j].symbol!=symbol) continue;
         if(lastClosedTime < zones[j].endTime)
         {
            double maxLow=MathMax(lowPrice, zones[j].low);
            double minHigh=MathMin(highPrice, zones[j].high);
            if(maxLow<=minHigh){ overlaps=true; break; }
            if(MathAbs(zones[j].high-highPrice)<point &&
               MathAbs(zones[j].low -lowPrice )<point)
            { duplicate=true; break; }
         }
      }
      if(overlaps||duplicate) continue;

      int zc=ArraySize(zones);
      if(zc>=MaxZonesTracked && zc>0)
      {
         if(DrawZonesOnChart)
         {
            ObjectDelete(0,zones[0].name);
            ObjectDelete(0,zones[0].name+"_lbl");
         }
         ArrayRemove(zones,0,1);
         zc--;
      }

      ArrayResize(zones, zc+1);
      zones[zc].symbol=symbol;
      zones[zc].high=highPrice;
      zones[zc].low=lowPrice;
      zones[zc].startTime=iTime(symbol,BaseTF,(startIndex>0?startIndex:breakoutShift+minBase));
      zones[zc].endTime=TimeCurrent() + (datetime)PeriodSeconds(BaseTF)*ZoneExtendBars;
      zones[zc].breakoutTime=lastClosedTime;
      zones[zc].isDemand=isDemand;
      zones[zc].tested=false;
      zones[zc].broken=false;
      zones[zc].name="SDZone_"+symbol+"_"+IntegerToString(zc)+"_"+TimeToString(zones[zc].startTime,TIME_DATE|TIME_SECONDS);

      added++;
   }
}

void UpdateZones(string symbol)
{
   int total=ArraySize(zones);
   if(total<=0) return;

   double prevHigh=iHigh(symbol,BaseTF,1);
   double prevLow =iLow (symbol,BaseTF,1);
   double prevClose=iClose(symbol,BaseTF,1);
   datetime lastClosedTime=iTime(symbol,BaseTF,1);

   for(int i=total-1;i>=0;i--)
   {
      if(zones[i].symbol!=symbol) continue;

      if(lastClosedTime>=zones[i].endTime)
      {
         if(DrawZonesOnChart)
         {
            ObjectDelete(0,zones[i].name);
            ObjectDelete(0,zones[i].name+"_lbl");
         }
         ArrayRemove(zones,i,1);
         continue;
      }

      bool overlap=(prevLow<=zones[i].high && prevHigh>=zones[i].low);
      if(overlap) zones[i].tested=true;

      if(zones[i].isDemand)
      {
         if(prevClose<zones[i].low) zones[i].broken=true;
      }
      else
      {
         if(prevClose>zones[i].high) zones[i].broken=true;
      }

      if(DrawZonesOnChart)
      {
         color zoneColor;
         if(zones[i].broken) zoneColor=clrDarkGray;
         else if(zones[i].tested) zoneColor=(zones[i].isDemand?clrBlueViolet:clrOrange);
         else zoneColor=(zones[i].isDemand?clrBlue:clrRed);

         ObjectDelete(0,zones[i].name);
         ObjectCreate(0,zones[i].name,OBJ_RECTANGLE,0,zones[i].startTime,zones[i].high,zones[i].endTime,zones[i].low);
         ObjectSetInteger(0,zones[i].name,OBJPROP_COLOR,zoneColor);
         ObjectSetInteger(0,zones[i].name,OBJPROP_FILL,true);
         ObjectSetInteger(0,zones[i].name,OBJPROP_BACK,true);

         datetime midTime=zones[i].startTime+(zones[i].endTime-zones[i].startTime)/2;
         double midPrice=(zones[i].high+zones[i].low)/2.0;

         string txt=(zones[i].isDemand?"Demand":"Supply");
         if(zones[i].tested) txt+=" (T)";
         if(zones[i].broken) txt+=" (X)";

         string labelName=zones[i].name+"_lbl";
         ObjectDelete(0,labelName);
         ObjectCreate(0,labelName,OBJ_TEXT,0,midTime,midPrice);
         ObjectSetString(0,labelName,OBJPROP_TEXT,txt);
         ObjectSetInteger(0,labelName,OBJPROP_COLOR,clrBlack);
         ObjectSetInteger(0,labelName,OBJPROP_ANCHOR,ANCHOR_CENTER);
      }
   }
}

string TFLabel(const ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      default:         return IntegerToString((int)tf);
   }
}

string StructVizPrefix(const string symbol,const ENUM_TIMEFRAMES tf)
{
   return "DDQN_STRUCT_" + symbol + "_" + TFLabel(tf) + "_";
}

void ClearStructureVisualizationForTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   string pfx = StructVizPrefix(symbol, tf);
   ObjectsDeleteAll(0, pfx);
}

void DrawSwingPointLabel(const string name, const datetime when, const double price, const string txt, const color clr)
{
   ObjectDelete(0, name);
   if(!ObjectCreate(0, name, OBJ_TEXT, 0, when, price))
      return;
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
}

void DrawZoneRectLabel(const string rectName, const string txtName, const datetime t1, const datetime t2, const double high, const double low, const color clr, const string txt)
{
   ObjectDelete(0, rectName);
   if(ObjectCreate(0, rectName, OBJ_RECTANGLE, 0, t1, high, t2, low))
   {
      ObjectSetInteger(0, rectName, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, rectName, OBJPROP_FILL, true);
      ObjectSetInteger(0, rectName, OBJPROP_BACK, true);
      ObjectSetInteger(0, rectName, OBJPROP_WIDTH, 1);
   }

   ObjectDelete(0, txtName);
   datetime tm = t1 + (t2 - t1) / 2;
   double mp = 0.5 * (high + low);
   if(ObjectCreate(0, txtName, OBJ_TEXT, 0, tm, mp))
   {
      ObjectSetString(0, txtName, OBJPROP_TEXT, txt);
      ObjectSetInteger(0, txtName, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, txtName, OBJPROP_ANCHOR, ANCHOR_CENTER);
      ObjectSetInteger(0, txtName, OBJPROP_FONTSIZE, 8);
   }
}

void DrawCandleFlagLabel(const string name, const datetime when, const double price, const string txt, const color clr)
{
   ObjectDelete(0, name);
   if(!ObjectCreate(0, name, OBJ_TEXT, 0, when, price))
      return;
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
}

void UpdateStructureVisualizationForTF(const string symbol,const ENUM_TIMEFRAMES tf)
{
   if(!(DrawSwingStructureOnChart || DrawSwingZonesOnChart || DrawZoneCandleFlagsOnChart))
      return;
   if(symbol != _Symbol)
      return;

   string pfx = StructVizPrefix(symbol, tf);
   ClearStructureVisualizationForTF(symbol, tf);

   if(DrawSwingStructureOnChart)
   {
      SwingPointMem swings[];
      int swingCount = 0;
      if(BuildSwingMemoryForTF_Core(symbol, tf, swings, swingCount))
      {
         int drawCount = MathMin(swingCount, MathMax(MaxDrawnSwingsPerTF, 1));
         int start = MathMax(0, swingCount - drawCount);
         int n = 0;
         for(int i=start; i<swingCount; ++i)
         {
            string nm = pfx + "SW_" + IntegerToString(n);
            string lbl = (swings[i].isHigh ? "SH" : "SL");
            color clr = (swings[i].isHigh ? clrTomato : clrLimeGreen);
            DrawSwingPointLabel(nm, swings[i].when, swings[i].price, TFLabel(tf) + " " + lbl, clr);
            n++;
         }
      }
   }

   SwingDerivedZone demands[], supplies[];
   int demandCount = 0, supplyCount = 0;
   bool haveZones = BuildSwingDerivedZonesForTF(symbol, tf, demands, demandCount, supplies, supplyCount);

   if(DrawSwingZonesOnChart && haveZones)
   {
      double atr = MathMax(GetATRValueTF(symbol, tf, 14, 1), 1e-8);
      int idxDem = -1, idxSup = -1;
      double ref = GetCloseSafe(symbol, tf, 1);
      SelectBestSwingZone(demands, demandCount, symbol, tf, ref, ref, ref, atr, idxDem);
      SelectBestSwingZone(supplies, supplyCount, symbol, tf, ref, ref, ref, atr, idxSup);

      int drawn = 0;
      if(idxDem >= 0 && drawn < MaxDrawnZonesPerTF)
      {
         SwingDerivedZone z = demands[idxDem];
         datetime t1 = z.pivotTime;
         datetime t2 = TimeCurrent() + (datetime)PeriodSeconds(tf) * 40;
         string tag = (z.lifeState > 0 ? "Active" : (z.lifeState == 0 ? "Candidate" : "Invalid"));
         DrawZoneRectLabel(pfx + "ZD_RECT_0", pfx + "ZD_TXT_0", t1, t2, z.high, z.low, clrLimeGreen, TFLabel(tf) + " Demand " + tag);
         drawn++;
      }
      if(idxSup >= 0 && drawn < MaxDrawnZonesPerTF)
      {
         SwingDerivedZone z = supplies[idxSup];
         datetime t1 = z.pivotTime;
         datetime t2 = TimeCurrent() + (datetime)PeriodSeconds(tf) * 40;
         string tag = (z.lifeState > 0 ? "Active" : (z.lifeState == 0 ? "Candidate" : "Invalid"));
         DrawZoneRectLabel(pfx + "ZS_RECT_0", pfx + "ZS_TXT_0", t1, t2, z.high, z.low, clrTomato, TFLabel(tf) + " Supply " + tag);
      }
   }

   if(DrawZoneCandleFlagsOnChart)
   {
      ZoneBranchFeaturePack zf;
      ComputeZoneBranchFeatures(symbol, tf, GetCloseSafe(symbol, tf, 1), GetCloseSafe(symbol, tf, 1), GetCloseSafe(symbol, tf, 1), zf);

      double open1 = iOpen(symbol, tf, 1);
      double high1 = iHigh(symbol, tf, 1);
      double low1  = iLow(symbol, tf, 1);
      double close1= GetCloseSafe(symbol, tf, 1);
      double range = MathMax(high1 - low1, SymbolInfoDouble(symbol,SYMBOL_POINT) * 5.0);
      double body  = MathAbs(close1 - open1);
      double upper = high1 - MathMax(open1, close1);
      double lower = MathMin(open1, close1) - low1;
      double closeLoc = Clamp((close1 - low1) / range, 0.0, 1.0);
      double activeDist = (zf.zoneSideState > 0.0 ? MathAbs(zf.nearestDemandDist) : MathAbs(zf.nearestSupplyDist));
      double zoneNearness = 1.0 - Clamp(activeDist, 0.0, 1.0);
      double zoneDepthCentrality = 1.0 - MathAbs(zf.zoneDepthPosition);
      double zoneContext = Clamp(0.45 * zoneNearness + 0.25 * zoneDepthCentrality + 0.20 * MathAbs(zf.zoneRelevance) + 0.10 * MathAbs(zf.zoneRetestBreakState), 0.0, 1.0);
      double rejection = 0.0;
      if(zf.zoneSideState > 0.0) rejection = Clamp((lower / range) * (closeLoc), 0.0, 1.0);
      else if(zf.zoneSideState < 0.0) rejection = Clamp((upper / range) * (1.0 - closeLoc), 0.0, 1.0);
      rejection *= zoneContext;
      double acceptance = 0.0;
      if(zf.zoneSideState > 0.0) acceptance = Clamp((1.0 - closeLoc) * (body / range), 0.0, 1.0);
      else if(zf.zoneSideState < 0.0) acceptance = Clamp((closeLoc) * (body / range), 0.0, 1.0);
      acceptance *= zoneContext;
      double indecision = Clamp((1.0 - body / range) * (0.35 + 0.65 * zoneContext), 0.0, 1.0);
      string txt = "";
      color clr = clrSilver;
      if(rejection >= 0.45) { txt = TFLabel(tf) + " Reject"; clr = clrDodgerBlue; }
      else if(acceptance >= 0.45) { txt = TFLabel(tf) + " Accept"; clr = clrOrangeRed; }
      else if(indecision >= 0.60) { txt = TFLabel(tf) + " Indecision"; clr = clrSlateBlue; }
      if(StringLen(txt) > 0)
         DrawCandleFlagLabel(pfx + "CF_0", iTime(symbol, tf, 1), high1, txt, clr);
   }
}

void UpdateStructureVisualization(const string symbol)
{
   if(!(DrawSwingStructureOnChart || DrawSwingZonesOnChart || DrawZoneCandleFlagsOnChart))
      return;
   ENUM_TIMEFRAMES tfExec = (TF_EXEC==PERIOD_CURRENT ? BaseTF : TF_EXEC);
   UpdateStructureVisualizationForTF(symbol, tfExec);
   UpdateStructureVisualizationForTF(symbol, TF_MID);
   UpdateStructureVisualizationForTF(symbol, TF_LONG);
   if(TF_STRUCT_EXT > 0)
      UpdateStructureVisualizationForTF(symbol, TF_STRUCT_EXT);
}



