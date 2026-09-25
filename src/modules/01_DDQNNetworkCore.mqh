//+------------------------------------------------------------------+
//| 01_DDQNNetworkCore.mqh                                          |
//| Modularized from the original Adaptive DDQN MT5 research EA.      |
//| DQN network structure plus branch-encoder initialization, forwar|
//| Logic below is preserved from the original monolithic source.     |
//+------------------------------------------------------------------+

struct DQNNetwork
{
   int    input_dim;
   int    hidden_dim;
   int    hidden_dim2;
   int    output_dim;
   int    fusion_dim;

   int    basket_h1;
   int    basket_h2;
   int    indicator_h1;
   int    indicator_h2;
   int    volatility_h1;
   int    volatility_h2;
   int    structure_h1;
   int    structure_h2;
   int    zone_h1;
   int    zone_h2;

   // branch encoders (2 dense layers each)
   double basket_W1[];
   double basket_b1[];
   double basket_W2[];
   double basket_b2[];

   double indicator_W1[];
   double indicator_b1[];
   double indicator_W2[];
   double indicator_b2[];

   double volatility_W1[];
   double volatility_b1[];
   double volatility_W2[];
   double volatility_b2[];

   double structure_W1[];
   double structure_b1[];
   double structure_W2[];
   double structure_b2[];

   double zone_W1[];
   double zone_b1[];
   double zone_W2[];
   double zone_b2[];

   // shared fusion trunk
   double W1[];
   double b1[];
   double W2[];
   double b2[];

   // dueling heads
   double WV[];
   double bV[];

   double WA[];
   double bA[];

   double feat_mean[];
   double feat_std[];
};

DQNNetwork gDQN[MAX_SYMBOLS][REGIME_COUNT];
DQNNetwork gTargetDQN[MAX_SYMBOLS][REGIME_COUNT];

int W1Index(const int input_dim, int h, int i)    { return h*input_dim + i; }
int WVIndex(const int hidden_dim, int v, int h)   { return v*hidden_dim + h; }
int WAIndex(const int hidden_dim, int o, int h)   { return o*hidden_dim + h; }

int DenseIndex(const int input_dim, int row, int col) { return row*input_dim + col; }

int HeadInputDim(const DQNNetwork &net)
{
   return (net.hidden_dim2>0 ? net.hidden_dim2 : net.hidden_dim);
}

ENUM_TIMEFRAMES DDEventTraceTFForSymbol(const string symbol)
{
   if(DDEventTraceUseBaseTF)
      return BaseTF;
   if(DDEventTraceTF!=PERIOD_CURRENT)
      return DDEventTraceTF;
   return BaseTF;
}


int BranchHidden1Size(const int inputCount)
{
   if(inputCount<=0) return 0;
   int minW=MathMax(2, BranchEncoderMinWidth);
   int cap=MathMax(minW, BranchEncoderH1Cap);
   int v=MathMax(minW, inputCount*2);
   return MathMin(v, cap);
}

int BranchHidden2Size(const int inputCount,const int h1)
{
   if(inputCount<=0 || h1<=0) return 0;
   int minW=MathMax(2, BranchEncoderMinWidth);
   int cap=MathMax(minW, BranchEncoderH2Cap);
   int v=MathMax(minW, inputCount);
   v=MathMin(v, cap);
   return MathMin(v, h1);
}

void InitBranchEncoder(const int inputCount,
                       const int h1,
                       const int h2,
                       double &W1[],
                       double &b1[],
                       double &W2[],
                       double &b2[])
{
   ArrayResize(W1, MathMax(inputCount,0)*MathMax(h1,0));
   ArrayResize(b1, MathMax(h1,0));
   ArrayResize(W2, MathMax(h1,0)*MathMax(h2,0));
   ArrayResize(b2, MathMax(h2,0));

   if(inputCount<=0 || h1<=0 || h2<=0)
      return;

   double scale1 = 1.0 / MathSqrt((double)MathMax(inputCount,1));
   double scale2 = 1.0 / MathSqrt((double)MathMax(h1,1));

   for(int i=0;i<ArraySize(W1);i++)
   {
      double r=(double)MathRand()/32767.0;
      W1[i]=(r*2.0-1.0)*scale1;
   }
   for(int i=0;i<ArraySize(b1);i++)
      b1[i]=0.0;

   for(int i=0;i<ArraySize(W2);i++)
   {
      double r=(double)MathRand()/32767.0;
      W2[i]=(r*2.0-1.0)*scale2;
   }
   for(int i=0;i<ArraySize(b2);i++)
      b2[i]=0.0;
}

void ForwardBranchEncoder(const double &x[],
                          const int start,
                          const int inputCount,
                          const double &W1[],
                          const double &b1[],
                          const double &W2[],
                          const double &b2[],
                          double &z1[],
                          double &a1[],
                          double &z2[],
                          double &a2[])
{
   int h1=ArraySize(b1);
   int h2=ArraySize(b2);

   ArrayResize(z1,h1);
   ArrayResize(a1,h1);
   ArrayResize(z2,h2);
   ArrayResize(a2,h2);

   if(inputCount<=0 || h1<=0 || h2<=0)
      return;

   int nx=ArraySize(x);

   for(int r=0;r<h1;r++)
   {
      double sum=b1[r];
      for(int c=0;c<inputCount;c++)
      {
         int xi=start+c;
         double xv=(xi>=0 && xi<nx ? x[xi] : 0.0);
         sum += W1[DenseIndex(inputCount,r,c)] * xv;
      }
      z1[r]=sum;
      a1[r]=SiLU(sum);
   }

   for(int r=0;r<h2;r++)
   {
      double sum=b2[r];
      for(int c=0;c<h1;c++)
         sum += W2[DenseIndex(h1,r,c)] * a1[c];
      z2[r]=sum;
      a2[r]=SiLU(sum);
   }
}

void BackpropBranchEncoder(const double &x[],
                           const int start,
                           const int inputCount,
                           double &W1[],
                           double &b1[],
                           double &W2[],
                           double &b2[],
                           const double &z1[],
                           const double &a1[],
                           const double &z2[],
                           double &gradOut[],
                           const double lr)
{
   int h1=ArraySize(b1);
   int h2=ArraySize(b2);
   if(inputCount<=0 || h1<=0 || h2<=0) return;

   double dZ2[];
   ArrayResize(dZ2,h2);
   for(int i=0;i<h2;i++)
      dZ2[i]=gradOut[i]*SiLUDerivativeFromPreAct(z2[i]);
   if(UseGradientClipping) ClipArrayInPlace(dZ2,GradientClipValue);

   double oldW2[];
   ArrayResize(oldW2,ArraySize(W2));
   for(int i=0;i<ArraySize(W2);i++) oldW2[i]=W2[i];

   for(int r=0;r<h2;r++)
   {
      for(int c=0;c<h1;c++)
      {
         int idx=DenseIndex(h1,r,c);
         double grad=dZ2[r]*a1[c];
         if(UseGradientClipping) grad=ClipScalar(grad,GradientClipValue);
         W2[idx] -= lr*grad;
      }
      double gradb=dZ2[r];
      if(UseGradientClipping) gradb=ClipScalar(gradb,GradientClipValue);
      b2[r] -= lr*gradb;
   }

   double dA1[];
   ArrayResize(dA1,h1);
   for(int c=0;c<h1;c++)
   {
      double s=0.0;
      for(int r=0;r<h2;r++)
         s += oldW2[DenseIndex(h1,r,c)] * dZ2[r];
      dA1[c]=s;
   }

   double dZ1[];
   ArrayResize(dZ1,h1);
   for(int i=0;i<h1;i++)
      dZ1[i]=dA1[i]*SiLUDerivativeFromPreAct(z1[i]);
   if(UseGradientClipping) ClipArrayInPlace(dZ1,GradientClipValue);

   int nx=ArraySize(x);
   for(int r=0;r<h1;r++)
   {
      for(int c=0;c<inputCount;c++)
      {
         int xi=start+c;
         double xv=(xi>=0 && xi<nx ? x[xi] : 0.0);
         int idx=DenseIndex(inputCount,r,c);
         double grad=dZ1[r]*xv;
         if(UseGradientClipping) grad=ClipScalar(grad,GradientClipValue);
         W1[idx] -= lr*grad;
      }
      double gradb=dZ1[r];
      if(UseGradientClipping) gradb=ClipScalar(gradb,GradientClipValue);
      b1[r] -= lr*gradb;
   }
}


