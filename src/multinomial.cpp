//  GBM by Greg Ridgeway  Copyright (C) 2003

#include <vector>
#include "multinomial.h"


void CMultinomial::UpdateParams
(
   double *adF,
   double *adOffset,
   double *adWeight,
   unsigned long cLength
)
{
   // Local variables
   unsigned long ii=0;
   unsigned long kk=0;

   // Set the probabilities for each observation in each class
   for (ii = 0; ii < mcRows; ii++)
   {
      double dClassSum = 0.0;
      for (kk = 0; kk < mcNumClasses; kk++)
      {
         int iIdx = ii + kk * mcRows;
         double dF = (adOffset == NULL) ? adF[iIdx] : adF[iIdx] + adOffset[iIdx];
         madProb[iIdx] = adWeight[iIdx] * std::exp(dF);
         dClassSum += adWeight[iIdx] * std::exp(dF);
      }

      dClassSum = (dClassSum > 0) ? dClassSum : 1e-8;

      for (kk = 0; kk < mcNumClasses; kk++)
      {
         madProb[ii + kk * mcRows] /= dClassSum;
      }
   }
}


void CMultinomial::ComputeWorkingResponse
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adF,
    double *adZ,
    double *adWeight,
    int *afInBag,
    unsigned long nTrain,
    int cIdxOff
)
{
    unsigned long i = 0;

    for(i=cIdxOff; i<nTrain+cIdxOff; i++)
    {
       adZ[i] = adY[i] - madProb[i];
    }

}


void CMultinomial::InitF
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adWeight,
    double &dInitF,
    unsigned long cLength
)
{
    dInitF = 0.0;
}

double CMultinomial::Deviance
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adWeight,
    double *adF,
    unsigned long cLength,
    int cIdxOff
)
{
    unsigned long ii=0;
    double dL = 0.0;
    double dW = 0.0;

    for(ii=cIdxOff; ii<cLength+cIdxOff; ii++)
    {
        dL += -adWeight[ii] * adY[ii] * std::log(madProb[ii]);
        dW += adWeight[ii];
    }

    return dL/dW;
}


void CMultinomial::FitBestConstant
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adW,
    double *adF,
    double *adZ,
    const std::vector<unsigned long>& aiNodeAssign,
    unsigned long nTrain,
    VEC_P_NODETERMINAL vecpTermNodes,
    unsigned long cTermNodes,
    unsigned long cMinObsInNode,
    int *afInBag,
    double *adFadj,
    int cIdxOff
)
{
      // Local variables
  unsigned long iNode = 0;
  unsigned long iObs = 0;
  
  // Call LocM for the array of values on each node
  for(iNode=0; iNode<cTermNodes; iNode++)
    {
      if(vecpTermNodes[iNode]->cN >= cMinObsInNode)
        {
	  // Get the number of nodes here
	  double dNum = 0.0;
	  double dDenom = 0.0;
	  for (iObs = 0; iObs < nTrain; iObs++)
	    {
	      if(afInBag[iObs] && (aiNodeAssign[iObs] == iNode))
                {
		  int iIdx = iObs + cIdxOff;
		  dNum += adW[iIdx] * adZ[iIdx];
		  dDenom += adW[iIdx] * fabs(adZ[iIdx]) * (1 - fabs(adZ[iIdx]));
                }
	    }
	  
	  dDenom = (dDenom > 0) ? dDenom : 1e-8;
	  
	  vecpTermNodes[iNode]->dPrediction = dNum / dDenom;
        }
    }
}

double CMultinomial::BagImprovement
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adWeight,
    double *adF,
    double *adFadj,
    int *afInBag,
    double dStepSize,
    unsigned long nTrain
)
{
    double dReturnValue = 0.0;
    double dW = 0.0;

   unsigned long ii;
   unsigned long kk;

   // Calculate the probabilities after the step
   std::vector<double> adStepProb(mcNumClasses * mcRows);

   // Assume that this is last class - calculate new prob as in updateParams but
   // using (F_ik + ss*Fadj_ik) instead of F_ik. Then calculate OOB improve
   for (ii = 0; ii < mcRows; ii++)
   {
      double dClassSum = 0.0;
      for (kk = 0; kk < mcNumClasses; kk++)
      {
         int iIdx = ii + kk * mcRows;
         double dF = (adOffset == NULL) ? adF[iIdx] : adF[iIdx] + adOffset[iIdx];
         dF += dStepSize * adFadj[iIdx];
         adStepProb[iIdx] = adWeight[iIdx] * std::exp(dF);
         dClassSum += adWeight[iIdx] * std::exp(dF);
      }

      dClassSum = (dClassSum > 0) ? dClassSum : 1e-8;

      for (kk = 0; kk < mcNumClasses; kk++)
      {
         adStepProb[ii + kk * mcRows] /= dClassSum;
      }
   }

   // Calculate the improvement
    for(ii=0; ii<nTrain; ii++)
    {
        if(!afInBag[ii])
      {
         for (kk = 0; kk < mcNumClasses; kk++)
         {
            int iIdx = ii + kk * mcRows;
                dReturnValue += adWeight[iIdx] * adY[iIdx] *
                               (std::log(adStepProb[iIdx]) - log(madProb[iIdx]));

            dW += adWeight[iIdx] * adY[iIdx];
         }
      }
    }

    return dReturnValue/dW;
}




