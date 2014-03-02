//  GBM by Greg Ridgeway  Copyright (C) 2003

#include <limits>
#include <cassert>

#include "quantile.h"

CQuantile::~CQuantile()
{

}

struct Pair {
    double w;
    double x;
    
    Pair(double w = 0.0, double x = 0.0) : w(w),x(x) {}

    bool operator<(const Pair& o) const { return x < o.x; }
};

/* 
 * Leonid Boytsov & Anna Belova: added the weighted quantile
 * Using the following approach: 
 * http://stats.stackexchange.com/questions/13169/defining-quantiles-over-a-weighted-sample/13223#13223
 */

static double GetWeightedQuantile(
    double dAlpha, 
    double* dVal, 
    double* dWeight, 
    unsigned long cLength
) {
    if (!cLength) return numeric_limits<double>::quiet_NaN(); // One shouldn't call with empty arrays, though
    vector<Pair>  data(cLength);

    double        total = 0.0;

    for (unsigned long i = 0; i < cLength; ++i) {
        total += dWeight[i];
        data[i] = Pair(dWeight[i], dVal[i]);
    }
    total *= (cLength - 1);

    sort(data.begin(), data.end());
    double dQuantPoint = total * dAlpha;

    if (dQuantPoint <= data[0].w) return data[0].x;
    if (dQuantPoint >= total) return data[cLength - 1].x;

    double sum = 0;

    for (unsigned long k = 1; k < cLength; ++k) {
        double x1 = data[k-1].x, x2 = data[k].x;
        double w1 = data[k-1].w, w2 = data[k].w;
        double sumNext = sum + (cLength - k) * w1 + k * w2;
    
        assert(dQuantPoint);

        if (sum <= dQuantPoint && dQuantPoint <= sumNext) {
           assert(sumNext > sum);
           return x1 + (x2 - x1) *  (dQuantPoint - sum) / (sumNext - sum);
        }
        sum = sumNext;
    }

    return data[cLength - 1].x;
}






GBMRESULT CQuantile::ComputeWorkingResponse
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adF, 
    double *adZ, 
    double *adWeight,
    bool *afInBag,
    unsigned long nTrain,
	int cIdxOff
)
{
    unsigned long i = 0;
    
    if(adOffset == NULL)
    {
        for(i=0; i<nTrain; i++)
        {
            adZ[i] = (adY[i] > adF[i]) ? dAlpha : -(1.0-dAlpha);
        }
    }
    else
    {
        for(i=0; i<nTrain; i++)
        {
            adZ[i] = (adY[i] > adF[i]+adOffset[i]) ? dAlpha : -(1.0-dAlpha);
        }
    }

    return GBM_OK;
}



// DEBUG: needs weighted quantile
// Leonid Boytsov & Anna Belova: added the weighted quantile
GBMRESULT CQuantile::InitF
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adWeight,
    double &dInitF, 
    unsigned long cLength
)
{
    double dOffset=0.0;
    unsigned long i=0;
    
    vecd.resize(cLength);
    for(i=0; i<cLength; i++)
    {
        dOffset = (adOffset==NULL) ? 0.0 : adOffset[i];
        vecd[i] = adY[i] - dOffset;
    }

    if(dAlpha==1.0)
    {
        dInitF = *max_element(vecd.begin(), vecd.end());
    } else
    {
#if 0
        nth_element(vecd.begin(), vecd.begin() + int(cLength*dAlpha), vecd.end());
        dInitF = *(vecd.begin() + int(cLength*dAlpha));
#else
        dInitF = GetWeightedQuantile(dAlpha, &vecd[0], adWeight, cLength);
#endif
    }
    
    return GBM_OK;
}


double CQuantile::Deviance
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
    unsigned long i=0;
    double dL = 0.0;
    double dW = 0.0;
    
    if(adOffset == NULL)
    {
        for(i=cIdxOff; i<cLength+cIdxOff; i++)
        {
            if(adY[i] > adF[i])
            {
                dL += adWeight[i]*dAlpha      *(adY[i] - adF[i]);
            }
            else
            {
                dL += adWeight[i]*(1.0-dAlpha)*(adF[i] - adY[i]);
            }
            dW += adWeight[i];
        }
    }
    else
    {
        for(i=cIdxOff; i<cLength+cIdxOff; i++)
        {
            if(adY[i] > adF[i] + adOffset[i])
            {
                dL += adWeight[i]*dAlpha      *(adY[i] - adF[i]-adOffset[i]);
            }
            else
            {
                dL += adWeight[i]*(1.0-dAlpha)*(adF[i]+adOffset[i] - adY[i]);
            }
            dW += adWeight[i];
        }
    }

    return dL/dW;
}


// DEBUG: needs weighted quantile
// Leonid Boytsov & Anna Belova: added the weighted quantile
GBMRESULT CQuantile::FitBestConstant
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adW,
    double *adF,
    double *adZ,
    const std::vector<unsigned long> &aiNodeAssign,
    unsigned long nTrain,
    VEC_P_NODETERMINAL vecpTermNodes,
    unsigned long cTermNodes,
    unsigned long cMinObsInNode,
    bool *afInBag,
    double *adFadj,
	int cIdxOff
)
{
    GBMRESULT hr = GBM_OK;

    unsigned long iNode = 0;
    unsigned long iObs = 0;
    unsigned long iVecd = 0;
    double dOffset;
    
    vecw.resize(nTrain);
    vecd.resize(nTrain); // should already be this size from InitF
    for(iNode=0; iNode<cTermNodes; iNode++)
    {
        if(vecpTermNodes[iNode]->cN >= cMinObsInNode)
        {
            iVecd = 0;
            for(iObs=0; iObs<nTrain; iObs++)
            {
                if(afInBag[iObs] && (aiNodeAssign[iObs] == iNode))
                {
                    dOffset = (adOffset==NULL) ? 0.0 : adOffset[iObs];

                    vecd[iVecd] = adY[iObs] - dOffset - adF[iObs];
                    vecw[iVecd] = adW[iObs];
                    iVecd++;
                }
            }

            if(dAlpha==1.0)
            {
                vecpTermNodes[iNode]->dPrediction = 
                    *max_element(vecd.begin(), vecd.begin()+iVecd);
            } else
            {
#if 0
                nth_element(vecd.begin(), 
                            vecd.begin() + int(iVecd*dAlpha), 
                            vecd.begin() + int(iVecd));
                vecpTermNodes[iNode]->dPrediction = 
                    *(vecd.begin() + int(iVecd*dAlpha));
#else
                vecpTermNodes[iNode]->dPrediction = GetWeightedQuantile(dAlpha, &vecd[0], &vecw[0], iVecd);
#endif
            }
         }
    }

    return hr;
}



double CQuantile::BagImprovement
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adWeight,
    double *adF,
    double *adFadj,
    bool *afInBag,
    double dStepSize,
    unsigned long nTrain
)
{
    double dReturnValue = 0.0;

    double dF = 0.0;
    double dW = 0.0;
    unsigned long i = 0;

    for(i=0; i<nTrain; i++)
    {
        if(!afInBag[i])
        {
            dF = adF[i] + ((adOffset==NULL) ? 0.0 : adOffset[i]);
            if(adY[i] > dF)
            {
                dReturnValue += adWeight[i]*dAlpha*(adY[i]-dF);
            }
            else
            {
                dReturnValue += adWeight[i]*(1-dAlpha)*(dF-adY[i]);
            }
            
            if(adY[i] > dF+dStepSize*adFadj[i])
            {
                dReturnValue -= adWeight[i]*dAlpha*
                                (adY[i] - dF-dStepSize*adFadj[i]);
            }
            else
            {
                dReturnValue -= adWeight[i]*(1-dAlpha)*
                                (dF+dStepSize*adFadj[i] - adY[i]);
            }
            dW += adWeight[i];
        }
    }

    return dReturnValue/dW;
}

