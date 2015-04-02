//------------------------------------------------------------------------------
//  GBM by Greg Ridgeway  Copyright (C) 2003
//
//  File:       gaussian.h
//
//  License:    GNU GPL (version 2 or later)
//
//  Contents:   gaussian object
//
//  Owner:      gregr@rand.org
//
//  History:    3/26/2001   gregr created
//              2/14/2003   gregr: adapted for R implementation
//
//------------------------------------------------------------------------------

#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include "distribution.h"

class CGaussian : public CDistribution
{

public:

    CGaussian();

    virtual ~CGaussian();


    void ComputeWorkingResponse(double *adY,
				double *adMisc,
				double *adOffset,
				double *adWeight,
				double *adF,
				double *adZ,
				int *afInBag,
				unsigned long nTrain,
				int cIdxOff);

    void InitF(double *adY,
	       double *adMisc,
	       double *adOffset,
	       double *adWeight,
	       double &dInitF,
	       unsigned long cLength);
    
    void FitBestConstant(double *adY,
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
			 int cIdxOff);

    double Deviance(double *adY,
                    double *adMisc,
                    double *adOffset,
                    double *adWeight,
                    double *adF,
                    unsigned long cLength,
		    int cIdxOff);

    double BagImprovement(double *adY,
                          double *adMisc,
                          double *adOffset,
                          double *adWeight,
                          double *adF,
                          double *adFadj,
                          int *afInBag,
                          double dStepSize,
                          unsigned long nTrain);
};

#endif // GAUSSIAN_H



