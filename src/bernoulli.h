//------------------------------------------------------------------------------
//  GBM by Greg Ridgeway  Copyright (C) 2003
//
//  File:       bernoulli.h
//
//  License:    GNU GPL (version 2 or later)
//
//  Contents:   bernoulli object
//
//  Owner:      gregr@rand.org
//
//  History:    3/26/2001   gregr created
//              2/14/2003   gregr: adapted for R implementation
//
//------------------------------------------------------------------------------

#ifndef BERNOULLI_H
#define BERNOULLI_H

#include "distribution.h"
#include "buildinfo.h"

class CBernoulli : public CDistribution
{

public:

    CBernoulli();

    virtual ~CBernoulli();

    void ComputeWorkingResponse(double *adY,
				double *adMisc,
				double *adOffset,
				double *adF,
				double *adZ,
				double *adWeight,
				int *afInBag,
				unsigned long nTrain,
				int cIdxOff);

    double Deviance(double *adY,
                    double *adMisc,
                    double *adOffset,
                    double *adWeight,
                    double *adF,
                    unsigned long cLength,
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
    
    double BagImprovement(double *adY,
                          double *adMisc,
                          double *adOffset,
                          double *adWeight,
                          double *adF,
                          double *adFadj,
                          int *afInBag,
                          double dStepSize,
                          unsigned long nTrain);

private:
    vector<double> vecdNum;
    vector<double> vecdDen;
};

#endif // BERNOULLI_H



