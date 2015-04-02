//------------------------------------------------------------------------------
//  GBM by Greg Ridgeway  Copyright (C) 2003
//
//  File:       gbm.h
//
//  License:    GNU GPL (version 2 or later)
//
//  Contents:   Entry point for gbm.dll
//
//  Owner:      gregr@rand.org
//
//  History:    2/14/2003   gregr created
//              6/11/2007   gregr added quantile regression
//                          written by Brian Kriegler
//
//------------------------------------------------------------------------------

#include <vector>
#include <string>
#include <memory>
#include "dataset.h"
#include "distribution.h"
#include "bernoulli.h"
#include "adaboost.h"
#include "poisson.h"
#include "gaussian.h"
#include "coxph.h"
#include "laplace.h"
#include "quantile.h"
#include "tdist.h"
#include "multinomial.h"
#include "pairwise.h"
#include "gbm_engine.h"
#include "locationm.h"
#include "huberized.h"
#include "gamma.h"
#include "tweedie.h"

std::auto_ptr<CDistribution> gbm_setup
(
    double *adY,
    double *adOffset,
    double *adX,
    int *aiXOrder,
    double *adWeight,
    double *adMisc,
    int cRows,
    int cCols,
    int *acVarClasses,
    int *alMonotoneVar,
    const std::string& family,
    int cTrees,
    int cLeaves,
    int cMinObsInNode,
    int cNumClasses,
    double dShrinkage,
    double dBagFraction,
    int cTrain,
    int cFeatures,
    CDataset *pData,
    int& cGroups
);


void gbm_transfer_to_R
(
 CGBM *pGBM,
 VEC_VEC_CATEGORIES &vecSplitCodes,
 int *aiSplitVar,
 double *adSplitPoint,
 int *aiLeftNode,
 int *aiRightNode,
 int *aiMissingNode,
 double *adErrorReduction,
 double *adWeight,
 double *adPred,
 int cCatSplitsOld
);


