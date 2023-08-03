/**
* @file MCSHDescriptorMain.h
* @brief This file contains the main functions for calculating HSMP and LPMP descriptors.
*
* @author Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
*		Andrew J. Medford <ajm@gatech.edu>
*
* Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
*/
#include "MP_types.h"

/**
* @brief function to parallelize LPMP descriptor calculation across communicators based on score.
*/
void taskPartition_RadialLegendre(const int length, const double rCutoff, const int *lList, const int numParallelComm, int *taskAssignmentList);

/**
* @brief function to parallelize HSMP descriptor calculation across communicators based on score.
*/
void taskPartition_RadialRStep(const int length, const double *rCutoffList, const int *lList, const int numParallelComm, int *taskAssignmentList);

/**
* @brief main function to calculate LPMP descriptors and save it to a master csv file.
*/
void MCSHDescriptorMain_RadialLegendre(const double *rho, MULTIPOLE_OBJ *mp, const int iterNum, const int commIndex, 
					const int numParallelComm, const MPI_Comm communicator, int DMVerts[6], const int spin_type, const int nFeatures);

/**
* @brief main function to calculate HSMP descriptors and save it to a master csv file.
*/
void MCSHDescriptorMain_RadialRStep(const double *rho, MULTIPOLE_OBJ *mp, const int iterNum, const int commIndex, 
				const int numParallelComm, const MPI_Comm communicator, int DMVerts[6], const int spin_type, const int nFeatures);
