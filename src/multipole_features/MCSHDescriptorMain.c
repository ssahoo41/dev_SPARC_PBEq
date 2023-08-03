/**
* @file MCSHDescriptorMain.h
* @brief This file contains the main functions for calculating HSMP and LPMP descriptors.
*
* @author Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
*		Andrew J. Medford <ajm@gatech.edu>
*
* Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
*/
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
#include <assert.h>
#include <mpi.h>
# include <math.h>
/* BLAS, LAPACK, LAPACKE routines */
#ifdef USE_MKL
    // #define MKL_Complex16 double complex
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#include "MCSHHelper.h"
#include "MCSH.h"
#include "MCSHDescriptorMain.h"
#include "MP_types.h"
#include "MCSHTools.h"

/**
* @brief function to parallelize descriptor calculation across communicators based on score.
*/
void taskPartition_RadialLegendre(const int length, const double rCutoff, const int *lList, const int numParallelComm, int *taskAssignmentList)
{	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double totalScore = 0;
	int i;
	for (i = 0; i < length; i++){	
		totalScore += scoreTask(rCutoff, lList[i]);
	}
	#ifdef DEBUG
	if (rank == 0) printf("after calc total score, total score: %f\n", totalScore);
	#endif

	double currentTotal = 0, currentRemaindingScore = totalScore;
	int currentRemaindingComm = numParallelComm, currentComm = 0;
	double currentTargetScore = currentRemaindingScore/currentRemaindingComm;

	#ifdef DEBUG
	if (rank == 0) printf("step -1, currentTotal: %f, currentRemaindingScore: %f, currentRemaindingComm: %d, currentComm: %d, currentTargetScore: %f \n", currentTotal, currentRemaindingScore, currentRemaindingComm, currentComm, currentTargetScore);	
	#endif

	for (i = 0; i < length; i++){
		taskAssignmentList[i] = currentComm;
		currentTotal += scoreTask(rCutoff, lList[i]);//, groupList[i]);
		#ifdef DEBUG
		if (rank == 0) printf("current total: %f\n", currentTotal);
		#endif
		if (currentTotal >= currentTargetScore ){
			currentComm++;
			currentRemaindingScore -= currentTotal; 
			currentRemaindingComm--; 
			currentTotal = 0;
			if (currentRemaindingScore != 0){
				currentTargetScore = currentRemaindingScore/currentRemaindingComm;
			}
		}
		#ifdef DEBUG
		if (rank == 0) printf("step %d, currentTotal: %f, currentRemaindingScore: %f, currentRemaindingComm: %d, currentComm: %d, currentTargetScore: %f \n", i, currentTotal, currentRemaindingScore, currentRemaindingComm, currentComm, currentTargetScore);
		#endif
	}	
	#ifdef DEBUG
	if (rank == 0) printf("Assigned tasks to the processors\n");
	#endif
	return;
}

/**
* @brief function to parallelize HSMP descriptor calculation across communicators based on score.
*/
void taskPartition_RadialRStep(const int length, const double *rCutoffList, const int *lList, const int numParallelComm, int *taskAssignmentList)
{	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double totalScore = 0;
	int i;
	for (i = 0; i < length; i++){	
		totalScore += scoreTask(rCutoffList[i], lList[i]);
	}
	#ifdef DEBUG
	if (rank == 0) printf("after calc total score, total score: %f\n", totalScore);
	#endif

	double currentTotal = 0, currentRemaindingScore = totalScore;
	int currentRemaindingComm = numParallelComm, currentComm = 0;
	double currentTargetScore = currentRemaindingScore/currentRemaindingComm;

	#ifdef DEBUG
	if (rank == 0) printf("step -1, currentTotal: %f, currentRemaindingScore: %f, currentRemaindingComm: %d, currentComm: %d, currentTargetScore: %f \n", currentTotal, currentRemaindingScore, currentRemaindingComm, currentComm, currentTargetScore);	
	#endif

	for (i = 0; i < length; i++){
		taskAssignmentList[i] = currentComm;
		currentTotal += scoreTask(rCutoffList[i], lList[i]);
		#ifdef DEBUG
		if (rank == 0) printf("current total: %f\n", currentTotal);
		#endif
		if (currentTotal >= currentTargetScore ){
			currentComm++;
			currentRemaindingScore -= currentTotal; 
			currentRemaindingComm--; 
			currentTotal = 0;
			if (currentRemaindingScore != 0){
				currentTargetScore = currentRemaindingScore/currentRemaindingComm;
			}
			
		}
		#ifdef DEBUG
		if (rank == 0) printf("step %d, currentTotal: %f, currentRemaindingScore: %f, currentRemaindingComm: %d, currentComm: %d, currentTargetScore: %f \n", i, currentTotal, currentRemaindingScore, currentRemaindingComm, currentComm, currentTargetScore);
		#endif
	}
	#ifdef DEBUG
	if (rank == 0) printf("Assigned tasks to the processors\n");
	#endif
	return;
}

/**
* @brief main function to calculate LPMP descriptors and save it to a master csv file.
*/
void MCSHDescriptorMain_RadialLegendre(const double *rho, MULTIPOLE_OBJ *mp, const int iterNum, const int commIndex, 
		const int numParallelComm, const MPI_Comm communicator, int DMVerts[6], const int spin_type, const int nFeatures)
{
	// Setting up rank and number of processors for featcomm_topo
	int rank, numProc;

	MPI_Comm_size(communicator, &numProc);
	MPI_Comm_rank(communicator, &rank);

	// Setting up rank for all set of processors
	int worldRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

	int imageDimX = mp->imageDimX;
	int imageDimY = mp->imageDimY;
	int imageDimZ = mp->imageDimZ;
	int imageSize = imageDimX * imageDimY * imageDimZ;
	
	int *LegendreOrderList = calloc( nFeatures, sizeof(int));
	int *lList = calloc( nFeatures, sizeof(int));
	
	getMainParameter_RadialLegendre(mp, LegendreOrderList, lList);

	if (commIndex >= 0){
		double normalizedU[9];
		normalizeU(mp->U, normalizedU);
		
		int *taskAssignmentList = calloc(nFeatures, sizeof(int));
		taskPartition_RadialLegendre(nFeatures, mp->MCSHMaxR, lList, numParallelComm, taskAssignmentList);

		// saving features to a 2D matrix
		int i, nFeat_local;
		nFeat_local = getnFeat_local(nFeatures, commIndex, taskAssignmentList);

		// define featureMatrixLocal
		double *featureMatrixLocal;
		if (rank == 0){
			featureMatrixLocal = malloc(imageSize * nFeat_local * sizeof(double));
			assert(featureMatrixLocal != NULL);
		}

		double *featureMatrixGlobal;
		// assigning memory to root processor of the communicator
		if (worldRank == 0){
			featureMatrixGlobal = malloc(imageSize * nFeatures * sizeof(double)); // allocate this to only root process of a communicator( can't do that)
			assert(featureMatrixGlobal != NULL);
		}
	
		char DescriptorFilename[128];
		
		if (worldRank == 0){
			snprintf(DescriptorFilename, 128, "LPMP_iter_%d_spin_%d_SH_%d_LP_%d_RCUT_%f.csv", iterNum, spin_type,
			mp->MCSHMaxOrder, mp->MCSHMaxRadialOrder, mp->MCSHMaxR);
			writeLPMPFirstRowToFile(DescriptorFilename, mp->MCSHMaxOrder, mp->MCSHMaxRadialOrder);
		}

		int count = 0;
		for (i = 0; i < nFeatures; i++){
		
			double *featureVectorGlobal;
			if(rank == 0){
				featureVectorGlobal = malloc(imageSize * sizeof(double));
				assert(featureVectorGlobal != NULL);
			}
			// count the numbers in taskAssignmentList, nFeature_local
			if (commIndex == taskAssignmentList[i]){
				//radial type: 2 for Legendre Polynomial
				prepareMCSHFeatureAndSave(rho, mp, mp->MCSHMaxR, lList[i], 2, LegendreOrderList[i], normalizedU, DMVerts, communicator, featureVectorGlobal);
				if (rank == 0){
					assert(count < nFeat_local);
					memcpy(featureMatrixLocal + (count*imageSize), featureVectorGlobal, imageSize*sizeof(double));
					count++;
					free(featureVectorGlobal);
				}
			}
		}
		
		int *nFeat_localarray;
		
		nFeat_localarray = calloc(numParallelComm, sizeof(int));
		assert(nFeat_localarray != NULL);
		get_all_nFeat_local(nFeatures, taskAssignmentList, numParallelComm, nFeat_localarray);

		// if (worldRank == 0){
		// 	for (int i = 0; i<numParallelComm; i++){
		// 		printf("nFeat_local: %d \n", nFeat_localarray[i]);
		// 	}
		// }

		int *displs, *rcounts;
		
		rcounts = calloc(numParallelComm, sizeof(int));
		displs = calloc(numParallelComm, sizeof(int));

		for (int i = 0; i < numParallelComm; i++){
			rcounts[i] = nFeat_localarray[i]*imageSize;
			displs[i+1] = displs[i] + rcounts[i];
		}

		MPI_Comm gather_comm;
		int color = rank;
		
		MPI_Comm_split(MPI_COMM_WORLD, color, 0, &gather_comm);

		if (rank == 0) {
			MPI_Gatherv(featureMatrixLocal, nFeat_local*imageSize, MPI_DOUBLE, featureMatrixGlobal, rcounts, displs, MPI_DOUBLE, 0, gather_comm);
		}

		if (worldRank == 0){
			writeFeatureMatrixToFile(DescriptorFilename, featureMatrixGlobal, nFeatures, imageDimX, imageDimY, imageDimZ);
		}

		if (rank == 0){
			free(featureMatrixLocal);
		}

		if (worldRank == 0){
			free(featureMatrixGlobal);
		}
		free(taskAssignmentList); free(nFeat_localarray);
	}
}

/**
* @brief main function to calculate HSMP descriptors and save it to a master csv file.
*/
void MCSHDescriptorMain_RadialRStep(const double *rho, MULTIPOLE_OBJ *mp, const int iterNum, const int commIndex, 
		const int numParallelComm, const MPI_Comm communicator, int DMVerts[6], const int spin_type, const int nFeatures)
{
	// Setting up rank and number of processors for featcomm_topo
	int rank, numProc;

	MPI_Comm_size(communicator, &numProc);
	MPI_Comm_rank(communicator, &rank);

	// Setting up rank for all set of processors
	int worldRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

	int imageDimX = mp->imageDimX;
	int imageDimY = mp->imageDimY;
	int imageDimZ = mp->imageDimZ;
	int imageSize = imageDimX * imageDimY * imageDimZ;

	double *rCutoffList = calloc( nFeatures, sizeof(double));
	int *lList = calloc( nFeatures, sizeof(int));

	getMainParameter_RadialRStep(mp, rCutoffList, lList);

	if (commIndex >= 0)
	{
		double normalizedU[9];
		normalizeU(mp->U, normalizedU);

		int *taskAssignmentList = calloc(nFeatures, sizeof(int));
		taskPartition_RadialRStep(nFeatures, rCutoffList, lList, numParallelComm, taskAssignmentList);

		// saving features to a 2D matrix
		int i, nFeat_local;
		nFeat_local = getnFeat_local(nFeatures, commIndex, taskAssignmentList);

		// define featureMatrixLocal
		double *featureMatrixLocal;
		if (rank == 0){
			featureMatrixLocal = malloc(imageSize * nFeat_local * sizeof(double));
			assert(featureMatrixLocal != NULL);
		}
		
		double *featureMatrixGlobal;
		// assigning memory to root processor of the communicator
		if (worldRank == 0){
			featureMatrixGlobal = malloc(imageSize * nFeatures * sizeof(double)); // allocate this to only root process of a communicator( can't do that)
			assert(featureMatrixGlobal != NULL);
		}
		char DescriptorFilename[128];
		
		if (worldRank == 0){
			snprintf(DescriptorFilename, 128, "HSMP_iter_%d_spin_%d_SH_%d_STEP_%f_RCUT_%f.csv", iterNum, spin_type,
			mp->MCSHMaxOrder, mp->MCSHRStepSize, mp->MCSHMaxR);
			writeHSMPFirstRowToFile(DescriptorFilename, mp->MCSHMaxOrder, mp->MCSHMaxR, mp->MCSHRStepSize);
		}

		int count = 0;

		for (i = 0; i < nFeatures; i++){
			double *featureVectorGlobal;
			
			if (commIndex == taskAssignmentList[i]){
				// radial type 1: r step
				// radial order: 0 (default, not relevant)
				prepareMCSHFeatureAndSave(rho, mp, rCutoffList[i], lList[i],
						1, 0, normalizedU, DMVerts, communicator, featureVectorGlobal);
				if (rank == 0){
					assert(count < nFeat_local);
					memcpy(featureMatrixLocal + (count*imageSize), featureVectorGlobal, imageSize*sizeof(double));
					count++;
					free(featureVectorGlobal);
				}
			}
		}
		
		int *nFeat_localarray;

		nFeat_localarray = calloc(numParallelComm, sizeof(int));
		assert(nFeat_localarray != NULL);
		get_all_nFeat_local(nFeatures, taskAssignmentList, numParallelComm, nFeat_localarray);

		// if (worldRank == 0){
		// 	for (int i = 0; i<numParallelComm; i++){
		// 		printf("nFeat_local: %d \n", nFeat_localarray[i]);
		// 	}
		// }

		int *displs, *rcounts;
		
		rcounts = calloc(numParallelComm, sizeof(int));
		displs = calloc(numParallelComm, sizeof(int));
		for (int i = 0; i < numParallelComm; i++){
			rcounts[i] = nFeat_localarray[i]*imageSize;
			displs[i+1] = displs[i] + rcounts[i];
		}

		MPI_Comm gather_comm;
		int color = rank;
		
		MPI_Comm_split(MPI_COMM_WORLD, color, 0, &gather_comm);

		if (rank == 0) {
			MPI_Gatherv(featureMatrixLocal, nFeat_local*imageSize, MPI_DOUBLE, featureMatrixGlobal, rcounts, displs, MPI_DOUBLE, 0, gather_comm);
		}

		if (worldRank == 0){
			writeFeatureMatrixToFile(DescriptorFilename, featureMatrixGlobal, nFeatures, imageDimX, imageDimY, imageDimZ);
		}

		if (rank == 0){
			free(featureMatrixLocal);
		}

		if (worldRank == 0){
			free(featureMatrixGlobal);
		}

		free(taskAssignmentList); free(nFeat_localarray);
	}
}



