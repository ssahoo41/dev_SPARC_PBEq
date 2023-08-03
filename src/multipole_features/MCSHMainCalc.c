/**
* @file MCSHMainCalc.c
* @brief This file sets up parallelization and calls the functions main functions to calculate the
*        Heaviside Multipole (HSMP) or Legendre Polynomial Multipole (LPMP) descriptors.
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
#include "isddft.h"
#include "MP_types.h"
#include "parallelization.h"
#include "ddbp_paral.h"
#include "MCSHTools.h"
#include "MCSHMainCalc.h"

/**
 * @brief   function to calculate grid-based multipole features from electron density after SCF calculation  
 */
void Calculate_MCSHDescriptors(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp, const int iterNum) {

    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    
    // Step 1: Intialization of parameters
    int nFeatures;
    int MCSHRadialType = pSPARC->MCSHRadialFunctionType;
    int imageSize = pSPARC->Nx * pSPARC->Ny * pSPARC->Nz;
    
    // Step 2: Calculate number of descriptors
    if (MCSHRadialType == 1) {
        nFeatures = getDescriptorListLength_RadialRStep(mp);
    }
    else if (MCSHRadialType == 2) {
        nFeatures = getDescriptorListLength_RadialLegendre(mp);
    }
    if (worldRank == 0){
        printf("Total number of features : %d \n", nFeatures);
    }
    
    // Step 3: Set up parallel communicators for the calculation of descriptors
    // * TODO: it is better to do this at the initialization stage instead
    // * of creating and destroying the communicators on-the-fly
    
    // find #featcomm's for paral. over features, and #processes for paral. over domain
    int numFeatComm, numDomainComm;
    
    dims_divide_2d(nFeatures, imageSize, worldSize, &numFeatComm, &numDomainComm);

    // overwrite these two parameters here if you want
    // numDomainComm = 1;
    // numFeatComm = worldSize / numDomainComm;
    // numFeatComm = 1;
    // numDomainComm = worldSize / numFeatComm;

    if (worldRank == 0) {
        printf("Feature process grid: numFeatComm x numDomainComm = %d x %d\n", numFeatComm, numDomainComm);
    }

    // create featcomm and then embed Cartesian topology within each featcomm for domain paral.
    MPI_Comm featcomm;
    // create featcomm's 
    // * Each featcomm is assigned a commIndex, idle processes has commIndex = -1
    int commIndex = create_subcomm(
        MPI_COMM_WORLD, &featcomm, &numFeatComm, nFeatures
    );

    int featcommSize = 0;
    if (featcomm != MPI_COMM_NULL)
        MPI_Comm_size(featcomm, &featcommSize);

    MPI_Comm featcomm_topo;
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int periods[3] = {1-pSPARC->BCx, 1-pSPARC->BCy, 1-pSPARC->BCz};
    int topo_dims[3] = {-1,-1,-1}; // ask the routine to find out the dims
    // embed a Cartesian topology (process grid) in each featcomm
    create_dmcomm(
        featcomm, commIndex, gridsizes, periods,
        1, topo_dims, &featcomm_topo, featcommSize
    );
    if (worldRank == 0) {
        printf("Each DomainComm (size: %d) is embeded a %d x %d x %d topology\n",
            numDomainComm, topo_dims[0], topo_dims[1], topo_dims[2]);
    }
    
    // assign subgrid to each process in featcomm_topo
    int DMVerts[6] = {0,0,0,0,0,0};
    assign_task_Cart(
        featcomm_topo, 3, topo_dims, gridsizes, DMVerts
    );


    int n_rho = pSPARC->Nspden/2*2+1;
    int length = pSPARC->Nd * n_rho;

    if (worldRank == 0){
        printf("Nd: %d, Nspden : %d, n_rho: %d, length: %d\n", pSPARC->Nd, pSPARC->Nspden, n_rho, length);
    }
    
    // memory allocation is required in all processors
    double *global_rho = malloc(length * sizeof(*global_rho));
    assert(global_rho != NULL);
    // collect the distributed rho data together and broadcast to all processes    

    gather_distributed_vector(pSPARC->electronDens, pSPARC->DMVertices, global_rho, gridsizes, pSPARC->dmcomm_phi, 1);
    MPI_Bcast( global_rho, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // test distribute gathered vector
    
// #define DEBUGDIST
#ifdef DEBUGDIST
	#define max(a,b) ((a)>(b)?(a):(b))
    double *distributed_rho = malloc(pSPARC->Nd_d * sizeof(*distributed_rho));
    distribute_global_vector(global_rho, gridsizes, distributed_rho, pSPARC->DMVertices, pSPARC->dmcomm_phi, 1);

	int is = pSPARC->DMVertices[0];
	int ie = pSPARC->DMVertices[1];
	int js = pSPARC->DMVertices[2];
	int je = pSPARC->DMVertices[3];
	int ks = pSPARC->DMVertices[4];
	int ke = pSPARC->DMVertices[5];
	// check convolve6 answer
	double err = 0.0;
	int outputIndex = 0;
	for (int k = ks; k <= ke; k++) {
		for (int j = js; j <= je; j++) {
			for (int i = is; i <= ie; i++) {
				err = max(fabs(pSPARC->electronDens[outputIndex] - distributed_rho[outputIndex]), err);
				outputIndex++;
			}
		}
	}
	assert(err < 1e-12);
	printf("Test passed!\n");
#endif

    // // write the density here to files
    // if (worldRank == 0){
    //     printDensity(pSPARC, global_rho, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    // }

    // calculate descriptors based on radial type and spin polarization
    double t1, t2;
    t1 = MPI_Wtime();

    // if MCSHRadialType == 1, call the function for MCSHHSMP and check spin polarization
    if (MCSHRadialType == 1){
        if (worldRank == 0)
        {
            printf("\n nFeatures: %d, Number of feature groups: %d, each contains a %d x %d x %d domain group.",
                nFeatures, numFeatComm, topo_dims[0], topo_dims[1], topo_dims[2]);
        }
        CalculateHSMPDescriptors(pSPARC, mp, iterNum, global_rho, commIndex, numFeatComm, featcomm_topo, DMVerts, nFeatures);
    }
    else if (MCSHRadialType == 2){
        if (worldRank == 0){
            printf("\n nFeatures: %d, Number of feature groups: %d, each contains a %d x %d x %d domain group.",
                    nFeatures, numFeatComm, topo_dims[0], topo_dims[1], topo_dims[2]);
        }
        CalculateLPMPDescriptors(pSPARC, mp, iterNum, global_rho, commIndex, numFeatComm, featcomm_topo, DMVerts, nFeatures);
    }
    else
    {
        printf("\nWARNING: Radial function type NOT recognized, multipole descriptor not calculated \n");
    }
    t2 = MPI_Wtime();
    if (worldRank == 0)
        printf("\n **** MCSH calculation took took: %f ms\n", (t2 - t1)*1000);

    free(global_rho);
    
    if (featcomm != MPI_COMM_NULL)
        MPI_Comm_free(&featcomm);
    if (featcomm_topo != MPI_COMM_NULL)
        MPI_Comm_free(&featcomm_topo);

}

/**
* @brief function to initialize structure for multipole feature calculation
*/
void Multipole_Initialize(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp) {

    mp->imageDimX = pSPARC->Nx;
    mp->imageDimY = pSPARC->Ny;
    mp->imageDimZ = pSPARC->Nz;

    mp->hx = pSPARC->delta_x;
    mp->hy = pSPARC->delta_y;
    mp->hz = pSPARC->delta_z;

    mp->MCSHMaxOrder = pSPARC->MCSHMaxMCSHOrder;
    mp->MCSHMaxR = pSPARC->MCSHMaxRCutoff;

    mp->accuracy = 6;
    double *Urow = pSPARC->LatUVec;
    double Ucol[9] = {Urow[0],Urow[3],Urow[6],Urow[1],Urow[4],Urow[7],Urow[2],Urow[5],Urow[8]};
    for(int i = 0; i < 9; i++)
        mp->U[i] = Ucol[i];

    if (pSPARC->MCSHRadialFunctionType == 1){
        mp->MCSHRStepSize = pSPARC->MCSHRStepSize;
    }
    else if (pSPARC->MCSHRadialFunctionType == 2){
        mp->MCSHMaxRadialOrder = pSPARC->MCSHRadialFunctionMaxOrder;
    }
    else {
        printf("\nWARNING: Radial function type NOT recognized, descriptors cannot be calculated \n");
    }
}

/**
 * @brief   function to collect distributed rho vector into a global vector
 */
void CollectElecDens(SPARC_OBJ *pSPARC, double *global_rho, double *global_psdchrgdens) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int nproc_dmcomm_phi, rank_dmcomm_phi, DMnd, i, j, k, index;
    MPI_Comm_size(pSPARC->dmcomm_phi, &nproc_dmcomm_phi);
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank_dmcomm_phi);
    
    int Nd = pSPARC->Nd;
    DMnd = pSPARC->Nd_d;
    
    double *rho, *b;
    rho = NULL;
    b = NULL;
    if (nproc_dmcomm_phi > 1) { // if there's more than one process, need to collect rho first
        // use DD2DD to collect distributed data
        int gridsizes[3], sdims[3], rdims[3], rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_dmcomm_phi) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int dims[3] = {1,1,1}, periods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &recv_comm);
        }
        D2D_OBJ d2d_sender, d2d_recvr;
        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;
        sdims[0] = pSPARC->npNdx_phi;
        sdims[1] = pSPARC->npNdy_phi;
        sdims[2] = pSPARC->npNdz_phi;
        rdims[0] = rdims[1] = rdims[2] = 1;
        rDMVert[0] = 0; rDMVert[1] = pSPARC->Nx-1;
        rDMVert[2] = 0; rDMVert[3] = pSPARC->Ny-1;
        rDMVert[4] = 0; rDMVert[5] = pSPARC->Nz-1;
        
        // set up D2D targets
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, rDMVert, pSPARC->dmcomm_phi, 
                       sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        if (rank_dmcomm_phi == 0) {
            int n_rho = pSPARC->Nspden/2*2+1;
            rho = (double*)malloc(pSPARC->Nd * n_rho * sizeof(double)); // allocating memory in only one processor
            b      = (double*)malloc(pSPARC->Nd * sizeof(double));
        }
        
        // send rho

        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->electronDens, rDMVert, 
            rho, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        
        if (pSPARC->Nspden > 1) { // send rho_up, rho_down
            D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->electronDens+DMnd, rDMVert, 
                rho+Nd, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
            D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->electronDens+2*DMnd, rDMVert, 
                rho+2*Nd, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);
        }
        
        // send b
        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices, pSPARC->psdChrgDens, rDMVert, 
            b, pSPARC->dmcomm_phi, sdims, recv_comm, rdims, pSPARC->dmcomm_phi);

        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, pSPARC->dmcomm_phi, recv_comm);
        if (rank_dmcomm_phi == 0) 
            MPI_Comm_free(&recv_comm);
    } else {
        rho = pSPARC->electronDens;
        b = pSPARC->psdChrgDens;
    }

    int n_rho = pSPARC->Nspden/2*2+1;
    int length = pSPARC->Nd * n_rho;
    if (rank_dmcomm_phi == 0)
    {
        memcpy(global_rho, rho, length * sizeof(double));
        memcpy(global_psdchrgdens, b, pSPARC->Nd * sizeof(double));
    }

    // free the collected data after printing to file
    if (nproc_dmcomm_phi > 1) {
        if (rank_dmcomm_phi == 0) {
            free(rho);
            free(b);
        }
    }
}

/**
 * @brief   function to calculate HSMP descriptors
 */
void CalculateHSMPDescriptors(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp, const int iterNum, double *elecDens, const int commIndex, const int numParallelComm,
                             const MPI_Comm communicator, int DMVerts[6], const int nFeatures){

    int imageSize = mp->imageDimX * mp->imageDimY * mp->imageDimZ;

    int spin_type;
    if (pSPARC->spin_typ == 0) {
        spin_type = 0;
        MCSHDescriptorMain_RadialRStep(elecDens, mp, iterNum, commIndex, numParallelComm,
            communicator, DMVerts, spin_type, nFeatures
            );
    }
    else if (pSPARC->spin_typ == 1) {
        for (spin_type = 1; spin_type < pSPARC->Nspden+1; spin_type++) {
            MCSHDescriptorMain_RadialRStep(elecDens+spin_type*imageSize, mp, iterNum,
            commIndex, numParallelComm, communicator, DMVerts, spin_type, nFeatures);
        }
    }
}

/**
 * @brief   function to calculate LPMP descriptors
 */
void CalculateLPMPDescriptors(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp, const int iterNum, double *elecDens, const int commIndex, 
                                const int numParallelComm, const MPI_Comm communicator, int DMVerts[6], const int nFeatures){

    int imageSize = mp->imageDimX * mp->imageDimY * mp->imageDimZ;

    int spin_type;
    
    if (pSPARC->spin_typ == 0) {
        
        spin_type = 0;

        MCSHDescriptorMain_RadialLegendre(elecDens, mp, iterNum, commIndex, numParallelComm,
            communicator, DMVerts, spin_type, nFeatures);

    }
    else if (pSPARC->spin_typ == 1) {
       
        for (spin_type = 1; spin_type < pSPARC->Nspden+1; spin_type++) {

            MCSHDescriptorMain_RadialLegendre(elecDens+spin_type*imageSize, mp, iterNum,
            commIndex, numParallelComm, communicator, DMVerts, spin_type, nFeatures
            );
        }
    }
}
