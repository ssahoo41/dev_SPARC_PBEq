/**
 * @file MCSHTools.c
 * @author Qimen Xu (qimenxu@gatech.edu)
 * @brief Some parallelization tools used in SPARC/DDBP
 *
 * @copyright Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 * 
 */

#include <complex.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <limits.h>

#include "isddft.h"
#include "parallelization.h"
#include "tools.h"
#include "MCSHTools.h"
#include "ddbp_basis.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/**
 * @brief Allgather a vector distributed in the psi-domain in SPARC.
 *
 *        This routine assumes the vector is originally distributed
 *        domain-wisely in each pSPARC->dmcomm. It first collects
 *        them into the root of each dmcomm, then broadcast the
 *        global vector.
 *        ! Note that this routine is almost the same as the function
 *        ! collect_Veff_global in ddbp/ddbp_basis.c. When the ddbp/
 *        ! is added, we should merge these two functions in one.
 *
 * @param pSPARC
 * @param vec_local Local part of the vector.
 * @param vec_global Global vector.
 */
void collect_global_vector_dmcomm(SPARC_OBJ *pSPARC, double *vec_local, double *vec_global)
{
    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start collecting global vector ... \n");
    #endif

    double t_comm = 0.0, t_bcast = 0.0;
    double t1, t2;

    int proc_active = (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) ? 0 : 1;
    int rank_kpt;
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kpt);
    int rank_dmcomm = -1;
    if (pSPARC->dmcomm != MPI_COMM_NULL)
        MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);

    // global grid size
    int Nx = pSPARC->Nx;
    int Ny = pSPARC->Ny;
    int Nz = pSPARC->Nz;
    int Nd = Nx * Ny * Nz;

    // start collecting vec_global
    if (proc_active == 1) {
        t1 = MPI_Wtime();
        MPI_Comm recv_comm = MPI_COMM_NULL;
        if (rank_dmcomm == 0) {
            int dims[3] = {1,1,1}, periods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &recv_comm);
        }
        t2 = MPI_Wtime();
        t_comm += t2 - t1;

        int gridsizes[3], sdims[3], rdims[3], rDMVert[6];
        gridsizes[0] = Nx; gridsizes[1] = Ny; gridsizes[2] = Nz;
        sdims[0] = pSPARC->npNdx;
        sdims[1] = pSPARC->npNdy;
        sdims[2] = pSPARC->npNdz;
        rdims[0] = rdims[1] = rdims[2] = 1;
        rDMVert[0] = 0; rDMVert[1] = Nx-1;
        rDMVert[2] = 0; rDMVert[3] = Ny-1;
        rDMVert[4] = 0; rDMVert[5] = Nz-1;

        D2D_OBJ d2d_sender, d2d_recvr;
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices_dmcomm, rDMVert,
            pSPARC->dmcomm, sdims, recv_comm, rdims, pSPARC->bandcomm);
        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices_dmcomm, vec_local,
            rDMVert, vec_global, pSPARC->dmcomm, sdims, recv_comm, rdims, pSPARC->bandcomm);
        Free_D2D_Target(&d2d_sender, &d2d_recvr, pSPARC->dmcomm, recv_comm);

        t1 = MPI_Wtime();
        MPI_Bcast(vec_global, Nd, MPI_DOUBLE, 0, pSPARC->dmcomm);
        t2 = MPI_Wtime();
        t_bcast += t2 - t1;

        if (rank_dmcomm == 0) MPI_Comm_free(&recv_comm);
    }

    t1 = MPI_Wtime();
    // bcast vec_global from active processes within kptcomm to idle processes (inter-comm bcast)
    intercomm_bcast(vec_global, Nd, MPI_DOUBLE, 0, pSPARC->kptcomm, proc_active);
    t2 = MPI_Wtime();
    t_bcast += t2 - t1;

    #ifdef DEBUG
    if (rank_t == 0) printf("== Setup rho ==: set up comms took %.3f ms\n", t_comm*1e3);
    if (rank_t == 0) printf("== Setup rho ==: Bcast took %.3f ms\n", t_bcast*1e3);
    #endif
}

/**
 * @brief Gather a distributed vector (domain paral) to the root process.
 * 
 * @param vec_local The local part of the vector.
 * @param DMVerts Domain vertices of the local vector.
 * @param vec_global The global vector (only used in the root process).
 * @param gridsizes The global grid sizes.
 * @param comm_topo The communicator where the vector is distributed (embeded
 *                  with a 3D Cartesian topology).
 * @param isCopy Flag to indicate whether to copy vec_local to vec_global in
 *               the special case when comm_topo has only 1 process.
 * @return int Error handle,
 *             -1: comm_topo is NULL, 0: copy sucess, 1: use vec_local directly.
 */

int gather_distributed_vector(double *vec_local, int DMVerts[6], double *vec_global, int gridsizes[3], MPI_Comm comm_topo, const int isCopy)
{
    if (comm_topo == MPI_COMM_NULL) return -1;
    int info = 0;

    // get information about the communicator (topology)
    int comm_dims[3], comm_periods[3], comm_coords[3];
    MPI_Cart_get(comm_topo, 3, comm_dims, comm_periods, comm_coords);
    int comm_size = comm_dims[0] * comm_dims[1] * comm_dims[2];
    int rank_comm;
    MPI_Comm_rank(comm_topo, &rank_comm);

    if (comm_size > 1) {
        // use D2D to collect distributed data
        int sdims[3], rdims[3], rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_comm) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int dims[3] = {1,1,1}, periods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &recv_comm);
        }
        D2D_OBJ d2d_sender, d2d_recvr;
        sdims[0] = comm_dims[0];
        sdims[1] = comm_dims[1];
        sdims[2] = comm_dims[2];
        rdims[0] = rdims[1] = rdims[2] = 1;
        rDMVert[0] = 0; rDMVert[1] = gridsizes[0]-1;
        rDMVert[2] = 0; rDMVert[3] = gridsizes[1]-1;
        rDMVert[4] = 0; rDMVert[5] = gridsizes[2]-1;

        // set up D2D targets
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, DMVerts, rDMVert, comm_topo, 
                       sdims, recv_comm, rdims, comm_topo);

        // redistribute vector through D2D
        D2D(&d2d_sender, &d2d_recvr, gridsizes, DMVerts, vec_local, rDMVert, 
            vec_global, comm_topo, sdims, recv_comm, rdims, comm_topo);

        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr,comm_topo, recv_comm);

        if (recv_comm != MPI_COMM_NULL)
            MPI_Comm_free(&recv_comm);
    } else {
        // in this case, the local vector is already the same as the global vector
        if (isCopy) { // copy vec_local into vec_global
            int Nd = gridsizes[0] * gridsizes[1] * gridsizes[2];
            memcpy(vec_global, vec_local, Nd * sizeof(*vec_local));
        } else {
            info = 1; // directly use vec_local as the global vector
        }
    }
    return info;
}

/**
 * @brief Distribute vector from the root process within a given communicator.
 * 
 * @param vec_global The global vector (only used in the root process).
 * @param gridsizes The global grid sizes.
 * @param vec_local The local part of the vector.
 * @param DMVerts Domain vertices of the local vector.

 * @param comm_topo The communicator where the vector is distributed (embeded
 *                  with a 3D Cartesian topology).
 * @param isCopy Flag to indicate whether to copy vec_local to vec_global in
 *               the special case when comm_topo has only 1 process.
 * @return int Error handle,
 *             -1: comm_topo is NULL, 0: copy sucess, 1: use vec_local directly.
 */
int distribute_global_vector(double *vec_global, int gridsizes[3], double *vec_local, int DMVerts[6], MPI_Comm comm_topo, const int isCopy)
{
    if (comm_topo == MPI_COMM_NULL) return -1;
    int info = 0;

    // get information about the communicator (topology)
    int comm_dims[3], comm_periods[3], comm_coords[3];
    MPI_Cart_get(comm_topo, 3, comm_dims, comm_periods, comm_coords);
    int comm_size = comm_dims[0] * comm_dims[1] * comm_dims[2];
    int rank_comm;
    MPI_Comm_rank(comm_topo, &rank_comm);

    if (comm_size > 1) {
        // use D2D to distribute data
        int sdims[3], rdims[3], rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_comm) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int dims[3] = {1,1,1}, periods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)

            // with all processors in dmcomm_phi?
            MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &recv_comm);
        }
        D2D_OBJ d2d_sender, d2d_recvr;
        sdims[0] = comm_dims[0];
        sdims[1] = comm_dims[1];
        sdims[2] = comm_dims[2];
        rdims[0] = rdims[1] = rdims[2] = 1;
        rDMVert[0] = 0; rDMVert[1] = gridsizes[0]-1;
        rDMVert[2] = 0; rDMVert[3] = gridsizes[1]-1;
        rDMVert[4] = 0; rDMVert[5] = gridsizes[2]-1;

        // set up D2D targets
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, DMVerts, rDMVert, comm_topo, 
                       sdims, recv_comm, rdims, comm_topo);

        // redistribute vector through D2D
        D2D(&d2d_recvr, &d2d_sender, gridsizes, rDMVert, vec_global, DMVerts, 
            vec_local, recv_comm, rdims, comm_topo, sdims, comm_topo);

        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr,comm_topo, recv_comm);

        if (recv_comm != MPI_COMM_NULL)
            MPI_Comm_free(&recv_comm);
    } else {
        // in this case, the global vector is same as local vector
        if (isCopy) { // copy vec local into vec global
            int Nd = gridsizes[0] * gridsizes[1] * gridsizes[2];
            memcpy(vec_local, vec_global, Nd * sizeof(*vec_local));
        } else {
            info = 1; // directly use vec_local as the global vector
        }
    }
    return info;
}

/**
* @brief Get the number of features in for a featcomm.
*/
int getnFeat_local(const int nFeatures, const int commIndex, const int *taskAssignmentList){
    
    int worldRank;
    int nFeat_local = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    for (int i = 0; i < nFeatures; i++){
        if (commIndex == taskAssignmentList[i]){
            nFeat_local = nFeat_local+1;
        }
    }
    return nFeat_local;
}

/**
 * @brief Get the all nFeat_local values.
 * 
 * @param nFeatures Total number of features.
 * @param taskAssignmentList The complete task assignment list.
 * @param len Length of the nFeat_localarray (= number of featcomms).
 * @param nFeat_localarray (OUTPUT) The list of local number of features for all featcomms.
 */
void get_all_nFeat_local(const int nFeatures, const int *taskAssignmentList, const int len, int *nFeat_localarray) {
    for (int i = 0; i < len; i++) {
        nFeat_localarray[i] = getnFeat_local(nFeatures, i, taskAssignmentList);
    }
}

// void getnFeat_localarray(const int nFeatures, const int commIndex, const int *taskAssignmentList, int *nFeat_localarray){

//     for (int i = 0; i < nFeatures; i++){
//         if(commIndex == taskAssignmentList[i]){
//             printf("commIndex: %d, taskAssignment: %d \n", commIndex, taskAssignmentList[i]);
//             nFeat_localarray[commIndex]++;
//         }
//     }
// }
