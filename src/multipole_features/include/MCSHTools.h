/**
 * @file MCSHTools.h
 * @author Qimen Xu (qimenxu@gatech.edu)
 * @brief Some parallelization tools used in SPARC/DDBP
 * 
 * @copyright Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 * 
 */

#ifndef MCSHTOOLS_H
#define MCSHTOOLS_H 

#include "isddft.h"
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
void collect_global_vector_dmcomm(SPARC_OBJ *pSPARC, double *vec_local, double *vec_global);

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
int gather_distributed_vector(double *vec_local, int DMVerts[6], double *vec_global, int gridsizes[3], MPI_Comm comm_topo, const int isCopy);

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
int distribute_global_vector(double *vec_global, int gridsizes[3], double *vec_local, int DMVerts[6], MPI_Comm comm_topo, const int isCopy);

/**
* @brief Get the number of features for a featcomm.
*/
int getnFeat_local(const int nFeatures, const int commIndex, const int *taskAssignmentList);

// void getnFeat_localarray(const int nFeatures, const int commIndex, const int *taskAssignmentList, int *nFeat_localarray);

/**
 * @brief Get the all nFeat_local values.
 * 
 * @param nFeatures Total number of features.
 * @param taskAssignmentList The complete task assignment list.
 * @param len Length of the nFeat_localarray (= number of featcomms).
 * @param nFeat_localarray (OUTPUT) The list of local number of features for all featcomms.
 */
void get_all_nFeat_local(const int nFeatures, const int *taskAssignmentList, const int len, int *nFeat_localarray);

#endif //MCSHTOOLS_H

