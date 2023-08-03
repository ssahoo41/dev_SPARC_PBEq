/**
 * @file    ddbp.h
 * @brief   This file contains the function declarations for the Discrete
 *          Discontinuous Basis Projection (DDBP) method.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef _DDBP_PARAL_H
#define _DDBP_PARAL_H

#include "isddft.h"



/**
 * @brief   Set up sub-communicators for DDBP method.
 *
 */
void Setup_Comms_DDBP(SPARC_OBJ *pSPARC);


/**
 * @brief   Set up Cartesian topology for domain parallelization of DDBP basis.
 *
 *          Need to do this after init_DDBP, i.e., after the grids are defined for the
 *          extended elements.
 */
void Embed_Cart_DDBP(SPARC_OBJ *pSPARC);


/**
 * @brief Allocate memory or initialize variables for DDBP.
 */
void allocate_init_DDBP_variables(SPARC_OBJ *pSPARC);


/**
 * @brief   Set up a Cartesian topology in a communicator for domain decomposition
 *          according to the given grid and dimensions (if given).
 *
 * @param comm        Given communicator to be embeded.
 * @param comm_index  Index corresponding to comm, if set to -1, this comm is not embeded
 * @param gridsizes   Number of grids in all three dimentions.
 * @param periods     Periodic flags, 1 - periodic, 0 - not periodic.
 * @param minsize     Minimum number of grids each process must contain in each dimention.
 * @param dims        The dimension of the Cartesian topology, if set to {-1,-1,-1}, will
 *                    be found automatically. (Input/Output)
 * @param subcomm     Subcommunicators. (Output)
 * @param nsubcomm    Maximum number of subcomms to create, might not be used in full.
 */
void create_dmcomm(
    const MPI_Comm comm, const int comm_index, const int gridsizes[3], const int periods[3],
    const int minsize, int dims[3], MPI_Comm *subcomm, int nsubcomm);



/**
 * @brief Split comm into nsubcomm subcomm's based over N objects.
 * @param comm        Given communicator to be splitted.
 * @param subcomm     Sub-communicators.
 * @param nsubcomm    Maximum number of subcomms to be created, might not be used in full,
 *                    if set to <= 0, will be changed to min(nproc,n_obj).
 * @param n_obj       Number of objects to be distributed across the subcomms.
 **/
int create_subcomm(const MPI_Comm comm, MPI_Comm *subcomm, int *nsubcomm, const int n_obj);


/**
 * @brief Create a bridge comm between subcomm's within a comm.
 *
 * @details Let comm be a communicator, and subcomm's are subcommunicators
 *          within comm. We create bridge communicators between all processes
 *          with the same rank in each subcomm.
 *
 * @param comm The global communicator that includes all the subcomm's.
 * @param subcomm The subcomm this process belongs to.
 * @param bridge_comm The brdige communicator between subcomm's (output).
 */
int create_bridge_comm(MPI_Comm comm, MPI_Comm subcomm, MPI_Comm *bridge_comm);


/**
 * @brief   Set up an inter-communicator between a sub-communicator
 *          within a communicator and the communicator including the
 *          rest processes.
 *
 * @param comm    A communicator.
 * @param subcomm A sub-communicator in the communicator containing ranks 0,
 *                1,...,nproc_sumbcomm-1, if a process in comm doesn't belong
 *                to subcomm, it should give MPI_COMM_NULL.
 * @param inter   The inter-communicator to be created.
 */
void create_Cart_inter_comm(MPI_Comm comm, MPI_Comm subcomm, MPI_Comm *inter);


/**
 * @brief Assign objects to nsubcomm subcomm's.
 * @param nsubcomm         Number of subcomms.
 * @param n_obj            Number of objects to be distributed across the subcomms.
 * @param subcomm_index    Index of the current subcomm (LOCAL).
 * @param obj_start_index  (OUTPUT) Start index of the assigned objects (LOCAL).
 * @param obj_end_index    (OUTPUT) End index of the assigned objects (LOCAL).
 * @param n_obj_subcomm    (OUTPUT) Number of the assigned objects (LOCAL).
 **/
void assign_task_subcomm(
    const int n_obj, const int nsubcomm, const int subcomm_index,
    int *obj_start_index, int *obj_end_index, int *n_obj_subcomm);


/**
 * @brief Assign objects to a communicator with Cartesian topology.
 *
 * @param comm             Communicator with Cartesian topology.
 * @param ndims            Number of dimensions of Cartesian grid.
 * @param dims             Number of processes in each dimension.
 * @param gridsizes        Number of grids/objects in each dimension.
 * @param DMvert (OUTPUT)  The vertices of the domain assigned to the current process.
 *
 **/
void assign_task_Cart(
    MPI_Comm comm, int ndims, const int dims[], const int gridsizes[], int DMvert[]
);


/**
 * @brief   Calculate number of rows/cols of a distributed array owned by
 *          the process (in one direction) using the block-cyclic way,
 *          except we force no cyclic.
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int block_decompose_BLCYC_fashion(const int n, const int p, const int rank);


/**
 * @brief   Calculate the start index of a distributed array owned by
 *          the process (in one direction) using the block-cyclic way,
 *          except we force no cyclic.
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int block_decompose_nstart_BLCYC_fashion(const int n, const int p, const int rank);


/**
 * @brief   Calculate which process owns the provided node of a distributed
 *          array (in one direction) using the block-cyclic way, except we
 *          force no cyclic.
 *
 * @param n           Number of nodes in the given direction of the global domain.
 * @param p           Total number of processes in the given direction of the
 *                    process topology.
 * @param node_index  Node index.
 */
int block_decompose_rank_BLCYC_fashion(
    const int n, const int p, const int node_indx
);


/**
 * @brief Assign objects to nsubcomm subcomm's using the block-cyclic fashion, except
 *        force no cyclic, but only block. This is slightly different from the block
 *        decomposition we do for domain parallelization.
 *
 * @param nsubcomm         Number of subcomms.
 * @param n_obj            Number of objects to be distributed across the subcomms.
 * @param subcomm_index    Index of the current subcomm (LOCAL).
 * @param obj_start_index  (OUTPUT) Start index of the assigned objects (LOCAL).
 * @param obj_end_index    (OUTPUT) End index of the assigned objects (LOCAL).
 * @param n_obj_subcomm    (OUTPUT) Number of the assigned objects (LOCAL).
 **/
void assign_task_BLCYC_fashion(
    const int n_obj, const int nsubcomm, const int subcomm_index,
    int *obj_start_index, int *obj_end_index, int *n_obj_subcomm
);


/**
 * @brief   Given the index of an element E_k, find which elemcomm it's assigned to.
 *
 *          This is accompanied by the routine above to assign elements. If the way
 *          to assign elements is changed, this has to be modified so that they're
 *          consistent.
 *
 * @param nelem Total number of elements.
 * @param npelem Number of element comms.
 * @param k Global index of an element.
 * @return int Elemcomm_index.
 */
int element_index_to_elemcomm_index(int nelem, int npelem, int k);


/**
 * @brief   Given the index of a basis, find which basiscomm it's assigned to.
 *
 *          This is accompanied by the routine above to assign basis. If the way
 *          to assign basis is changed, this has to be modified so that they're
 *          consistent. Note that we assume all elements have the same number of
 *          basis functions, therefore communication is not required.
 *
 * @param DDBP_info  Pointer to a DDBP_INFO object.
 * @param k          Global index of an element. (Currently unused.)
 * @param n          Index of a basis within element E_k.
 */
int basis_index_to_basiscomm_index(DDBP_INFO *DDBP_info, int k, int n);


/**
 * @brief   Given the indices of elecomm, basiscomm, dmcomm, determine the
 *          rank of the corresponding process within kptcomm.
 *
 *          This is accompanied by the routines to create elemcomm, basiscomm,
 *          and dmcomm. If the way these sub-communicators are changed, this has
 *          to be modified so that they're consistent. Note that we assume no
 *          domain paral, so dmcomm only contains a single process.
 *
 * @param DDBP_info       Pointer to a DDBP_INFO object.
 * @param elemcomm_index  Index of elemcomm.
 * @param basiscomm_index Index of basiscomm.
 * @param dmcomm_rank     Rank of process in the dmcomm.
 */
int indices_to_rank_kptcomm(
    DDBP_INFO *DDBP_info, int elemcomm_index, int basiscomm_index,
    int dmcomm_rank, MPI_Comm kptcomm
);


/**
 * @brief Set the up haloX info for DDBP array.
 * 
 * @param DDBP_info DDBP_info object.
 * @param X DDBP array.
 * @param bandcomm Communicator where X is distributed. // TODO: not used anymore
 */
void setup_haloX_DDBP_Array(DDBP_INFO *DDBP_info, DDBP_ARRAY *X, MPI_Comm bandcomm);


/**
 * @brief   Start the non-blocking data transfer (halo exchange) between neighbor
 *          elements.
 *
 * @param sendbuf  The buffer array with data that will be sent out.
 * @param recvbuf  The buffer array which will be used to receive data.
 * @param kptcomm  The global kptcomm that includes all the elemcomm's.
 */
void DDBP_element_Ineighbor_alltoallv(
    haloX_t *haloX, const void *sendbuf, void *recvbuf, MPI_Comm kptcomm);


/**
 * @brief   Start the REMOTE non-blocking data transfer (halo exchange) between
 *          neighbor elements.
 *
 *          This routine only starts the MPI_Isend and MPI_Irecv if the targe
 *          is not the process itself. It SKIPS all the communication to/from
 *          itself and leaves that part of data untouched.
 *
 * @param sendbuf  The buffer array with data that will be sent out.
 * @param recvbuf  The buffer array which will be used to receive data.
 * @param kptcomm  The global kptcomm that includes all the elemcomm's.
 */
void DDBP_remote_element_Ineighbor_alltoallv(
    haloX_t *haloX, const void *sendbuf, void *recvbuf, MPI_Comm kptcomm);


/**
 * @brief   Start the LOCAL data transfer (halo exchange) between neighbor
 *          elements.
 *
 *          This routine only performs the data transfer if the target process
 *          is the process itself. This routine is to be used together with
 *          its REMOTE counterpart
 *              DDBP_remote_element_Ineighbor_alltoallv().
 *
 *
 * @param sendbuf  The buffer array with data that will be sent out.
 * @param recvbuf  The buffer array which will be used to receive data.
 * @param kptcomm  The global kptcomm that includes all the elemcomm's.
 */
void DDBP_local_element_Ineighbor_alltoallv(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, haloX_t *haloX,
    const void *sendbuf, void *recvbuf, MPI_Comm kptcomm);


/**
 * @brief Initialize data structure for element parallelization to
 *        domain parallelization data transfer.
 */
void E2D_Init(E2D_INFO *E2D_info, int *Edims, int nelem, DDBP_ELEM *elem_list,
    int *gridsizes, int *BCs, int Ncol,
    int send_ns, int send_ncol, MPI_Comm s_rowcomm, int s_rowsize, int s_rowcomm_index,
    MPI_Comm s_colcomm, int s_colsize, int s_colcomm_index,
    int recv_ns, int recv_ncol, int *recv_DMVerts, MPI_Comm r_rowcomm, int r_rowsize, MPI_Comm r_colcomm,
    int *r_coldims, int r_colcomm_index,
    MPI_Comm union_comm);

// start to transfer element distribution to domain distribution, non-blocking
// * set up sendbuf and recvbuf
// * initialize the MPI_Isend and MPI_Irecv
void E2D_Iexec(E2D_INFO *E2D_info, const void **sdata);

// wait for nonblocking send and receive in E2D to be completed before moving on
void E2D_Wait(E2D_INFO *E2D_info, void *rdata);

// call E2D_Iexec and E2D_Wait (non-blocking)
void E2D_Exec(E2D_INFO *E2D_info, const void **sdata, void *rdata);

// free E2D_info
void E2D_Finalize(E2D_INFO *E2D_info);

// non-blocking routine that packs the above routines in one function
void E2D(E2D_INFO *E2D_info, int *Edims, int nelem, DDBP_ELEM *elem_list,
    int *gridsizes, int *BCs, int Ncol, const void **sdata, int send_ns,
    int send_ncol, MPI_Comm s_rowcomm, int s_rowsize, int s_rowcomm_index,
    MPI_Comm s_colcomm, int s_colsize, int s_colcomm_index,
    void *rdata, int recv_ns, int recv_ncol, int *recv_DMVerts,
    MPI_Comm r_rowcomm, int r_rowsize, MPI_Comm r_colcomm, int *r_coldims, int r_colcomm_index,
    MPI_Comm union_comm);

#endif // _DDBP_PARAL_H

