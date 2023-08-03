/**
 * @file    ddbp_types.h
 * @brief   This file contains the DDBP types definition.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef _DDBP_TYPES_H
#define _DDBP_TYPES_H

#include <mpi.h>

// declare datatypes (defined elsewhere)
typedef struct _SPARC_OBJ SPARC_OBJ;
typedef struct _ATOM_NLOC_INFLUENCE_OBJ ATOM_NLOC_INFLUENCE_OBJ;
typedef struct _NLOC_PROJ_OBJ NLOC_PROJ_OBJ;
typedef struct _PSD_OBJ PSD_OBJ;
typedef struct _DDBP_ELEM DDBP_ELEM;

#define INIT_CAPACITY 4
typedef int value_type;

/**
 * @brief  Data type for dynamic array.
 */
typedef struct {
    value_type *array;
    size_t len;  // length of the array (used)
    size_t capacity; // total capacity of memory available
} dyArray;


/**
 * @brief  Data type which contains target info for performing halo exchange
 *         between DDBP neighbor elements.
 */
typedef struct _haloX_t
{
    // for all 6 neighbors (possibly coincides with itself or repeated)
    int n_neighbors; // this is currently a constant, i.e., 6
    // order of neighbors: x-, x+, y-, y+, z-, z+ ("x-" means left in x dir)
    int issend[6]; // start index of data to be sent
    int iesend[6]; // end index of data to be sent
    int jssend[6]; // start index of data to be sent
    int jesend[6]; // end index of data to be sent
    int kssend[6]; // start index of data to be sent
    int kesend[6]; // end index of data to be sent
    int isrecv[6]; // start index of data to be received
    int ierecv[6]; // end index of data to be received
    int jsrecv[6]; // start index of data to be received
    int jerecv[6]; // end index of data to be received
    int ksrecv[6]; // start index of data to be received
    int kerecv[6]; // end index of data to be received
    int neighbor_indices[6]; // TODO: for debugging purpose
    int neighbor_ranks[6]; // ranks (kptcomm) of the neighbor processes to transfer data with
    int sendcounts[6]; // the number of elements to send to neighbor i
    int recvcounts[6]; // the number of elements to receive from neighbor i
    int sdispls[6]; // the displacement (offset from sbuf) from which to send
                    // data to neighbor i
    int rdispls[6]; // the displacement (offset from rbuf) to which data from
                    // neighbor i should be written
    int stags[6]; // send tags
    int rtags[6]; // recv tags
    MPI_Datatype sendtype;
    MPI_Datatype recvtype;
    MPI_Request requests[12]; // first 6 for sending, last 6 for receiving
} haloX_t;


/**
 * @brief  Data type which contains target info for performing halo exchange of
 *         arrays expressed in the DDBP basis between DDBP neighbor elements.
 */
typedef struct _haloX_DDBP_t
{
    // for all 6 neighbors (possibly coincides with itself or repeated)
    int n_neighbors; // this is currently a constant, i.e., 6
    int neighbor_indices[6]; // TODO: for debugging purpose
    int neighbor_ranks[6]; // ranks (kptcomm) of the neighbor processes to transfer data with
    int sendcounts[6]; // the number of elements to send to neighbor i
    int recvcounts[6]; // the number of elements to receive from neighbor i
    int sdispls[6]; // the displacement (offset from sbuf) from which to send
                    // data to neighbor i
    int rdispls[6]; // the displacement (offset from rbuf) to which data from
                    // neighbor i should be written
    int stags[6]; // send tags
    int rtags[6]; // recv tags
    MPI_Datatype sendtype;
    MPI_Datatype recvtype;
    MPI_Request requests[12]; // first 6 for sending, last 6 for receiving
} haloX_DDBP_t;


typedef
struct _DDBP_VNL
{
    int Ntypes; // global number of atom types
    ATOM_NLOC_INFLUENCE_OBJ *AtmNloc; // influencing atoms
    // TODO: create a DDBP adapted nonlocal projector datatype!
    NLOC_PROJ_OBJ *nlocProj; // nonlocal projectors expressed in DDBP basis
    double dV; // integration weights for nonlocal projectors
} DDBP_VNL;


/**
 * @brief DDBP Hamiltonian element row blocks.
 * @details The DDBP Hamiltonian is a block sparse matrix, each block row
 *          corresponds to an element and its neighbors. This struct stores the
 *          nonzero blocks correspond to a row block obtained by an element.
 */
typedef
struct _DDBP_HAMILT_ERBLKS
{
    int nblks; // number of nonzero blocks (unique neighbor elements <= 6+1).
    int blksz; // size of each (square) block
    double *h_kj[7]; // nonzero blocks of the DDBP Hamiltonian (local part).
    char isserial; // is the block distributed or serial, 'T'-true, 'F'-false
    // TODO: create a DDBP adapted nonlocal projector datatype!
    // NLOC_PROJ_OBJ *nlocProj; // nonlocal projectors expressed in DDBP basis
    DDBP_VNL Vnl_DDBP;
} DDBP_HAMILT_ERBLKS;


/**
 * @brief  DDBP Hamiltonian object.
 *
 *         This is the data type that defines the DDBP Hamiltonian, i.e., the
 *         global Hamiltonian expressed in the DDBP basis. The DDBP Hamiltonian
 *         Can be defined by
 *                      H_ddbp := V^T * H * V,
 *         where H := -1/2 D^2 + Veff + Vnl is the global Hamiltonian, D^2 is
 *         Laplacian operator, V is the DDBP basis (a block-diagonal matrix).
 *           We separate the local part of H_ddbp and the nonlocal part, as we do
 *         for the global Hamiltonian. Since the local part of H_ddbp is a block
 *         sparse matrix, we only store the nonzero blocks. For the nonlocal part,
 *         we only store the projectors expressed in the DDBP basis, and perform
 *         the nonlocal operator-vector multiplication separately.
 */
typedef
struct _DDBP_HAMILTONIAN
{
    int nelem; // number of local elements
    DDBP_ELEM *elem_list; // ! (might be redundant) list of local elements
    DDBP_HAMILT_ERBLKS **H_DDBP_Ek_list; // list of rows of DDBP Hamiltonian
    // atomic info (pointer pointing to the global arrays, no extra mem allocated)
    int Ntypes; // global number of atom types
    int n_atom; // global total number of atoms
    int *nAtomv; // pointer to the global nAtomv, do not allocate/free mem
    int *localPsd; // pointer to the global localPsd, do not allocate/free mem
    int *IP_displ; // pointer to the global IP_displ, do not allocate/free mem
    PSD_OBJ *psd; // pointer to the global psd, do not allocate/free mem
} DDBP_HAMILTONIAN;


/**
 * @brief Array distributed in DDBP elements.
 * 
 */
typedef
struct _DDBP_ARRAY
{
    int BCs[3]; // global Boundary Conditions of the array
    int Edims[3]; // global number of elements in each dir
    int nelem; // number of local elements
    DDBP_ELEM *elem_list; // list of local elements
    int ncol; // number of columns of each element array
              // we assume it's the same for all elements
    int *nrows; // number of rows of each element array
                // this is equal to nALB in each element
    double **array; // list of arrays in each element
    haloX_t *haloX_info; // list of haloX info for each element
} DDBP_ARRAY;


/**
 * @brief  DDBP element structure.
 */
typedef
struct _DDBP_ELEM {
    int index; // element global index
    int coords[3]; // element global coordinates
    int EBCx; // elment BC in x dir
    int EBCy; // elment BC in y dir
    int EBCz; // elment BC in z dir
    int nALB; // total number of adaptive local basis (ALB) functions in this element
    int ALB_ns; // start global index of the basis functions
    int ALB_ne; // end global index of the basis functions
    int n_atom; // number of atoms in the element
    int n_atom_ex; // number of atoms in the extended element
    dyArray atom_list; // list of atoms (indices) in the element
    dyArray atom_list_ex; // list of atoms (indices) in the extended element

    int n_element_nbhd; // number of neighbor elements
    // DDBP_ELEM *element_nbhd[]; // list of neighbor elements
    int *element_nbhd_list; // number of neighbor elements (including itself)

    // element domain/grid
    double xs; // start coordinate of the element
    double xe; // end coordinate of the element
    double ys; // start coordinate of the element
    double ye; // end coordinate of the element
    double zs; // start coordinate of the element
    double ze; // end coordinate of the element
    double xs_sg; // start coordinate of the element Snapped to Grid
    double xe_sg; // end coordinate of the element Snapped to Grid
    double ys_sg; // start coordinate of the element Snapped to Grid
    double ye_sg; // end coordinate of the element Snapped to Grid
    double zs_sg; // start coordinate of the element Snapped to Grid
    double ze_sg; // end coordinate of the element Snapped to Grid
    int is; // start index of the element in x dir
    int ie; // end index of the element in x dir
    int js; // start index of the element in y dir
    int je; // end index of the element in y dir
    int ks; // start index of the element in z dir
    int ke; // end index of the element in z dir
    int nx; // number of grid points in the element in x dir
    int ny; // number of grid points in the element in y dir
    int nz; // number of grid points in the element in z dir
    int nd; // total number of grid points in the element
    // extended element/grid
    double buffer_x; // buffer size in x dir
    double buffer_y; // buffer size in y dir
    double buffer_z; // buffer size in z dir
    double xs_ex; // start coordinate of the extended element
    double xe_ex; // end coordinate of the extended element
    double ys_ex; // start coordinate of the extended element
    double ye_ex; // end coordinate of the extended element
    double zs_ex; // start coordinate of the extended element
    double ze_ex; // end coordinate of the extended element
    double xs_ex_sg; // start coordinate of the extended element Snapped to Grid
    double xe_ex_sg; // end coordinate of the extended element Snapped to Grid
    double ys_ex_sg; // start coordinate of the extended element Snapped to Grid
    double ye_ex_sg; // end coordinate of the extended element Snapped to Grid
    double zs_ex_sg; // start coordinate of the extended element Snapped to Grid
    double ze_ex_sg; // end coordinate of the extended element Snapped to Grid
    int is_ex; // start index of the extended element in x dir
    int ie_ex; // end index of the extended element in x dir, Caution: map back to domain for periodic BC
    int js_ex; // start index of the extended element in y dir
    int je_ex; // end index of the extended element in y dir, Caution: map back to domain for periodic BC
    int ks_ex; // start index of the extended element in z dir
    int ke_ex; // end index of the extended element in z dir, Caution: map back to domain for periodic BC
    int nx_ex; // number of grid points in the extended element in x dir
    int ny_ex; // number of grid points in the extended element in y dir
    int nz_ex; // number of grid points in the extended element in z dir
    int nd_ex; // total number of grid points in the extended element

    int DMVert_ex_topo[6]; // local extended domain vertices in elemcomm_topo for storing basis (LOCAL)
    int nx_ex_d_topo;      // gridsize of distributed extended domain in x-dir in each elemcomm_topo process (LOCAL)
    int ny_ex_d_topo;      // gridsize of distributed extended domain in y-dir in each elemcomm_topo process (LOCAL)
    int nz_ex_d_topo;      // gridsize of distributed extended domain in z-dir in each elemcomm_topo process (LOCAL)
    int nd_ex_d_topo;      // total number of grids of distributed domain in each elemcomm_topo process (LOCAL)

    // local domain (for domain paral.), note that these are relative to the extended element
    int DMVert_ex[6]; // local extended domain vertices in dmcomm for storing basis (LOCAL)
    int nx_ex_d;      // gridsize of distributed extended domain in x-dir in each dmcomm process (LOCAL)
    int ny_ex_d;      // gridsize of distributed extended domain in y-dir in each dmcomm process (LOCAL)
    int nz_ex_d;      // gridsize of distributed extended domain in z-dir in each dmcomm process (LOCAL)
    int nd_ex_d;      // total number of grids of distributed domain in each dmcomm process (LOCAL)

    int DMVert[6]; // local domain vertices in dmcomm for storing basis, relative to the extended element (LOCAL)
    int nx_d;      // gridsize of distributed domain in x-dir in each dmcomm process (LOCAL)
    int ny_d;      // gridsize of distributed domain in y-dir in each dmcomm process (LOCAL)
    int nz_d;      // gridsize of distributed domain in z-dir in each dmcomm process (LOCAL)
    int nd_d;      // total number of grids of distributed domain in each dmcomm process (LOCAL)

    double *Veff_loc_dmcomm_prev; // previous Veff in the extended element
    double *v_tilde; // basis functions in the extended element
    double *v;       // basis functions in the element (restricted to E_k and orthognalized)
    double *v_prev;  // basis functions from the previous SCF iteration
    double _Complex *v_tilde_cmplx; // basis functions (complex if not gamma-point) in the extended element
    double _Complex *v_cmplx;       // basis functions (complex if not gamma-point) in the element (restricted to E_k and orthognalized)
    double _Complex *v_prev_cmplx;  // basis functions from the previous SCF iteration
    int desc_v[9];

    double ***Mvvp; // overlap matrix of currenct basis and previous basis: Mvvp = v^T * v_prev
    int desc_Mvvp[9];

    double *Hv; // H * v (non-zero block) within the element, where H is the global Hamiltonian
    double *sendbuf; // temporary buffers for communication between elements
    double *recvbuf; // temporary buffers for communication between elements
    haloX_t haloX_Hv; // (send and recv) info for halo exchange of Hv data

    // element Hamiltonian for basis construction
    SPARC_OBJ *ESPRC; // SPARC object for each element, required for mat-vec

    // nonlocal projectors that fall in the current element (used for creating projected nloc DDBP projectors)
    ATOM_NLOC_INFLUENCE_OBJ *AtmNloc; // atom info. for atoms that have nonlocal influence on the current element (LOCAL)
    NLOC_PROJ_OBJ *nlocProj;  // nonlocal projectors in the current element (LOCAL)

    // DDBP Hamiltonian regarding this element row
    DDBP_HAMILT_ERBLKS H_DDBP_Ek;
} DDBP_ELEM;


/**
 * @brief  DDBP info type.
 */
typedef
struct _DDBP_INFO {
    // input parameters
    int Nex; // number of DDBP elements in x dir
    int Ney; // number of DDBP elements in y dir
    int Nez; // number of DDBP elements in z dir
    int Ne_tot; // total number of DDBP elements
    double buffer_x; // buffer size in x dir
    double buffer_y; // buffer size in y dir
    double buffer_z; // buffer size in z dir
    int BCx;  // global boundary condition in x dir
    int BCy;  // global boundary condition in y dir
    int BCz;  // global boundary condition in z dir
    int EBCx; // element boundary condition in x dir
    int EBCy; // element boundary condition in y dir
    int EBCz; // element boundary condition in z dir
    int nALB_atom; // number of Adaptive Local Basis (ALB) per atom

    int npelem; // number of elemcomms
    int npband; // number of bandcomms
    int npbasis; // number of basiscomms for basis generation
    int npdm; // number of dmcomm's for basis generation

    int Nstates; // number of states

    // other variables
    int nALB_tot; // total number of ALB for the simulation
    int fd_order; // finite-difference order for calculating DDBP basis
    int n_atom;
    int *atom_types; // types of each atom
    double *rcs_x; // the size of the rc box in x dir
    double *rcs_y; // the size of the rc box in y dir
    double *rcs_z; // the size of the rc box in z dir

    double Lex; // element side in x dir
    double Ley; // element side in y dir
    double Lez; // element side in z dir

    MPI_Comm elemcomm; // communicator for the parallelization of DDBP elements
    int ictxt_elemcomm; // row context
    int ictxt_elemcomm_eigentopo; // process grid context
    int elemcomm_index; // index of the current elemcomm
    int elem_start_index; // start index of the assigned elements
    int elem_end_index; // end index of the assigned elements
    int n_elem_elemcomm; // total number of elements assigned to the current elemcomm
    // int *elem_list; // list of elements assigned to the current elemcomm
    DDBP_ELEM *elem_list; // list of elements assigned to the current elemcomm

    MPI_Comm bandcomm;
    int bandcomm_index;
    int band_start_index;// start index of DDBP bands assigned to current bandcomm (LOCAL)
    int band_end_index;  // end index of DDBP bands assigned to current bandcomm (LOCAL)
    int n_band_bandcomm;

    MPI_Comm elemcomm_topo; // communicator for the domain parallelization in the current elemcomm
    int elemcomm_topo_dims[3]; // dimensions of the Cartesian topology embeded in elemcomm

    MPI_Comm elemcomm_topo_inter; // inter-communicator between elemcomm_topo and the rest in elemcomm

    MPI_Comm basiscomm; // communicator for the parallelization of DDBP basis functions
    int basiscomm_index; // index of the current basiscomm
    int basis_start_index; // start index of the assigned basis functions
    int basis_end_index; // end index of the assigned basis functions
    int n_basis_basiscomm; // total number of basis functions assigned to the current basiscomm

    MPI_Comm dmcomm; // communicator for the domain parallelization of DDBP basis functions
    int dmcomm_dims[3]; // dimensions of the Cartesian topology embeded in dmcomm

    MPI_Comm blacscomm; // communicator for the projection of element Hamiltonian
    int ictxt_blacs; // handle for ScaLAPACK context within blacscomm for original (row)
    int ictxt_blacs_topo; // handle for ScaLAPACK context within blacscomm for block-cyclic

    // Kohn-Sham eigensolver
    DDBP_HAMILTONIAN H_DDBP; // DDBP Hamiltonian
    DDBP_ARRAY **xorb; // KS orbital in DDBP basis
    DDBP_ARRAY yorb; // temp copy of KS orbital (one kpt/spin)
    int desc_xorb[9]; // descriptor for xorb (for each element)
    double ****psi; // KS orbital on original FD grid (spin,kpt,elem,i)
    int **desc_psi; // descriptor for psi (elem,i), same for all kpoints
    double **rho; // electron density distributed over elements (elem,i)
    DDBP_ARRAY Lanczos_x0; // init guess for Lanczos
    double eigmax[2]; // store extreme eigvals for Chebyshev filtering bounds
    double eigmin[2]; // store extreme eigvals for Chebyshev filtering bounds
    double *lambda; // eigenvalues
    // subspace Hamiltonian Hp, Mp
    int nr_Hp_BLCYC;
    int nc_Hp_BLCYC;
    int nr_Mp_BLCYC;
    int nc_Mp_BLCYC;
    int nr_Q_BLCYC;
    int nc_Q_BLCYC;
    int desc_Hp_BLCYC[9];
    int desc_Mp_BLCYC[9];
    int desc_Q_BLCYC[9];
    double *Hp; // subspace Hamiltonian
    double *Mp; // subspace mass matrix
    double *Q;  // eigenvectors
    double _Complex *Hp_kpt; // subspace Hamiltonian (for k-points)
    double _Complex *Mp_kpt; // subspace mass matrix (for k-points)
    double _Complex *Q_kpt;  // eigenvectors (for k-points)
} DDBP_INFO;



typedef struct _E2D_INFO {
    // global matrix size // TODO: check if this is needed during data transfer
    int gridsizes[3]; // global grid sizes for the vector
    int Ncol; // global number of columns

    // info for senders
    int is_sender; // flag to indicate if the current process is active as a sender
    int nelem; // local number of elements
    int **elem_verts; // element vertices for each element in current process (nelem,i)
    int elem_band_nstart; // (global) band start index (element distribution)
    int elem_band_nend; // (global) band end index (element distribution)
    int nproc_to_send; // number of target processes to send data to
    int *ranks_to_send; // target ranks to send data to (in union communicator)
    int *sendcounts; // #values to send to each target rank
    int *sdispls; // displacement of data in buffer for each target rank (nproc_to_send+1), last entry is the sum of sendcounts
    int *send_nstarts; // (global) column start index to be sent to each send rank (sendrank)
    int *send_nends; // (global) column end index to be sent to each send rank (sendrank)
    int *send_nelems; // nelem that overlap with local domain for send ranks (sendrank)
    int **send_elem_inds; // local element indices that overlap with local domain in send rank (sendrank,nelem_x)
    int ***send_elem_verts; // (global) vertices of the elements (overlapping part) to send ranks (sendrank,nelem,i)
    int **send_elem_displs; // displacement of data in buffer in each element data (sendrank,nelem)
    MPI_Request *send_requests;
    MPI_Datatype sendtype;
    void *sendbuf; // send data buffer

    // info for receivers
    int is_recver; // flag to indicate if the current process is active as a receiver
    int dm_band_nstart; // (global) band start index (domain distribution)
    int dm_band_nend; // (global) band end index (domain distribution)
    int dm_vert[6]; // domain vertices as a receiver, receiver always has one domain    
    int nproc_to_recv; // number of target processes to receive data from
    int *ranks_to_recv; // target ranks to receive data from
    int *recvcounts; // recv counts from each recv rank
    int *rdispls; // (nproc_to_recv + 1), the last entry is the sum
    int *recv_nstarts; // (global) column start index received from each recv rank (recvrank)
    int *recv_nends; // (global) column end index received from each recv rank (recvrank)
    int *recv_nelems; // nelem that overlap with local domain from recv ranks (recvrank)
    int **recv_elem_inds; // global element indices that overlap with local domain from recv rank (recv,nelem_x), not used
    int ***recv_elem_verts; // (global) vertices of the elements from recv ranks (recvrank,nelem,i)
    int **recv_elem_displs; // displacement of data in buffer in each element data (recvrank,nelem)
    MPI_Request *recv_requests;
    MPI_Datatype recvtype;
    void *recvbuf; // receive data buffer

    MPI_Comm union_comm; // union comm containg all the senders and receivers (can be bigger)
} E2D_INFO;


#endif // _DDBP_TYPES_H

