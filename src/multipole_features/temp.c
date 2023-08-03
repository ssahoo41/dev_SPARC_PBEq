/**
 * @brief Distribut vector from the root process within a given communicator.
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
int distribut_global_vector(
    const double *vec_global, const int gridsizes[3], double *vec_local,
    const int DMVerts[6], MPI_Comm comm_topo, const int isCopy)
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
        // in this case, the local vector is already the same as the global vector
        if (isCopy) { // copy vec_local into vec_global
            int Nd = gridsizes[0] * gridsizes[1] * gridsizes[2];
            memcpy(vec_local, vec_global, Nd * sizeof(*vec_local));
        } else {
            info = 1; // directly use vec_local as the global vector
        }
    }
    return info;
}
