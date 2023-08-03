/**
 * @file    cyclix_forces.c
 * @brief   This file contains functions for force calculation in systems with cyclix geometry
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */


#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

/* BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

// this is for checking existence of files
#include "cyclix_forces.h"
#include "cyclix_tools.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "lapVecRoutines.h"
#include "lapVecRoutinesKpt.h"
#include "tools.h" 
#include "isddft.h"
#include "initialization.h"
#include "electrostatics.h"

#define TEMP_TOL 1e-12

/**
 * @brief    Calculate nonlocal force components for cyclix system with gamma point.
 */
void Calculate_nonlocal_forces_cyclix(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, n, np, ldispl, ndc, ityp, iat, ncol, DMnd, atom_index, count, l, m, lmax, spn_i, nspin, size_s;
    nspin = pSPARC->Nspin_spincomm; // number of spin in my spin communicator
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    size_s = ncol * DMnd;
    double *force_nloc, *alpha, *beta, *x_ptr, *dx_ptr, *dx_ptr1, *x_rc, *dx_rc, *dx_rc1, *x_rc_ptr, *dx_rc_ptr, *dx_rc_ptr1;
    double fJ_x, fJ_y, fJ_z, val_x, val_y, val_z, val2_x, val2_y, val2_z, g_nk, *beta_x, *beta_y,
           *beta_z;
    double *Y2 = (double *) malloc( size_s * sizeof(double));
    
    force_nloc = (double *)calloc(3 * pSPARC->n_atom, sizeof(double));
    alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 4, sizeof(double));
    double x0_i, y0_i, z0_i, y0, z0;
    int szp;
    double ty, tz;

#ifdef DEBUG 
    if (!rank) printf("Start Calculating nonlocal forces\n");
#endif
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * count;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            //lmax = pSPARC->psd[ityp].lmax;
            if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
            for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat]; 
                x_rc = (double *)malloc( ndc * ncol * sizeof(double));
                atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                
                /* first find inner product <Psi_n, Chi_Jlm>, here we calculate <Chi_Jlm, Psi_n> instead */
                
                for (n = 0; n < ncol; n++) {
                    x_ptr = pSPARC->Xorb + spn_i * size_s + n * DMnd;
                    x_rc_ptr = x_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        // x_rc[n*ndc+i] = pSPARC->Xorb[n*DMnd+pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                        *(x_rc_ptr + i) = *(x_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                    }
                }
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi_cyclix[iat], ndc, 
                            x_rc, ndc, 1.0, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);
                free(x_rc);
            }
        }
        count++;
    }    
    
    /* find inner product <Chi_Jlm, dPsi_n> */
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        // find dPsi in direction dim
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, pSPARC->Yorb, 0, pSPARC->dmcomm);
        beta_x = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (nspin * 1 + count);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, Y2, 1, pSPARC->dmcomm);
        beta_y = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (nspin * 2 + count);
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            szp = pSPARC->nlocProj[ityp].nproj * ncol;
            double *PX1 = (double *) malloc( szp * sizeof(double));
            double *PX2 = (double *) malloc( szp * sizeof(double));
                
            if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
            for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                x0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
                y0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                z0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                    
                ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat]; 
                dx_rc = (double *)malloc( ndc * ncol * sizeof(double));
                dx_rc1 = (double *)malloc( ndc * ncol * sizeof(double));
                atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                for (n = 0; n < ncol; n++) {
                    dx_ptr = pSPARC->Yorb + n * DMnd;
                    dx_ptr1 = Y2 + n * DMnd;
                    dx_rc_ptr = dx_rc + n * ndc;
                    dx_rc_ptr1 = dx_rc1 + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                        *(dx_rc_ptr1 + i) = *(dx_ptr1 + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                    }
                }
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi_cyclix[iat], ndc, 
                            dx_rc, ndc, 0.0, PX1, pSPARC->nlocProj[ityp].nproj); 
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi_cyclix[iat], ndc, 
                            dx_rc1, ndc, 0.0, PX2, pSPARC->nlocProj[ityp].nproj); 
                
                // Apply the rotation matrix
                y0 = pSPARC->atom_pos[3*atom_index+1];
                z0 = pSPARC->atom_pos[3*atom_index+2];
                ty = (y0 - y0_i)/pSPARC->range_y;
                tz = (z0 - z0_i)/pSPARC->range_z;
                RotMat_cyclix(pSPARC, ty, tz);
                for(i = 0; i < szp; i++) {
                    Rotate_vector_cyclix(pSPARC, PX1 + i, PX2 + i);
                    beta_x[pSPARC->IP_displ[atom_index]*ncol + i] += PX1[i];
                    beta_y[pSPARC->IP_displ[atom_index]*ncol + i] += PX2[i];
                }
                free(dx_rc);
                free(dx_rc1);
            }
            free(PX1);
            free(PX2);
        }
        count++;
    }

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        // find dPsi in direction dim
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, pSPARC->Yorb, 2, pSPARC->dmcomm);
        beta_z = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (nspin * 3 + count);
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
            for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {    
                ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat]; 
                dx_rc = (double *)malloc( ndc * ncol * sizeof(double));
                atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                for (n = 0; n < ncol; n++) {
                    dx_ptr = pSPARC->Yorb + n * DMnd;
                    dx_rc_ptr = dx_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                    }
                }
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi_cyclix[iat], ndc, 
                            dx_rc, ndc, 1.0, beta_z + pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);
                free(dx_rc);
            }
        }
        count++;
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 4, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal force */
    // go over all atoms and find nonlocal force components
    beta_x = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin;
    beta_y = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 2;
    beta_z = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 3;
    count = 0; atom_index = 0;
    double spn_fac;


    for(spn_i = 0; spn_i < nspin; spn_i++) {
        atom_index = 0;
        spn_fac = 2.0/pSPARC->Nspin * 2.0;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            int lloc = pSPARC->localPsd[ityp];
            lmax = pSPARC->psd[ityp].lmax;
            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                fJ_x = fJ_y = fJ_z = 0.0;
                //alpha_J = alpha + pSPARC->IP_displ[atom_index]*ncol;
                //beta_Jx = beta_x + pSPARC->IP_displ[atom_index]*ncol;
                //beta_Jy = beta_y + pSPARC->IP_displ[atom_index]*ncol;
                //beta_Jz = beta_z + pSPARC->IP_displ[atom_index]*ncol;
                for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                    g_nk = pSPARC->occ[spn_i*pSPARC->Nstates+n];
                    val2_x = val2_y = val2_z = 0.0;
                    ldispl = 0;
                    for (l = 0; l <= lmax; l++) {
                        // skip the local l
                        if (l == lloc) {
                            ldispl += pSPARC->psd[ityp].ppl[l];
                            continue;
                        }
                        for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                            val_x = val_y = val_z = 0.0;
                            for (m = -l; m <= l; m++) {
                                val_x += alpha[count] * beta_x[count];
                                val_y += alpha[count] * beta_y[count];
                                val_z += alpha[count] * beta_z[count];
                                count++;
                            }
                            val2_x += val_x * pSPARC->psd[ityp].Gamma[ldispl+np];
                            val2_y += val_y * pSPARC->psd[ityp].Gamma[ldispl+np];
                            val2_z += val_z * pSPARC->psd[ityp].Gamma[ldispl+np];
                        }
                        ldispl += pSPARC->psd[ityp].ppl[l];
                    }
                    fJ_x += val2_x * g_nk;
                    fJ_y += val2_y * g_nk;
                    fJ_z += val2_z * g_nk;
                }
                
                force_nloc[atom_index*3  ] -= spn_fac * fJ_x;
                force_nloc[atom_index*3+1] -= spn_fac * fJ_y;
                force_nloc[atom_index*3+2] -= spn_fac * fJ_z;
                
                atom_index++;
            }
        }
    }    
    
    // sum over all spin
    if (pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(force_nloc, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
    }
    
    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(force_nloc, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    }
    
#ifdef DEBUG    
    if (!rank) {
        printf("force_nloc = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", force_nloc[i*3], force_nloc[i*3+1], force_nloc[i*3+2]);
        }
    }    
    if (!rank) {
        printf("force_loc = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", pSPARC->forces[i*3], pSPARC->forces[i*3+1], pSPARC->forces[i*3+2]);
        }
    }
#endif
    
    if (!rank) {
        for (i = 0; i < 3 * pSPARC->n_atom; i++) {
            pSPARC->forces[i] += force_nloc[i];
        }
    }
    
    free(force_nloc);
    free(alpha);
    free(Y2);
}




/**
 * @brief    Calculate nonlocal force components for cyclix system with kpts.
 */
void Calculate_nonlocal_forces_kpt_cyclix(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, k, n, np, ldispl, ndc, ityp, iat, ncol, DMnd, atom_index, count, l, m, lmax, kpt, Nk, size_k, spn_i, nspin, size_s;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;    
    size_s = size_k * Nk;
    double _Complex *alpha, *beta, *x_ptr, *dx_ptr, *dx_ptr1, *x_rc, *dx_rc, *dx_rc1, *x_rc_ptr, *dx_rc_ptr, *dx_rc_ptr1, *beta_x, *beta_y, *beta_z;
    double *force_nloc, fJ_x, fJ_y, fJ_z, val_x, val_y, val_z, val2_x, val2_y, val2_z, g_nk;
    double _Complex *Y2 = (double _Complex *) malloc( size_k * sizeof(double _Complex));
    
    force_nloc = (double *)calloc(3 * pSPARC->n_atom, sizeof(double));
    alpha = (double _Complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 4, sizeof(double _Complex));
    
    //printf("max displ %d, nkpts %d \n",pSPARC->IP_displ[pSPARC->n_atom], Nk);
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, kpt_vec[3], theta, x0_i, y0_i, z0_i, y0, z0;
    double _Complex bloch_fac, b;
    int szp;
    double ty, tz;
    
#ifdef DEBUG 
    if (!rank) printf("Start Calculating nonlocal forces\n");
#endif
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++) {
            k1 = pSPARC->k1_loc[kpt];
            k2 = pSPARC->k2_loc[kpt];
            k3 = pSPARC->k3_loc[kpt];
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * count;
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                //lmax = pSPARC->psd[ityp].lmax;
                if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    x0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
                    y0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                    z0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                    theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
                    bloch_fac = cos(theta) - sin(theta) * I;
                    b = 1.0;
                    ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                    x_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    //printf("03\n");
                    /* first find inner product <Psi_n, Chi_Jlm>, here we calculate <Chi_Jlm, Psi_n> instead */
                    for (n = 0; n < ncol; n++) {
                        x_ptr = pSPARC->Xorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd;
                        x_rc_ptr = x_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            *(x_rc_ptr + i) = conj(*(x_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]));
                        }
                    }

                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c_cyclix[iat], ndc, 
                                x_rc, ndc, &b, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);                    
                    free(x_rc);
                    
                }
            }
            count++;
        }
    }    
    
    /* find inner product <Chi_Jlm, dPsi_n> */
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++) {
            kpt_vec[0] = k1 = pSPARC->k1_loc[kpt];
            kpt_vec[1] = k2 = pSPARC->k2_loc[kpt];
            kpt_vec[2] = k3 = pSPARC->k3_loc[kpt];
            // find dPsi in direction dim
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, pSPARC->Yorb_kpt, 0, kpt_vec, pSPARC->dmcomm);
            beta_x = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin * 1 + count);
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, Y2, 1, kpt_vec, pSPARC->dmcomm);
            beta_y = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin * 2 + count);
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                szp = pSPARC->nlocProj[ityp].nproj * ncol;
                double _Complex *PX1 = (double _Complex *)malloc( szp * sizeof(double _Complex));
                double _Complex *PX2 = (double _Complex *)malloc( szp * sizeof(double _Complex));
                //lmax = pSPARC->psd[ityp].lmax;
                if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    x0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
                    y0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                    z0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                    theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
                    bloch_fac = cos(theta) + sin(theta) * I;
                    b = 0.0;
                    ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat]; 
                    dx_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
                    dx_rc1 = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    for (n = 0; n < ncol; n++) {
                        dx_ptr = pSPARC->Yorb_kpt + n * DMnd;
                        dx_ptr1 = Y2 + n * DMnd;
                        dx_rc_ptr = dx_rc + n * ndc;
                        dx_rc_ptr1 = dx_rc1 + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            *(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                            *(dx_rc_ptr1 + i) = *(dx_ptr1 + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                        }
                    }
                   
                    // Matrix -matrix multiplication
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c_cyclix[iat], ndc, 
                                dx_rc, ndc, &b, PX1, pSPARC->nlocProj[ityp].nproj);
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c_cyclix[iat], ndc, 
                                dx_rc1, ndc, &b, PX2, pSPARC->nlocProj[ityp].nproj); 
                    
                    // Apply the rotation matrix
                    y0 = pSPARC->atom_pos[3*atom_index+1];
                    z0 = pSPARC->atom_pos[3*atom_index+2];
                    ty = (y0 - y0_i)/pSPARC->range_y;
                    tz = (z0 - z0_i)/pSPARC->range_z;
                    RotMat_cyclix(pSPARC, ty, tz);
                    for(i = 0; i < szp; i++) {
                        Rotate_vector_complex_cyclix(pSPARC, PX1 + i, PX2 + i);
                        beta_x[pSPARC->IP_displ[atom_index]*ncol + i] += PX1[i];
                        beta_y[pSPARC->IP_displ[atom_index]*ncol + i] += PX2[i];
                    }
                    free(dx_rc);
                    free(dx_rc1);
                }
                free(PX1);
                free(PX2);
            }
            count++; 
        }
    }
    free(Y2);

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++) {
            kpt_vec[0] = k1 = pSPARC->k1_loc[kpt];
            kpt_vec[1] = k2 = pSPARC->k2_loc[kpt];
            kpt_vec[2] = k3 = pSPARC->k3_loc[kpt];
            // find dPsi in direction dim
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, pSPARC->Yorb_kpt, 2, kpt_vec, pSPARC->dmcomm);
            beta_z = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin * 3 + count);
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                //lmax = pSPARC->psd[ityp].lmax;
                if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    x0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
                    y0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                    z0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                    theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
                    bloch_fac = cos(theta) + sin(theta) * I;
                    b = 1.0;
                    ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat]; 
                    dx_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    for (n = 0; n < ncol; n++) {
                        dx_ptr = pSPARC->Yorb_kpt + n * DMnd;
                        dx_rc_ptr = dx_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            *(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                        }
                    }
                   
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c_cyclix[iat], ndc, 
                                dx_rc, ndc, &b, beta_z + pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);
                    free(dx_rc);
                    
                }
            }
            count++; 
        }
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }
    
    /* calculate nonlocal force */
    // go over all atoms and find nonlocal force components
    int Ns = pSPARC->Nstates;
    double kpt_spn_fac;
    beta_x = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin;
    beta_y = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 2;
    beta_z = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 3;
    count = 0; 
    
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            kpt_spn_fac = (2.0/pSPARC->Nspin) * 2.0 * pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
            atom_index = 0;
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                int lloc = pSPARC->localPsd[ityp];
                lmax = pSPARC->psd[ityp].lmax;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    fJ_x = fJ_y = fJ_z = 0.0;
                    for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                        g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                        val2_x = val2_y = val2_z = 0.0;
                        ldispl = 0;
                        for (l = 0; l <= lmax; l++) {
                            // skip the local l
                            if (l == lloc) {
                                ldispl += pSPARC->psd[ityp].ppl[l];
                                continue;
                            }
                            for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                                val_x = val_y = val_z = 0.0;
                                for (m = -l; m <= l; m++) {
                                    val_x += creal(alpha[count]) * creal(beta_x[count]) - cimag(alpha[count]) * cimag(beta_x[count]);
                                    val_y += creal(alpha[count]) * creal(beta_y[count]) - cimag(alpha[count]) * cimag(beta_y[count]);
                                    val_z += creal(alpha[count]) * creal(beta_z[count]) - cimag(alpha[count]) * cimag(beta_z[count]);
                                    count++;
                                }
                                val2_x += val_x * pSPARC->psd[ityp].Gamma[ldispl+np];
                                val2_y += val_y * pSPARC->psd[ityp].Gamma[ldispl+np];
                                val2_z += val_z * pSPARC->psd[ityp].Gamma[ldispl+np];
                            }
                            ldispl += pSPARC->psd[ityp].ppl[l];
                        }
                        fJ_x += val2_x * g_nk;
                        fJ_y += val2_y * g_nk;
                        fJ_z += val2_z * g_nk;
                    }
                    force_nloc[atom_index*3  ] -= kpt_spn_fac * fJ_x;
                    force_nloc[atom_index*3+1] -= kpt_spn_fac * fJ_y;
                    force_nloc[atom_index*3+2] -= kpt_spn_fac * fJ_z;
                    atom_index++;
                }
            }
        }
    }    
    
    // sum over all spin
    if (pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(force_nloc, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
    }



    // sum over all kpoints
    if (pSPARC->npkpt > 1) {
        if (pSPARC->kptcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        else
            MPI_Reduce(force_nloc, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
       
            //MPI_Allreduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }
    
    
    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(force_nloc, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    }
    
#ifdef DEBUG    
    if (!rank) {
        printf("force_nloc = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", force_nloc[i*3], force_nloc[i*3+1], force_nloc[i*3+2]);
        }
    }    
    if (!rank) {
        printf("force_loc = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", pSPARC->forces[i*3], pSPARC->forces[i*3+1], pSPARC->forces[i*3+2]);
        }
    }
#endif
    
    if (!rank) {
        for (i = 0; i < 3 * pSPARC->n_atom; i++) {
            pSPARC->forces[i] += force_nloc[i];
        }
    }
    
    free(force_nloc);
    free(alpha);
}


/**
 * @brief    Calculate local force components
 */ 
void Calculate_local_forces_cyclix(SPARC_OBJ *pSPARC)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; // consider broadcasting the force components or force residual
    
    int ityp, iat, i, j, k, p, ip, jp, kp, i_DM, j_DM, k_DM, FDn, count, count_interp,
        Nx, Ny, Nz, DMnx, DMny, DMnz, DMnd, nx, ny, nz, nd, nxp, nyp, nzp, nd_ex, 
        icor, jcor, kcor, atom_index, *ind_interp, dK, dJ, dI;
    double x0_i, y0_i, z0_i, x, y, z, *R, 
           *VJ, *VJ_ref, *VcJ,
           DVcJ_x_val, DVcJ_y_val, DVcJ_z_val, force_x, force_y, force_z, force_corr_x, 
           force_corr_y, force_corr_z, *R_interp, *VJ_interp;
    double DVJ_x_val, DVJ_y_val, DVJ_z_val, y0, z0;
    double ty, tz;       
    double inv_4PI = 0.25 / M_PI, w2_diag, rchrg;
    int *pshifty_ex, *pshiftz_ex;
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
#ifdef DEBUG    
    if (!rank) printf("Start calculating local components of forces ...\n");
#endif
    ////////////////////////////
    double t1, t2, t_sort = 0.0;
    ////////////////////////////
    double *Lap_wt, *Lap_stencil;
    
    FDn = pSPARC->order / 2;
    w2_diag = (pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_z[0]) * -inv_4PI;

    Lap_wt = (double *)malloc((5*(FDn+1))*sizeof(double));
    Lap_stencil = Lap_wt+5;
    Lap_stencil_coef_compact(pSPARC, FDn, Lap_stencil, -inv_4PI);


    Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz;
    DMnx = pSPARC->Nx_d; DMny = pSPARC->Ny_d; DMnz = pSPARC->Nz_d;
    DMnd = pSPARC->Nd_d;
    
    // initialize force components to zero
    for (i = 0; i < 3 * pSPARC->n_atom; i++) {
        pSPARC->forces[i] = 0.0;
    }
    
    // Create indices for laplacian
    pshifty_ex = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz_ex = (int *)malloc( (FDn+1) * sizeof(int));
    
    if (pshifty_ex == NULL || pshiftz_ex == NULL) {
        printf("\nMemory allocation failed in local forces!\n");
        exit(EXIT_FAILURE);
    }
    
    // find gradient of phi
    double *Dphi_x, *Dphi_y, *Dphi_z, *DVc_x, *DVc_y, *DVc_z;
    Dphi_x = (double *)malloc( DMnd * sizeof(double));
    Dphi_y = (double *)malloc( DMnd * sizeof(double));
    Dphi_z = (double *)malloc( DMnd * sizeof(double));
    DVc_x = (double *)malloc( DMnd * sizeof(double));
    DVc_y = (double *)malloc( DMnd * sizeof(double));
    DVc_z = (double *)malloc( DMnd * sizeof(double));
    
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_z, 2, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->Vc, DVc_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->Vc, DVc_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->Vc, DVc_z, 2, pSPARC->dmcomm_phi);
    
    //printf("rank %d\n", rank);
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        for (iat = 0; iat < pSPARC->Atom_Influence_local[ityp].n_atom; iat++) {
            // coordinates of the image atom
            //printf("A\n");
            x0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3];
            y0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 1];
            z0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 2];
            // original atom index this image atom corresponds to
            atom_index = pSPARC->Atom_Influence_local[ityp].atom_index[iat];
            
            // number of finite-difference nodes in each direction of overlap rb region
            nx = pSPARC->Atom_Influence_local[ityp].xe[iat] - pSPARC->Atom_Influence_local[ityp].xs[iat] + 1;
            ny = pSPARC->Atom_Influence_local[ityp].ye[iat] - pSPARC->Atom_Influence_local[ityp].ys[iat] + 1;
            nz = pSPARC->Atom_Influence_local[ityp].ze[iat] - pSPARC->Atom_Influence_local[ityp].zs[iat] + 1;
            nd = nx * ny * nz;
            // number of finite-difference nodes in each direction of extended rb (+ order/2) region
            nxp = nx + pSPARC->order;
            nyp = ny + pSPARC->order;
            nzp = nz + pSPARC->order;
            nd_ex = nxp * nyp * nzp; // total number of nodes
            
            pshifty_ex[0] = pshiftz_ex[0] = 0;
            for (p = 1; p <= FDn; p++) {
                // for x_ex
                pshifty_ex[p] = p * nxp;
                pshiftz_ex[p] = pshifty_ex[p] * nyp;
            }
            
            // number of finite-difference nodes in each direction of extended rb (+ order) region
            //nx2p = nxp + pSPARC->order;
            //ny2p = nyp + pSPARC->order;
            //nz2p = nzp + pSPARC->order;
            //nd_2ex = nx2p * ny2p * nz2p; // total number of nodes
            
            // radii^2 of the finite difference grids of the FDn-extended-rb-region
            R  = (double *)malloc(sizeof(double) * nd_ex);
            if (R == NULL) {
                printf("\nMemory allocation failed!\n");
                exit(EXIT_FAILURE);
            } 
            //printf("B\n");
            
            // left corner of the 2FDn-extended-rb-region
//            icor = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->order;
//            jcor = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->order;
//            kcor = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->order;
            icor = pSPARC->Atom_Influence_local[ityp].xs[iat] - FDn;
            jcor = pSPARC->Atom_Influence_local[ityp].ys[iat] - FDn;
            kcor = pSPARC->Atom_Influence_local[ityp].zs[iat] - FDn;
            
            // relative coordinate of image atoms
            // x0_i_shift =  x0_i - pSPARC->delta_x * icor; 
            // y0_i_shift =  y0_i - pSPARC->delta_y * jcor;
            // z0_i_shift =  z0_i - pSPARC->delta_z * kcor;
            
            // find distance between atom and finite-difference grids
            count = 0; count_interp = 0;
            for (k = kcor; k < kcor+nzp; k++) {
                z = k * pSPARC->delta_z;
                for (j = jcor; j < jcor+nyp; j++) {
                    y = j * pSPARC->delta_y;
                    for (i = icor; i < icor+nxp; i++) {
                        x = pSPARC->xin + i * pSPARC->delta_x;
                        CalculateDistance(pSPARC, x, y, z, x0_i, y0_i, z0_i, &R[count]);
                        if (R[count] <= rchrg) count_interp++;
                        count++;
                    }
                }
            }
            
            
            
            VJ_ref = (double *)malloc( nd_ex * sizeof(double) );
            if (VJ_ref == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            
            // Calculate pseudopotential reference
            Calculate_Pseudopot_Ref(R, nd_ex, pSPARC->REFERENCE_CUTOFF, -pSPARC->Znucl[ityp], VJ_ref);
            
            VJ = (double *)malloc( nd_ex * sizeof(double) );
            if (VJ == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            
            // avoid sorting positions larger than rchrg
            VJ_interp = (double *)malloc( count_interp * sizeof(double) );
            R_interp = (double *)malloc( count_interp * sizeof(double) );
            ind_interp = (int *)malloc( count_interp * sizeof(int) );
            count = 0;
            for (i = 0; i < nd_ex; i++) {
                if (R[i] <= rchrg) {
                    ind_interp[count] = i; // store index
                    R_interp[count] = R[i]; // store radius value
                    count++;
                } else {
                    VJ[i] = -pSPARC->Znucl[ityp] / R[i];
                }
            }
            
            
            t1 = MPI_Wtime();
            
            // sort R_interp and then apply cubic spline interpolation to find VJ
            // notice here we extract out positions within radius rchrg
            //printf("rank = %d, R[%d] = %.13e\n", rank, len_interp-1, R[len_interp-1]); // R is not sorted!
            SortSplineInterp(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
                             R_interp, VJ_interp, count_interp, pSPARC->psd[ityp].SplinerVlocD); 

            t2 = MPI_Wtime();
            t_sort += t2 - t1;
            
            for (i = 0; i < count_interp; i++) {
                if (R_interp[i] < TEMP_TOL) {
                    VJ[ind_interp[i]] = pSPARC->psd[ityp].Vloc_0;
                } else {
                    VJ[ind_interp[i]] =  VJ_interp[i]/R_interp[i];
                }
            }
            free(VJ_interp); VJ_interp = NULL;
            free(R_interp); R_interp = NULL;
            free(ind_interp); ind_interp = NULL;
            free(R); R = NULL;
            
            // calculate VcJ in the extended-rb-domain
            VcJ = (double *)malloc( nd_ex * sizeof(double) );
            if (VcJ == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            for (i = 0; i < nd_ex; i++) {
                VcJ[i] = VJ_ref[i] - VJ[i];
            }
            
            // calculate bJ, bJ_ref and gradient of VcJ in the rb-domain
            dK = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dJ = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            dI = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];

            double *bJ = (double*)malloc(nd * sizeof(double));
            double *bJ_ref = (double*)malloc(nd * sizeof(double));
                        
            double xin = pSPARC->xin + pSPARC->Atom_Influence_local[ityp].xs[iat] * pSPARC->delta_x;           
            Calc_lapV(pSPARC, VJ, FDn, nxp, nyp, nzp, nx, ny, nz, Lap_wt, w2_diag, xin, -inv_4PI, bJ);
            Calc_lapV(pSPARC, VJ_ref, FDn, nxp, nyp, nzp, nx, ny, nz, Lap_wt, w2_diag, xin, -inv_4PI, bJ_ref);
            //printf("rank %d\n", rank);
            y0 = pSPARC->atom_pos[3*atom_index+1];
            z0 = pSPARC->atom_pos[3*atom_index+2];
            ty = (y0 - y0_i)/pSPARC->range_y;
            tz = (z0 - z0_i)/pSPARC->range_z;
            RotMat_cyclix(pSPARC, ty, tz);
            
            force_x = force_y = force_z = 0.0;
            force_corr_x = force_corr_y = force_corr_z = 0.0;
            for (k = 0; k < nz; k++) {
                kp = k + FDn;
                k_DM = k + dK;
                int kshift_DM = k_DM * DMnx * DMny;
                int kshift_p = kp * nxp * nyp;
                int kshift = k * nx * ny; 
                for (j = 0; j < ny; j++) {
                    jp = j + FDn;
                    j_DM = j + dJ;
                    int jshift_DM = kshift_DM + j_DM * DMnx;
                    int jshift_p = kshift_p + jp * nxp;
                    int jshift = kshift + j * nx;
                    //#pragma simd
                    for (i = 0; i < nx; i++) {
                        ip = i + FDn;
                        i_DM = i + dI;
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;
                        int ishift = jshift + i;
                        DVcJ_x_val = DVcJ_y_val = DVcJ_z_val = 0.0;
                        DVJ_x_val = DVJ_y_val = DVJ_z_val = 0.0;
                        //#pragma simd
                        
                        Dpseudopot_cyclix(pSPARC, VcJ, FDn, ishift_p, pshifty_ex, pshiftz_ex, &DVcJ_x_val, &DVcJ_y_val, &DVcJ_z_val, i_DM + pSPARC->DMVertices[0], j_DM + pSPARC->DMVertices[2], k_DM + pSPARC->DMVertices[4]);
                        Dpseudopot_cyclix(pSPARC, VJ, FDn, ishift_p, pshifty_ex, pshiftz_ex, &DVJ_x_val, &DVJ_y_val, &DVJ_z_val, i_DM + pSPARC->DMVertices[0], j_DM + pSPARC->DMVertices[2], k_DM + pSPARC->DMVertices[4]);
                        
                        // find integrals in the force expression
                        double b_plus_b_ref = pSPARC->psdChrgDens[ishift_DM] + pSPARC->psdChrgDens_ref[ishift_DM];
                        double bJ_plus_bJ_ref = bJ[ishift] + bJ_ref[ishift];
                        
                        double fl_x = bJ[ishift] * (Dphi_x[ishift_DM] - DVJ_x_val) * pSPARC->Intgwt_phi[ishift_DM];
                        double fc_x= (DVcJ_x_val * b_plus_b_ref - DVc_x[ishift_DM] * bJ_plus_bJ_ref) * pSPARC->Intgwt_phi[ishift_DM];
                        double fl_y = bJ[ishift] * (Dphi_y[ishift_DM] - DVJ_y_val) * pSPARC->Intgwt_phi[ishift_DM];
                        double fc_y = (DVcJ_y_val * b_plus_b_ref - DVc_y[ishift_DM] * bJ_plus_bJ_ref) * pSPARC->Intgwt_phi[ishift_DM];
                        double fl_z = bJ[ishift] * (Dphi_z[ishift_DM] - DVJ_z_val) * pSPARC->Intgwt_phi[ishift_DM]; 
                        double fc_z = (DVcJ_z_val * b_plus_b_ref - DVc_z[ishift_DM] * bJ_plus_bJ_ref) * pSPARC->Intgwt_phi[ishift_DM];
                        
                        Rotate_vector_cyclix(pSPARC, &fl_x, &fl_y);
                        Rotate_vector_cyclix(pSPARC, &fc_x, &fc_y);
                        
                        force_x -= fl_x; force_y -= fl_y; force_z -= fl_z;
                        
                        force_corr_x += fc_x; force_corr_y += fc_y; force_corr_z += fc_z;

                        
                        
                    }
                }
            }
                       
            pSPARC->forces[atom_index*3  ] += (force_x + 0.5 * force_corr_x);
            pSPARC->forces[atom_index*3+1] += (force_y + 0.5 * force_corr_y);
            pSPARC->forces[atom_index*3+2] += (force_z + 0.5 * force_corr_z);
            
            free(VJ); VJ = NULL;
            free(VJ_ref); VJ_ref = NULL;
            free(bJ); bJ = NULL;
            free(bJ_ref); bJ_ref = NULL;
            free(VcJ); VcJ = NULL;
            //printf("H\n");
             
        }   
    }

    free(Lap_wt);            
    free(Dphi_x);
    free(Dphi_y);
    free(Dphi_z);
    free(DVc_x);
    free(DVc_y);
    free(DVc_z);
    free(pshifty_ex);
    free(pshiftz_ex);
    
    t1 = MPI_Wtime();
    // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
    //printf("rank in %d", rank);
    MPI_Allreduce(MPI_IN_PLACE, pSPARC->forces, 3*pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    //printf("rank out %d", rank);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("time for sorting and interpolate pseudopotential: %.3f ms, time for Allreduce/Reduce: %.3f ms \n", t_sort*1e3, (t2-t1)*1e3);
#endif

}

/**
 * @brief    Calculate xc force components
 */ 
void Calculate_forces_xc_cyclix(SPARC_OBJ *pSPARC, double *forces_xc) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);

    int ityp, iat, i, j, k, p, ip, jp, kp, di, dj, dk, i_DM, j_DM, k_DM, FDn, count, count_interp,
        DMnx, DMny, DMnd, nx, ny, nz, nd, nxp, nyp, nzp, nd_ex, nx2p, ny2p, nz2p, nd_2ex, 
        icor, jcor, kcor, *pshifty_ex, *pshiftz_ex, *ind_interp;
    double x0_i, y0_i, z0_i, x, y, z, *R, *R_interp;
    double rchrg;
    double t1, t2, t_sort = 0.0;

    FDn = pSPARC->order / 2;
    
    DMnx = pSPARC->Nx_d;
    DMny = pSPARC->Ny_d;
    // DMnz = pSPARC->Nz_d;
    DMnd = pSPARC->Nd_d * (2*pSPARC->Nspin - 1);

    // Create indices for laplacian
    //pshifty = (int *)malloc( (FDn+1) * sizeof(int));
    //pshiftz = (int *)malloc( (FDn+1) * sizeof(int));
    pshifty_ex = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz_ex = (int *)malloc( (FDn+1) * sizeof(int));
    
    if (pshifty_ex == NULL || pshiftz_ex == NULL) {
        printf("\nMemory allocation failed in local forces!\n");
        exit(EXIT_FAILURE);
    }

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        for (iat = 0; iat < pSPARC->Atom_Influence_local[ityp].n_atom; iat++) {
            // coordinates of the image atom
            x0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3];
            y0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 1];
            z0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 2];

            // original atom index this image atom corresponds to
            int atom_index = pSPARC->Atom_Influence_local[ityp].atom_index[iat];
            
            // number of finite-difference nodes in each direction of overlap rb region
            nx = pSPARC->Atom_Influence_local[ityp].xe[iat] - pSPARC->Atom_Influence_local[ityp].xs[iat] + 1;
            ny = pSPARC->Atom_Influence_local[ityp].ye[iat] - pSPARC->Atom_Influence_local[ityp].ys[iat] + 1;
            nz = pSPARC->Atom_Influence_local[ityp].ze[iat] - pSPARC->Atom_Influence_local[ityp].zs[iat] + 1;
            nd = nx * ny * nz;
            // number of finite-difference nodes in each direction of extended_rb (rb + order/2) region
            nxp = nx + pSPARC->order;
            nyp = ny + pSPARC->order;
            nzp = nz + pSPARC->order;
            nd_ex = nxp * nyp * nzp; // total number of nodes
            // number of finite-difference nodes in each direction of extended_extended_rb (rb + order) region
            nx2p = nxp + pSPARC->order;
            ny2p = nyp + pSPARC->order;
            nz2p = nzp + pSPARC->order;
            nd_2ex = nx2p * ny2p * nz2p; // total number of nodes
            
            // radii^2 of the finite difference grids of the extended_extended_rb region
            R  = (double *)malloc(sizeof(double) * nd_2ex);
            assert(R != NULL);
            
            // left corner of the 2FDn-extended-rb-region
            icor = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->order;
            jcor = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->order;
            kcor = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->order;
                        
            // relative coordinate of image atoms
            //x0_i_shift =  x0_i - pSPARC->delta_x * icor; 
            //y0_i_shift =  y0_i - pSPARC->delta_y * jcor;
            //z0_i_shift =  z0_i - pSPARC->delta_z * kcor;
            

            count = 0; count_interp = 0;
            for (k = kcor; k < kcor+nz2p; k++) {
                z = k * pSPARC->delta_z;
                for (j = jcor; j < jcor+ny2p; j++) {
                    y = j * pSPARC->delta_y;
                    for (i = icor; i < icor+nx2p; i++) {
                        x = pSPARC->xin + i * pSPARC->delta_x;
                        CalculateDistance(pSPARC, x, y, z, x0_i, y0_i, z0_i, &R[count]);
                        if (R[count] <= rchrg) count_interp++;
                        count++;
                    }
                }
            }
                        
            // VJ = (double *)malloc( nd_2ex * sizeof(double) );
            double *rhocJ = (double *)calloc( nd_2ex,sizeof(double) );
            assert(rhocJ != NULL);
            
            // avoid interpolating positions larger than rchrg
            R_interp = (double *)malloc( count_interp * sizeof(double) );
            ind_interp = (int *)malloc( count_interp * sizeof(int) );
            double *rhocJ_interp = (double *)calloc(count_interp, sizeof(double));
            count = 0;
            for (i = 0; i < nd_2ex; i++) {
                if (R[i] <= rchrg) {
                    ind_interp[count] = i; // store index
                    R_interp[count] = R[i]; // store radius value
                    count++;
                }
            }
            
            t1 = MPI_Wtime();
            // sort R_interp and then apply cubic spline interpolation to find rhocJ
            SplineInterpMain(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rho_c_table, pSPARC->psd[ityp].size, 
                         R_interp, rhocJ_interp, count_interp, pSPARC->psd[ityp].SplineRhocD,pSPARC->psd[ityp].is_r_uniform);
            t2 = MPI_Wtime();
            t_sort += t2 - t1;

            for (i = 0; i < count_interp; i++) {
                rhocJ[ind_interp[i]] = rhocJ_interp[i];
            }

            free(rhocJ_interp); rhocJ_interp = NULL;
            free(R_interp); R_interp = NULL;
            free(ind_interp); ind_interp = NULL;
            free(R); R = NULL;
            
            // shift vectors initialized
            //pshifty[0] = pshiftz[0] = 
            pshifty_ex[0] = pshiftz_ex[0] = 0;
            for (p = 1; p <= FDn; p++) {
                //pshifty[p] = p * nxp;
                //pshiftz[p] = pshifty[p] * nyp;
                pshifty_ex[p] = p * nx2p;
                pshiftz_ex[p] = pshifty_ex[p] *ny2p;
            }
            
            //double xin = pSPARC->xin + pSPARC->Atom_Influence_local[ityp].xs[iat] * pSPARC->delta_x;

            // calculate gradient of bJ, bJ_ref, VJ, VJ_ref in the rb-domain
            dk = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dj = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            di = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];

            // calculate drhocJ, 3 components
            double *drhocJ_x = malloc(nd_ex * sizeof(double));
            double *drhocJ_y = malloc(nd_ex * sizeof(double));
            double *drhocJ_z = malloc(nd_ex * sizeof(double));
            assert(drhocJ_x != NULL && drhocJ_y != NULL && drhocJ_z != NULL);

            double y0 = pSPARC->atom_pos[3*atom_index+1];
            double z0 = pSPARC->atom_pos[3*atom_index+2];
            double ty = (y0 - y0_i)/pSPARC->range_y;
            double tz = (z0 - z0_i)/pSPARC->range_z;
            RotMat_cyclix(pSPARC, ty, tz);
            
            for (int k2p = FDn, kp = 0; k2p < nz2p-FDn; k2p++,kp++) {
                int kshift_2p = k2p * nx2p * ny2p;
                int kshift_p = kp * nxp * nyp;
                for (int j2p = FDn, jp = 0; j2p < ny2p-FDn; j2p++,jp++) {
                    int jshift_2p = kshift_2p + j2p * nx2p;
                    int jshift_p = kshift_p + jp * nxp;
                    for (int i2p = FDn, ip = 0; i2p < nx2p-FDn; i2p++,ip++) {
                        int ishift_2p = jshift_2p + i2p;
                        int ishift_p = jshift_p + ip;
                        double drhocJ_x_val, drhocJ_y_val, drhocJ_z_val;
                        drhocJ_x_val = drhocJ_y_val = drhocJ_z_val = 0.0;
                        Dpseudopot_cyclix(pSPARC, rhocJ, FDn, ishift_2p, pshifty_ex, pshiftz_ex, &drhocJ_x_val, &drhocJ_y_val, &drhocJ_z_val, ip + di + pSPARC->DMVertices[0] - FDn, jp + dj + pSPARC->DMVertices[2] - FDn, kp + dk + pSPARC->DMVertices[4] - FDn);
                        drhocJ_x[ishift_p] = drhocJ_x_val;
                        drhocJ_y[ishift_p] = drhocJ_y_val;
                        drhocJ_z[ishift_p] = drhocJ_z_val;
                    }
                }
            }

            // find int Vxc(x) * drhocJ(x) dx

            double *Vxc = pSPARC->XCPotential;
            double force_xc_x, force_xc_y, force_xc_z;
            force_xc_x = force_xc_y = force_xc_z = 0.0;
            for (k = 0, kp = FDn, k_DM = dk; k < nz; k++, kp++, k_DM++) {
                int kshift_DM = k_DM * DMnx * DMny;
                int kshift_p = kp * nxp * nyp;
                //int kshift = k * nx * ny;  
                for (j = 0, jp = FDn, j_DM = dj; j < ny; j++, jp++, j_DM++) {
                    int jshift_DM = kshift_DM + j_DM * DMnx;
                    int jshift_p = kshift_p + jp * nxp;
                    //int jshift = kshift + j * nx;
                    for (i = 0, ip = FDn, i_DM = di; i < nx; i++, ip++, i_DM++) {
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;
                        //int ishift = jshift + i;
                        double drhocJ_x_val = drhocJ_x[ishift_p];
                        double drhocJ_y_val = drhocJ_y[ishift_p];
                        double drhocJ_z_val = drhocJ_z[ishift_p];
                        double Vxc_val; 
                        if (pSPARC->Nspin == 1)
                            Vxc_val = Vxc[ishift_DM];
                        else
                            Vxc_val = 0.5 * (Vxc[ishift_DM] + Vxc[pSPARC->Nd_d+ishift_DM]);
                        double f_xc_x = Vxc_val * drhocJ_x_val * pSPARC->Intgwt_phi[ishift_DM];
                        double f_xc_y = Vxc_val * drhocJ_y_val * pSPARC->Intgwt_phi[ishift_DM];
                        double f_xc_z = Vxc_val * drhocJ_z_val * pSPARC->Intgwt_phi[ishift_DM];

                        Rotate_vector_cyclix(pSPARC, &f_xc_x, &f_xc_y);

                        force_xc_x += f_xc_x;
                        force_xc_y += f_xc_y;
                        force_xc_z += f_xc_z;
                    }
                }
            }
            forces_xc[atom_index*3  ] += force_xc_x;
            forces_xc[atom_index*3+1] += force_xc_y;
            forces_xc[atom_index*3+2] += force_xc_z;
            free(rhocJ);
            free(drhocJ_x);
            free(drhocJ_y);
            free(drhocJ_z);
        }
    }

    // sum over all domains
    MPI_Allreduce(MPI_IN_PLACE, forces_xc, 3*pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);

    //free(pshifty);
    //free(pshiftz);
    free(pshifty_ex);
    free(pshiftz_ex);
}



void Dpseudopot_cyclix(SPARC_OBJ *pSPARC, double *VJ, int FDn, int ishift_p, int *pshifty_ex, int *pshiftz_ex, double *DVJ_x_val, double *DVJ_y_val, double *DVJ_z_val, double xs, double ys, double zs) {
    double x, y, z, xc, yc, zc, c11, c12, c21, c22, c31;
    x = pSPARC->xin + xs * pSPARC->delta_x; y = ys * pSPARC->delta_y; z = zs * pSPARC->delta_z;
    xc = x; yc = y; zc = z;
    nonCart2Cart_coord(pSPARC, &xc, &yc, &zc);
    c11 = xc/(x);
    c12 = -yc/(x*x);
    c21 = yc/(x);
    c22 = xc/(x*x);
    c31 = -pSPARC->twist;
    double DX, DY, DZ;
    for (int p = 1; p <= FDn; p++) {
        DX = (VJ[ishift_p+p] - VJ[ishift_p-p]) * pSPARC->D1_stencil_coeffs_x[p];
        DY = (VJ[ishift_p+pshifty_ex[p]] - VJ[ishift_p-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
        DZ = (VJ[ishift_p+pshiftz_ex[p]] - VJ[ishift_p-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];
        *DVJ_x_val += c11 * DX + c12 * DY;
        *DVJ_y_val += c21 * DX + c22 * DY;
        *DVJ_z_val += c31 * DY + DZ;
    }    
}




void Rotate_vector_cyclix(SPARC_OBJ *pSPARC, double *fx, double *fy) {
    double f1, f2;
    f1 = *fx; f2 = *fy;
    *fx = pSPARC->RotM_cyclix[0] * f1 + pSPARC->RotM_cyclix[1] * f2;
    *fy = pSPARC->RotM_cyclix[3] * f1 + pSPARC->RotM_cyclix[4] * f2;
}

void Rotate_vector_complex_cyclix(SPARC_OBJ *pSPARC, double _Complex *fx, double _Complex *fy) {
    double _Complex f1, f2;
    f1 = *fx; f2 = *fy;
    *fx = pSPARC->RotM_cyclix[0] * f1 + pSPARC->RotM_cyclix[1] * f2;
    *fy = pSPARC->RotM_cyclix[3] * f1 + pSPARC->RotM_cyclix[4] * f2;
}
