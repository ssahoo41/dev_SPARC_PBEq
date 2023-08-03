/**
 * @file    electronDensity.c
 * @brief   This file contains the functions for calculating electron density.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <complex.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>

#include "electronicGroundState.h"
#include "electronDensity.h"
#include "eigenSolver.h"
#include "eigenSolverKpt.h"
#include "isddft.h"
#include "sq3.h"
#include "ddbp.h"
#include "extFPMD.h"



/*
@ brief: Main function responsible to find electron density
*/
void Calculate_elecDens(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error){
    int i;

    double *rho = (double *) calloc(pSPARC->Nd_d_dmcomm * (pSPARC->Nspden/2*2+1), sizeof(double));
    double t1 = MPI_Wtime();
    
    // Currently only involves Chebyshev filtering eigensolver
    if (pSPARC->isGammaPoint){
        eigSolve_CheFSI(rank, pSPARC, SCFcount, error);
        if(pSPARC->SQ3Flag == 1){
            SubDensMat(pSPARC, pSPARC->Ds_cmc, pSPARC->Efermi, pSPARC->ChebComp);
        }
        if (pSPARC->DDBP_Flag == 1) {
            DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
            int Nstates = pSPARC->Nstates;
            int nspin = pSPARC->Nspin_spincomm;
            int nkpt = pSPARC->Nkpts_kptcomm;
            Calculate_density_psi_DDBP(
                DDBP_info->n_elem_elemcomm, DDBP_info->elem_list,
                DDBP_info->psi, DDBP_info->rho, pSPARC->occ, pSPARC->dV,
                pSPARC->isGammaPoint, pSPARC->spin_typ, nspin, nkpt, Nstates,
                pSPARC->spin_start_indx, DDBP_info->band_start_index,
                DDBP_info->band_end_index, DDBP_info->elemcomm
            );

            // #define CHECK_RHO
            #ifdef CHECK_RHO
                // check if \int{rho} = Nelectron
                double int_rho = 0.0;
                for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
                    DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
                    int nd_k = E_k->nd_d;
                    double *rho_k = DDBP_info->rho[k];
                    for (int i = 0; i < nd_k; i++) {
                        int_rho += rho_k[i];
                    }
                }
                int_rho *= pSPARC->dV;
                MPI_Allreduce(MPI_IN_PLACE, &int_rho, 1, MPI_DOUBLE, MPI_SUM, DDBP_info->bandcomm);
                double sum_occ = 0.0;
                for (int i = 0; i < Nstates; i++) {
                    sum_occ += 2.0*pSPARC->occ[i];
                }
                printf("rank = %2d, checking rho: sum_occ = %.16f, int_rho = %.16f\n", rank, sum_occ, int_rho);
                // warning: this is only for checking spin-unpolarized test
                // sleep(1);
                if (pSPARC->spin_typ == 0 && rank == 0)
                    assert(fabs(int_rho - sum_occ) < 1e-10);
            #endif
        } else {
            if(pSPARC->spin_typ == 0)
                CalculateDensity_psi(pSPARC, rho);
            else if(pSPARC->spin_typ == 1)
                CalculateDensity_psi_spin(pSPARC, rho);
            else if(pSPARC->spin_typ == 2)
                assert(pSPARC->spin_typ <= 1);
        }
    } else {
        eigSolve_CheFSI_kpt(rank, pSPARC, SCFcount, error);
        if(pSPARC->spin_typ == 0)
            CalculateDensity_psi_kpt(pSPARC, rho);
        else if(pSPARC->spin_typ == 1)
            CalculateDensity_psi_kpt_spin(pSPARC, rho);
        else if(pSPARC->spin_typ == 2)
            assert(pSPARC->spin_typ <= 1);
    }

    // add high energy electron density for ext-FPMD method
    if (pSPARC->ext_FPMD_Flag != 0) {
        highE_rho_extFPMD(pSPARC, 1.0, rho, pSPARC->Nd_d_dmcomm);
    }

    double t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("rank = %d, Calculating density took %.3f ms\n",rank,(t2-t1)*1e3);       
    if(!rank) printf("rank = %d, starting to transfer density...\n",rank);
#endif
    
    // transfer density from psi-domain to phi-domain
    t1 = MPI_Wtime();
    
    if (pSPARC->DDBP_Flag == 1) {
        // TODO: remove after check
        // first overwrite the original value so that we're sure the data transfer is complete
        // for (int i = 0; i < pSPARC->Nd_d; i++) {
        //     pSPARC->electronDens[i] = -124;
        // }
        
        for (int i = 0; i < 2*pSPARC->Nspin-1; i++) {
            // transfter density from elem distribution to domain distribution
            DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
            int Nstates = pSPARC->Nstates;
            int nspin = pSPARC->Nspin_spincomm;
            int nkpt = pSPARC->Nkpts_kptcomm;
            // element distribution to domain distribution
            int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
            int BCs[3] = {pSPARC->BCx, pSPARC->BCy, pSPARC->BCz};
            int dmcomm_phi_dims[3] = {pSPARC->npNdx_phi, pSPARC->npNdy_phi, pSPARC->npNdz_phi};
            int send_ncol = DDBP_info->bandcomm_index == 0 ? 1 : 0;
            int recv_ncol = 1;
            int Edims[3] = {DDBP_info->Nex, DDBP_info->Ney, DDBP_info->Nez};
            E2D_INFO E2D_info;
            E2D_Init(&E2D_info, Edims, DDBP_info->n_elem_elemcomm, DDBP_info->elem_list,
                gridsizes, BCs, 1,
                0, send_ncol, DDBP_info->elemcomm, DDBP_info->npband, DDBP_info->elemcomm_index,
                DDBP_info->bandcomm, DDBP_info->npelem, DDBP_info->bandcomm_index,
                0, recv_ncol, pSPARC->DMVertices, MPI_COMM_SELF, 1, pSPARC->dmcomm_phi,
                &dmcomm_phi_dims[0], 0, pSPARC->kptcomm
            );

            E2D_Iexec(&E2D_info, (const void **) DDBP_info->rho);
            E2D_Wait(&E2D_info, pSPARC->electronDens + i*pSPARC->Nd_d);
            E2D_Finalize(&E2D_info);
        }
    } else {
        for (i = 0; i < pSPARC->Nspden/2*2+1; i++)
            TransferDensity(pSPARC, rho + i*pSPARC->Nd_d_dmcomm, pSPARC->electronDens + i*pSPARC->Nd_d);
    }

    t2 = MPI_Wtime();

    #ifdef CHECK_RHO
    // check electron density
    // check if \int{rho} = Nelectron
    double int_rho = 0.0;
    for (int i = 0; i < pSPARC->Nd_d; i++) {
        int_rho += pSPARC->electronDens[i];
    }
    
    int_rho *= pSPARC->dV;
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL)
        MPI_Allreduce(MPI_IN_PLACE, &int_rho, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    double sum_occ = 0.0;
    for (int i = 0; i < pSPARC->Nstates; i++) {
        sum_occ += 2.0*pSPARC->occ[i];
    }
    if (pSPARC->ext_FPMD_Flag != 0) {
        usleep(20000);
	    double HighECharge = calculate_highE_Charge_extFPMD(pSPARC, pSPARC->Efermi);
        printf("== CHECK_RHO ==: rank = %d, Efermi = %f, HighECharge = %f\n", rank, pSPARC->Efermi, HighECharge);
        sum_occ += HighECharge;
    }
    printf("rank = %2d, after transfering rho: sum_occ = %.16f, int_rho = %.16f\n", rank, sum_occ, int_rho);
    // warning: this is only for checking spin-unpolarized test
    if (pSPARC->spin_typ == 0 && rank == 0)
        assert(fabs(int_rho - sum_occ) < 1e-8 * sum_occ);
    #endif

#ifdef DEBUG
    if(!rank) printf("rank = %d, Transfering density took %.3f ms\n", rank, (t2 - t1) * 1e3);
#endif

    // int length = pSPARC->Nd_d_dmcomm * (pSPARC->Nspden/2*2+1); // new length
    // // int length = (2*pSPARC->Nspin-1) * pSPARC->Nd_d_dmcomm; // this data can be distributed
    // if (!rank) printf("length = %d\n", length);
    // memcpy(pSPARC->scfElectronDens, rho, length * sizeof(double));
    // void BroadcastRho(SPARC_OBJ *pSPARC, double *rho, int rank, int length);
    
    // BroadcastRho(pSPARC, pSPARC->scfElectronDens, rank, length);
    free(rho);
    // if (!rank) printf("broadcasting rho successful \n");
}
/** 
 * Ray's function for broadcasting of rho into SCF electrondens
 */

// void BroadcastRho(SPARC_OBJ *pSPARC, double *rho, int length)
// {
//     double t1, t2;
//     // #ifdef DEBUG
//     //     if (rank == 0) printf("rank = %d, --- Calculate rho: took %.3f ms\n", rank, (t2-t1)*1e3);
//     // #endif

//         t1 = MPI_Wtime();
//         MPI_Bcast( rho, length, MPI_DOUBLE,  0, MPI_COMM_WORLD);
//         t2 = MPI_Wtime();
// }

/**
 * @brief   Calculate electron density with given states in psi-domain.
 *
 *          Note that here rho is distributed in psi-domain, which needs
 *          to be transmitted to phi-domain for solving the poisson 
 *          equation.
 */
void CalculateDensity_psi(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int i, n, Nd, count, nstart, nend;
    double g_nk;
    int CS_Flag = pSPARC->CS_Flag;
    int Nt = 0;
    if (CS_Flag == 1)
        Nt = pSPARC->CS_Nt;
    int Ns = pSPARC->Nstates;
    Nd = pSPARC->Nd_d_dmcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    int spinor;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    
    t1 = MPI_Wtime();
    // calculate rho based on local bands
    if (pSPARC->SQ3Flag == 1) {
        update_rho_psi(pSPARC, rho, Nd, Ns, nstart, nend);
    } else if (CS_Flag == 1) {
        count = 0;
        for (n = nstart; n <= nend; n++) {
            g_nk = 2.0;
            double *psi_n = pSPARC->Xorb + Nd*(n-nstart);
            for (i = 0; i < Nd; i++, count++) {
                rho[i] += g_nk * psi_n[i] * psi_n[i];
            }
        }
        for (n = nstart; n <= nend; n++) {
            if (n < Ns - Nt) continue;
            g_nk = -2.0 * (1-pSPARC->occ[n]);
            double *psi_n = pSPARC->Yorb + Nd*(n-nstart);
            for (i = 0; i < Nd; i++) {
                rho[i] += g_nk * psi_n[i] * psi_n[i];
            }
        }
    } else {
        count = 0;
        for (n = nstart; n <= nend; n++) {
            // g_nk = 2.0 * smearing_FermiDirac(pSPARC->Beta,pSPARC->lambda[n],pSPARC->Efermi);
            g_nk = pSPARC->occfac * pSPARC->occ[n];
            for (spinor = 0; spinor < pSPARC->Nspinor; spinor ++) {
                for (i = 0; i < Nd; i++, count++) {
                    rho[i] += g_nk * pSPARC->Xorb[count] * pSPARC->Xorb[count];
                }
            }
        }
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    // sum over all band groups
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(rho, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    if (!pSPARC->CyclixFlag) {
        t1 = MPI_Wtime();
        double vscal = 1.0 / pSPARC->dV;
        // scale electron density by 1/dV
        // TODO: this can be done in phi-domain over more processes!
        //       Perhaps right after transfer to phi-domain is complete.
        for (i = 0; i < pSPARC->Nd_d_dmcomm; i++) {
            rho[i] *= vscal; 
        }
        
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }
}


/**
 * @brief   Calculate electron density with given states in psi-domain with spin on.
 *
 *          Note that here rho is distributed in psi-domain, which needs
 *          to be transmitted to phi-domain for solving the poisson 
 *          equation.
 */
void CalculateDensity_psi_spin(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int i, n, Ns, Nd, count, nstart, nend, spn_i, sg;
    double g_nk;
    Ns = pSPARC->Nstates;
    Nd = pSPARC->Nd_d_dmcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    int spinor;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;

    t1 = MPI_Wtime();
    
    // calculate rho based on local bands
    count = 0;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        sg = spn_i + pSPARC->spin_start_indx;
        for (n = nstart; n <= nend; n++) {
            // g_nk = 2.0 * smearing_FermiDirac(pSPARC->Beta,pSPARC->lambda[n],pSPARC->Efermi);
            g_nk = pSPARC->occfac * pSPARC->occ[n+spn_i*Ns];
            for (spinor = 0; spinor < pSPARC->Nspinor; spinor++) {
                for (i = 0; i < Nd; i++, count++) {
                    rho[(sg+spinor+1)*Nd + i] += g_nk * pSPARC->Xorb[count] * pSPARC->Xorb[count];
                }
            }
        }
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
    // sum over spin comm
    t1 = MPI_Wtime();
    if(pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(rho, rho, 3*pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
    }
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    // sum over all band groups
    if (pSPARC->npband > 1 && pSPARC->spincomm_index == 0) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(rho, rho, 3*pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    } // TODO: can be made only 2*Nd

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    if (!pSPARC->CyclixFlag) {
        t1 = MPI_Wtime();
        double vscal = 1.0 / pSPARC->dV;
        // scale electron density by 1/dV
        // TODO: this can be done in phi-domain over more processes!
        //       Perhaps right after transfer to phi-domain is complete.
        for (i = 0; i < 2*Nd; i++) {
            rho[Nd+i] *= vscal;
        }    
        
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    t1 = MPI_Wtime();
    for (i = 0; i < Nd; i++) {
        rho[i] = rho[Nd+i] + rho[2*Nd+i]; 
    }
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("rank = %d, --- Calculate rho: forming total rho took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}



void CalculateDensity_psi_kpt(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int i, n, k, Ns, count, nstart, nend, spinor;
    double g_nk, occfac;
    Ns = pSPARC->Nstates;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    occfac = pSPARC->occfac;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    
    t1 = MPI_Wtime();
    
    // calculate rho based on local bands
    count = 0;
    
    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
        for (n = nstart; n <= nend; n++) {
            g_nk = occfac * (pSPARC->kptWts_loc[k] / pSPARC->Nkpts) * pSPARC->occ[k*Ns+n];
            for (spinor = 0; spinor < pSPARC->Nspinor; spinor ++) {
                for (i = 0; i < pSPARC->Nd_d_dmcomm; i++) {
                    rho[i] += g_nk * (pow(creal(pSPARC->Xorb_kpt[count]), 2.0) 
                                    + pow(cimag(pSPARC->Xorb_kpt[count]), 2.0));
                    count++;
                }
            }
        }
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    
    // sum over all k-point groups
    if (pSPARC->npkpt > 1) {    
        if (pSPARC->kptcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        else
            MPI_Reduce(rho, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
    }
    
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
    
    t1 = MPI_Wtime();
    // sum over all band groups (only in the first k point group)
    if (pSPARC->npband > 1 && pSPARC->kptcomm_index == 0) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(rho, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    if (!pSPARC->CyclixFlag) {
        t1 = MPI_Wtime();
        double vscal = 1.0 / pSPARC->dV;
        // scale electron density by 1/dV
        // TODO: this can be done in phi-domain over more processes!
        //       Perhaps right after transfer to phi-domain is complete.
        for (i = 0; i < pSPARC->Nd_d_dmcomm; i++) {
            rho[i] *= vscal;
        }
        
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }
}




void CalculateDensity_psi_kpt_spin(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int i, n, k, Ns, Nd, Nk, count, nstart, nend, sg, spn_i, spinor;
    double g_nk, occfac = pSPARC->occfac;
    Ns = pSPARC->Nstates;
    Nd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    
    t1 = MPI_Wtime();
    // calculate rho based on local bands
    count = 0;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        sg = spn_i + pSPARC->spin_start_indx;
        for (k = 0; k < Nk; k++) {
            for (n = nstart; n <= nend; n++) {
                g_nk = (pSPARC->kptWts_loc[k] / pSPARC->Nkpts) * occfac * pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                for (spinor = 0; spinor < pSPARC->Nspinor; spinor ++) {
                    for (i = 0; i < pSPARC->Nd_d_dmcomm; i++) {
                        rho[i+(sg+spinor+1)*Nd] += g_nk * (pow(creal(pSPARC->Xorb_kpt[count]), 2.0) + pow(cimag(pSPARC->Xorb_kpt[count]), 2.0));
                        count++;
                    }
                }
            }
        }
    }
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    // sum over spin comm group
    t1 = MPI_Wtime();
    if(pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(rho, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
    }
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    // sum over all k-point groups
    if (pSPARC->spincomm_index == 0 &&  pSPARC->npkpt > 1) {    
        if (pSPARC->kptcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        else
            MPI_Reduce(rho, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
    }
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
    
    t1 = MPI_Wtime();
    // sum over all band groups (only in the first k point group)
    if (pSPARC->npband > 1 && pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(rho, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    if (!pSPARC->CyclixFlag) {
        t1 = MPI_Wtime();
        double vscal = 1.0 / pSPARC->dV;
        // scale electron density by 1/dV
        // TODO: this can be done in phi-domain over more processes!
        //       Perhaps right after transfer to phi-domain is complete.
        for (i = 0; i < 2*Nd; i++) {
            rho[Nd+i] *= vscal; 
        }
        
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }
    
    t1 = MPI_Wtime();
    for (i = 0; i < Nd; i++) {
        rho[i] = rho[Nd+i] + rho[2*Nd+i]; 
    }
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("rank = %d, --- Calculate rho: forming total rho took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}

