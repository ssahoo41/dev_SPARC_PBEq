/**
* @file convolutionalGGAMultipoleXC.c
* @brief This file contains declaration of functions required for multipole dependent convolutional GGA.
*
* @author Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
*		Andrew J. Medford <ajm@gatech.edu>
*
* Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#include "exchangeCorrelation.h"
#include "initialization.h"
#include "isddft.h"
#include "gradVecRoutines.h"
#include "tools.h"
#include "vdWDFexchangeLinearCorre.h"
#include "vdWDFnonlinearCorre.h"
#include "convgradVecRoutines.h"
#include "convolutionalGGAMultipoleXC.h"
#include "ConvolutionalGGAexchangeCorrelation.h"
#include "MCSH.h"
#include "MCSHHelper.h"
#include "MCSHMainCalc.h"
#include "MCSHTools.h"
#include "MP_types.h"

/**
 * @brief   function to calculate XC potential using GGA_CONV_PBE_MULTIPOLE
 *
 */
void Calculate_Vxc_GGA_CONV_PBE_MULTIPOLE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // array for X potential and C potential, ex and ec energy densities
    double *XPotential, *CPotential, *e_c, *e_x, *Dxdgrho, *Dcdgrho;
    
    int DMnd, i;
    DMnd = pSPARC->Nd_d;
    XPotential = (double *) malloc(DMnd * sizeof(double));
    CPotential = (double *) malloc(DMnd * sizeof(double));
    e_c = (double *) malloc(DMnd * sizeof(double));
    e_x = (double *) malloc(DMnd * sizeof(double));
    Dxdgrho = (double *) malloc(DMnd * sizeof(double));
    Dcdgrho = (double *) malloc(DMnd * sizeof(double));

    Calculate_Vx_GGA_CONV_PBE_MULTIPOLE(pSPARC, xc_cst, rho, XPotential, e_x, Dxdgrho);
    Calculate_Vc_GGA_CONV_PBE(pSPARC, xc_cst, rho, CPotential, e_c, Dcdgrho);

    #ifdef DEBUG
    double *global_CPotential;
    global_CPotential = (double *) malloc(pSPARC->Nd * sizeof(double));
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    gather_distributed_vector(CPotential, pSPARC->DMVertices, global_CPotential, gridsizes, pSPARC->dmcomm_phi, 1);
    char CPotentialFilename[128];
    if (rank == 0){
        snprintf(CPotentialFilename, 128, "pbe_corr_potential.csv");
        writeMatToFile(CPotentialFilename, global_CPotential, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    }
    #endif

    for(i = 0; i < DMnd; i++){
        pSPARC->XCPotential[i] = XPotential[i] + CPotential[i];
        pSPARC->e_xc[i] = e_x[i] + e_c[i];
        pSPARC->Dxcdgrho[i] = Dxdgrho[i] + Dcdgrho[i];
    }

    free(XPotential); free(CPotential);
    free(global_CPotential);
    free(e_x); free(e_c); 
    free(Dxdgrho); free(Dcdgrho);
}

// Next steps:
// 1. Dxdgrho transformation part- for non-orthogonal cells (is it required for features)
// 2. Stress and pressure calculation using multipole features

/**
 * @brief   function to calculate exchange potential for PBEq where the additional terms for potential
 *          are neglected. The exchange enhancement factor is dependent on spatially-resolved alpha 
 *          which is a function of monopole feature.
 */
void Calculate_Vx_GGA_CONV_PBE_MULTIPOLE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *XPotential, double *e_x, double *Dxdgrho){

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vx (GGA_CONV_PBE_MULTIPOLE)\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }

    int DMnd, i;
    DMnd = pSPARC->Nd_d;

    MULTIPOLE_OBJ mp;
    Multipole_Initialize(pSPARC, &mp);
    double m = pSPARC->m_val;
    double n = pSPARC->n_val;

    // JUST EXCHANGE
    double rho_updn, rho_updnm1_3, rhom1_3, rhotot_inv, rhotmo6, rhoto6, rhomot, ex_lsd, rho_inv, coeffss, ss;
    double divss, dfxdss, fx, ex_gga, dssdn, dfxdn, dssdg, dfxdg, ex;

    double temp1, temp2, temp3;
    
    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;
    double *global_rho, *global_sigma;

    double Df_alpha, Dalpha_qp;
    // all array pointers for HSMP features
    double *alpha;
    double *feat_qp_monopole, *Dfeat_qp_mp, *DDfeat_qp_mp, *global_Df_featmp;

    Drho_x = (double *) malloc(DMnd * sizeof(double));
    Drho_y = (double *) malloc(DMnd * sizeof(double));
    Drho_z = (double *) malloc(DMnd * sizeof(double));
    DDrho_x = (double *) malloc(DMnd * sizeof(double));
    DDrho_y = (double *) malloc(DMnd * sizeof(double));
    DDrho_z = (double *) malloc(DMnd * sizeof(double));
    sigma = (double *) malloc(DMnd * sizeof(double));

    Dfeat_qp_mp = (double *) malloc(DMnd * sizeof(double));
    DDfeat_qp_mp = (double *)malloc(DMnd * sizeof(double));
    feat_qp_monopole = (double *) malloc(DMnd * sizeof(double));
    
    alpha = (double *) malloc(DMnd * sizeof(double));
    // memory allocation for global variables
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    global_rho = (double *) malloc(pSPARC->Nd * sizeof(double));
    global_Df_featmp = (double *) malloc(pSPARC->Nd * sizeof(double));

    // gathering distributed rho vector into global vector and broadcast to all processors
    gather_distributed_vector(rho, pSPARC->DMVertices, global_rho, gridsizes, pSPARC->dmcomm_phi, 1);
    MPI_Bcast(global_rho, pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    
    // including monopole feature: feat_qp_monopole
    double *global_monopole;
    global_monopole = (double *) malloc(pSPARC->Nd * sizeof(double));
    Conv_feat_vectors_dir(pSPARC, &mp, pSPARC->DMVertices, 1, global_rho, feat_qp_monopole, "000", pSPARC->dmcomm_phi);
    gather_distributed_vector(feat_qp_monopole, pSPARC->DMVertices, global_monopole, gridsizes, pSPARC->dmcomm_phi, 1);
    char MonopoleFilename[128];
    if (rank == 0){
        snprintf(MonopoleFilename, 128, "feature_monopole.csv");
        writeMatToFile(MonopoleFilename, global_monopole, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    }
    Construct_alpha(pSPARC, feat_qp_monopole, m, n, DMnd, alpha);
    double *global_alpha;
    global_alpha = (double *) malloc(pSPARC->Nd * sizeof(double));
    gather_distributed_vector(alpha, pSPARC->DMVertices, global_alpha, gridsizes, pSPARC->dmcomm_phi, 1);
    char alphaFilename[128];
    if (rank == 0){
        snprintf(alphaFilename, 128, "alpha.csv");
        writeMatToFile(alphaFilename, global_alpha, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    }
    // using SPARC's convolution routine for calculating derivatives
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, Drho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, Drho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, Drho_z, 2, pSPARC->dmcomm_phi);

    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        lapcT = (double *) malloc(6 * sizeof(double));
        lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
        for(i = 0; i < DMnd; i++){
            sigma[i] = Drho_x[i] * (lapcT[0] * Drho_x[i] + lapcT[1] * Drho_y[i]) + Drho_y[i] * (lapcT[3] * Drho_y[i] + lapcT[4] * Drho_z[i]) +
                       Drho_z[i] * (lapcT[5] * Drho_z[i] + lapcT[2] * Drho_x[i]); 
        }
        free(lapcT);
    } else {
        for(i = 0; i < DMnd; i++){
            sigma[i] = Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i];
        }
    }

    // Compute exchange

    for(i = 0; i < DMnd; i++){
        rho_updn = rho[i]/2.0;
        rho_updnm1_3 = pow(rho_updn, -xc_cst->third);
        rhom1_3 = xc_cst->twom1_3 * rho_updnm1_3;
        rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        rhotmo6 = sqrt(rhom1_3);
        rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // First take care of the exchange part of the functional
        rhomot = rho_updnm1_3;
        ex_lsd = -xc_cst->threefourth_divpi * xc_cst->sixpi2_1_3 * (rhomot * rhomot * rho_updn);

        // Perdew-Burke-Ernzerhof GGA, exchange part
        rho_inv = rhomot * rhomot * rhomot;
        coeffss = (1.0/4.0) * xc_cst->sixpi2m1_3 * xc_cst->sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
        ss = (sigma[i]/4.0) * coeffss; // s^2

        divss = 1.0/(1.0 + ((xc_cst->mu_divkappa * ss)/alpha[i])); // alpha is added here
        dfxdss = xc_cst->mu * pow(divss, alpha[i]) * divss;
    
        fx = 1.0 + xc_cst->kappa * (1.0 - pow(divss, alpha[i]));
        ex_gga = ex_lsd * fx;
        dssdn = (-8.0/3.0) * (ss * rho_inv);
        dfxdn = dfxdss * dssdn;
        XPotential[i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);

        dssdg = 2.0 * coeffss;
        dfxdg = dfxdss * dssdg;
        Dxdgrho[i] = 0.5 * ex_lsd * rho_updn * dfxdg;// second part of the derivative
        ex = ex_gga * rho_updn;
        Df_alpha = -xc_cst->kappa * pow(divss, alpha[i])*(-log(1/divss) + ((xc_cst->mu_divkappa * ss * divss)/alpha[i])); // fixing bug (deleted the fix for sometime)
        Dalpha_qp = -((4.0-0.75) * m)/(2.0 + exp(- m * (feat_qp_monopole[i] - n)) + exp(m * (feat_qp_monopole[i] - n)));
        Dfeat_qp_mp[i] = ex_lsd * 2 * rho_updn * Df_alpha * Dalpha_qp;

        // If non spin-polarized, treat spin down contribution now, similar to spin up
        ex = ex * 2.0;
        e_x[i] = ex * rhotot_inv;
        // WARNING: Dxcdgrho = 0.5 * dvxcdgrho1 here in M-SPARC!! But the same in the end.     
    }

    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(i = 0; i < DMnd; i++){
            temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * Dxdgrho[i];
            temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * Dxdgrho[i];
            temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * Dxdgrho[i];
            Drho_x[i] = temp1;
            Drho_y[i] = temp2;
            Drho_z[i] = temp3;
        }
    } else {
        for(i = 0; i < DMnd; i++){
            Drho_x[i] *= Dxdgrho[i];
            Drho_y[i] *= Dxdgrho[i];
            Drho_z[i] *= Dxdgrho[i];
        }
    }

    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);

    for(i = 0; i < DMnd; i++){
        //if(pSPARC->electronDens[i] != 0.0)
        XPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
    }

    // Save X potential contribution from PBE, no extra contribution added so far
    #ifdef DEBUG
    if (rank == 0) printf("Saving X potential contribution from PBE.\n");
    double *pbeX_potential;
    pbeX_potential = (double *)malloc(pSPARC->Nd * sizeof(double));
    gather_distributed_vector(XPotential, pSPARC->DMVertices, pbeX_potential, gridsizes, pSPARC->dmcomm_phi, 1);
    char PBEPotentialFilename[128];
    if (rank == 0){
        snprintf(PBEPotentialFilename, 128, "pbe_x_potential.csv");
        writeMatToFile(PBEPotentialFilename, pbeX_potential, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    }
    free(pbeX_potential);
    #endif

    // Extra potential contribution from HSMP part saved to a different file
    #ifdef DEBUG
    if (rank == 0) printf("Saving extra potential contribution from HSMP.\n");
    #endif
    double *extra_potential, *global_extra_potential;
    extra_potential = (double *)malloc(DMnd * sizeof(double));
    for (i = 0; i < DMnd; i++){
        extra_potential[i] = 0;
    }
    #ifdef DEBUG
    global_extra_potential = (double *)malloc(pSPARC->Nd * sizeof(double));
    #endif

    gather_distributed_vector(Dfeat_qp_mp, pSPARC->DMVertices, global_Df_featmp, gridsizes, pSPARC->dmcomm_phi, 1);
    MPI_Bcast(global_Df_featmp, pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    Conv_feat_vectors_dir(pSPARC, &mp, pSPARC->DMVertices, 1, global_Df_featmp, DDfeat_qp_mp, "000", pSPARC->dmcomm_phi);
    // monopole contribution
    for (i = 0; i < DMnd; i++){
        XPotential[i] += DDfeat_qp_mp[i];
        extra_potential[i] += DDfeat_qp_mp[i];
    }

    #ifdef DEBUG
    gather_distributed_vector(extra_potential, pSPARC->DMVertices, global_extra_potential, gridsizes, pSPARC->dmcomm_phi, 1);
    char extraPotentialFilename[128];
    if (rank == 0){
        snprintf(extraPotentialFilename, 128, "extra_X_potential.csv");
        writeMatToFile(extraPotentialFilename, global_extra_potential, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    }
    #endif
    //Deallocate memory
    free(extra_potential); free(global_extra_potential);
    free(Drho_x); free(Drho_y); free(Drho_z);
    free(global_rho); 
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);
    free(alpha);
    free(global_monopole); free(feat_qp_monopole);
    free(Dfeat_qp_mp); free(DDfeat_qp_mp);
    free(global_Df_featmp);
}

/**
* @brief function to calculate XC potential for GGA_CONV_PBE_MULTIPOLE XC potential for spin-polarized case
*
*/
void Calculate_Vxc_GSGA_CONV_PBE_MULTIPOLE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho) {
    
    double *XPotential, *CPotential, *e_c, *e_x, *e_xc, *Dxdgrho, *Dcdgrho;
    int DMnd, i, ncopy, spin_i;
    spin_i = pSPARC->Nspden;
    ncopy = spin_i/2*2+1;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    DMnd = pSPARC->Nd_d;
    
    XPotential = (double *) malloc(spin_i * DMnd * sizeof(double));
    CPotential = (double *) malloc(spin_i * DMnd * sizeof(double));
    e_c = (double *) malloc(DMnd * sizeof(double));
    e_x = (double *) malloc(DMnd * sizeof(double));
    e_xc = (double *) malloc(DMnd * sizeof(double));
    Dxdgrho = (double *) malloc(DMnd * spin_i * sizeof(double));
    Dcdgrho = (double *) malloc(DMnd * sizeof(double));

    Calculate_Vx_GSGA_CONV_PBE_MULTIPOLE(pSPARC, xc_cst, rho, XPotential, e_x, Dxdgrho);
    Calculate_Vc_GSGA_CONV_PBE(pSPARC, xc_cst, rho, CPotential, e_c, Dcdgrho);

    for(i = 0; i < DMnd ; i++) {
        pSPARC->Dxcdgrho[i] = Dcdgrho[i];
        pSPARC->Dxcdgrho[DMnd + i] = Dxdgrho[i];
        pSPARC->Dxcdgrho[2*DMnd + i] = Dxdgrho[DMnd + i];
    }

    for(i = 0; i < DMnd; i++){
        pSPARC->XCPotential[i] = XPotential[i] + CPotential[i]; // potential for spin-up
        pSPARC->XCPotential[DMnd + i] = XPotential[DMnd + i] + CPotential[DMnd + i]; // potential for spin-down
        pSPARC->e_xc[i] = e_x[i] + e_c[i];
    }

    free(XPotential); free(CPotential);
    free(e_xc); free(e_x);
    free(e_c); free(Dxdgrho); free(Dcdgrho);
}

/**
 * @brief   function to calculate exchange potential for spin-polarized case for GGA_CONV_PBE_MULTIPOLE
 *          similar to exchange potential for spin paired case
 */
void Calculate_Vx_GSGA_CONV_PBE_MULTIPOLE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *XPotential, double *e_x, double *Dxdgrho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vx (GSGA_CONV_PBE_MULTIPOLE) ...\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    MULTIPOLE_OBJ mp;
    Multipole_Initialize(pSPARC, &mp);
    double m = pSPARC->m_val;
    double n = pSPARC->n_val;

    // JUST EXCHANGE
    double rho_updn, rho_updnm1_3, rhom1_3, rhotot_inv, rhotmo6, rhoto6, rhomot, ex_lsd, rho_inv, coeffss, ss;
    double divss, dfxdss, fx, ex_gga, dssdn, dfxdn, dssdg, dfxdg, ex;

    double temp1, temp2, temp3;
    int DMnd, i, spn_i, spin_typ, Nd;
    Nd = pSPARC->Nd;
    DMnd = pSPARC->Nd_d;
    spin_typ = pSPARC->Nspden;

    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;
    double *global_rho;

    double Df_alpha, Dalpha_qp;
    // all array pointers for HSMP features
    double *alpha;
    double *feat_qp_monopole, *Dfeat_qp_mp, *DDfeat_qp_mp, *global_Df_featmp;

    Drho_x = (double *) malloc(2 * DMnd * sizeof(double));
    Drho_y = (double *) malloc(2 * DMnd * sizeof(double));
    Drho_z = (double *) malloc(2 * DMnd * sizeof(double));
    DDrho_x = (double *) malloc(2 * DMnd * sizeof(double));
    DDrho_y = (double *) malloc(2 * DMnd * sizeof(double));
    DDrho_z = (double *) malloc(2 * DMnd * sizeof(double));
    sigma = (double *) malloc(2 * DMnd * sizeof(double));

    Dfeat_qp_mp = (double *) malloc(2 * DMnd * sizeof(double));
    DDfeat_qp_mp = (double *) malloc(2 * DMnd * sizeof(double));
    feat_qp_monopole = (double *) malloc(2 * DMnd * sizeof(double));
    alpha = (double *) malloc(2 * DMnd * sizeof(double));

    // memory allocation for global variables
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    global_rho = (double *) malloc(2 * Nd * sizeof(double));
    global_Df_featmp = (double *) malloc(2 * Nd * sizeof(double));


    // // gathering distributed rho vector into global vector and broadcast to all processors
    gather_distributed_vector(rho + DMnd, pSPARC->DMVertices, global_rho, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(rho + 2 * DMnd, pSPARC->DMVertices, global_rho + Nd, gridsizes, pSPARC->dmcomm_phi, 1);
    MPI_Bcast(global_rho, 2 * Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);

    // including monopole feature: feat_qp_monopole
    Conv_feat_vectors_dir(pSPARC, &mp, pSPARC->DMVertices, 2, global_rho, feat_qp_monopole, "000", pSPARC->dmcomm_phi);
    double *global_monopole_up, *global_monopole_dn;
    global_monopole_up = (double *) malloc(pSPARC->Nd * sizeof(double));
    global_monopole_dn = (double *) malloc(pSPARC->Nd * sizeof(double));
    gather_distributed_vector(feat_qp_monopole, pSPARC->DMVertices, global_monopole_up, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(feat_qp_monopole + DMnd, pSPARC->DMVertices, global_monopole_dn, gridsizes, pSPARC->dmcomm_phi, 1);
    char MonopoleUpFilename[128];
    char MonopoleDnFilename[128];
    if (rank == 0){
        snprintf(MonopoleUpFilename, 128, "feature_monopole_up.csv");
        snprintf(MonopoleDnFilename, 128, "feature_monopole_dn.csv");
        writeMatToFile(MonopoleUpFilename, global_monopole_up, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
        writeMatToFile(MonopoleDnFilename, global_monopole_dn, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    }

    Construct_alpha(pSPARC, feat_qp_monopole, m, n, DMnd, alpha);
    Construct_alpha(pSPARC, feat_qp_monopole + DMnd, m, n, DMnd, alpha + DMnd);
    double *global_alpha_up, *global_alpha_dn;
    global_alpha_up = (double *) malloc(pSPARC->Nd * sizeof(double));
    global_alpha_dn = (double *) malloc(pSPARC->Nd * sizeof(double));
    gather_distributed_vector(alpha, pSPARC->DMVertices, global_alpha_up, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(alpha + DMnd, pSPARC->DMVertices, global_alpha_dn, gridsizes, pSPARC->dmcomm_phi, 1);
    char alphaUpFilename[128];
    char alphaDnFilename[128];
    if (rank == 0){
        snprintf(alphaUpFilename, 128, "alpha_up.csv");
        snprintf(alphaDnFilename, 128, "alpha_dn.csv");
        writeMatToFile(alphaUpFilename, global_alpha_up, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
        writeMatToFile(alphaDnFilename, global_alpha_dn, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    }
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, rho+DMnd, Drho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, rho+DMnd, Drho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, rho+DMnd, Drho_z, 2, pSPARC->dmcomm_phi);
    
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        lapcT = (double *) malloc(6 * sizeof(double));
        lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
        for(i = 0; i < 2*DMnd; i++){
            sigma[i] = Drho_x[i] * (lapcT[0] * Drho_x[i] + lapcT[1] * Drho_y[i]) + Drho_y[i] * (lapcT[3] * Drho_y[i] + lapcT[4] * Drho_z[i]) +
                       Drho_z[i] * (lapcT[5] * Drho_z[i] + lapcT[2] * Drho_x[i]);
        }
        free(lapcT);
    } else {
        for(i = 0; i < 2*DMnd; i++){
            sigma[i] = Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i];
        }
    }

    for(i = 0; i < DMnd; i++) {
        rhom1_3 = pow(rho[i],-xc_cst->third);
        rhotot_inv = pow(rhom1_3,3.0);
        rhotmo6 = sqrt(rhom1_3);
        rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // First take care of the exchange part of the functional
        ex = 0.0;
        for(spn_i = 0; spn_i < 2; spn_i++){
            rho_updn = rho[DMnd + spn_i*DMnd + i];
            rho_updnm1_3 = pow(rho_updn, -xc_cst->third);
            rhomot = rho_updnm1_3;
            ex_lsd = -xc_cst->threefourth_divpi * xc_cst->sixpi2_1_3 * (rhomot * rhomot * rho_updn);
            rho_inv = rhomot * rhomot * rhomot;
            coeffss = (1.0/4.0) * xc_cst->sixpi2m1_3 * xc_cst->sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
            ss = sigma[spn_i*DMnd + i] * coeffss;
            
            divss = 1.0/(1.0 + ((xc_cst->mu_divkappa * ss)/alpha[spn_i*DMnd + i])); // alpha is added here
            dfxdss = xc_cst->mu * pow(divss, alpha[spn_i*DMnd + i]) * divss;
            
			fx = 1.0 + xc_cst->kappa * (1.0 - pow(divss, alpha[spn_i*DMnd + i]));
            ex_gga = ex_lsd * fx;
            dssdn = (-8.0/3.0) * (ss * rho_inv);
            dfxdn = dfxdss * dssdn;
            XPotential[spn_i*DMnd + i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);// spin up and spin down

            dssdg = 2.0 * coeffss;
            dfxdg = dfxdss * dssdg;
            Dxdgrho[spn_i*DMnd + i] = ex_lsd * rho_updn * dfxdg; // spin up and spin down for second part of the derivative
            ex += ex_gga * rho_updn;
            Df_alpha = -xc_cst->kappa * pow(divss, alpha[spn_i*DMnd + i])*(-log(1/divss) + ((xc_cst->mu_divkappa * ss * divss)/alpha[spn_i*DMnd + i]));
            Dalpha_qp = - ((4.0 - 0.75) * m)/(2.0 + exp(-m * (feat_qp_monopole[spn_i*DMnd + i]-n)) + exp(m*(feat_qp_monopole[spn_i*DMnd + i]-n)));
            Dfeat_qp_mp[spn_i*DMnd+i] = ex_lsd * rho_updn * Df_alpha * Dalpha_qp;
        }
        e_x[i] = ex * rhotot_inv;
    }

    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(i = 0; i < 2*DMnd; i++){
            temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * Dxdgrho[i];
            temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * Dxdgrho[i];
            temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * Dxdgrho[i];
            Drho_x[i] = temp1;
            Drho_y[i] = temp2;
            Drho_z[i] = temp3;
        }
    } else {
       for(i = 0; i < 2*DMnd; i++){
            Drho_x[i] *= Dxdgrho[i];
            Drho_y[i] *= Dxdgrho[i];
            Drho_z[i] *= Dxdgrho[i];
        }
    }
    
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);
    
    for(i = 0; i < DMnd; i++){
        XPotential[i] += - DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
        XPotential[DMnd + i] += - DDrho_x[DMnd + i] - DDrho_y[DMnd + i] - DDrho_z[DMnd + i];
    }  

    // add the contribution from the monopole
    for (int i = 0; i < 2; i++){
        gather_distributed_vector(Dfeat_qp_mp + i*DMnd, pSPARC->DMVertices, global_Df_featmp + i*Nd, gridsizes, pSPARC->dmcomm_phi, 1);
    }
    
    MPI_Bcast(global_Df_featmp, 2*Nd, MPI_DOUBLE, 0, pSPARC->dmcomm_phi);
    Conv_feat_vectors_dir(pSPARC, &mp, pSPARC->DMVertices, 2, global_Df_featmp, DDfeat_qp_mp, "000", pSPARC->dmcomm_phi);

    for(i=0; i < DMnd; i++){
        XPotential[i] +=  DDfeat_qp_mp[i];
        XPotential[DMnd + i] += DDfeat_qp_mp[DMnd + i];
    }


    // Deallocate memory
    free(feat_qp_monopole); free(Dfeat_qp_mp); free(DDfeat_qp_mp); free(global_Df_featmp);
    free(Drho_x); free(Drho_y); free(Drho_z);
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);
    free(global_rho);
    free(alpha);
}

/**
 * @brief function to calculate dipole feature by taking L2 norm of individual components 
 *
 */
void Construct_feature(SPARC_OBJ *pSPARC, const double *Dx, const double *Dy, const double *Dz, const int DMnd, double *featureVector){
    double feat_sq, *global_feat;
    int Nd = pSPARC->Nd;
    global_feat = (double *) malloc(Nd * sizeof(double));
    int rank;
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < DMnd; i++){
        feat_sq = Dx[i]*Dx[i] + Dy[i]*Dy[i] + Dz[i]*Dz[i];
        featureVector[i] = sqrt(feat_sq);
    }
    gather_distributed_vector(featureVector, pSPARC->DMVertices, global_feat, gridsizes, pSPARC->dmcomm_phi, 1);
    char FeatFilename[128];
    if (rank == 0){
        snprintf(FeatFilename, 128, "feature_dipole.csv");
        writeMatToFile(FeatFilename, global_feat, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    }
}

/**
 * @brief function to calculate alpha using any feature vector
 *
 */
void Construct_alpha(SPARC_OBJ *pSPARC, const double *featureVector, const double m, const double n, const int DMnd, double *alpha){
    double *global_alpha;
    int Nd = pSPARC->Nd;
    global_alpha = (double *) malloc(Nd * sizeof(double));
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    for (int i = 0; i < DMnd; i++){
        alpha[i] = 0.75 + ((4.0-0.75)/(1.0+exp(m*(featureVector[i] - n))));
        // alpha[i] = 0.52; //// comment out this line for constant alpha calculations
    }
}
