/**
 * @file    ConvolutionalGGAexchangeCorrelation.h
 * @brief   This file contains declaration of functions required for convolutional GGA.
 *			In convolutional GGA, the exchange potential is calculated using 3D convolutions
 *			of finite-difference stencils with electron density and correlation potential is
 *			 calculated using SPARC's original implementation.
 * @author  Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
 *          Andrew J. Medford <ajm@gatech.edu>
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
#include "ConvolutionalGGAexchangeCorrelation.h"
#include "MCSHTools.h"
#include "MCSHHelper.h"

/**
 * @brief  function to calculate the XC potential using GGA_CONV_PBE for spin-paired case
 *
 */
void Calculate_Vxc_GGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho) 
{
    // variable for X potential and C potential, ex and ec energy densities
    double *XPotential, *CPotential, *e_c, *e_x, *e_xc, *Dxdgrho, *Dcdgrho;
    int DMnd, i;
    DMnd = pSPARC->Nd_d;
    XPotential = (double *) malloc(DMnd * sizeof(double));
    CPotential = (double *) malloc(DMnd * sizeof(double));
    // XCPotential = (double *) malloc(DMnd * sizeof(double));
    e_c = (double *) malloc(DMnd * sizeof(double));
    e_x = (double *) malloc(DMnd * sizeof(double));
    e_xc = (double *) malloc(DMnd * sizeof(double));
    Dxdgrho = (double *) malloc(DMnd * sizeof(double));
    Dcdgrho = (double *) malloc(DMnd * sizeof(double));
    // Dxcdgrho = (double *) malloc(DMnd * sizeof(double));

    Calculate_Vx_GGA_CONV_PBE(pSPARC, xc_cst, rho, XPotential, e_x, Dxdgrho);
    Calculate_Vc_GGA_CONV_PBE(pSPARC, xc_cst, rho, CPotential, e_c, Dcdgrho);

    for(i = 0; i < DMnd ; i++) {
        pSPARC->Dxcdgrho[i] = Dxdgrho[i] + Dcdgrho[i];
    }

    for(i = 0; i < DMnd; i++){
        pSPARC->XCPotential[i] = XPotential[i] + CPotential[i];
        pSPARC->e_xc[i] = e_x[i] + e_c[i];
    }

// // #define DEBUGCONV
// #ifdef DEBUGCONV
// 	#define max(a,b) ((a)>(b)?(a):(b))
// 	// double *ref_gradient_result = calloc( DMnd, sizeof(double));
	
// 	int is = pSPARC->DMVertices[0];
// 	int ie = pSPARC->DMVertices[1];
// 	int js = pSPARC->DMVertices[2];
// 	int je = pSPARC->DMVertices[3];
// 	int ks = pSPARC->DMVertices[4];
// 	int ke = pSPARC->DMVertices[5];
// 	// check convolve6 answer
// 	double err_vxc = 0.0;
//     // double err_dxcdgrho = 0.0;
//     double err_e_xc = 0.0;

// 	int outputIndex = 0;
// 	for (int k = ks; k <= ke; k++) {
// 		for (int j = js; j <= je; j++) {
// 			for (int i = is; i <= ie; i++) {
// 				// int ind_global = k * imageDimX * imageDimY + j * imageDimX + i;
// 				err_vxc = max(fabs(pSPARC->XCPotential[outputIndex] - XCPotential[outputIndex]), err_vxc);
//                 // err_dxcdgrho = max(fabs(pSPARC->Dxcdgrho[outputIndex] - Dxcdgrho[outputIndex]), err_dxcdgrho);
//                 err_e_xc = max(fabs(pSPARC->e_xc[outputIndex] - e_xc[outputIndex]), err_e_xc);
// 				outputIndex++;
// 			}
// 		}
// 	}
//     printf("error in XC: %.15f, error in exc: %.15f \n", err_vxc, err_e_xc);
// 	assert(err_vxc < 1e-8);
//     assert(err_e_xc < 1e-8);
// 	printf("Test passed!\n");
// #endif

    free(XPotential); free(CPotential);
    free(e_xc); free(e_x);
    free(e_c); free(Dxdgrho); free(Dcdgrho);
}

/**
 * @brief   function to calculate the exchange potential using GGA_CONV_PBE
 * 
 */
void Calculate_Vx_GGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *XPotential, double *e_x, double *Dxdgrho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vx (GGA_CONV_PBE) ...\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    // JUST EXCHANGE
    double rho_updn, rho_updnm1_3, rhom1_3, rhotot_inv, rhotmo6, rhoto6, rhomot, ex_lsd, rho_inv, coeffss, ss;
    double divss, dfxdss, fx, ex_gga, dssdn, dfxdn, dssdg, dfxdg, ex;

    double temp1, temp2, temp3;
    int DMnd, i;
    DMnd = pSPARC->Nd_d;

    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;
    double *global_rho, *global_Drho_x, *global_Drho_y, *global_Drho_z;

    Drho_x = (double *) malloc(DMnd * sizeof(double));
    Drho_y = (double *) malloc(DMnd * sizeof(double));
    Drho_z = (double *) malloc(DMnd * sizeof(double));
    DDrho_x = (double *) malloc(DMnd * sizeof(double));
    DDrho_y = (double *) malloc(DMnd * sizeof(double));
    DDrho_z = (double *) malloc(DMnd * sizeof(double));
    sigma = (double *) malloc(DMnd * sizeof(double));

    // memory allocation for global variables
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    global_rho = (double *) malloc(pSPARC->Nd * sizeof(double));
    global_Drho_x = (double *) malloc(pSPARC->Nd * sizeof(double));
    global_Drho_y = (double *) malloc(pSPARC->Nd * sizeof(double));
    global_Drho_z = (double *) malloc(pSPARC->Nd * sizeof(double));

    // gathering distributed rho vector into global vector and broadcast to all processors
    gather_distributed_vector(rho, pSPARC->DMVertices, global_rho, gridsizes, pSPARC->dmcomm_phi, 1);
    MPI_Bcast(global_rho, pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);

    // using convolution for calculating gradient
    // TODO: First check is that the derivative with the value you get from SPARC gradient function
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 1, global_rho, Drho_x, 0, pSPARC->dmcomm_phi);
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 1, global_rho, Drho_y, 1, pSPARC->dmcomm_phi);
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 1, global_rho, Drho_z, 2, pSPARC->dmcomm_phi); 

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

        if (strcmpi(pSPARC->XC,"GGA_CONV_PBE") == 0) {
            divss = 1.0/(1.0 + xc_cst->mu_divkappa * ss);
            dfxdss = xc_cst->mu * (divss * divss);
            //d2fxdss2 = -xc_cst->mu * 2.0 * xc_cst->mu_divkappa * (divss * divss * divss);
        } else {
            printf("Unrecognized GGA functional: %s\n",pSPARC->XC);
            exit(EXIT_FAILURE);
        }
        fx = 1.0 + xc_cst->kappa * (1.0 - divss);
            
        ex_gga = ex_lsd * fx;
        dssdn = (-8.0/3.0) * (ss * rho_inv);
        dfxdn = dfxdss * dssdn;
        XPotential[i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);

        dssdg = 2.0 * coeffss;
        dfxdg = dfxdss * dssdg;
        Dxdgrho[i] = 0.5 * ex_lsd * rho_updn * dfxdg;
        ex = ex_gga * rho_updn;

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
    gather_distributed_vector(Drho_x, pSPARC->DMVertices, global_Drho_x, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(Drho_y, pSPARC->DMVertices, global_Drho_y, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(Drho_z, pSPARC->DMVertices, global_Drho_z, gridsizes, pSPARC->dmcomm_phi, 1);
    MPI_Bcast(global_Drho_x, pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    MPI_Bcast(global_Drho_y, pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    MPI_Bcast(global_Drho_z, pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);

    // convolution will be the derivative of global_Drho_x
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 1, global_Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 1, global_Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 1, global_Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);

    for(i = 0; i < DMnd; i++){
        //if(pSPARC->electronDens[i] != 0.0)
        XPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
    }
    // Deallocate memory
    free(Drho_x); free(Drho_y); free(Drho_z);
    free(global_rho);
    free(global_Drho_x); free(global_Drho_y); free(global_Drho_z); 
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);
}

/**
 * @brief   function to calculate the correlation potential using standard GGA_PBE implementation
 *
 */
void Calculate_Vc_GGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *CPotential, double *e_c, double *Dcdgrho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vc (GGA_CONV_PBE or GGA_CONV_PBE_MULTIPOLE) ...\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    double rho_updn, rho_updnm1_3, rhom1_3, rhotot_inv, rhotmo6, rhoto6, rhomot, ex_lsd, rho_inv;
    double ec, rs, sqr_rs, rsm1_2;
    double ec0_q0, ec0_q1, ec0_q1p, ec0_den, ec0_log, ecrs0, decrs0_drs, ecrs, decrs_drs;
    double phi_zeta_inv, phi3_zeta, gamphi3inv, bb, dbb_drs, exp_pbe, cc, dcc_dbb, dcc_drs, coeff_aa, aa, daa_drs;
    double grrho2, dtt_dg, tt, xx, dxx_drs, dxx_dtt, pade_den, pade, dpade_dxx, dpade_drs, dpade_dtt, coeff_qq, qq, dqq_drs, dqq_dtt;
    double arg_rr, div_rr, rr, drr_dqq, drr_drs, drr_dtt, hh, dhh_dtt, dhh_drs, drhohh_drho; 

    double temp1, temp2, temp3;
    int DMnd, i;
    DMnd = pSPARC->Nd_d;

    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;
    Drho_x = (double *) malloc(DMnd * sizeof(double));
    Drho_y = (double *) malloc(DMnd * sizeof(double));
    Drho_z = (double *) malloc(DMnd * sizeof(double));
    DDrho_x = (double *) malloc(DMnd * sizeof(double));
    DDrho_y = (double *) malloc(DMnd * sizeof(double));
    DDrho_z = (double *) malloc(DMnd * sizeof(double));
    sigma = (double *) malloc(DMnd * sizeof(double));

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

    // Compute correlation
    phi_zeta_inv = 1.0;
    phi3_zeta = 1.0;
    gamphi3inv = xc_cst->gamma_inv;
    
    for(i = 0; i < DMnd; i++){
        rho_updn = rho[i]/2.0;
        rho_updnm1_3 = pow(rho_updn, -xc_cst->third);
        rhom1_3 = xc_cst->twom1_3 * rho_updnm1_3;
        rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        rhotmo6 = sqrt(rhom1_3);
        rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // Then takes care of the LSD correlation part of the functional
        rs = xc_cst->rsfac * rhom1_3;
        sqr_rs = xc_cst->sq_rsfac * rhotmo6;
        rsm1_2 = xc_cst->sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD
        ec0_q0 = -2.0 * xc_cst->ec0_aa * (1.0 + xc_cst->ec0_a1 * rs);
        ec0_q1 = 2.0 * xc_cst->ec0_aa * (xc_cst->ec0_b1 * sqr_rs + xc_cst->ec0_b2 * rs + xc_cst->ec0_b3 * rs * sqr_rs + xc_cst->ec0_b4 * rs * rs);
        ec0_q1p = xc_cst->ec0_aa * (xc_cst->ec0_b1 * rsm1_2 + 2.0 * xc_cst->ec0_b2 + 3.0 * xc_cst->ec0_b3 * sqr_rs + 4.0 * xc_cst->ec0_b4 * rs);
        ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
        ecrs0 = ec0_q0 * ec0_log;
        decrs0_drs = -2.0 * xc_cst->ec0_aa * xc_cst->ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        ecrs = ecrs0;
        decrs_drs = decrs0_drs;
        //decrs_dzeta = 0.0;
        //zeta = 0.0;

        // Add LSD correlation functional to GGA exchange functional
        e_c[i] = ecrs;
        CPotential[i] = ecrs - (rs/3.0) * decrs_drs;

        // From ec to bb
        bb = ecrs * gamphi3inv;
        dbb_drs = decrs_drs * gamphi3inv;
        // dbb_dzeta = gamphi3inv * (decrs_dzeta - 3.0 * ecrs * phi_logder);

        // From bb to cc
        exp_pbe = exp(-bb);
        cc = 1.0/(exp_pbe - 1.0);
        dcc_dbb = cc * cc * exp_pbe;
        dcc_drs = dcc_dbb * dbb_drs;
        // dcc_dzeta = dcc_dbb * dbb_dzeta;

        // From cc to aa
        coeff_aa = xc_cst->beta * xc_cst->gamma_inv * phi_zeta_inv * phi_zeta_inv;
        aa = coeff_aa * cc;
        daa_drs = coeff_aa * dcc_drs;
        //daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // Introduce tt : do not assume that the spin-dependent gradients are collinear
        grrho2 = sigma[i];
        dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * xc_cst->coeff_tt;
        // Note that tt is (the t variable of PBE divided by phi) squared
        tt = 0.5 * grrho2 * dtt_dg;

        // Get xx from aa and tt
        xx = aa * tt;
        dxx_drs = daa_drs * tt;
        //dxx_dzeta = daa_dzeta * tt;
        dxx_dtt = aa;

        // From xx to pade
        pade_den = 1.0/(1.0 + xx * (1.0 + xx));
        pade = (1.0 + xx) * pade_den;
        dpade_dxx = -xx * (2.0 + xx) * pow(pade_den,2);
        dpade_drs = dpade_dxx * dxx_drs;
        dpade_dtt = dpade_dxx * dxx_dtt;
        //dpade_dzeta = dpade_dxx * dxx_dzeta;

        // From pade to qq
        coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        qq = coeff_qq * pade;
        dqq_drs = coeff_qq * dpade_drs;
        dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;
        //dqq_dzeta = coeff_qq * (dpade_dzeta - 2.0 * pade * phi_logder);

        // From qq to rr
        arg_rr = 1.0 + xc_cst->beta * xc_cst->gamma_inv * qq;
        div_rr = 1.0/arg_rr;
        rr = xc_cst->gamma * log(arg_rr);
        drr_dqq = xc_cst->beta * div_rr;
        drr_drs = drr_dqq * dqq_drs;
        drr_dtt = drr_dqq * dqq_dtt;
        //drr_dzeta = drr_dqq * dqq_dzeta;

        // From rr to hh
        hh = phi3_zeta * rr;
        dhh_drs = phi3_zeta * drr_drs;
        dhh_dtt = phi3_zeta * drr_dtt;
        //dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

        // The GGA correlation energy is added
        e_c[i] += hh;

        // From hh to the derivative of the energy wrt the density
        drhohh_drho = hh - xc_cst->third * rs * dhh_drs - (7.0/3.0) * tt * dhh_dtt; //- zeta * dhh_dzeta 
        CPotential[i] += drhohh_drho;
        Dcdgrho[i] = rho[i] * dtt_dg * dhh_dtt;
    }
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(i = 0; i < DMnd; i++){
            temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * Dcdgrho[i];
            temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * Dcdgrho[i];
            temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * Dcdgrho[i];
            Drho_x[i] = temp1;
            Drho_y[i] = temp2;
            Drho_z[i] = temp3;
        }
    } else {
        for(i = 0; i < DMnd; i++){
            Drho_x[i] *= Dcdgrho[i];
            Drho_y[i] *= Dcdgrho[i];
            Drho_z[i] *= Dcdgrho[i];
        }
    }

    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);
    for(i = 0; i < DMnd; i++){
        //if(pSPARC->electronDens[i] != 0.0)
        CPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
    }

    // Deallocate memory
    free(Drho_x); free(Drho_y); free(Drho_z);
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);
}

/**
 * @brief  function to calculate the XC potential using GGA_CONV_PBE for spin-polarized case
 *
 */
void Calculate_Vxc_GSGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho) {
    // variable for X potential and C potential, ex and ec energy densities
    double *XPotential, *CPotential, *e_c, *e_x, *e_xc, *Dxdgrho, *Dcdgrho;
    int DMnd, i;
    int ncopy = pSPARC->Nspden/2*2+1;

    DMnd = pSPARC->Nd_d;
    
    XPotential = (double *) malloc(pSPARC->Nspden * DMnd * sizeof(double));// 2*DMnd
    CPotential = (double *) malloc(pSPARC->Nspden * DMnd * sizeof(double));// 2*DMnd
    e_c = (double *) malloc(DMnd * sizeof(double));
    e_x = (double *) malloc(DMnd * sizeof(double));
    e_xc = (double *) malloc(DMnd * sizeof(double));
    Dxdgrho = (double *) malloc(DMnd * pSPARC->Nspden * sizeof(double));// up and down
    Dcdgrho = (double *) malloc(DMnd * sizeof(double));

    Calculate_Vx_GSGA_CONV_PBE(pSPARC, xc_cst, rho, XPotential, e_x, Dxdgrho);
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
* @brief function to calculate exchange potential for spin polarized case using GGA_CONV_PBE
*
*/
void Calculate_Vx_GSGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *XPotential, double *e_x, double *Dxdgrho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vx (GSGA_CONV_PBE) ...\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    // JUST EXCHANGE
    double rho_updn, rho_updnm1_3, rhom1_3, rhotot_inv, rhotmo6, rhoto6, rhomot, ex_lsd, rho_inv, coeffss, ss;
    double divss, dfxdss, fx, ex_gga, dssdn, dfxdn, dssdg, dfxdg, ex;

    double temp1, temp2, temp3;
    int DMnd, i, spn_i;
    DMnd = pSPARC->Nd_d;

    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;
    double *global_rho, *global_Drho_x, *global_Drho_y, *global_Drho_z;

    Drho_x = (double *) malloc(2 * DMnd * sizeof(double));
    Drho_y = (double *) malloc(2 * DMnd * sizeof(double));
    Drho_z = (double *) malloc(2 * DMnd * sizeof(double));
    DDrho_x = (double *) malloc(2 * DMnd * sizeof(double));
    DDrho_y = (double *) malloc(2 * DMnd * sizeof(double));
    DDrho_z = (double *) malloc(2 * DMnd * sizeof(double));
    sigma = (double *) malloc(2 * DMnd * sizeof(double));

    // // memory allocation for global variables
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    global_rho = (double *) malloc(2 * pSPARC->Nd * sizeof(double));// 2*Nd (rho_up and rho_dn)
    global_Drho_x = (double *) malloc(2 * pSPARC->Nd * sizeof(double));
    global_Drho_y = (double *) malloc(2 * pSPARC->Nd * sizeof(double));
    global_Drho_z = (double *) malloc(2 * pSPARC->Nd * sizeof(double));

    // // gathering distributed rho vector into global vector and broadcast to all processors (rho_up and rho_dn)
    gather_distributed_vector(rho + DMnd, pSPARC->DMVertices, global_rho, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(rho + 2 * DMnd, pSPARC->DMVertices, global_rho + pSPARC->Nd, gridsizes, pSPARC->dmcomm_phi, 1);
    MPI_Bcast(global_rho, 2 * pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);

    // // Gradient of rho_up and rho_dn
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 2, global_rho, Drho_x, 0, pSPARC->dmcomm_phi);
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 2, global_rho, Drho_y, 1, pSPARC->dmcomm_phi);
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 2, global_rho, Drho_z, 2, pSPARC->dmcomm_phi);
    
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
            ss = sigma[spn_i*DMnd + i] * coeffss;// sigma is 2 * DMnd
            
            if (strcmpi(pSPARC->XC,"GGA_CONV_PBE") == 0) {
                divss = 1.0/(1.0 + xc_cst->mu_divkappa * ss);
                dfxdss = xc_cst->mu * (divss * divss);
            } else {
                printf("Unrecognized GGA functional: %s\n",pSPARC->XC);
                exit(EXIT_FAILURE);
            }
            
			fx = 1.0 + xc_cst->kappa * (1.0 - divss);
            ex_gga = ex_lsd * fx;
            dssdn = (-8.0/3.0) * (ss * rho_inv);
            dfxdn = dfxdss * dssdn;
            XPotential[spn_i*DMnd + i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);// spin up and spin down

            dssdg = 2.0 * coeffss;
            dfxdg = dfxdss * dssdg;
            Dxdgrho[spn_i*DMnd + i] = ex_lsd * rho_updn * dfxdg; // spin up and spin down
            ex += ex_gga * rho_updn;
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
    
    gather_distributed_vector(Drho_x, pSPARC->DMVertices, global_Drho_x, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(Drho_y, pSPARC->DMVertices, global_Drho_y, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(Drho_z, pSPARC->DMVertices, global_Drho_z, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(Drho_x + DMnd, pSPARC->DMVertices, global_Drho_x + pSPARC->Nd, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(Drho_y + DMnd, pSPARC->DMVertices, global_Drho_y + pSPARC->Nd, gridsizes, pSPARC->dmcomm_phi, 1);
    gather_distributed_vector(Drho_z + DMnd, pSPARC->DMVertices, global_Drho_z + pSPARC->Nd, gridsizes, pSPARC->dmcomm_phi, 1);

    MPI_Bcast(global_Drho_x, 2 * pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    MPI_Bcast(global_Drho_y, 2 * pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    MPI_Bcast(global_Drho_z, 2 * pSPARC->Nd, MPI_DOUBLE,  0, MPI_COMM_WORLD);

    // // convolution will be the derivative of global_Drho_x
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 2, global_Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 2, global_Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Conv_gradient_vectors_dir(pSPARC, pSPARC->DMVertices, 2, global_Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);
    
    for(i = 0; i < DMnd; i++){
        XPotential[i] += - DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
        XPotential[DMnd + i] += - DDrho_x[DMnd + i] - DDrho_y[DMnd + i] - DDrho_z[DMnd + i];
    }  

    // Deallocate memory
    free(Drho_x); free(Drho_y); free(Drho_z);
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);
}

/**
* @brief function to calculate correlation potential for spin polarized case using GGA_CONV_PBE
*
*/
void Calculate_Vc_GSGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *CPotential, double *e_c, double *Dcdgrho){

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vc (GSGA_CONV_PBE) ...\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return;
    }
    double rhom1_3, rhotot_inv, zeta, zetp, zetm, zetpm1_3, zetmm1_3;
    double rho_updn, rho_updnm1_3, rhotmo6, rhoto6, rhomot, ex_lsd, rho_inv, coeffss, ss;
    double divss, dfxdss, fx, ex_gga, dssdn, dfxdn, dssdg, dfxdg, exc, rs, sqr_rs, rsm1_2;
    double ec0_q0, ec0_q1, ec0_q1p, ec0_den, ec0_log, ecrs0, decrs0_drs, mac_q0, mac_q1, mac_q1p, mac_den, mac_log, macrs, dmacrs_drs;
    double ec1_q0, ec1_q1, ec1_q1p, ec1_den, ec1_log, ecrs1, decrs1_drs, zetp_1_3, zetm_1_3, f_zeta, fp_zeta, zeta4;
    double gcrs, ecrs, dgcrs_drs, decrs_drs, dfzeta4_dzeta, decrs_dzeta, vxcadd;
    double phi_zeta, phip_zeta, phi_zeta_inv, phi_logder, phi3_zeta, gamphi3inv;
    double bb, dbb_drs, dbb_dzeta, exp_pbe, cc, dcc_dbb, dcc_drs, dcc_dzeta, coeff_aa, aa, daa_drs, daa_dzeta;
    double grrho2, dtt_dg, tt, xx, dxx_drs, dxx_dzeta, dxx_dtt, pade_den, pade, dpade_dxx, dpade_drs, dpade_dtt, dpade_dzeta;
    double coeff_qq, qq, dqq_drs, dqq_dtt, dqq_dzeta, arg_rr, div_rr, rr, drr_dqq, drr_drs, drr_dtt, drr_dzeta;
    double hh, dhh_dtt, dhh_drs, dhh_dzeta, drhohh_drho;

    double temp1, temp2, temp3;
    int DMnd, i, spn_i;
    DMnd = pSPARC->Nd_d;

    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;

    // only one column needed for correlation part
    Drho_x = (double *) malloc(DMnd * sizeof(double));
    Drho_y = (double *) malloc(DMnd * sizeof(double));
    Drho_z = (double *) malloc(DMnd * sizeof(double));
    DDrho_x = (double *) malloc(DMnd * sizeof(double));
    DDrho_y = (double *) malloc(DMnd * sizeof(double));
    DDrho_z = (double *) malloc(DMnd * sizeof(double));
    sigma = (double *) malloc(DMnd * sizeof(double));

    // Gradient of total electron density (only the total electron density- need to verify)
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

    for(i = 0; i < DMnd; i++) {
        rhom1_3 = pow(rho[i],-xc_cst->third);
        rhotot_inv = pow(rhom1_3,3.0);
        zeta = (rho[DMnd + i] - rho[2*DMnd + i]) * rhotot_inv;
        zetp = 1.0 + zeta * xc_cst->alpha_zeta;
        zetm = 1.0 - zeta * xc_cst->alpha_zeta;
        zetpm1_3 = pow(zetp,-xc_cst->third);
        zetmm1_3 = pow(zetm,-xc_cst->third);
        rhotmo6 = sqrt(rhom1_3);
        rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;
        e_c[i] = 0.0;

        // Take care of the LSD correlation part of the functional
        rs = xc_cst->rsfac * rhom1_3;
        sqr_rs = xc_cst->sq_rsfac * rhotmo6;
        rsm1_2 = xc_cst->sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD
        ec0_q0 = -2.0 * xc_cst->ec0_aa * (1.0 + xc_cst->ec0_a1 * rs);
        ec0_q1 = 2.0 * xc_cst->ec0_aa *(xc_cst->ec0_b1 * sqr_rs + xc_cst->ec0_b2 * rs + xc_cst->ec0_b3 * rs * sqr_rs + xc_cst->ec0_b4 * rs * rs);
        ec0_q1p = xc_cst->ec0_aa * (xc_cst->ec0_b1 * rsm1_2 + 2.0 * xc_cst->ec0_b2 + 3.0 * xc_cst->ec0_b3 * sqr_rs + 4.0 * xc_cst->ec0_b4 * rs);
        ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
        ecrs0 = ec0_q0 * ec0_log;
        decrs0_drs = -2.0 * xc_cst->ec0_aa * xc_cst->ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        mac_q0 = -2.0 * xc_cst->mac_aa * (1.0 + xc_cst->mac_a1 * rs);
        mac_q1 = 2.0 * xc_cst->mac_aa * (xc_cst->mac_b1 * sqr_rs + xc_cst->mac_b2 * rs + xc_cst->mac_b3 * rs * sqr_rs + xc_cst->mac_b4 * rs * rs);
        mac_q1p = xc_cst->mac_aa * (xc_cst->mac_b1 * rsm1_2 + 2.0 * xc_cst->mac_b2 + 3.0 * xc_cst->mac_b3 * sqr_rs + 4.0 * xc_cst->mac_b4 * rs);
        mac_den = 1.0/(mac_q1 * mac_q1 + mac_q1);
        mac_log = -log( mac_q1 * mac_q1 * mac_den );
        macrs = mac_q0 * mac_log;
        dmacrs_drs = -2.0 * xc_cst->mac_aa * xc_cst->mac_a1 * mac_log - mac_q0 * mac_q1p * mac_den;

        //zeta = (rho(:,2) - rho(:,3)) .* rhotot_inv;
        ec1_q0 = -2.0 * xc_cst->ec1_aa * (1.0 + xc_cst->ec1_a1 * rs);
        ec1_q1 = 2.0 * xc_cst->ec1_aa * (xc_cst->ec1_b1 * sqr_rs + xc_cst->ec1_b2 * rs + xc_cst->ec1_b3 * rs * sqr_rs + xc_cst->ec1_b4 * rs * rs);
        ec1_q1p = xc_cst->ec1_aa * (xc_cst->ec1_b1 * rsm1_2 + 2.0 * xc_cst->ec1_b2 + 3.0 * xc_cst->ec1_b3 * sqr_rs + 4.0 * xc_cst->ec1_b4 * rs);
        ec1_den = 1.0/(ec1_q1 * ec1_q1 + ec1_q1);
        ec1_log = -log( ec1_q1 * ec1_q1 * ec1_den );
        ecrs1 = ec1_q0 * ec1_log;
        decrs1_drs = -2.0 * xc_cst->ec1_aa * xc_cst->ec1_a1 * ec1_log - ec1_q0 * ec1_q1p * ec1_den;
        
        // xc_cst->alpha_zeta is introduced in order to remove singularities for fully polarized systems.
        zetp_1_3 = (1.0 + zeta * xc_cst->alpha_zeta) * pow(zetpm1_3,2.0);
        zetm_1_3 = (1.0 - zeta * xc_cst->alpha_zeta) * pow(zetmm1_3,2.0);

        f_zeta = ( (1.0 + zeta * xc_cst->alpha_zeta2) * zetp_1_3 + (1.0 - zeta * xc_cst->alpha_zeta2) * zetm_1_3 - 2.0 ) * xc_cst->factf_zeta;
        fp_zeta = ( zetp_1_3 - zetm_1_3 ) * xc_cst->factfp_zeta;
        zeta4 = pow(zeta, 4.0);

        gcrs = ecrs1 - ecrs0 + macrs * xc_cst->fsec_inv;
        ecrs = ecrs0 + f_zeta * (zeta4 * gcrs - macrs * xc_cst->fsec_inv);
        dgcrs_drs = decrs1_drs - decrs0_drs + dmacrs_drs * xc_cst->fsec_inv;
        decrs_drs = decrs0_drs + f_zeta * (zeta4 * dgcrs_drs - dmacrs_drs * xc_cst->fsec_inv);
        dfzeta4_dzeta = 4.0 * pow(zeta,3.0) * f_zeta + fp_zeta * zeta4;
        decrs_dzeta = dfzeta4_dzeta * gcrs - fp_zeta * macrs * xc_cst->fsec_inv;

        e_c[i]  = ecrs;
        vxcadd = ecrs - rs * xc_cst->third * decrs_drs - zeta * decrs_dzeta;
        CPotential[i] = vxcadd + decrs_dzeta;
        CPotential[DMnd+i] = vxcadd - decrs_dzeta;

        // Eventually add the GGA correlation part of the PBE functional
        // The definition of phi has been slightly changed, because
        // the original PBE one gives divergent behaviour for fully polarized points

        phi_zeta = ( zetpm1_3 * (1.0 + zeta * xc_cst->alpha_zeta) + zetmm1_3 * (1.0 - zeta * xc_cst->alpha_zeta)) * 0.5;
        phip_zeta = (zetpm1_3 - zetmm1_3) * xc_cst->third * xc_cst->alpha_zeta;
        phi_zeta_inv = 1.0/phi_zeta;
        phi_logder = phip_zeta * phi_zeta_inv;
        phi3_zeta = phi_zeta * phi_zeta * phi_zeta;
        gamphi3inv = xc_cst->gamma_inv * phi_zeta_inv * phi_zeta_inv * phi_zeta_inv;

        // From ec to bb
        bb = ecrs * gamphi3inv;
        dbb_drs = decrs_drs * gamphi3inv;
        dbb_dzeta = gamphi3inv * (decrs_dzeta - 3.0 * ecrs * phi_logder);

        // From bb to cc
        exp_pbe = exp(-bb);
        cc = 1.0/(exp_pbe - 1.0);
        dcc_dbb = cc * cc * exp_pbe;
        dcc_drs = dcc_dbb * dbb_drs;
        dcc_dzeta = dcc_dbb * dbb_dzeta;

        // From cc to aa
        coeff_aa = xc_cst->beta * xc_cst->gamma_inv * phi_zeta_inv * phi_zeta_inv;
        aa = coeff_aa * cc;
        daa_drs = coeff_aa * dcc_drs;
        daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // From cc to aa
        coeff_aa = xc_cst->beta * xc_cst->gamma_inv * phi_zeta_inv * phi_zeta_inv;
        aa = coeff_aa * cc;
        daa_drs = coeff_aa * dcc_drs;
        daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // Introduce tt : do not assume that the spin-dependent gradients are collinear
        grrho2 = sigma[i];
        dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * xc_cst->coeff_tt;
        // Note that tt is (the t variable of PBE divided by phi) squared
        tt = 0.5 * grrho2 * dtt_dg;

        // Get xx from aa and tt
        xx = aa * tt;
        dxx_drs = daa_drs * tt;
        dxx_dzeta = daa_dzeta * tt;
        dxx_dtt = aa;

        // From xx to pade
        pade_den = 1.0/(1.0 + xx * (1.0 + xx));
        pade = (1.0 + xx) * pade_den;
        dpade_dxx = -xx * (2.0 + xx) * pow(pade_den,2.0);
        dpade_drs = dpade_dxx * dxx_drs;
        dpade_dtt = dpade_dxx * dxx_dtt;
        dpade_dzeta = dpade_dxx * dxx_dzeta;

        // From pade to qq
        coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        qq = coeff_qq * pade;
        dqq_drs = coeff_qq * dpade_drs;
        dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;
        dqq_dzeta = coeff_qq * (dpade_dzeta - 2.0 * pade * phi_logder);

        // From qq to rr
        arg_rr = 1.0 + xc_cst->beta * xc_cst->gamma_inv * qq;
        div_rr = 1.0/arg_rr;
        rr = xc_cst->gamma * log(arg_rr);
        drr_dqq = xc_cst->beta * div_rr;
        drr_drs = drr_dqq * dqq_drs;
        drr_dtt = drr_dqq * dqq_dtt;
        drr_dzeta = drr_dqq * dqq_dzeta;

        // From rr to hh
        hh = phi3_zeta * rr;
        dhh_drs = phi3_zeta * drr_drs;
        dhh_dtt = phi3_zeta * drr_dtt;
        dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

        // The GGA correlation energy is added
        e_c[i] += hh;

        // From hh to the derivative of the energy wrt the density
        drhohh_drho = hh - xc_cst->third * rs * dhh_drs - zeta * dhh_dzeta - (7.0/3.0) * tt * dhh_dtt; 
        CPotential[i] += drhohh_drho + dhh_dzeta;
        CPotential[DMnd + i] += drhohh_drho - dhh_dzeta;

        // From hh to the derivative of the energy wrt to the gradient of the
        // density, divided by the gradient of the density
        // (The v3.3 definition includes the division by the norm of the gradient)
        Dcdgrho[i] = (rho[i] * dtt_dg * dhh_dtt);// correlation part
    }

    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(i = 0; i < DMnd; i++){
            temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * Dcdgrho[i];
            temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * Dcdgrho[i];
            temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * Dcdgrho[i];
            Drho_x[i] = temp1;
            Drho_y[i] = temp2;
            Drho_z[i] = temp3;
        }
    } else {
       for(i = 0; i < DMnd; i++){
            Drho_x[i] *= Dcdgrho[i];
            Drho_y[i] *= Dcdgrho[i];
            Drho_z[i] *= Dcdgrho[i];
        }
    }

    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);

    for(i = 0; i < DMnd; i++){
        CPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
        CPotential[DMnd + i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
    }
    // Deallocate memory
    free(Drho_x); free(Drho_y); free(Drho_z);
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);
}
