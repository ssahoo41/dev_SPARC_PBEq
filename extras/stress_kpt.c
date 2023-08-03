/**
 * @brief    Calculate nonlocal + kinetic components of stress.
 */
void Calculate_nonlocal_kinetic_stress_kpt(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, j, k, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, i_DM, j_DM, k_DM;
    int dim, dim2, atom_index, count, count2, l, m, lmax, kpt, Nk, size_k, spn_i, nspin, size_s;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    
    double complex *alpha, *beta, *psi_ptr, *dpsi_ptr, *psi_rc, *psi_rc_ptr, *dpsi_full;
    double complex *beta1_x1, *beta2_x1, *beta2_x2, *beta3_x1, *beta3_x2, *beta3_x3;
    double complex *dpsi_xi, *dpsi_xj, *dpsi_x1, *dpsi_x2, *dpsi_x3, *dpsi_xi_lv, *dpsi_xi_rc, *dpsi_xi_rc_ptr;
    double SJ[6], eJ, temp_k, temp_e, temp_s[6], temp2_e, temp2_s[6], g_nk, gamma_jl, kptwt,  R1, R2, R3, x1_R1, x2_R2, x3_R3;
    double dpsii_dpsij, energy_nl = 0.0, stress_k[6], stress_nl[6], StXmRjp;
    
    for (i = 0; i < 6; i++) SJ[i] = temp_s[i] = temp2_s[i] = stress_nl[i] = stress_k[i] = 0;

    dpsi_full = (double complex *)malloc( 2 * size_s * nspin * sizeof(double complex) );  // dpsi_y, dpsi_z in cartesian coordinates, dpsi_x saved in Yorb_kpt 
    assert(dpsi_full != NULL);
    
    alpha = (double complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 7, sizeof(double complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, kpt_vec;
    double complex bloch_fac, a, b;
#ifdef DEBUG 
    if (!rank) printf("Start calculating stress contributions from kinetic and nonlocal psp. \n");
#endif

    if (pSPARC->cell_typ == 0){
        for (dim = 0; dim < 3; dim++) {
            // count = 0;
            for(spn_i = 0; spn_i < nspin; spn_i++) {
                for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                    k1 = pSPARC->k1_loc[kpt];
                    k2 = pSPARC->k2_loc[kpt];
                    k3 = pSPARC->k3_loc[kpt];
                    kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                    // find dPsi in direction dim
                    dpsi_xi = (dim == 0) ? pSPARC->Yorb_kpt : (dpsi_full + (dim-1)*size_s*nspin );
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, dpsi_xi+spn_i*size_s+kpt*size_k, dim, kpt_vec, pSPARC->dmcomm);
                }
            }
        }
    } else {
        dpsi_xi_lv = (double complex *)malloc( size_k * sizeof(double complex) );  // dpsi_x, dpsi_y, dpsi_z along lattice vecotrs
        assert(dpsi_xi_lv != NULL);
        dpsi_x1 = pSPARC->Yorb_kpt;
        dpsi_x2 = dpsi_full;
        dpsi_x3 = dpsi_full + size_s*nspin;
        for (dim = 0; dim < 3; dim++) {
            // count = 0;
            for(spn_i = 0; spn_i < nspin; spn_i++) {
                for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                    k1 = pSPARC->k1_loc[kpt];
                    k2 = pSPARC->k2_loc[kpt];
                    k3 = pSPARC->k3_loc[kpt];
                    kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                    // find dPsi in direction dim along lattice vector directions
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, dpsi_xi_lv, dim, kpt_vec, pSPARC->dmcomm);
                    // find dPsi in direction dim in cartesian coordinates
                    for (i = 0; i < size_k; i++) {
                        if (dim == 0) {
                            dpsi_x1[i + spn_i*size_s+kpt*size_k] = pSPARC->gradT[0]*dpsi_xi_lv[i];
                            dpsi_x2[i + spn_i*size_s+kpt*size_k] = pSPARC->gradT[1]*dpsi_xi_lv[i];
                            dpsi_x3[i + spn_i*size_s+kpt*size_k] = pSPARC->gradT[2]*dpsi_xi_lv[i];
                        } else {
                            dpsi_x1[i + spn_i*size_s+kpt*size_k] += pSPARC->gradT[0+3*dim]*dpsi_xi_lv[i];
                            dpsi_x2[i + spn_i*size_s+kpt*size_k] += pSPARC->gradT[1+3*dim]*dpsi_xi_lv[i];
                            dpsi_x3[i + spn_i*size_s+kpt*size_k] += pSPARC->gradT[2+3*dim]*dpsi_xi_lv[i];
                        }
                    }
                }
            }
        }
        free(dpsi_xi_lv);
    }


    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++){
            k1 = pSPARC->k1_loc[kpt];
            k2 = pSPARC->k2_loc[kpt];
            k3 = pSPARC->k3_loc[kpt];
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * count;
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                if (!pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
                    R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                    R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                    theta = -k1 * (floor(R1/Lx) * Lx) - k2 * (floor(R2/Ly) * Ly) - k3 * (floor(R3/Lz) * Lz);
                    bloch_fac = cos(theta) - sin(theta) * I;
                    a = bloch_fac * pSPARC->dV;
                    b = 1.0;
                    
                    ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                    psi_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    
                    /* first find inner product <Chi_Jlm, Psi_n> */
                    for (n = 0; n < ncol; n++) {
                        psi_ptr = pSPARC->Xorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd;
                        psi_rc_ptr = psi_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            *(psi_rc_ptr + i) = conj(*(psi_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]));
                        }
                    }
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &a, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                psi_rc, ndc, &b, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiplied dV to get inner-product
                    free(psi_rc);    
                }
            }
            count++;
        }
    }       

    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    count2 = 1;
    for (dim = 0; dim < 3; dim++) {
        dpsi_xi = (dim == 0) ? pSPARC->Yorb_kpt : dpsi_full + (dim-1)*size_s*nspin;
        for (dim2 = dim; dim2 < 3; dim2++) {
            count = 0;
            for(spn_i = 0; spn_i < nspin; spn_i++) {
                for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                    k1 = pSPARC->k1_loc[kpt];
                    k2 = pSPARC->k2_loc[kpt];
                    k3 = pSPARC->k3_loc[kpt];
                    beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin * count2 + count);
                    
                    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                        if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                        for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                            R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3];
                            R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                            R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                            theta = -k1 * (floor(R1/Lx) * Lx) - k2 * (floor(R2/Ly) * Ly) - k3 * (floor(R3/Lz) * Lz);
                            bloch_fac = cos(theta) + sin(theta) * I;
                            b = 1.0;
                            ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                            dpsi_xi_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
                            assert(dpsi_xi_rc);
                            atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                            for (n = 0; n < ncol; n++) {
                                dpsi_ptr = dpsi_xi + spn_i * size_s + kpt * size_k + n * DMnd;
                                dpsi_xi_rc_ptr = dpsi_xi_rc + n * ndc;

                                for (i = 0; i < ndc; i++) {
                                    indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                                    k_DM = indx / (DMnx * DMny);
                                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                                    i_DM = indx % DMnx;
                                    x1_R1 = (i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - R1;
                                    x2_R2 = (j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - R2;
                                    x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                                    StXmRjp = pSPARC->LatUVec[0+dim2] * x1_R1 + pSPARC->LatUVec[3+dim2] * x2_R2 + pSPARC->LatUVec[6+dim2] * x3_R3;
                                    *(dpsi_xi_rc_ptr + i) = *(dpsi_ptr + indx) * StXmRjp;
                                }
                            }
                        
                            /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                            *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                            *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                            *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).

                            */      
                            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                        dpsi_xi_rc, ndc, &b, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);                        
                            free(dpsi_xi_rc);
                        }
                    }
                    count ++;
                }
            }    
            count2 ++;
        }
    }
    
    double complex *dpsi_xi_ptr, *dpsi_xj_ptr;
    // Kinetic stress
    count = 0;
    for (dim = 0; dim < 3; dim++) {
        dpsi_xi = (dim == 0) ? pSPARC->Yorb_kpt : dpsi_full + (dim-1)*size_s*nspin;
        for (dim2 = dim; dim2 < 3; dim2++) {
            dpsi_xj = (dim2 == 0) ? pSPARC->Yorb_kpt : dpsi_full + (dim2-1)*size_s*nspin;
            for(spn_i = 0; spn_i < nspin; spn_i++) {
                for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                    temp_k = 0;
                    for(n = 0; n < ncol; n++){
                        dpsi_xi_ptr = dpsi_xi + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_xi
                        dpsi_xj_ptr = dpsi_xj + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_xj

                        dpsii_dpsij = 0;
                        for(i = 0; i < DMnd; i++){
                            dpsii_dpsij += creal(*(dpsi_xi_ptr + i)) * creal(*(dpsi_xj_ptr + i)) + cimag(*(dpsi_xi_ptr + i)) * cimag(*(dpsi_xj_ptr + i));
                        }
                        g_nk = pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                        temp_k += dpsii_dpsij * g_nk;
                    }
                    stress_k[count] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k;
                }
            }
            count ++;
        }
    }
    free(dpsi_full);

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 7, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal stress */
    // go over all atoms and find nonlocal stress
    beta1_x1 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin;
    beta2_x1 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 2;
    beta3_x1 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 3;
    beta2_x2 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 4;
    beta3_x2 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 5;
    beta3_x3 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 6;

    double alpha_r, alpha_i;
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                lmax = pSPARC->psd[ityp].lmax;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    eJ = 0.0; for(i = 0; i < 6; i++) SJ[i] = 0.0;
                    for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                        g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                        temp2_e = 0.0; for(i = 0; i < 6; i++) temp2_s[i] = 0.0;
                        ldispl = 0;
                        for (l = 0; l <= lmax; l++) {
                            // skip the local l
                            if (l == pSPARC->localPsd[ityp]) {
                                ldispl += pSPARC->psd[ityp].ppl[l];
                                continue;
                            }
                            for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                                temp_e = 0.0; for(i = 0; i < 6; i++) temp_s[i] = 0.0;
                                for (m = -l; m <= l; m++) {
                                    alpha_r = creal(alpha[count]); alpha_i = cimag(alpha[count]);
                                    temp_e += pow(alpha_r, 2.0) + pow(alpha_i, 2.0);
                                    temp_s[0] += alpha_r * creal(beta1_x1[count]) - alpha_i * cimag(beta1_x1[count]);
                                    temp_s[1] += alpha_r * creal(beta2_x1[count]) - alpha_i * cimag(beta2_x1[count]);
                                    temp_s[2] += alpha_r * creal(beta3_x1[count]) - alpha_i * cimag(beta3_x1[count]);
                                    temp_s[3] += alpha_r * creal(beta2_x2[count]) - alpha_i * cimag(beta2_x2[count]);
                                    temp_s[4] += alpha_r * creal(beta3_x2[count]) - alpha_i * cimag(beta3_x2[count]);
                                    temp_s[5] += alpha_r * creal(beta3_x3[count]) - alpha_i * cimag(beta3_x3[count]);
                                    count++;
                                }
                                gamma_jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                temp2_e += temp_e * gamma_jl;
                                for(i = 0; i < 6; i++)
                                    temp2_s[i] += temp_s[i] * gamma_jl;
                            }
                            ldispl += pSPARC->psd[ityp].ppl[l];
                        }
                        eJ += temp2_e * g_nk;
                        for(i = 0; i < 6; i++)
                            SJ[i] += temp2_s[i] * g_nk;
                    }
                    
                    kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                    energy_nl -= kptwt * eJ;
                    for(i = 0; i < 6; i++)
                        stress_nl[i] -= kptwt * SJ[i];
                }
            }
        }
    }     
    
    for(i = 0; i < 6; i++)
        stress_nl[i] *= (2.0/pSPARC->Nspin) * 2.0;
    
    energy_nl *= (2.0/pSPARC->Nspin)/pSPARC->dV;   

    pSPARC->stress_nl[0] = stress_nl[0] + energy_nl;
    pSPARC->stress_nl[1] = stress_nl[1];
    pSPARC->stress_nl[2] = stress_nl[2];
    pSPARC->stress_nl[3] = stress_nl[3] + energy_nl;
    pSPARC->stress_nl[4] = stress_nl[4];
    pSPARC->stress_nl[5] = stress_nl[5] + energy_nl;
    for(i = 0; i < 6; i++)
        pSPARC->stress_k[i] = stress_k[i];
    
    // sum over all spin
    if (pSPARC->npspin > 1) {    
        if (pSPARC->spincomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        } else{
            MPI_Reduce(pSPARC->stress_nl, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
            MPI_Reduce(pSPARC->stress_k, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        }
    }


    // sum over all kpoints
    if (pSPARC->npkpt > 1) {    
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        } else{
            MPI_Reduce(pSPARC->stress_nl, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);

            MPI_Reduce(pSPARC->stress_k, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        }
    }

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;

        for(i = 0; i < 6; i++) {
            pSPARC->stress_nl[i] /= cell_measure;
            pSPARC->stress_k[i] /= cell_measure;
        }

    }

#ifdef DEBUG    
    if (!rank){
        printf("\nNon-local contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_nl, NULL);
        printf("\nKinetic contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_k, NULL);  
    } 
#endif
    free(alpha);
}