/**
 * @brief   Compute column of density matrix using Clenshaw curtis Quadrature
 */
void Clenshaw_curtis_density_matrix_col(SPARC_OBJ *pSPARC, double ***DMcol, int i, int j, int k, int nd) 
{
    int ii, jj, kk, *nloc, FDn, nq;
    double ***Hv, ***t0, ***t1, ***t2, *di;

    SQ_OBJ *pSQ = pSPARC->pSQ;
    nloc = pSQ->nloc;
    FDn = pSPARC->order / 2;

    // Compute quadrature coefficients
    double bet0, bet, lambda_fp;
    bet0 = pSPARC->Beta;
    di = (double *) calloc(sizeof(double), pSPARC->SQ_npl_c+1);
    bet = bet0 * pSQ->zee[nd];
    lambda_fp = (pSPARC->Efermi - pSQ->chi[nd]) / pSQ->zee[nd];
    ChebyshevCoeff_SQ(pSPARC->SQ_npl_c, pSQ->Ci[nd], di, smearing_function, bet, lambda_fp, pSPARC->elec_T_type);
    free(di);
    
    // Start to find density matrix
    Hv = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	t0 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	t1 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	t2 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);

	for(kk = 0; kk < 2*nloc[2]+1; kk++) {
		Hv[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		t0[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		t1[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		t2[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);

		for(jj = 0; jj < 2*nloc[1]+1; jj++) {
			Hv[kk][jj] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
			t0[kk][jj] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
			t1[kk][jj] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
			t2[kk][jj] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
		}
	}

    /// For each FD node, loop over quadrature order to find Chebyshev expansion components, Use the HsubTimesVec function this.
    for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
        for(jj = -nloc[1];jj <= nloc[1]; jj++) {
            for(ii = -nloc[0]; ii <= nloc[0]; ii++) {
                t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]=0.0;
                if(ii==0 && jj==0 && kk==0)
                    t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]=1.0;
            }
        }
    }

    HsubTimesVec(pSPARC,t0,i,j,k,Hv); // Hv=Hsub*t0. Here t0 is the vector and i,j,k are the indices of node in proc domain to which the Hsub corresponds to and the indices are w.r.t proc+Rcut domain

    for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
        for(jj = -nloc[1]; jj <= nloc[1]; jj++) {
            for(ii = -nloc[0]; ii <= nloc[0]; ii++) {
                t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]=(Hv[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] - pSQ->chi[nd]*t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]) / pSQ->zee[nd];

                DMcol[kk+nloc[2]+FDn][jj+nloc[1]+FDn][ii+nloc[0]+FDn] = 0.0;
            }
        }
    }
    
    // loop over quadrature order
    for(nq = 0; nq <= pSPARC->SQ_npl_c; nq++) {
        if(nq == 0) {
            for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
                for(jj = -nloc[1]; jj <= nloc[1]; jj++) {
                    for(ii = -nloc[0]; ii <= nloc[0]; ii++) {
                        DMcol[kk+nloc[2]+FDn][jj+nloc[1]+FDn][ii+nloc[0]+FDn] += 
                            (double)(t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]*pSQ->Ci[nd][nq]/pSPARC->dV);
                    }
                }
            }
        } else if(nq == 1) {
            for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
                for(jj = -nloc[1]; jj <= nloc[1]; jj++) {
                    for(ii = -nloc[0];ii <= nloc[0]; ii++) {
                        DMcol[kk+nloc[2]+FDn][jj+nloc[1]+FDn][ii+nloc[0]+FDn] += 
                            (double)(t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]*pSQ->Ci[nd][nq]/pSPARC->dV);
                    }
                }
            }

        } else {
            HsubTimesVec(pSPARC, t1, i, j, k, Hv); // Hv=Hsub*t1

            for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
                for(jj = -nloc[1]; jj <= nloc[1]; jj++) {
                    for(ii = -nloc[0]; ii <= nloc[0]; ii++) {
                        t2[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] = 
                            (2*(Hv[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] - pSQ->chi[nd]*t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]) / pSQ->zee[nd]) - t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]];

                        DMcol[kk+nloc[2]+FDn][jj+nloc[1]+FDn][ii+nloc[0]+FDn] += (double)(t2[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] * pSQ->Ci[nd][nq]/pSPARC->dV);

                        t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] = t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]];
                        t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] = t2[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]];
                    }
                }
            }
        }
    }
    
    for(kk = 0; kk < 2*nloc[2]+1; kk++) {
        for(jj = 0; jj < 2*nloc[1]+1; jj++) {
            free(Hv[kk][jj]);
            free(t0[kk][jj]);
            free(t1[kk][jj]);
            free(t2[kk][jj]);
        }
        free(Hv[kk]);
        free(t0[kk]);
        free(t1[kk]);
        free(t2[kk]);
    }
    free(Hv);
    free(t0);
    free(t1);
    free(t2);
}

/**
 * @brief   Compute chebyshev coefficients Ci (length npl_c+1) to fit the function "func"
 * 
 * NOTE:    THIS FUNCTION CURRENTLY USES DISCRETE ORTHOGONALITY PROPERTY OF CHEBYSHEV POLYNOMIALS WHICH ONLY GIVE "approximate" 
 *          COEFFICIENTS. SO IF THE FUNCTION IS NOT SMOOTH ENOUGH THE COEFFICIENTS WILL NOT BE ACCURATE ENOUGH. 
 *          SO THIS FUNCTION HAS TO BE REPLACED WITH ITS CONTINUOUS COUNTER PART WHICH EVALUATES OSCILLATORY INTEGRALS AND USES FFT.
 */
void ChebyshevCoeff_SQ(int npl_c, double *Ci, double *d, double (*fun)(double, double, double, int), 
                const double beta, const double lambda_f, const int type) {
    int k, j;
    double y, fac1, fac2, sum;

    fac1 = M_PI/(npl_c + 1);
    for (k = 0; k < npl_c + 1; k++) {
        y = cos(fac1 * (k + 0.5));
        // d[k] = (*func)(y, lambda_fp, bet);
        d[k] = fun(beta, y, lambda_f, type);
    }

    fac2 = 2.0 / (npl_c + 1);
    for (j = 0; j < npl_c + 1; j++) {
        sum = 0.0;
        for (k = 0; k < npl_c + 1; k++) {
            // sum = sum + d[k] * cos((M_PI * (j - 1 + 1)) * ((double)((k - 0.5 + 1) / (npl_c + 1))));
            sum = sum + d[k] * cos(fac1 * j * (k + 0.5));
        }
        Ci[j] = fac2 * sum;
    }
    Ci[0] = Ci[0] / 2.0;
}

/**
 * @brief   Lanczos algorithm computing only minimal eigenvales
 * 
 * TODO:    There are 2 different Lanczos algorithm with confusing names. Change them or combine them
 */
void LanczosAlgorithm_new(SPARC_OBJ *pSPARC, int i, int j, int k, double *lambda_min, int nd, int choice) {
    int rank, ii, jj, kk, ll, p, max_iter = 500;
    int *nloc, flag = 0, count;
    double ***vk, ***vkp1, val, *aa, *bb, **lanc_V, lambda_min_temp;
    double lmin_prev = 0.0, dl = 1.0, dm=1.0 ,lmax_prev=0.0;
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    SQ_OBJ *pSQ = pSPARC->pSQ;
    nloc = pSQ->nloc;


    aa = (double *) calloc(sizeof(double), pSQ->Nd_loc);
    bb = (double *) calloc(sizeof(double), pSQ->Nd_loc);

    vk = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
    vkp1 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        vk[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
        vkp1[kk] =  (double **) calloc(sizeof(double *), 2*nloc[1]+1);
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            vk[kk][jj] = (double *) calloc(sizeof(double ), 2*nloc[0]+1);
            vkp1[kk][jj] = (double *) calloc(sizeof(double ), 2*nloc[0]+1);
        }
    }
    
    lanc_V = (double **) calloc(sizeof(double *), max_iter+1);
    for (p = 0; p < max_iter + 1; p++)
        lanc_V[p] = (double *) calloc(sizeof(double), pSQ->Nd_loc);

    Vector2Norm_local(pSQ->vec[nd], nloc, &val);

    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                pSQ->vec[nd][kk][jj][ii] = pSQ->vec[nd][kk][jj][ii] / val;
            }
        }
    }

    HsubTimesVec(pSPARC, pSQ->vec[nd], i, j, k, vk); 
    VectorDotProduct_local(pSQ->vec[nd], vk, nloc, &val);

    aa[0] = val;
    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                vk[kk][jj][ii] = vk[kk][jj][ii] - aa[0] * pSQ->vec[nd][kk][jj][ii];
            }
        }
    }
    Vector2Norm_local(vk, nloc, &val);
    bb[0] = val;
    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                vk[kk][jj][ii] = vk[kk][jj][ii] / bb[0];
                lanc_V[flag][kk * (2 * nloc[0] + 1) * (2 * nloc[1] + 1) + jj * (2 * nloc[0] + 1) + ii] = vk[kk][jj][ii];
            }
        }
    }

    flag += 1;
    count = 0;
    while ((count < max_iter && dl > pSPARC->TOL_LANCZOS) || (choice == 1 && dm > pSPARC->TOL_LANCZOS)) {
        HsubTimesVec(pSPARC, vk, i, j, k, vkp1); // vkp1=Hsub*vk
        VectorDotProduct_local(vk, vkp1, nloc, &val); // val=vk'*vkp1
        aa[count + 1] = val;
        for (kk = 0; kk <= 2 * nloc[2]; kk++) {
            for (jj = 0; jj <= 2 * nloc[1]; jj++) {
                for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                    vkp1[kk][jj][ii] = vkp1[kk][jj][ii] - aa[count + 1] * vk[kk][jj][ii] - bb[count] * pSQ->vec[nd][kk][jj][ii];
                }
            }
        }
        Vector2Norm_local(vkp1, nloc, &val);
        bb[count + 1] = val;

        for (kk = 0; kk <= 2 * nloc[2]; kk++) {
            for (jj = 0; jj <= 2 * nloc[1]; jj++) {
                for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                    pSQ->vec[nd][kk][jj][ii] = vk[kk][jj][ii];
                    vk[kk][jj][ii] = vkp1[kk][jj][ii] / bb[count + 1];
                    lanc_V[flag][kk * (2 * nloc[0] + 1) * (2 * nloc[1] + 1) + jj * (2 * nloc[0] + 1) + ii] = vk[kk][jj][ii];
                }
            }
        }
        flag += 1;
        // Eigendecompose the tridiagonal matrix
        TridiagEigenSolve_new(pSPARC, aa, bb, count + 2, lambda_min, choice, nd);
        
        dl = fabs(( * lambda_min) - lmin_prev);
        dm = fabs((pSQ->lambda_max[nd])-lmax_prev);
        lmin_prev = * lambda_min;
        lmax_prev = pSQ->lambda_max[nd];
        count = count + 1;
    }
    lambda_min_temp = *lambda_min; *lambda_min -= pSPARC->TOL_LANCZOS;
    if (choice == 1) {
        pSQ->lambda_max[nd] += pSPARC->TOL_LANCZOS;
        // free(pSQ->low_eig_vec);
    }
    // if(choice == 0)
    //     free(pSQ->low_eig_vec);
    
    // eigenvector corresponding to lowest eigenvalue of the tridiagonal matrix of the current node
    pSQ->low_eig_vec = (double *) calloc(sizeof(double), count + 1);
    TridiagEigenSolve_new(pSPARC, aa, bb, count + 1, &lambda_min_temp, 2, nd);
    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++)
                pSQ->vec[nd][kk][jj][ii] = 0.0;
        }
    }

    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                for (ll = 0; ll < count + 1; ll++)
                    pSQ->vec[nd][kk][jj][ii] += lanc_V[ll][kk * (2 * nloc[0] + 1) * (2 * nloc[1] + 1) + jj * (2 * nloc[0] + 1) + ii] * pSQ->low_eig_vec[ll];
            }
        }
    }

    free(pSQ->low_eig_vec);

#ifdef DEBUG
    if (!rank && !nd) {
        printf("\nrank %d, nd %d, Lanczos took %d iterations. \n", rank, nd, count);
    }
#endif

    if (count == max_iter)
        printf("WARNING: Lanczos exceeded max_iter. count=%d, dl=%f \n", count, dl);
    
    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            free(vk[kk][jj]);
            free(vkp1[kk][jj]);
        }
        free(vk[kk]);
        free(vkp1[kk]);
    }
    free(vk);
    free(vkp1);
    free(aa);
    free(bb);
    for (p = 0; p < max_iter + 1; p++)
        free(lanc_V[p]);
    free(lanc_V);
}

/**
 * @brief   Tridiagonal solver for eigenvalues. Part of Lanczos algorithm.
 */
void TridiagEigenSolve_new(SPARC_OBJ *pSPARC, double *diag, double *subdiag, 
                            int n, double *lambda_min, int choice, int nd) {

    int i, k, m, l, iter, index;
    double s, r, p, g, f, dd, c, b;
    // d has diagonal and e has subdiagonal
    double *d, *e, **z; 
    SQ_OBJ* pSQ  = pSPARC->pSQ;

    d = (double *) calloc(sizeof(double), n);
    e = (double *) calloc(sizeof(double), n);
    z = (double **) calloc(sizeof(double *), n);

    for (i = 0; i < n; i++)
        z[i] = (double *) calloc(sizeof(double), n);

    if (choice == 2) {
        for (i = 0; i < n; i++) {
            z[i][i] = 1.0;
        }
    }

    //create copy of diag and subdiag in d and e
    for (i = 0; i < n; i++) {
        d[i] = diag[i];
        e[i] = subdiag[i];
    }

    // e has the subdiagonal elements 
    // ignore last element(n-1) of e, make it zero
    e[n - 1] = 0.0;

    for (l = 0; l <= n - 1; l++) {
        iter = 0;
        do {
            for (m = l; m <= n - 2; m++) {
                dd = fabs(d[m]) + fabs(d[m + 1]);
                if ((double)(fabs(e[m]) + dd) == dd) break;
            }
            if (m != l) {
                if (iter++ == 200) {
                    printf("Too many iterations in Tridiagonal solver\n");
                    exit(1);
                }
                g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                r = sqrt(g * g + 1.0); // pythag
                g = d[m] - d[l] + e[l] / (g + SIGN(r, g)); 
                s = c = 1.0;
                p = 0.0;

                for (i = m - 1; i >= l; i--) {
                    f = s * e[i];
                    b = c * e[i];
                    e[i + 1] = (r = sqrt(g * g + f * f));
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p = s * r);
                    g = c * r - b;
                    if (choice == 2) {
                        for (k = 0; k < n; k++) {
                            f = z[k][i + 1];
                            z[k][i + 1] = s * z[k][i] + c * f;
                            z[k][i] = c * z[k][i] - s * f;
                        }
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }

    // go over the array d to find the smallest and largest eigenvalue
    *lambda_min = d[0];
    if (choice == 1)
        pSQ->lambda_max[nd] = d[0];
    index = 0;
    for (i = 1; i < n; i++) {
        if (choice == 1) {
            if (d[i] > pSQ->lambda_max[nd]) {
                pSQ->lambda_max[nd] = d[i];
            }
        }
        if (d[i] < *lambda_min) { 
            *lambda_min = d[i];
            index = i;
        }
    }

    if (choice == 2) {
        for (i = 0; i < n; i++)
            pSQ->low_eig_vec[i] = z[i][index];
    }

    free(d);
    free(e);
    for (i = 0; i < n; i++)
        free(z[i]);
    free(z);

    return;
}