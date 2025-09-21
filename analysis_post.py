#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for vidualising results after running the analysis-form_csv.py
created tables with processed results.

@author: Piotr Bentkowski - bentkowski.piotr@gmail.com
"""
import matplotlib
import pandas as pd

import analysis_from_csv as afc

matplotlib.use('agg')


lower_bnd_adhr = 0.0
upper_bnd_adhr = 0.1
lower_bnd_nona = 0.0
upper_bnd_nona = 0.4
max_transmit = 800

simParams, des = afc.get_params()
pSI_psi, pSI_ne = afc.get_pSI()

# results = afc.aggregate_results_csv(simParams, pSI_ne)
result_file_csv = 'results_table.csv'
results = pd.read_csv(result_file_csv, low_memory=False)

afc.print_params(simParams, pSI_psi, pSI_ne)
afc.plot_violin_R(results, simParams, pSI_psi, pSI_ne,
                  lower_bnd_nona, upper_bnd_nona)
afc.plot_data(results, simParams, pSI_psi, pSI_ne,
              lower_bnd_adhr, upper_bnd_adhr,
              lower_bnd_nona, upper_bnd_nona)
afc.plot_data_points(results, simParams, pSI_psi, pSI_ne,
                     lower_bnd_adhr, upper_bnd_adhr,
                     lower_bnd_nona, upper_bnd_nona)
afc.plot_transm_hist_all_asyms(results, 20)
afc.plot_RvsR_for_given_paramset(results, None, 0, max_transmit)

# # LInes below require source raw results
# all_runz, all_maxes, x_max, y_max = afc.get_epi_trjec_all(results, simParams)
# afc.plot_all_epi_trjec(all_runz, all_maxes, x_max, y_max)
