#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: craig
"""
import pandas as pd
import numpy as np
np.random.seed(23)

con = pd.read_csv('csv/conditions.csv')
con.columns = [col.lower() for col in con.columns]

# =============================================================================
# Get Diabetic Patients
# =============================================================================
mask = con['description'].str.contains('iabet')

diabetics = list(con[mask]['patient'].unique()) # length 37884

# =============================================================================
# Get observations for diabetic patients
# =============================================================================
obs = pd.read_csv('csv/observations.csv')
obs.columns = [col.lower() for col in obs.columns]

mask_ob = obs['patient'].isin(diabetics)
obs[mask_ob]

# =============================================================================
# Find prevalent biomarkers among diabetic patients
# =============================================================================
biomarks = [k for k,v
            in dict(obs[mask_ob]['description'].value_counts()).items()
            if v > 4000
]

mask_ob_mark = obs['description'].isin(biomarks)

dfobs = obs[mask_ob & mask_ob_mark][['encounter',
                                       'patient',
                                       'description',
                                       'value'
                                       ]
]

dfobs.index = list(range(dfobs.shape[0]))

# =============================================================================
# Pivot the selected data to get redings for each patient
# =============================================================================
dfobs.reset_index(inplace=True)

dfobs_pivot = dfobs.pivot(index='index', columns='description', values='value')

diabetic_readings = pd.concat([dfobs, dfobs_pivot], axis=1)

# =============================================================================
# Trim the diabetic readings data frame
# =============================================================================
diabetic_readings.drop(['description', 'value'], axis=1, inplace=True)

diabetic_readings[list(diabetic_readings.columns[3:])] = \
    diabetic_readings[list(diabetic_readings.columns[3:])] \
    .applymap(lambda x: 0 if type(x)!=str else float(x))

readings = diabetic_readings.groupby(['encounter', 'patient']).sum()

# =============================================================================
# Reconstitute the grouped data frame
# =============================================================================
encounters = [readings.index[i][0] for i in range(readings.shape[0])]

readings['encounter'] = encounters

patients = [readings.index[i][1] for i in range(readings.shape[0])]

readings['patient'] = patients

readings.index = list(range(readings.shape[0]))

readings.drop('index', axis=1, inplace=True)
