#!/usr/bin/env python
# coding: utf-8

# In[91]:


from glob import glob
import numpy as np
import pandas as pd
import pyrotd
import matplotlib.pyplot as plt
import os
from warnings import filterwarnings
filterwarnings("ignore")
from obspy.geodetics import gps2dist_azimuth
def distcal(stla,stlo,evla,evlo):
    return gps2dist_azimuth(stla,stlo,evla,evlo)[0]/1000


# In[92]:


fl = glob("mocchaueq/*")


# In[97]:


time_step = 1/200
osc_damping = 0.05
period = np.array([0.01,0.02,0.022,0.025,0.029,0.03,0.032,0.035,0.036,0.04,0.042,0.044,0.045,0.046,
               0.048,0.05,0.055,0.06,0.065,0.067,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.13,
               0.133,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.22,0.24,0.25,0.26,0.28,0.29,0.3,0.32,0.34,
               0.35,0.36,0.38,0.4,0.42,0.44,0.45,0.46,0.48,0.5,0.55,0.6,0.65,0.667,0.7,0.75,0.8,0.85,0.9,0.95,
               1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.5,2.6,2.8,3,3.2,3.4,3.5,3.6,3.8,4,4.2,4.4,4.6,4.8,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]);
osc_freqs = np.flip(1/period)
col_names1 = ['EQID','STAID','MW','H_km','evlat','evlon','stlat','stlon']
# create dataframe-2 containing the spectral acclereation
col_names2 = [str(a) + " hz" for a in osc_freqs]
col_names_all = col_names1 + col_names2
df_rot_psa = pd.DataFrame(columns = col_names_all)
for i in range(0,len(fl)):
# for i in range(0,10):
    fn = fl[i]
    eqid = fn.split("_")[-2]
    sta = fn.split("_")[-3].split("\\")[-1]
    header = pd.read_csv(fn, nrows = 7, delimiter=": ",index_col=0).T
    header = header.reset_index()
    header.columns = ['sta','stalat', 'stalon', 'evelat', 'evelon', 'ml','mw','dep']
    data = pd.read_csv(fn, delim_whitespace = True, skiprows=17, index_col = False, names=["t","V","H1","H2"])
    time = data.t # time
    ug1 = data.H1 # E component
    ug2 = data.H2 # H component
    header = header[['sta','mw','dep','evelat','evelon','stalat','stalon']]
    eq_info = np.stack([header[v_n] for v_n in header], axis=1).reshape(-1)
    psa_ug1 = pyrotd.calc_spec_accels(time_step, ug1, osc_freqs, osc_damping)
    psa_ug2 = pyrotd.calc_spec_accels(time_step, ug2, osc_freqs, osc_damping)
    gm_psa = np.sqrt(0.5*psa_ug2['spec_accel']**2 + 0.5*psa_ug1['spec_accel']**2)
    tempt =  np.concatenate([[eqid],eq_info, gm_psa])
    df_rot_psa.loc[i] = tempt


# In[98]:


df_rot_psa.to_csv('Sa_table_mc.csv')


# In[99]:



# In[ ]:




