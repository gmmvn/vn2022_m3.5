#!/usr/bin/env python
# coding: utf-8

# In[28]:


from glob import glob
import pandas as pd
import obspy
from obspy import read_inventory
import os
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
def distcal(stla,stlo,evla,evlo):
    return gps2dist_azimuth(stla,stlo,evla,evlo)[0]/1000
import numpy as np
from scipy.optimize import curve_fit
from obspy.taup import TauPyModel
model = TauPyModel(model="iasp91")
import matplotlib.pyplot as plt


# In[9]:


def get_disp(tr):
    """Integrate acceleration to displacement.
    Args:
        tr (StationTrace):
            Trace of acceleration data. This is the trace where the Cache values will
            be set.
        config (dict):
            Configuration dictionary (or None). See get_config().
    Returns:
        StationTrace.
    """
    acc = tr.copy()
    try:
        disp = acc.integrate().integrate()
    except Exception as e:
        raise e
    return disp

def correct_baseline(trace):
    """
    Performs a baseline correction following the method of Ancheta
    et al. (2013). This removes low-frequency, non-physical trends
    that remain in the time series following filtering.
    Args:
        trace (obspy.core.trace.Trace):
            Trace of strong motion data.
        config (dict):
            Configuration dictionary (or None). See get_config().
    Returns:
        trace: Baseline-corrected trace.
    """
    # Integrate twice to get the displacement time series
    disp = get_disp(trace)

    # Fit a sixth order polynomial to displacement time series, requiring
    # that the 1st and 0th order coefficients are zero
    time_values = (
        np.linspace(0, trace.stats.npts - 1, trace.stats.npts) * trace.stats.delta
    )
    poly_cofs = list(curve_fit(_poly_func, time_values, disp.data)[0])
    poly_cofs += [0, 0]

    # Construct a polynomial from the coefficients and compute
    # the second derivative
    polynomial = np.poly1d(poly_cofs)
    polynomial_second_derivative = np.polyder(polynomial, 2)

    # Subtract the second derivative of the polynomial from the
    # acceleration trace
    trace.data -= polynomial_second_derivative(time_values)

    return trace

def _poly_func(x, a, b, c, d, e):
    """
    Model polynomial function for polynomial baseline correction.
    """
    return a * x ** 6 + b * x ** 5 + c * x ** 4 + d * x ** 3 + e * x ** 2


# In[3]:


def add0(s):
    if len(str(s)) < 2:
        return "0" + str(s)
    else:
        return str(s)

def add00(s):
    if len(str(s)) == 1:
        return "00" + str(s)
    elif len(str(s)) == 2: 
        return "0" + str(s)
    else:
        return str(s)


# In[4]:


def convert_utc(yy,mm,dd,hh,mi,s):
    return UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi), int(s))
    


# In[5]:


respf = "/home/nghianc/auto_pick/instrument_response/XML"
eqcat = pd.read_csv("vn_cat.csv",index_col=0)
evl = sorted([a.split("_")[-2] for a in glob("../sac_data/*")])
eqcat['utctime'] = eqcat.apply(lambda x: convert_utc(x.year, x.month, x.day, x.hour, x.minute, x.second),axis=1)


# In[41]:


record = []
for i in range(0,len(evl)):
    print("Read event {}".format(evl[i]))
    dfinfo = eqcat[eqcat['utctime'] == UTCDateTime(evl[i])].reset_index(drop=True)
    sacl = glob("../sac_data/*{}*/*".format(evl[i]))
    stal = list(set([a.split("/")[-1].split(".")[0] for a in sacl]))
    fs,nf = [],[]
    for sta in stal:
        try:
            inv = read_inventory(os.path.join(respf,"VN.{}.xml".format(sta)))
    #         print("Found station {}".format(sta))
            fs.append(sta)
        except:
            print("Station {} not found".format(sta))
            nf.append(sta)
            pass
    sacpr = [a for a in [a for a in sacl if a.split("/")[-1].split(".")[0] in fs] if "HHN" in a]
    sacpr = [a.replace("HHN","HH*") for a in sacpr]
    # for sacfile in sacpr:
    for file in sacpr:
        sta = file.split("/")[-1].split(".")[0]
        inv = read_inventory(os.path.join(respf,"VN.{}.xml".format(sta)))
        for chan in inv[0][0]:
            chan.start_date = UTCDateTime(2000,1,1)
        try:
            ev = evl[i]
            st = obspy.read(file)
            sr = 100
            trz = st.select(component="Z")[0].interpolate(sampling_rate=sr)
            tre = st.select(component="E")[0].interpolate(sampling_rate=sr)
            trn = st.select(component="N")[0].interpolate(sampling_rate=sr)
            pre_filt = (0.005, 0.006, 30.0, 35.0)
            for tr in [trz,tre,trn]:
                tr.detrend("demean")
                tr = correct_baseline(tr)
                tr.stats.channel = tr.stats.channel.replace("HH ","HH")
                tr.remove_response(inventory=inv, pre_filt=pre_filt, output="ACC",
                       water_level=60)

            df = trz.stats.sampling_rate



            stla = "{:.3f}".format(inv[0][0].latitude)
            stlo = "{:.3f}".format(inv[0][0].longitude)
            evla = "{:.3f}".format(dfinfo.Lat[0])
            evlo = "{:.3f}".format(dfinfo.Lon[0])
            evdp = "{:.3f}".format(dfinfo.Depth[0])
            mag = "{:.1f}".format(dfinfo.Ml[0])
            dis = distcal(float(stla),float(stlo),float(evla),float(evlo))
            t0 = UTCDateTime(evl[i])
            ml = mag
        #     mw = 

#             arrivals = model.get_travel_times(source_depth_in_km=evdp,
#                                       distance_in_degree=dis,
#                                       phase_list=["P"])
            print(ev,UTCDateTime(t0))
            print("Read station {} - event {} - distance {}".format(sta,ev,dis))
            trn=trn.trim(t0,t0+180)
            trz=trz.trim(t0,t0+180)
            tre=tre.trim(t0,t0+180)
    #         trn.plot()
    #         trz.plot()
    #         tre.plot()
            if trn.data.mean() == 0 or tre.data.mean() == 0 or trz.data.mean() == 0:
                print("Missing component!")
                nfile = "ascii/{}_{}_mc_{}hz.cwb".format(sta,ev,sr)
            else:
                nfile = "ascii/{}_{}_{}hz.cwb".format(sta,ev,sr)
            print(nfile)
            if dis < 300:
                with open(nfile,"w") as f:
                    print("%StationCode:",sta,file=f)
                    print("%StationLat:",stla,file=f)
                    print("%StationLon:",stlo,file=f)
                    print("%EventLat:",evla,file=f)
                    print("%EventLon:",evlo,file=f)
                    print("%EventML: {}".format(ml),file=f)
            #         print("%EventMW: {:.2f}".format(mw),file=f)
                    print("%EventDep: {}".format(evdp),file=f)
                    print("%InstrumentKind: N.A.",file=f)
                    print("%StartTime:",t0,file=f)
                    print("%SampleRate(Hz): 100",file=f)
                    print("%AmplitudeUnit: cm/ss",file=f)
                    print("%AmplitudeMAX. U: {:.4f}~ {:.4f}".format(max(trz.data)*100,min(trz.data)*100),file=f)
                    print("%AmplitudeMAX. N: {:.4f}~ {:.4f}".format(max(trn.data)*100,min(trn.data)*100),file=f)
                    print("%AmplitudeMAX. E: {:.4f}~ {:.4f}".format(max(tre.data)*100,min(tre.data)*100),file=f)
                    print("%DataSequence: Time U(+); N(+), E(+)",file=f)
                    print("%Data: 4F10.3",file=f)
                    for k in range(0,len(trz.data)):
                        t = 0 + 1/sr*k
                        print("{:11.3f}{:10.4f}{:10.4f}{:10.4f}".format(t,trz.data[k]*100,trn.data[k]*100,tre.data[k]*100),file=f)
                    rec = {"eqid":ev,"evla":evla,"evlo":evlo,"evdp":evdp,"sta":sta,"stla":stla,"stlo":stlo,"dis":dis}
                    record.append(rec)
        except IndexError:
    #         raise
            print("!!!File {} has error!!!".format(file))
record = pd.DataFrame(record)
record.to_csv("recordVN.csv")


# In[24]:


# arrivals = model.get_travel_times(source_depth_in_km=float(evdp),
#                           distance_in_degree=dis/110,
#                           phase_list=["S"])


# In[26]:


# arr = arrivals[0].time


# In[ ]:


# plt.figure(figsize=(20,6))
# plt.plot(trz.times(),trz.data,"k")
# plt.plot([arr,arr],[1,-1])
# plt.ylim(-max(trz.data),max(trz.data))
# plt.show()


# In[ ]:




