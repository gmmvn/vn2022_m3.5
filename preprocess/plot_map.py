
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pygmt
import matplotlib.pyplot as plt
import numpy as np
from obspy.core.event import read_events
from cat_util import *
# In[ ]:


# ast = pd.read_excel("Thong ke may_2020.xlsx").dropna(thresh=3).reset_index(drop=True)
# ast.columns = ["netwrok","sta","time","non1","lat","lon","h","datalogger","s1","s2"]
# ast[["lat","lon"]]= ast[["lat","lon"]].astype(float)
# ast.head()
# ll = 0.02

startLatA,startLonA,endLatA,endLonA = 1,98.5,3.5,98.5
startLatB,startLonB,endLatB,endLonB = 2.25,97,2.25,100
# In[5]:

cat_auto = read_events("cat_done.xml",format="QUAKEML")
#cat_auto = cat_auto.filter("latitude > -7.5","latitude < -7","longitude > 110.2", "longitude < 110.6")
df = summarize_catalog(cat_auto,magnitude_type="ML")
df.depth = df.depth/1000
neq = df
neq.columns = ['event_id', 'origin_time', 'Latitud', 'Longitud', 'Depth',
       'Ml', 'magnitude_type']

#neq = neq[neq.GAP < 200]
neq.head()


# In[6]:


st = pd.read_json("../station_list_amba.json",).T
st[['lat','lon','h']] = pd.DataFrame(st.coords.tolist(), index= st.index)
st[['lat','lon','h']] = st[['lat','lon','h']].astype(float)
st['sta'] = st.index
st = st.reset_index(drop=True)
st.head()




# In[9]:


minlon, maxlon = 97,100
minlat, maxlat = 1, 3.5



# Visualization
#plt.figure()
#plt.subplot(211)
fig = pygmt.Figure()


topo_data = '@earth_relief_03s' #30 arc second global relief (SRTM15+V2.1 @ 1.0 km)
pygmt.makecpt(
    cmap='grayC',
    series='-100/10000/200',
    continuous=True

)
fig.grdimage(
    grid=topo_data,
    region=[minlon, maxlon, minlat, maxlat], 
    projection='M20c',
    shading=True,
    frame="a",
    N=1,
    )


fig.plot(
    x=st.lon,
    y=st.lat, 
    style='t0.55c', 
    color="black", 
    pen='1p,black', 
)

pygmt.makecpt(
    cmap='turbo',
    series=[0,50],
    continuous=False
)


fig.colorbar(frame='af+l"Hypocenter depth (km)"',position="JML+o2.0c/0c")

#fig.coast(borders=["1/1.0p,black,-"],)
fig.coast(shorelines=True)
fig.plot(
    x=neq.Longitud,
    y=neq.Latitud, 
    style='c', 
    size=0.15*1.3** neq.Ml,
    cmap=True,
    color=neq.Depth, 
    pen='1p,black', 
    )
for m in [2,3,4]:
    mag = 0.15*1.3**m
    fig.plot(
        x=[0],
        y=[0],
        color="white",
        style='c{}'.format(mag), 
        pen='black',label="M={}".format(m)
        )
fig.legend(position='JBR+jBR+o0.2c',box='+gwhite+p1p')

fig.basemap(map_scale="x0.6i/0.8i+c0+w20")

fig.savefig("figs/cat_new_all.png")
fig.show()


#%%
# minlon, maxlon = 109.7, 111.2
# minlat, maxlat = -8, -6.5
fig = pygmt.Figure()
topo_data = '@earth_relief_03s'
with fig.subplot(nrows=2, ncols=2, 
   figsize=("30c", "30c"), 
   sharex="b", 
   sharey="l",
   margins="0c"
    ):
    with fig.set_panel(panel=0):
        pro1 = "M13c"
        pygmt.makecpt(
    cmap='grayC',
    series='-100/10000/200',
    continuous=True

)
        fig.grdimage(
    grid=topo_data,
    region=[minlon, maxlon, minlat, maxlat], 
    projection='M13c',
    shading=True,
    frame=["WsNe"],
    N=1,
    )
        fig.coast(shorelines=True, projection=pro1)
        # fig.plot(data="map/fault_lv1.txt", pen="2.0,red3", projection=pro1)
        # fig.plot(data="map/DGXuyencap2.txt", pen="1.0", projection=pro1)
        # fig.plot(data="map/DGXuyencap3.txt", pen="0.5", projection=pro1)

        fig.plot(
    x=st.lon,
    y=st.lat, 
    projection=pro1,
    style='t0.45c', 
    color="black", 
    pen='1p,black', 
)
        pygmt.makecpt(
            cmap='turbo',
            series=[0,50],
            continuous=False
        )

        fig.plot(
            x=neq.Longitud,
            y=neq.Latitud, 
            projection=pro1,
            style='c', 
            size=0.15*1.3** neq.Ml,
            cmap=True,
            color=neq.Depth, 
            pen='1p,black', 
            )


        fig.colorbar(frame='af+l"Hypocenter depth (km)"',
        position="JMB+o0.0c/5c", 
        projection=pro1
        )
        #fig.legend(position='JBL+jBL+o0.2c',box='+gwhite+p1p', projection=pro1)
        for m in [2,3,4]:
            mag = 0.15*1.3**m
            fig.plot(
                x=[0],
                y=[0],
                color="white",
                style='c{}'.format(mag), 
                pen='black',label="M={}".format(m),
                projection=pro1
                )
        fig.legend(position='JBR+jBR+o0.2c',box='+gwhite+p1p', projection=pro1)

        fig.basemap(map_scale="x0.6i/0.6i+c0+w20", projection=pro1)
        fig.plot(x=[startLonA,endLonA],y=[startLatA,endLatA],pen="1.0p,black", projection=pro1)
        fig.text(x=startLonA,y=startLatA,text="A",font="10p,Helvetica,black",fill="white", projection=pro1)
        fig.text(x=endLonA,y=endLatA,text="A'",font="10p,Helvetica,black",fill="white", projection=pro1)

        fig.plot(x=[startLonB,endLonB],y=[startLatB,endLatB],pen="1.0p,black", projection=pro1)
        fig.text(x=startLonB,y=startLatB,text="B",font="10p,Helvetica,black",fill="white", projection=pro1)
        fig.text(x=endLonB,y=endLatB,text="B'",font="10p,Helvetica,black",fill="white", projection=pro1)
    with fig.set_panel(panel=1,clearance=['w-1c']):  
        pro2 = "X3c/13c"
        fig.basemap(region=[-5, 60, minlat, maxlat], projection=pro2,frame=["wSne","x+lDepth(km)"])
        fig.plot(
            x=neq.Depth,
            y=neq.Latitud, 
            projection=pro2,
            style='c', 
            size=0.15*1.3** neq.Ml,
            color="white",
            pen='1p,black', 
            )
    with fig.set_panel(panel=2,clearance=['s11c']):  
        pro3 = "X13.0c/-3c"
        fig.basemap(region=[minlon, maxlon, -5, 60], projection=pro3,frame=["Wsne","y+lDepth(km)"])
        
        fig.plot(
            x=neq.Longitud,
            y=neq.Depth, 
            projection=pro3,
            style='c', 
            size=0.15*1.3** neq.Ml,
            color="white", 
            pen='1p,black', 
            )


fig.savefig("figs/cat_new_cc_all.png")     
fig.show()
# In[9]:

# minlon, maxlon = 109.7, 111.2
# minlat, maxlat = -8, -6.5



# Visualization
fig = pygmt.Figure()
topo_data = '@earth_relief_03s' #30 arc second global relief (SRTM15+V2.1 @ 1.0 km)
pygmt.makecpt(
    cmap='grayC',
    series='-100/10000/200',
    continuous=True
)
fig.grdimage(
    grid=topo_data,
    region=[minlon, maxlon, minlat, maxlat], 
    projection='M20c',
    shading=True,
    frame="a",
    N=1
    )
fig.coast(
    region=[minlon, maxlon, minlat, maxlat], 
    shorelines=True,
    )

fig.plot(
    x=st.lon,
    y=st.lat, 
    style='t0.55c', 
    color="black", 
    pen='1p,black', 
 )

pygmt.makecpt(
    cmap='turbo',
    series=[0,50],
    continuous=False
)

fig.plot(
    x=neq.Longitud,
    y=neq.Latitud, 
    style='c', 
    sizes=0.2*1.3** neq.Ml,
    cmap=True,
    color=neq.Depth, 
    pen='1p,black', 
    )

fig.plot(
    x=[np.NaN],
    y=[np.NaN],
    color="black",
    style='c0.2c', 
    pen='0.1p,black',
    label='"EQT catalogue"'
    )
fig.colorbar(frame='af+l"Depth (km)"')

fig.coast(borders=["1/1.0p,black,-"],)
#fig.plot(data="map/fault_lv1_edit.txt", pen="2.0,red3",label='"Rank 1 fault"')
#fig.plot(data="map/fault_lv2_edit.txt", pen="1.0",label='"Rank 2 rault"')
fig.legend(position='JBL+jBL+o0.2c',box='+gwhite+p1p')


# fig.text(text="Hanoi", x=106.1, y=21.05, font="10p,Helvetica,black",fill="white")
# fig.text(text="LAOS", x=103, y=20, font="20p,Helvetica-Bold,black")
# #fig.text(text="VIETNAM", x=105.5, y=21.7, font="20p,Helvetica-Bold,white")
# fig.text(text="CHINA", x=107.4, y=23, font="20p,Helvetica-Bold,black")

# fig.plot(
#     x=[103.12,103.34],
#     y=[21.163,21.75],
#     color="yellow",
#     sizes=0.1*1.3** np.array([6.7,6.8]),
#     style='a', 
#     pen='black'
#     )

for m in [3,4,5]:
    mag = 0.1*1.3**m
    fig.plot(
        x=[0],
        y=[0],
        color="mediumpurple",
        style='c{}'.format(mag), 
        pen='black',label="M={}".format(m)
        )
#fig.legend(position='JBR+jBR+o0.2c',box='+gwhite+p1p')
fig.plot(x=[startLonA,endLonA],y=[startLatA,endLatA],pen="1.0p,black")
fig.text(x=startLonA,y=startLatA,text="A",font="10p,Helvetica,black",fill="white")
fig.text(x=endLonA,y=endLatA,text="A'",font="10p,Helvetica,black",fill="white")

fig.plot(x=[startLonB,endLonB],y=[startLatB,endLatB],pen="1.0p,black")
fig.text(x=startLonB,y=startLatB,text="B",font="10p,Helvetica,black",fill="white")
fig.text(x=endLonB,y=endLatB,text="B'",font="10p,Helvetica,black",fill="white")

fig.basemap(map_scale="x0.6i/0.8i+c0+w20")
fig.savefig("figs/cat_all.png")
fig.show()



# In[12]:


# minlon, maxlon = 109.7, 111.2
# minlat, maxlat = -8, -6.5



# Visualization
fig = pygmt.Figure()
topo_data = '@earth_relief_03s' #30 arc second global relief (SRTM15+V2.1 @ 1.0 km)
pygmt.makecpt(
    cmap='grayC',
    series='-100/10000/200',
    continuous=True
)
fig.grdimage(
    grid=topo_data,
    region=[minlon, maxlon, minlat, maxlat], 
    projection='M20c',
    shading=True,
    frame="a",
    N=1
    )
fig.coast(
    region=[minlon, maxlon, minlat, maxlat], 
    shorelines=True,
    )
fig.coast(shorelines=True)

fig.plot(
    x=st.lon,
    y=st.lat, 
    style='t0.55c', 
    color="black", 
    pen='1p,black', 

)
for j in range(0,len(st)):
    fig.text(
    text=st.sta[j],
    x=st.lon[j]+0.01,
    y=st.lat[j]+0.05,
    font="12p,Helvetica,black",
    fill="white")

fig.savefig("figs/stations.png")
fig.show()



# In[14]:


# desta = pd.read_csv("desta.csv")
# desta = st.merge(desta,on="sta")
# desta = desta.sort_values("det").reset_index(drop=True)
# desta.head(20)


# In[15]:


# minlon, maxlon = 102, 104
# minlat, maxlat = 21, 23



# # Visualization
# fig = pygmt.Figure()
# topo_data = '@earth_relief_03s' #30 arc second global relief (SRTM15+V2.1 @ 1.0 km)
# pygmt.makecpt(
#     cmap='grayC',
#     series='-100/10000/200',
#     continuous=True
# )
# fig.grdimage(
#     grid=topo_data,
#     region=[minlon, maxlon, minlat, maxlat], 
#     projection='M20c',
#     shading=True,
#     frame="a",
#     N=1
#     )
# fig.coast(
#     region=[minlon, maxlon, minlat, maxlat], 
#     shorelines=True,
#     )

# fig.coast(borders=["1/1.0p,black,-"],)
# fig.plot(data="map/fault_lv1.txt", pen="2.0,red3",label='"Rank I fault"')
# fig.plot(data="map/DGXuyencap2.txt", pen="1.0",label='"Rank II rault"')
# fig.legend(position='JBL+jBL+o0.2c',box='+gwhite+p1p')

# pygmt.makecpt(cmap="jet", series=[desta.det.min(), desta.det.max()])

# fig.plot(
#     x=desta.lon,
#     y=desta.lat, 
#     style='t0.55c', 
#     color=desta.det, 
#     cmap=True,
#     pen='black', 
#     )
# fig.colorbar(
#     frame='+l"Number of detections"'
#     )
# for j in range(0,len(desta)):
#     fig.text(
#     text=desta.sta[j],
#     x=desta.lon[j]+0.01,
#     y=desta.lat[j]+0.05,
#     font="12p,Helvetica,black",
#     fill="white")
# fig.savefig("figs/detection_map.png",dpi=200)
# fig.show()


# In[16]:


# plt.figure()
# desta[["sta","det"]].plot(kind="bar",x="sta",xlabel="Station",ylabel="Number of detections",legend=False)
# plt.savefig("figs/dect.pdf",bbox_inches="tight",dpi=600)



#%%
import pygmt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
import os, subprocess

minlon, maxlon = 109.7, 111.2
minlat, maxlat = -8, -6.5
grid = pygmt.datasets.load_earth_relief("03s", region=[minlon, maxlon, minlat, maxlat])
# %%

def cross_plot(startLat,startLon,endLat,endLon,cross_name="AA'",fault=None):
    global eq_proj, x_p
    n=4000
    x = xr.DataArray(np.linspace(startLon,endLon,n), dims='along_course')
    y = xr.DataArray(np.linspace(startLat,endLat,n), dims='along_course')
    elevation_profile = grid.interp(lon=x, lat=y, method='linear')
    x_p = ((endLon - startLon)**2+(endLat - startLat)**2)**0.5*111
    xx = np.linspace(0,x_p,n)
    cat[["Longitud","Latitud","Depth","Ml"]].to_csv("project.tmp",sep=" ",index=None,header=None)
    #subprocess.Popen('gmt project project.tmp -C$startLon/$startLat -E$endLon/$endLat -L0/$x_p -W-50/50 -Fpz -Q > point.tmp')

    pygmt.project(data="project.tmp",outfile="point.tmp",center=[startLon,startLat],endpoint=[endLon,endLat],width=[-50,50],length=[0,x_p],convention="pz")
    
    eq_proj = pd.read_csv("point.tmp",delim_whitespace=True,names=["range","dep","Ml"])
    eq_proj.range = eq_proj.range * 110
    s = 1.2*3**eq_proj.Ml/1
    s = s.values
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col', 
                                gridspec_kw={'height_ratios': [1, 4]},
                                figsize=(6, 4))


    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    axes[0].plot(xx,elevation_profile/1000,color="black")
    if fault != None:
        distf = ((fault[0] - startLon)**2 + (fault[1] - startLat)**2)**0.5*110
        axes[0].scatter(distf,0.5,color="k",marker="v",s=40)
        axes[0].text(distf-0.5,0.8,fault[2])

    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(True)
    axes[0].spines['left'].set_visible(True)
    axes[0].get_xaxis().set_ticks([])
    axes[0].set_ylabel("Elevation (km)")

    scatter  = axes[1].scatter(eq_proj.range,eq_proj.dep,s=s,edgecolor="k",facecolors='none')
    axes[1].set_ylim(60,-2)
    axes[1].set_xlim(0,x_p)
    axes[1].set_ylabel("Depth (km)")
    axes[1].set_xlabel("Distance (km)")
    axes[1].get_xaxis().set_ticks(np.arange(0,x_p,10))

    for m in [2,3,4]:
        axes[1].scatter([],[],edgecolor="k",facecolors='none',s=1.2*3**m/1,label="M={}".format(m))
    plt.legend()
    plt.savefig("figs/cross_NLL_neq_all{}.png".format(cross_name[0]),bbox_inches="tight",dpi = 200)


#%%

cat_nll = neq[["Longitud","Latitud","Depth","Ml"]].fillna(1)
cat = cat_nll

cross_plot(startLatA,startLonA,endLatA,endLonA,cross_name="AA'")
cross_plot(startLatB,startLonB,endLatB,endLonB,cross_name="BB'")
# %%
