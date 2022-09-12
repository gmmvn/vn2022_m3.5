# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
dfmc = pd.read_csv("../data/Mocchau_GM_2021_Oct5.csv")
dfn = pd.read_excel("../data/Sa_table_h.xlsx",index_col=0)
# %%
stal = dfmc.STAID.unique()
dfnmc = dfn[dfn.STAID.isin(stal)].reset_index(drop=True)
# %%
i = 0 
sta = stal[i]
dfmct = dfmc.query("STAID == '{}' & MW == 5.0".format(sta)).reset_index(drop=True)
dfnmct = dfnmc.query("STAID == '{}' & MW == 5.0".format(sta)).reset_index(drop=True)
ymc = dfmct.iloc[0,6:].to_list()
xmc = [float(a.strip("T").strip("s")) for a in dfmct.columns[6:].to_list()]
# %%
plt.figure()
plt.plot(xmc,ymc)
plt.show()
# %%
