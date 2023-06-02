import os # for file location
import numpy as np
import pandas as pd
import pyam
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import colors
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
colors = pyam.plotting.PYAM_COLORS
pd.DataFrame({'name': list(colors.keys()), 'color': list(colors.values())})

#GWP constants (100y, IPCC AR6)
gwp_ch4 = 25
gwp_hfc = 1430
gwp_n2o = 298
gwp_pfc = 7390
gwp_sf6 = 25200

#Import data
df_nz = pyam.IamDataFrame(data='NatComms_Net-zero_all data.csv')
df_nz_mag = pyam.IamDataFrame(data='NatComms_Net-zero_MAGICC data.csv')

#Rename scenarios and regions for use later
df_nz.rename(scenario={'GP_NDC2030': 'NDC', 'GP_CurPol': 'CurPol', 'SSP2_SPA2_19I_RE': '1.5C', 'NDC_NZ_Pledge': 'NDC-NZ',
                       'NZ_Pledge':'NZ', 'NZ_INC':'NZ-Br','NZ_Str':'NZ-Str'}, inplace=True)
df_nz.rename(region={'INDIA': 'India', 'RUS': 'Russia', 'CHN': 'China', 'BRA': 'Brazil'}, inplace=True)

#Agreggate variables for new regions
df_nz.aggregate_region(df_nz.variable, region='EU', subregions=('WEU','CEU'), components=False, method='sum', weight=None, append=True, drop_negative_weights=True)
df_nz.aggregate_region(df_nz.variable, region='OECD', subregions=('OCE','WEU','CEU','CAN','JAP','KOR','MEX','TUR','USA'), components=False, method='sum', weight=None, append=True, drop_negative_weights=True)
df_nz.aggregate_region(df_nz.variable, region='non-OECD_target', subregions=('Brazil','China','India','INDO','Russia','SAF','SEAS','STAN','UKR','WAF'), components=False, method='sum', weight=None, append=True, drop_negative_weights=True)
df_nz.aggregate_region(df_nz.variable, region='non-OECD_nontarget', subregions=('EAF','ME','NAF','RCAM','RSAF','RSAM','RSAS'), components=False, method='sum', weight=None, append=True, drop_negative_weights=True)

#Transformation of units to be used later on
df_nz.divide("Emissions|Kyoto Gases", 1000, "Emissions|Kyoto Gases [Gt]",ignore_units=True,append=True)
df_nz.divide("Emissions|CO2", 1000, "Emissions|CO\u2082 [Gt]",ignore_units=True,append=True)
df_nz.divide("Emissions|CO2|AFOLU", 1000, "Emissions|CO2|AFOLU [Gt]",ignore_units=True,append=True)
df_nz.divide("Emissions|CO2|Energy|Demand|Industry", 1000, "Emissions|CO2|Energy|Demand|Industry [Gt]",ignore_units=True,append=True)
df_nz.divide("Emissions|CO2|Energy|Demand|Transportation", 1000, "Emissions|CO2|Energy|Demand|Transportation [Gt]",ignore_units=True,append=True)
df_nz.divide("Emissions|CO2|Energy|Demand|Residential and Commercial", 1000, "Emissions|CO2|Energy|Demand|Residential and Commercial [Gt]",ignore_units=True,append=True)
df_nz.divide("Emissions|CO2|Energy|Supply|Electricity", 1000, "Emissions|CO2|Energy|Supply|Electricity [Gt]",ignore_units=True,append=True)
df_nz.divide("Emissions|CO2|Energy and Industrial Processes", 1000, "Emissions|CO2|Energy and Industrial Processes [Gt]",ignore_units=True,append=True)

df_nz.subtract("Emissions|Kyoto Gases", "Emissions|CO2", "Emissions|non-CO\u2082",ignore_units=True,append=True)
df_nz.subtract("Emissions|Kyoto Gases [Gt]", "Emissions|CO\u2082 [Gt]", "Emissions|non-CO\u2082 [Gt]",ignore_units=True,append=True)

df_nz.multiply("Emissions|CH4", 28, "Emissions|CH4 [Mt CO\u2082eq]",ignore_units=True,append=True)
df_nz.divide("Emissions|CH4 [Mt CO\u2082eq]", 1000, "Emissions|CH4 [Gt CO\u2082eq]",ignore_units=True,append=True)
df_nz.multiply("Emissions|N2O", 273, "Emissions|N2O [kt CO\u2082eq]",ignore_units=True,append=True)
df_nz.divide("Emissions|N2O [kt CO\u2082eq]", 1000000, "Emissions|N2O [Gt CO\u2082eq]",ignore_units=True,append=True)
df_nz.multiply("Emissions|HFC", 1530, "Emissions|HFC [kt CO\u2082eq]",ignore_units=True,append=True)
df_nz.divide("Emissions|HFC [kt CO\u2082eq]", 1000000, "Emissions|HFC [Gt CO\u2082eq]",ignore_units=True,append=True)
df_nz.multiply("Emissions|PFC", 7380, "Emissions|PFC [kt CO\u2082eq]",ignore_units=True,append=True)
df_nz.divide("Emissions|PFC [kt CO\u2082eq]", 1000000, "Emissions|PFC [Gt CO\u2082eq]",ignore_units=True,append=True)
df_nz.multiply("Emissions|SF6", 25200, "Emissions|SF6 [kt CO\u2082eq]",ignore_units=True,append=True)
df_nz.divide("Emissions|SF6 [kt CO\u2082eq]", 1000000, "Emissions|SF6 [Gt CO\u2082eq]",ignore_units=True,append=True)

Fgases_vars=["Emissions|HFC [Gt CO\u2082eq]", "Emissions|PFC [Gt CO\u2082eq]","Emissions|SF6 [Gt CO\u2082eq]"]
df_nz.aggregate("Emissions|Fgases [Gt CO\u2082eq]", components=Fgases_vars, append=True)

#Create energy intensity variables
df_nz.divide("Primary Energy", "GDP|PPP", "Energy intensity|[EJ/billion 2010 USD]",ignore_units=True,append=True)
df_nz.multiply("Energy intensity|[EJ/billion 2010 USD]", 1000, "Energy intensity|[MJ/2010 USD]",ignore_units=True,append=True)

#Create unabated FF in electricity variables
unabatedFF_vars=["Secondary Energy|Electricity|Coal|w/o CCS", "Secondary Energy|Electricity|Gas|w/o CCS","Secondary Energy|Electricity|Oil|w/o CCS"]
df_nz.aggregate("Unabated FF|Electricity", components=unabatedFF_vars, append=True)
df_nz.divide("Unabated FF|Electricity", "Secondary Energy|Electricity", "Unabated FF|Electricity|perc",ignore_units=True,append=True)
df_nz.multiply("Unabated FF|Electricity|perc", 100, "Unabated FF|Electricity [%]",ignore_units=True,append=True)

#Rename long variables for graphs
df_nz.rename(variable={'Emissions|CO2|AFOLU [Gt]': 'CO\u2082|AFOLU', 'Emissions|CO2|Energy|Demand|Industry [Gt]': 'CO\u2082|Industry', 'Emissions|CO2|Energy|Demand|Transportation [Gt]': 'CO\u2082|Transport',
                       'Emissions|CO2|Energy|Demand|Residential and Commercial [Gt]': 'CO\u2082|Buildings',  'Emissions|CO2|Energy|Supply|Electricity [Gt]': 'CO\u2082|Electricity',
                       'Emissions|CO2|Energy and Industrial Processes [Gt]': 'CO\u2082|Energy and Industry', 'Emissions|non-CO\u2082 [Gt]': 'non-CO\u2082',
                       'Emissions|CH4 [Gt CO\u2082eq]':'non-CO\u2082 - CH\u2084','Emissions|N2O [Gt CO\u2082eq]':'non-CO\u2082 - N\u2082O',
                       'Emissions|Fgases [Gt CO\u2082eq]':'non-CO\u2082 - F-gases' }, inplace=True)

#Declare variables to be used in sectoral results
var_sectors=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport", "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082 - CH\u2084","non-CO\u2082 - F-gases","non-CO\u2082 - N\u2082O")

#Primary energy variable creation
df_nz.divide("Primary Energy|Nuclear", 0.4 , "Primary Energy|Nuclear [sub]",ignore_units=True,append=True)
df_nz.divide("Primary Energy|Non-Biomass Renewables|Hydro", 0.4 , "Primary Energy|Non-Biomass Renewables|Hydro [sub]",ignore_units=True,append=True)
df_nz.divide("Primary Energy|Non-Biomass Renewables|Solar", 0.4 , "Primary Energy|Non-Biomass Renewables|Solar [sub]",ignore_units=True,append=True)
df_nz.divide("Primary Energy|Non-Biomass Renewables|Wind", 0.4 , "Primary Energy|Non-Biomass Renewables|Wind [sub]",ignore_units=True,append=True)
df_nz.divide("Primary Energy|Other", 0.4 , "Primary Energy|Other [sub]",ignore_units=True,append=True)
df_nz.rename(variable={"Primary Energy|Coal|w/ CCS":"CoalCCS", "Primary Energy|Coal|w/o CCS":"Coal", "Primary Energy|Oil|w/ CCS":'OilCCS',"Primary Energy|Oil|w/o CCS":'Oil',
                                      "Primary Energy|Gas|w/ CCS":'GasCCS', "Primary Energy|Gas|w/o CCS":'Gas',"Primary Energy|Nuclear [sub]":'Nuclear',"Primary Energy|Non-Biomass Renewables|Hydro [sub]":'Hydropower',
                                      "Primary Energy|Non-Biomass Renewables|Solar [sub]":'Solar', "Primary Energy|Non-Biomass Renewables|Wind [sub]":'Wind', "Primary Energy|Biomass|Modern":'Mod.Biomass',
                                      "Primary Energy|Biomass|Traditional":'Trad.Biomass',"Primary Energy|Biomass|Electricity|w/ CCS":'Mod.BECCS',"Primary Energy|Other [sub]":'Other'}, inplace=True)
#Create variable set to be used for primary energy graphs
var_primary=("CoalCCS", "Coal", "GasCCS", "Gas","Oil","OilCCS","Nuclear","Hydropower", "Solar", "Wind", "Mod.Biomass", "Trad.Biomass","Mod.BECCS","Other")
cmp = ListedColormap(['black', 'gray','#425563','#517891','darkblue','lawngreen','seagreen','pink','sienna','sandybrown','red','gold','darkgreen', 'darkkhaki'])

'''
#Plot Global GHG, mean temp increase (line graphs), probabilistic temp bands (bar graphs) and peak temps (dots) for all scenarios 
'''
color_map = {'CurPol': '#b0724e','NDC': '#f69320','NZ': 'lightsteelblue','NZ-Al': 'lightskyblue',
             'NZ-Br': '#448ee4','NZ-Str': '#0343df','1.5C': '#044a05', 'Historical':'black'}
pyam.run_control().update({'color': {'scenario': color_map}})

fig = plt.figure()
fig.set_figheight(12)
fig.set_figwidth(16)
ax1 = plt.subplot2grid(shape=(6, 21), loc=(0, 0), rowspan = 4,colspan=10)
ax2 = plt.subplot2grid(shape=(6, 21), loc=(0, 11), rowspan = 4, colspan=10)
ax3 = plt.subplot2grid(shape=(6, 7), loc=(4, 0), rowspan=2, colspan=1)
ax4 = plt.subplot2grid(shape=(6, 7), loc=(4, 1), rowspan=2, colspan=1, sharey=ax3)
ax5 = plt.subplot2grid(shape=(6, 7), loc=(4, 2), rowspan=2, colspan=1, sharey=ax3)
ax6 = plt.subplot2grid(shape=(6, 7), loc=(4, 3), rowspan=2, colspan=1, sharey=ax3)
ax7 = plt.subplot2grid(shape=(6, 7), loc=(4, 4), rowspan=2, colspan=1, sharey=ax3)
ax8 = plt.subplot2grid(shape=(6, 7), loc=(4, 5), rowspan=2, colspan=1, sharey=ax3)
ax9 = plt.subplot2grid(shape=(6, 7), loc=(4, 6), rowspan=2, colspan=1, sharey=ax3)

df_nz.filter(variable='Emissions|Kyoto Gases [Gt]', region="World").plot(ax=ax1,color='scenario',title=(''),linewidth=2,zorder=3)
ax1.set(xlabel="",ylabel='Global GHG emissions - GWP100 AR6 (Gt CO\u2082eq $\mathregular{yr^{-1}}$)')
ax1.set_xlim([2005,2100])
ax1.yaxis.get_label().set_fontsize(16)
ax1.yaxis.labelpad = -7
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
handles, labels = ax1.get_legend_handles_labels()
order = [1,3,4,5,6,7,0]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower left',bbox_to_anchor=(0,0.15),
           labelspacing = 0.3,prop={'size': 17},framealpha=0.7,edgecolor="none",frameon=True) 
ax1.axhline(y=0, color='black', linestyle='-', linewidth=2)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(True)
ax1.yaxis.set_tick_params(length=0)
ax1.yaxis.grid(zorder=0,color = "grey", linestyle = "dotted")
ax1.text(1994, 70, 'a', fontweight='bold',fontsize=16)

df_nz.filter(variable='Temperature|Global Mean', region='World').plot(ax=ax2,color='scenario',title=(''),legend=False,linewidth=2,zorder=3)
ax2.set(xlabel="",ylabel="Global mean temperature increase ($^{o}$C)")
ax2.yaxis.get_label().set_fontsize(16)
ax2.yaxis.labelpad = -1
ax2.xaxis.set_tick_params(labelsize=14)
ax2.yaxis.set_tick_params(labelsize=14)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(True)
ax2.yaxis.set_tick_params(length=0)
ax2.yaxis.grid(color = "grey", linestyle = "dotted")
ax2.yaxis.set_label_position("left")
ax2.yaxis.tick_left()
ax2.text(1988, 3.52, 'b', fontweight='bold',fontsize=16)

data_CurPol2=df_nz_mag.filter(model="IMAGE", scenario='CurPol', variable="Exceedance Probability|*", region="World")
data_CurPol2.plot.bar(ax=ax3,bars="year", x="variable", width=0.9,legend=False,zorder=3,cmap=ListedColormap(['#b0724e']),edgecolor = "black",alpha = 0.7,align='center')
ax3.set_title('CurPol', fontsize=16)
ax3.set_ylabel('Probability of temperature increase (%)', fontsize=14)
ax3.set_xlabel('')
f = np.arange(len(data_CurPol2.variable))
ax3.set_xticks(f)
ax3.set_xticklabels(data_CurPol2.unit,ha='left')
ax3.yaxis.set_tick_params(labelsize=14)
ax3.yaxis.grid(zorder=-1,color = "grey", linestyle = "dotted")
ax3.set_axisbelow(True)
ax3.text(-4, 65, 'c', fontweight='bold',fontsize=16)
ax31 = ax3.twinx()
ax31.set_ylim([1,4])
x=[5]
y=df_nz_mag.filter(variable="Temperature|Peak",scenario='CurPol', region='World',year=[2100]).data.value[0]
ax31.scatter(x,y,marker='o', color='black',zorder=3)
ax31.set_yticks([])
ax31.yaxis.grid(zorder=0,color = "grey", linestyle = "dotted")

data_NDC2=df_nz_mag.filter(model="IMAGE", scenario='NDC', variable="Exceedance Probability|*", region="World")
data_NDC2.plot.bar(ax=ax4,bars="year", x="variable", width=0.9,legend=False,zorder=3,cmap=ListedColormap(['#f69320']),edgecolor = "black",alpha = 0.7,align='center')
ax4.set_title('NDC', fontsize=16)
ax4.set_ylabel('')
ax4.set_xlabel('')
x = np.arange(len(data_NDC2.variable))
ax4.set_xticks(x)
ax4.set_xticklabels(data_NDC2.unit,ha='left')
ax4.yaxis.set_tick_params(length=0)
ax4.tick_params(labelleft=False)
ax4.yaxis.labelpad = 0
ax4.yaxis.grid(color = "grey", linestyle = "dotted")
ax41 = ax4.twinx()
ax41.set_ylim([1,4])
x=[4]
y=df_nz_mag.filter(variable="Temperature|Peak",scenario='NDC', region='World',year=[2100]).data.value[0]
ax41.scatter(x,y,marker='o', color='black')
ax41.yaxis.set_tick_params(length=0)
ax41.tick_params(labelright=False)

data_NDCNZ2=df_nz_mag.filter(model="IMAGE", scenario='NZ', variable="Exceedance Probability|*", region="World")
data_NDCNZ2.plot.bar(ax=ax5,bars="year", x="variable", width=0.9,legend=False,zorder=3,cmap=ListedColormap(['lightsteelblue']),edgecolor = "black",alpha = 0.7,align='center')
ax5.set_title('NZ', fontsize=16)
ax5.set_ylabel('')
ax5.set_xlabel('')
x = np.arange(len(data_NDCNZ2.variable))
ax5.set_xticks(x)
ax5.set_xticklabels(data_NDCNZ2.unit,ha='left')
ax5.yaxis.set_tick_params(length=0)
ax5.yaxis.labelpad = 0
ax5.yaxis.grid(color = "grey", linestyle = "dotted")
ax51 = ax5.twinx()
ax51.set_ylim([1,4])
ax51.get_shared_y_axes().join(ax51, ax31)
x=[2]
y=df_nz_mag.filter(variable="Temperature|Peak",scenario='NZ', region='World',year=[2100]).data.value[0]
ax51.scatter(x,y,marker='o', color='black')
ax51.yaxis.set_tick_params(length=0)

data_NZ2=df_nz_mag.filter(model="IMAGE", scenario='NZ-Al', variable="Exceedance Probability|*", region="World")
data_NZ2.plot.bar(ax=ax6,bars="year", x="variable", width=0.9,legend=False,zorder=3,cmap=ListedColormap(['lightskyblue']),edgecolor = "black",alpha = 0.7,align='center')
ax6.set_title('NZ-Al', fontsize=16)
ax6.set_ylabel('')
ax6.set_xlabel('')
x = np.arange(len(data_NZ2.variable))
ax6.set_xticks(x)
ax6.set_xticklabels(data_NZ2.unit,ha='left')
ax6.yaxis.set_tick_params(length=0)
ax6.yaxis.labelpad = 0
ax6.yaxis.grid(color = "grey", linestyle = "dotted")
ax61 = ax6.twinx()
ax61.set_ylim([1,4])
ax61.get_shared_y_axes().join(ax61, ax31)
x=[2]
y=df_nz_mag.filter(variable="Temperature|Peak",scenario='NZ-Al', region='World',year=[2100]).data.value[0]
ax61.scatter(x,y,marker='o', color='black')
ax61.yaxis.set_tick_params(length=0)

data_NZINC2=df_nz_mag.filter(model="IMAGE", scenario='NZ-Br', variable="Exceedance Probability|*",region="World")
data_NZINC2.plot.bar(ax=ax7,bars="year", x="variable", width=0.9,legend=False,zorder=3,cmap=ListedColormap(['#448ee4']),edgecolor = "black",alpha = 0.7,align='center')
ax7.set_title('NZ-Br', fontsize=16)
ax7.set_ylabel('')
ax7.set_xlabel('')
x = np.arange(len(data_NZINC2.variable))
ax7.set_xticks(x)
ax7.set_xticklabels(data_NZINC2.unit,ha='left')
ax7.yaxis.set_tick_params(length=0)
ax7.yaxis.labelpad = 0
ax7.yaxis.grid(color = "grey", linestyle = "dotted")
ax71 = ax7.twinx()
ax71.set_ylim([1,4])
ax71.get_shared_y_axes().join(ax71, ax31)
x=[1]
y=df_nz_mag.filter(variable="Temperature|Peak",scenario='NZ-Br', region='World',year=[2100]).data.value[0]
ax71.scatter(x,y,marker='o', color='black')
ax71.yaxis.set_tick_params(length=0)

data_NZStr2=df_nz_mag.filter(model="IMAGE", scenario='NZ-Str', variable="Exceedance Probability|*", region="World")
data_NZStr2.plot.bar(ax=ax8,bars="year", x="variable", width=0.9,legend=False,zorder=3,cmap=ListedColormap(['#0343df']),edgecolor = "black",alpha = 0.7,align='center')
ax8.set_title('NZ-Str', fontsize=16)
ax8.set_ylabel('')
ax8.set_xlabel('')
x = np.arange(len(data_NZStr2.variable))
ax8.set_xticks(x)
ax8.set_xticklabels(data_NZStr2.unit,ha='left')
ax8.yaxis.set_tick_params(length=0)
ax8.yaxis.labelpad = 0
ax8.yaxis.grid(color = "grey", linestyle = "dotted")
ax81 = ax8.twinx()
ax81.set_ylim([1,4])
ax81.get_shared_y_axes().join(ax81, ax31)
x=[1]
y=df_nz_mag.filter(variable="Temperature|Peak",scenario='NZ-Str', region='World',year=[2100]).data.value[0]
ax81.scatter(x,y,marker='o', color='black')
ax81.yaxis.set_tick_params(length=0)

data_15C2=df_nz_mag.filter(model="IMAGE", scenario='1.5C', variable="Exceedance Probability|*", region="World")
data_15C2.plot.bar(ax=ax9,bars="year", x="variable", width=0.9,legend=False,zorder=3,cmap=ListedColormap(['#044a05']),edgecolor = "black",alpha = 0.7,align='center')
ax9.set_title('1.5C', fontsize=16)
ax9.set_ylabel('')
ax9.set_xlabel('')
x = np.arange(len(data_15C2.variable))
ax9.set_xticks(x)
ax9.set_xticklabels(data_15C2.unit,ha='left')
ax9.yaxis.set_tick_params(length=0)
ax9.tick_params(labelleft=False)
ax9.yaxis.labelpad = 0
ax9.yaxis.grid(color = "grey", linestyle = "dotted")
ax91 = ax9.twinx()
ax91.set_ylim([1,4])
ax91.get_shared_y_axes().join(ax91, ax31)
x=[0]
y=df_nz_mag.filter(variable="Temperature|Peak",scenario='1.5C', region='World',year=[2100]).data.value[0]
ax91.scatter(x,y,marker='o', color='black')
ax91.yaxis.set_tick_params(labelsize=14)
ax91.set_ylabel('Peak temperature increase ($^{o}$C)', fontsize=14)
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.6, wspace=0.3)

'''
Sectoral waterfall graph
'''
df_wf=df_nz.filter(region=('World','OECD','non-OECD_target','non-OECD_nontarget'), scenario=('CurPol','NDC','NZ','NZ-Al','NZ-Br','NZ-Str','1.5C'),year=[2050],variable=('CO\u2082|AFOLU','CO\u2082|Industry','CO\u2082|Transport','CO\u2082|Buildings', 'CO\u2082|Electricity', 'non-CO\u2082'))
df_wf.subtract("NZ","CurPol", "diffNDCNZCP",axis="scenario",ignore_units=True,append=True)
df_wf.subtract("NZ-Al","NZ", "diffNZNDCNZ",axis="scenario",ignore_units=True,append=True)
df_wf.subtract("NZ-Br","NZ-Al", "diffNZINCNZ",axis="scenario",ignore_units=True,append=True)
df_wf.subtract("NDC","CurPol", "diffNDCCP",axis="scenario",ignore_units=True,append=True)
df_wf.subtract("NZ","NDC", "diffNDCNZ",axis="scenario",ignore_units=True,append=True)
df_wf.subtract("NZ-Str","NZ-Br", "diffNZStrNZINC",axis="scenario",ignore_units=True,append=True)
TE_vars=["CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport","CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082"]
df_wf.aggregate("TE",components=TE_vars,append=True)

fig, axs = plt.subplot_mosaic([['a'], ['b'],['c']], constrained_layout=True, figsize=(24,20))
for label, ax in axs.items():
    trans = mtransforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=22, weight = 'bold', verticalalignment='top',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    
cmp_wf = "Set2"
ind = np.arange(start=0, stop=55, step=1.31)

args_CurPol_2050 = dict(scenario=("CurPol"), year=[2050])
args_CurPol_2050= df_nz.filter(**args_CurPol_2050, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="OECD", keep=True)
args_CurPol_2050.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[41], width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=cmp_wf,legend=False)

args_CurPol_base=dict(scenario=("CurPol"), year=[2050])
data_CurPol_base=df_wf.filter(**args_CurPol_base, variable="TE").filter(region="OECD", keep=True)
axs['a'].axhline(y=data_CurPol_base.data.value[0], xmin=0.02, xmax=0.18,color='black', linestyle='--', linewidth=1, zorder=3)

args2 = dict(scenario=("diffNDCCP"), year=[2050])
data2 = df_wf.filter(**args2, variable=("non-CO\u2082")).filter(region="OECD", keep=True)
data2.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),  hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args3 = dict(scenario=("diffNDCCP"), year=[2050])
data3 = df_wf.filter(**args3, variable=("CO\u2082|Transport")).filter(region="OECD", keep=True)
data3.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args4 = dict(scenario=("diffNDCCP"), year=[2050])
data4 = df_wf.filter(**args4, variable=("CO\u2082|Industry")).filter(region="OECD", keep=True)
data4.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args5 = dict(scenario=("diffNDCCP"), year=[2050])
data5 = df_wf.filter(**args5, variable=("CO\u2082|Electricity")).filter(region="OECD", keep=True)
data5.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0]+data4.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args6 = dict(scenario=("diffNDCCP"), year=[2050])
data6 = df_wf.filter(**args6, variable=("CO\u2082|Buildings")).filter(region="OECD", keep=True)
data6.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0]+data4.data.value[0]+data5.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args7 = dict(scenario=("diffNDCCP"), year=[2050])
data7 = df_wf.filter(**args7, variable=("CO\u2082|AFOLU")).filter(region="OECD", keep=True)
data7.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0]+data4.data.value[0]+data5.data.value[0]+data6.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args8 = dict(scenario=("NDC"), year=[2050])
dataNDC = df_nz.filter(**args8, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="OECD", keep=True)
dataNDC.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True, position=ind[38],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NDC_base=dict(scenario=("NDC"), year=[2050])
data_NDC_base=df_wf.filter(**args_NDC_base, variable="TE").filter(region="OECD", keep=True)
axs['a'].axhline(y=data_NDC_base.data.value[0], xmin=0.19, xmax=0.36,color='black', linestyle='--', linewidth=1, zorder=3)

args9 = dict(scenario=("diffNDCNZ"), year=[2050])
data9 = df_wf.filter(**args9, variable=("non-CO\u2082")).filter(region="OECD", keep=True)
data9.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),  hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args10 = dict(scenario=("diffNDCNZ"), year=[2050])
data10 = df_wf.filter(**args10, variable=("CO\u2082|Transport")).filter(region="OECD", keep=True)
data10.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args11 = dict(scenario=("diffNDCNZ"), year=[2050])
data11 = df_wf.filter(**args11, variable=("CO\u2082|Industry")).filter(region="OECD", keep=True)
data11.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0]+data10.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args12 = dict(scenario=("diffNDCNZ"), year=[2050])
data12 = df_wf.filter(**args12, variable=("CO\u2082|Electricity")).filter(region="OECD", keep=True)
data12.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0]+data10.data.value[0]+data11.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args13 = dict(scenario=("diffNDCNZ"), year=[2050])
data13 = df_wf.filter(**args13, variable=("CO\u2082|Buildings")).filter(region="OECD", keep=True)
data13.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0]+data10.data.value[0]+data11.data.value[0]+data12.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args14 = dict(scenario=("diffNDCNZ"), year=[2050])
data14 = df_wf.filter(**args14, variable=("CO\u2082|AFOLU")).filter(region="OECD", keep=True)
data14.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0]+data10.data.value[0]+data11.data.value[0]+data12.data.value[0]+data13.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args15 = dict(scenario=("NZ"), year=[2050])
dataNDCNZ = df_nz.filter(**args15, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="OECD", keep=True)
dataNDCNZ.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True, position=ind[35],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NDCNZ_base=dict(scenario=("NZ"), year=[2050])
data_NDCNZ_base=df_wf.filter(**args_NDCNZ_base, variable="TE").filter(region="OECD", keep=True)
axs['a'].axhline(y=data_NDCNZ_base.data.value[0], xmin=0.37, xmax=0.54,color='black', linestyle='--', linewidth=1, zorder=3)

args16 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data16 = df_wf.filter(**args16, variable=("non-CO\u2082")).filter(region="OECD", keep=True)
data16.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args17 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data17 = df_wf.filter(**args17, variable=("CO\u2082|Transport")).filter(region="OECD", keep=True)
data17.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args18 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data18 = df_wf.filter(**args18, variable=("CO\u2082|Industry")).filter(region="OECD", keep=True)
data18.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args19 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data19 = df_wf.filter(**args19, variable=("CO\u2082|Electricity")).filter(region="OECD", keep=True)
data19.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data17.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args20 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data20 = df_wf.filter(**args20, variable=("CO\u2082|Buildings")).filter(region="OECD", keep=True)
data20.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data18.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args21 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data21 = df_wf.filter(**args21, variable=("CO\u2082|AFOLU")).filter(region="OECD", keep=True)
data21.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data18.data.value[0]+data20.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args22 = dict(scenario=("NZ-Al"), year=[2050])
dataNZ = df_nz.filter(**args22, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="OECD", keep=True)
dataNZ.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True, position=ind[32],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NZ_base=dict(scenario=("NZ-Al"), year=[2050])
data_NZ_base=df_wf.filter(**args_NZ_base, variable="TE").filter(region="OECD", keep=True)
axs['a'].axhline(y=data_NZ_base.data.value[0], xmin=0.55, xmax=0.72,color='black', linestyle='--', linewidth=1, zorder=3)

args23 = dict(scenario=("diffNZINCNZ"), year=[2050])
data23 = df_wf.filter(**args23, variable=("non-CO\u2082")).filter(region="OECD", keep=True)
data23.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args24 = dict(scenario=("diffNZINCNZ"), year=[2050])
data24 = df_wf.filter(**args24, variable=("CO\u2082|Transport")).filter(region="OECD", keep=True)
data24.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args25 = dict(scenario=("diffNZINCNZ"), year=[2050])
data25 = df_wf.filter(**args25, variable=("CO\u2082|Industry")).filter(region="OECD", keep=True)
data25.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args26 = dict(scenario=("diffNZINCNZ"), year=[2050])
data26 = df_wf.filter(**args26, variable=("CO\u2082|Electricity")).filter(region="OECD", keep=True)
data26.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0]+data25.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args27 = dict(scenario=("diffNZINCNZ"), year=[2050])
data27 = df_wf.filter(**args27, variable=("CO\u2082|Buildings")).filter(region="OECD", keep=True)
data27.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0]+data25.data.value[0]+data26.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args28 = dict(scenario=("diffNZINCNZ"), year=[2050])
data28 = df_wf.filter(**args28, variable=("CO\u2082|AFOLU")).filter(region="OECD", keep=True)
data28.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0]+data25.data.value[0]+data26.data.value[0]+data27.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args29 = dict(scenario=("NZ-Br"), year=[2050])
dataNZINC = df_nz.filter(**args29, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="OECD", keep=True)
dataNZINC.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True, position=ind[29],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NZINC_base=dict(scenario=("NZ-Br"), year=[2050])
data_NZINC_base=df_wf.filter(**args_NZINC_base, variable="TE").filter(region="OECD", keep=True)
axs['a'].axhline(y=data_NZINC_base.data.value[0], xmin=0.73, xmax=0.9,color='black', linestyle='--', linewidth=1, zorder=3)

args30 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data30 = df_wf.filter(**args30, variable=("non-CO\u2082")).filter(region="OECD", keep=True)
data30.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]), hatch='//',width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args31 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data31 = df_wf.filter(**args31, variable=("CO\u2082|Transport")).filter(region="OECD", keep=True)
data31.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args32 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data32 = df_wf.filter(**args32, variable=("CO\u2082|Industry")).filter(region="OECD", keep=True)
data32.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args33 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data33 = df_wf.filter(**args33, variable=("CO\u2082|Electricity")).filter(region="OECD", keep=True)
data33.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args34 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data34 = df_wf.filter(**args34, variable=("CO\u2082|Buildings")).filter(region="OECD", keep=True)
data34.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0]+data32.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args35 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data35 = df_wf.filter(**args35, variable=("CO\u2082|AFOLU")).filter(region="OECD", keep=True)
data35.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0]+data32.data.value[0]+data34.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args36 = dict(scenario=("NZ-Str"), year=[2050])
dataNZstr = df_nz.filter(**args36, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="OECD", keep=True)
dataNZstr.plot.bar(ax=axs['a'],bars="variable", x="year", stacked=True, position=ind[26],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NZStr_base=dict(scenario=("NZ-Str"), year=[2050])
data_NZStr_base=df_wf.filter(**args_NZStr_base, variable="TE").filter(region="OECD", keep=True)
axs['a'].axhline(y=data_NZStr_base.data.value[0], xmin=0.91, xmax=0.985,color='black', linestyle='--', linewidth=1, zorder=3)
axs['a'].set_xlim((-2.73,-1.62))
axs['a'].set_ylim((-5,15))
axs['a'].set_title('OECD', fontsize=24)
axs['a'].set_ylabel('Gt CO\u2082eq $\mathregular{yr^{-1}}$', fontsize=24)
axs['a'].set_xlabel('') 
axs['a'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['a'].xaxis.set_tick_params(labelsize=22)
axs['a'].yaxis.set_tick_params(labelsize=22)
axs['a'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['a'].set_facecolor('gainsboro')

#non-OECD_target
args_CurPol_2050 = dict(scenario=("CurPol"), year=[2050])
args_CurPol_2050= df_nz.filter(**args_CurPol_2050, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_target", keep=True)
args_CurPol_2050.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[41], width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=cmp_wf,legend=False)

args_CurPol_base=dict(scenario=("CurPol"), year=[2050])
data_CurPol_base=df_wf.filter(**args_CurPol_base, variable="TE").filter(region="non-OECD_target", keep=True)
axs['b'].axhline(y=data_CurPol_base.data.value[0], xmin=0.02, xmax=0.18,color='black', linestyle='--', linewidth=1, zorder=3)

args2 = dict(scenario=("diffNDCCP"), year=[2050])
data2 = df_wf.filter(**args2, variable=("non-CO\u2082")).filter(region="non-OECD_target", keep=True)
data2.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),  hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args3 = dict(scenario=("diffNDCCP"), year=[2050])
data3 = df_wf.filter(**args3, variable=("CO\u2082|Transport")).filter(region="non-OECD_target", keep=True)
data3.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args4 = dict(scenario=("diffNDCCP"), year=[2050])
data4 = df_wf.filter(**args4, variable=("CO\u2082|Industry")).filter(region="non-OECD_target", keep=True)
data4.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args5 = dict(scenario=("diffNDCCP"), year=[2050])
data5 = df_wf.filter(**args5, variable=("CO\u2082|Electricity")).filter(region="non-OECD_target", keep=True)
data5.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0]+data4.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args6 = dict(scenario=("diffNDCCP"), year=[2050])
data6 = df_wf.filter(**args6, variable=("CO\u2082|Buildings")).filter(region="non-OECD_target", keep=True)
data6.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0]+data4.data.value[0]+data5.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args7 = dict(scenario=("diffNDCCP"), year=[2050])
data7 = df_wf.filter(**args7, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_target", keep=True)
data7.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args8 = dict(scenario=("NDC"), year=[2050])
dataNDC = df_nz.filter(**args8, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_target", keep=True)
dataNDC.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True, position=ind[38],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NDC_base=dict(scenario=("NDC"), year=[2050])
data_NDC_base=df_wf.filter(**args_NDC_base, variable="TE").filter(region="non-OECD_target", keep=True)
axs['b'].axhline(y=data_NDC_base.data.value[0], xmin=0.19, xmax=0.36,color='black', linestyle='--', linewidth=1, zorder=3)

args9 = dict(scenario=("diffNDCNZ"), year=[2050])
data9 = df_wf.filter(**args9, variable=("non-CO\u2082")).filter(region="non-OECD_target", keep=True)
data9.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),  hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args10 = dict(scenario=("diffNDCNZ"), year=[2050])
data10 = df_wf.filter(**args10, variable=("CO\u2082|Transport")).filter(region="non-OECD_target", keep=True)
data10.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args11 = dict(scenario=("diffNDCNZ"), year=[2050])
data11 = df_wf.filter(**args11, variable=("CO\u2082|Industry")).filter(region="non-OECD_target", keep=True)
data11.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0]+data10.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args12 = dict(scenario=("diffNDCNZ"), year=[2050])
data12 = df_wf.filter(**args12, variable=("CO\u2082|Electricity")).filter(region="non-OECD_target", keep=True)
data12.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0]+data10.data.value[0]+data11.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args13 = dict(scenario=("diffNDCNZ"), year=[2050])
data13 = df_wf.filter(**args13, variable=("CO\u2082|Buildings")).filter(region="non-OECD_target", keep=True)
data13.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0]+data10.data.value[0]+data11.data.value[0]+data12.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args14 = dict(scenario=("diffNDCNZ"), year=[2050])
data14 = df_wf.filter(**args14, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_target", keep=True)
data14.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0]+data10.data.value[0]+data11.data.value[0]+data12.data.value[0]+data13.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args15 = dict(scenario=("NZ"), year=[2050])
dataNDCNZ = df_nz.filter(**args15, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_target", keep=True)
dataNDCNZ.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True, position=ind[35],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NDCNZ_base=dict(scenario=("NZ"), year=[2050])
data_NDCNZ_base=df_wf.filter(**args_NDCNZ_base, variable="TE").filter(region="non-OECD_target", keep=True)
axs['b'].axhline(y=data_NDCNZ_base.data.value[0], xmin=0.37, xmax=0.54,color='black', linestyle='--', linewidth=1, zorder=3)

args16 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data16 = df_wf.filter(**args16, variable=("non-CO\u2082")).filter(region="non-OECD_target", keep=True)
data16.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args17 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data17 = df_wf.filter(**args17, variable=("CO\u2082|Transport")).filter(region="non-OECD_target", keep=True)
data17.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args18 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data18 = df_wf.filter(**args18, variable=("CO\u2082|Industry")).filter(region="non-OECD_target", keep=True)
data18.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data17.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args19 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data19 = df_wf.filter(**args19, variable=("CO\u2082|Electricity")).filter(region="non-OECD_target", keep=True)
data19.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data17.data.value[0]+data18.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args20 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data20 = df_wf.filter(**args20, variable=("CO\u2082|Buildings")).filter(region="non-OECD_target", keep=True)
data20.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data17.data.value[0]+data18.data.value[0]+data19.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args21 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data21 = df_wf.filter(**args21, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_target", keep=True)
data21.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data17.data.value[0]+data18.data.value[0]+data19.data.value[0]+data20.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args22 = dict(scenario=("NZ-Al"), year=[2050])
dataNZ = df_nz.filter(**args22, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_target", keep=True)
dataNZ.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True, position=ind[32],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NZ_base=dict(scenario=("NZ-Al"), year=[2050])
data_NZ_base=df_wf.filter(**args_NZ_base, variable="TE").filter(region="non-OECD_target", keep=True)
axs['b'].axhline(y=data_NZ_base.data.value[0], xmin=0.55, xmax=0.72,color='black', linestyle='--', linewidth=1, zorder=3)

args23 = dict(scenario=("diffNZINCNZ"), year=[2050])
data23 = df_wf.filter(**args23, variable=("non-CO\u2082")).filter(region="non-OECD_target", keep=True)
data23.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args24 = dict(scenario=("diffNZINCNZ"), year=[2050])
data24 = df_wf.filter(**args24, variable=("CO\u2082|Transport")).filter(region="non-OECD_target", keep=True)
data24.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args25 = dict(scenario=("diffNZINCNZ"), year=[2050])
data25 = df_wf.filter(**args25, variable=("CO\u2082|Industry")).filter(region="non-OECD_target", keep=True)
data25.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args26 = dict(scenario=("diffNZINCNZ"), year=[2050])
data26 = df_wf.filter(**args26, variable=("CO\u2082|Electricity")).filter(region="non-OECD_target", keep=True)
data26.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0]+data25.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args27 = dict(scenario=("diffNZINCNZ"), year=[2050])
data27 = df_wf.filter(**args27, variable=("CO\u2082|Buildings")).filter(region="non-OECD_target", keep=True)
data27.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0]+data25.data.value[0]+data26.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args28 = dict(scenario=("diffNZINCNZ"), year=[2050])
data28 = df_wf.filter(**args28, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_target", keep=True)
data28.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0]+data25.data.value[0]+data26.data.value[0]+data27.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args29 = dict(scenario=("NZ-Br"), year=[2050])
dataNZINC = df_nz.filter(**args29, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_target", keep=True)
dataNZINC.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True, position=ind[29],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NZINC_base=dict(scenario=("NZ-Br"), year=[2050])
data_NZINC_base=df_wf.filter(**args_NZINC_base, variable="TE").filter(region="non-OECD_target", keep=True)
axs['b'].axhline(y=data_NZINC_base.data.value[0], xmin=0.73, xmax=0.9,color='black', linestyle='--', linewidth=1, zorder=3)

args30 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data30 = df_wf.filter(**args30, variable=("non-CO\u2082")).filter(region="non-OECD_target", keep=True)
data30.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]), hatch='//',width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args31 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data31 = df_wf.filter(**args31, variable=("CO\u2082|Transport")).filter(region="non-OECD_target", keep=True)
data31.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args32 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data32 = df_wf.filter(**args32, variable=("CO\u2082|Industry")).filter(region="non-OECD_target", keep=True)
data32.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args33 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data33 = df_wf.filter(**args33, variable=("CO\u2082|Electricity")).filter(region="non-OECD_target", keep=True)
data33.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0]+data32.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args34 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data34 = df_wf.filter(**args34, variable=("CO\u2082|Buildings")).filter(region="non-OECD_target", keep=True)
data34.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0]+data32.data.value[0]+data33.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args35 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data35 = df_wf.filter(**args35, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_target", keep=True)
data35.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0]+data32.data.value[0]+data33.data.value[0]+data34.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args36 = dict(scenario=("NZ-Str"), year=[2050])
dataNZstr = df_nz.filter(**args36, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_target", keep=True)
dataNZstr.plot.bar(ax=axs['b'],bars="variable", x="year", stacked=True, position=ind[26],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NZStr_base=dict(scenario=("NZ-Str"), year=[2050])
data_NZStr_base=df_wf.filter(**args_NZStr_base, variable="TE").filter(region="non-OECD_target", keep=True)
axs['b'].axhline(y=data_NZStr_base.data.value[0], xmin=0.91, xmax=0.985,color='black', linestyle='--', linewidth=1, zorder=3)
axs['b'].set_xlim((-2.73,-1.62))
axs['b'].set_ylim((-8,32))
handles, labels = axs['b'].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs['b'].set_title('non-OECD with net-zero target', fontsize=24)
axs['b'].set_ylabel('Gt CO\u2082eq $\mathregular{yr^{-1}}$', fontsize=24)
axs['b'].set_xlabel('') 
axs['b'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['b'].xaxis.set_tick_params(labelsize=22)
axs['b'].yaxis.set_tick_params(labelsize=22)
axs['b'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['b'].set_facecolor('gainsboro')

#non-OECD_nontarget
args_CurPol_2050 = dict(scenario=("CurPol"), year=[2050])
args_CurPol_2050= df_nz.filter(**args_CurPol_2050, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
args_CurPol_2050.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[41], width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=cmp_wf,legend=False)

args_CurPol_base=dict(scenario=("CurPol"), year=[2050])
data_CurPol_base=df_wf.filter(**args_CurPol_base, variable="TE").filter(region="non-OECD_nontarget", keep=True)
axs['c'].axhline(y=data_CurPol_base.data.value[0], xmin=0.02, xmax=0.18,color='black', linestyle='--', linewidth=1, zorder=3)

args2 = dict(scenario=("diffNDCCP"), year=[2050])
data2 = df_wf.filter(**args2, variable=("non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
data2.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),  hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args3 = dict(scenario=("diffNDCCP"), year=[2050])
data3 = df_wf.filter(**args3, variable=("CO\u2082|Transport")).filter(region="non-OECD_nontarget", keep=True)
data3.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args4 = dict(scenario=("diffNDCCP"), year=[2050])
data4 = df_wf.filter(**args4, variable=("CO\u2082|Industry")).filter(region="non-OECD_nontarget", keep=True)
data4.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args5 = dict(scenario=("diffNDCCP"), year=[2050])
data5 = df_wf.filter(**args5, variable=("CO\u2082|Electricity")).filter(region="non-OECD_nontarget", keep=True)
data5.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0]+data4.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args6 = dict(scenario=("diffNDCCP"), year=[2050])
data6 = df_wf.filter(**args6, variable=("CO\u2082|Buildings")).filter(region="non-OECD_nontarget", keep=True)
data6.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0]+data4.data.value[0]+data5.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args7 = dict(scenario=("diffNDCCP"), year=[2050])
data7 = df_wf.filter(**args7, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_nontarget", keep=True)
data7.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[39], bottom=data_CurPol_base.data.value[0]+data2.data.value[0]+data3.data.value[0]+data4.data.value[0]+data5.data.value[0]+data6.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args8 = dict(scenario=("NDC"), year=[2050])
dataNDC = df_nz.filter(**args8, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
dataNDC.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True, position=ind[38],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NDC_base=dict(scenario=("NDC"), year=[2050])
data_NDC_base=df_wf.filter(**args_NDC_base, variable="TE").filter(region="non-OECD_nontarget", keep=True)
axs['c'].axhline(y=data_NDC_base.data.value[0], xmin=0.19, xmax=0.36,color='black', linestyle='--', linewidth=1, zorder=3)

args9 = dict(scenario=("diffNDCNZ"), year=[2050])
data9 = df_wf.filter(**args9, variable=("non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
data9.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),  hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args10 = dict(scenario=("diffNDCNZ"), year=[2050])
data10 = df_wf.filter(**args10, variable=("CO\u2082|Transport")).filter(region="non-OECD_nontarget", keep=True)
data10.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args11 = dict(scenario=("diffNDCNZ"), year=[2050])
data11 = df_wf.filter(**args11, variable=("CO\u2082|Industry")).filter(region="non-OECD_nontarget", keep=True)
data11.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data10.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args12 = dict(scenario=("diffNDCNZ"), year=[2050])
data12 = df_wf.filter(**args12, variable=("CO\u2082|Electricity")).filter(region="non-OECD_nontarget", keep=True)
data12.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data10.data.value[0]+data11.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args13 = dict(scenario=("diffNDCNZ"), year=[2050])
data13 = df_wf.filter(**args13, variable=("CO\u2082|Buildings")).filter(region="non-OECD_nontarget", keep=True)
data13.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data10.data.value[0]+data11.data.value[0]+data12.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args14 = dict(scenario=("diffNDCNZ"), year=[2050])
data14 = df_wf.filter(**args14, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_nontarget", keep=True)
data14.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[36], bottom=data_NDC_base.data.value[0]+data9.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args15 = dict(scenario=("NZ"), year=[2050])
dataNDCNZ = df_nz.filter(**args15, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
dataNDCNZ.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True, position=ind[35],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NDCNZ_base=dict(scenario=("NZ"), year=[2050])
data_NDCNZ_base=df_wf.filter(**args_NDCNZ_base, variable="TE").filter(region="non-OECD_nontarget", keep=True)
axs['c'].axhline(y=data_NDCNZ_base.data.value[0], xmin=0.37, xmax=0.54,color='black', linestyle='--', linewidth=1, zorder=3)

args16 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data16 = df_wf.filter(**args16, variable=("non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
data16.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args17 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data17 = df_wf.filter(**args17, variable=("CO\u2082|Transport")).filter(region="non-OECD_nontarget", keep=True)
data17.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args18 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data18 = df_wf.filter(**args18, variable=("CO\u2082|Industry")).filter(region="non-OECD_nontarget", keep=True)
data18.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data17.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args19 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data19 = df_wf.filter(**args19, variable=("CO\u2082|Electricity")).filter(region="non-OECD_nontarget", keep=True)
data19.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data17.data.value[0]+data18.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args20 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data20 = df_wf.filter(**args20, variable=("CO\u2082|Buildings")).filter(region="non-OECD_nontarget", keep=True)
data20.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data17.data.value[0]+data18.data.value[0]+data19.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args21 = dict(scenario=("diffNZNDCNZ"), year=[2050])
data21 = df_wf.filter(**args21, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_nontarget", keep=True)
data21.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[33], bottom=data_NDCNZ_base.data.value[0]+data16.data.value[0]+data17.data.value[0]+data18.data.value[0]+data19.data.value[0]+data20.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args22 = dict(scenario=("NZ-Al"), year=[2050])
dataNZ = df_nz.filter(**args22, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
dataNZ.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True, position=ind[32],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NZ_base=dict(scenario=("NZ-Al"), year=[2050])
data_NZ_base=df_wf.filter(**args_NZ_base, variable="TE").filter(region="non-OECD_nontarget", keep=True)
axs['c'].axhline(y=data_NZ_base.data.value[0], xmin=0.55, xmax=0.72,color='black', linestyle='--', linewidth=1, zorder=3)

args23 = dict(scenario=("diffNZINCNZ"), year=[2050])
data23 = df_wf.filter(**args23, variable=("non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
data23.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]),hatch='//', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args24 = dict(scenario=("diffNZINCNZ"), year=[2050])
data24 = df_wf.filter(**args24, variable=("CO\u2082|Transport")).filter(region="non-OECD_nontarget", keep=True)
data24.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args25 = dict(scenario=("diffNZINCNZ"), year=[2050])
data25 = df_wf.filter(**args25, variable=("CO\u2082|Industry")).filter(region="non-OECD_nontarget", keep=True)
data25.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args26 = dict(scenario=("diffNZINCNZ"), year=[2050])
data26 = df_wf.filter(**args26, variable=("CO\u2082|Electricity")).filter(region="non-OECD_nontarget", keep=True)
data26.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0]+data25.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args27 = dict(scenario=("diffNZINCNZ"), year=[2050])
data27 = df_wf.filter(**args27, variable=("CO\u2082|Buildings")).filter(region="non-OECD_nontarget", keep=True)
data27.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0]+data25.data.value[0]+data26.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args28 = dict(scenario=("diffNZINCNZ"), year=[2050])
data28 = df_wf.filter(**args28, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_nontarget", keep=True)
data28.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[30], bottom=data_NZ_base.data.value[0]+data23.data.value[0]+data24.data.value[0]+data25.data.value[0]+data26.data.value[0]+data27.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args29 = dict(scenario=("NZ-Br"), year=[2050])
dataNZINC = df_nz.filter(**args29, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
dataNZINC.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True, position=ind[29],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NZINC_base=dict(scenario=("NZ-Br"), year=[2050])
data_NZINC_base=df_wf.filter(**args_NZINC_base, variable="TE").filter(region="non-OECD_nontarget", keep=True)
axs['c'].axhline(y=data_NZINC_base.data.value[0], xmin=0.73, xmax=0.9,color='black', linestyle='--', linewidth=1, zorder=3)

args30 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data30 = df_wf.filter(**args30, variable=("non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
data30.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0],
               cmap=ListedColormap([179/256, 179/256, 179/256]), hatch='//',width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args31 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data31 = df_wf.filter(**args31, variable=("CO\u2082|Transport")).filter(region="non-OECD_nontarget", keep=True)
data31.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([229/256, 196/256, 148/256]),hatch='\\',legend=False)

args32 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data32 = df_wf.filter(**args32, variable=("CO\u2082|Industry")).filter(region="non-OECD_nontarget", keep=True)
data32.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0], width=.05,linewidth=0.5,
               edgecolor = "black", zorder=3,cmap=ListedColormap([166/256, 216/256, 84/256]),hatch='//',legend=False)

args33 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data33 = df_wf.filter(**args33, variable=("CO\u2082|Electricity")).filter(region="non-OECD_nontarget", keep=True)
data33.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0]+data32.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([231/256, 138/256, 195/256]),hatch='\\',legend=False)

args34 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data34 = df_wf.filter(**args34, variable=("CO\u2082|Buildings")).filter(region="non-OECD_nontarget", keep=True)
data34.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0]+data32.data.value[0]+data33.data.value[0], 
               width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap=ListedColormap([252/256, 141/256, 98/256]),hatch='//',legend=False)

args35 = dict(scenario=("diffNZStrNZINC"), year=[2050])
data35 = df_wf.filter(**args35, variable=("CO\u2082|AFOLU")).filter(region="non-OECD_nontarget", keep=True)
data35.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True,position=ind[27], bottom=data_NZINC_base.data.value[0]+data30.data.value[0]+data31.data.value[0]+data32.data.value[0]+data33.data.value[0]+data34.data.value[0],
               cmap=ListedColormap([102/256, 194/256, 165/256]),hatch='\\', width=.05,linewidth=0.5,edgecolor = "black", zorder=3,legend=False)

args36 = dict(scenario=("NZ-Str"), year=[2050])
dataNZstr = df_nz.filter(**args36, variable=("CO\u2082|AFOLU", "CO\u2082|Industry", "CO\u2082|Transport",
                                      "CO\u2082|Buildings", "CO\u2082|Electricity", "non-CO\u2082")).filter(region="non-OECD_nontarget", keep=True)
dataNZstr.plot.bar(ax=axs['c'],bars="variable", x="year", stacked=True, position=ind[26],width=.05,linewidth=0.5,edgecolor = "black", zorder=3,cmap = cmp_wf,legend=False)

args_NZStr_base=dict(scenario=("NZ-Str"), year=[2050])
data_NZStr_base=df_wf.filter(**args_NZStr_base, variable="TE").filter(region="non-OECD_nontarget", keep=True)
axs['c'].axhline(y=data_NZStr_base.data.value[0], xmin=0.91, xmax=0.985,color='black', linestyle='--', linewidth=1, zorder=3)
axs['c'].set_xlim((-2.73,-1.62))
axs['c'].set_ylim((-5,15))
axs['c'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['c'].set_title('non-OECD without net-zero target', fontsize=24)
axs['c'].set_ylabel('Gt CO\u2082eq $\mathregular{yr^{-1}}$', fontsize=24)
axs['c'].set_xlabel('') 
axs['c'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['c'].xaxis.set_tick_params(labelsize=22)
axs['c'].yaxis.set_tick_params(labelsize=22)
axs['c'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['c'].set_facecolor('gainsboro')
axs['c'].set_xticks([-2.655, -2.46, -2.265, -2.07, -1.87, -1.675])
axs['c'].set_xticklabels(['CurPol','NDC','NZ','NZ-Al','NZ-Br', 'NZ-Str'],fontsize=22,rotation=45)
custom_legend = [Line2D([0], [0], color='black', linestyle='--',label='Total emissions'),
                Patch(facecolor=([179/256, 179/256, 179/256]), edgecolor=([179/256, 179/256, 179/256]),label='non-CO\u2082'),
                Patch(facecolor=([229/256, 196/256, 148/256]), edgecolor=([229/256, 196/256, 148/256]),label='CO\u2082|Transport'),
                Patch(facecolor=([166/256, 216/256, 84/256]), edgecolor=([166/256, 216/256, 84/256]),label='CO\u2082|Industry'),
                Patch(facecolor=([231/256, 138/256, 195/256]), edgecolor=([231/256, 138/256, 195/256]),label='CO\u2082|Electricity'),
                Patch(facecolor=([252/256, 141/256, 98/256]), edgecolor=([252/256, 141/256, 98/256]),label='CO\u2082|Buildings'),                                
                Patch(facecolor=([102/256, 194/256, 165/256]), edgecolor=([102/256, 194/256, 165/256]),label='CO\u2082|AFOLU')]
axs['c'].legend(handles=custom_legend, loc='upper right',bbox_to_anchor=(1, 1),prop=dict(size=20),ncol=1,framealpha=0) 

'''
Plot continuous graphs for emissions
'''
fig, axs = plt.subplot_mosaic([['a','b','c'],['d','e','f'],['g','h','i']], constrained_layout=True, figsize=(24,20),sharey=False)
for label, ax in axs.items():
    #label physical distance in and down:
    trans = mtransforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=22, weight="bold", verticalalignment='top',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

cmp2 = ListedColormap(['#66c2a5','#fc8d62','#e78ac3','#a6d854','#e5c494','#b3b3b3','#8da0cb','black'])

df_nz.filter(scenario="CurPol", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100],variable=var_sectors, region='OECD',keep=True).plot.stack(ax=axs['a'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['a'].set_title('CurPol', fontsize=26)
axs['a'].set_ylabel('Gt CO\u2082eq $\mathregular{yr^{-1}}$ (GWP-100 AR6)', fontsize=24)
axs['a'].set_xlabel('') 
axs['a'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['a'].yaxis.grid(False)
axs['a'].xaxis.set_tick_params(labelsize=22)
axs['a'].yaxis.set_tick_params(labelsize=22)
axs['a'].yaxis.labelpad = -5
axs['a'].set_ylim((-6,15))
axs['a'].yaxis.set_major_locator(MaxNLocator(integer=True))
df_nz.filter(scenario="NZ", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='OECD',keep=True).plot.stack(ax=axs['b'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
df_test3 = df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='OECD',keep=True)
axs['b'].set_title('NZ', fontsize=26)
axs['b'].set_ylabel('', fontsize=16)
axs['b'].set_xlabel('') 
axs['b'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['b'].xaxis.set_tick_params(labelsize=22)
axs['b'].set_ylim((-6,15))
axs['b'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['b'].set_yticklabels([])
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100],variable=var_sectors, region='OECD',keep=True).plot.stack(ax=axs['c'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['c'].set_title('NZ-Br', fontsize=26)
axs['c'].set_ylabel('', fontsize=20)
axs['c'].set_xlabel('') 
axs['c'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['c'].yaxis.grid(False)
axs['c'].xaxis.set_tick_params(labelsize=22)
axs['c'].set_ylabel("OECD", fontsize=24,rotation=270)
axs['c'].yaxis.set_label_position("right")
axs['c'].yaxis.labelpad = 22
axs['c'].set_ylim((-6,15))
axs['c'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['c'].set_yticklabels([])
df_nz.filter(scenario="CurPol", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_target',keep=True).plot.stack(ax=axs['d'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['d'].set_title('', fontsize=16)
axs['d'].set_ylabel('Gt CO\u2082eq $\mathregular{yr^{-1}}$ (GWP-100 AR6)', fontsize=24)
axs['d'].set_xlabel('') 
axs['d'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['d'].xaxis.set_tick_params(labelsize=22)
axs['d'].yaxis.set_tick_params(labelsize=22)
axs['d'].set_ylim((-7,35))
df_nz.filter(scenario="NZ", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_target',keep=True).plot.stack(ax=axs['e'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['e'].set_title('', fontsize=16)
axs['e'].set_ylabel('', fontsize=16)
axs['e'].set_xlabel('') 
axs['e'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['e'].xaxis.set_tick_params(labelsize=22)
axs['e'].set_ylim((-7,35))
axs['e'].set_yticklabels([])
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_target',keep=True).plot.stack(ax=axs['f'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['f'].set_title('', fontsize=16)
axs['f'].set_ylabel('', fontsize=16)
axs['f'].set_xlabel('') 
axs['f'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['f'].xaxis.set_tick_params(labelsize=22)
axs['f'].set_ylim((-7,35))
axs['f'].set_yticklabels([])
axs['f'].set_ylabel("non-OECD with net-zero target", fontsize=24,rotation=270)
axs['f'].yaxis.set_label_position("right")
axs['f'].yaxis.labelpad = 22
df_nz.filter(scenario="CurPol", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['g'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['g'].set_title('', fontsize=16)
axs['g'].set_ylabel('Gt CO\u2082eq $\mathregular{yr^{-1}}$ (GWP-100 AR6)', fontsize=24)
axs['g'].set_xlabel('') 
axs['g'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['g'].xaxis.set_tick_params(labelsize=22)
axs['g'].yaxis.set_tick_params(labelsize=22)
axs['g'].set_ylim((-7,25))
df_nz.filter(scenario="NZ", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['h'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['h'].set_title('', fontsize=16)
axs['h'].set_ylabel('', fontsize=20)
axs['h'].set_xlabel('') 
axs['h'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['h'].xaxis.set_tick_params(labelsize=22)
axs['h'].set_ylim((-7,25))
axs['h'].set_yticklabels([])
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['i'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=True,cmap=cmp2)
custom_legend = [Line2D([0], [0], color='black', linestyle='--',label='Total'),
                Patch(facecolor=([102/256, 194/256, 165/256]), edgecolor=([102/256, 194/256, 165/256]),label='CO\u2082|AFOLU'),
                Patch(facecolor=([252/256, 141/256, 98/256]), edgecolor=([252/256, 141/256, 98/256]),label='CO\u2082|Buildings'),
                Patch(facecolor=([231/256, 138/256, 195/256]), edgecolor=([231/256, 138/256, 195/256]),label='CO\u2082|Electricity'),
                Patch(facecolor=([166/256, 216/256, 84/256]), edgecolor=([166/256, 216/256, 84/256]),label='CO\u2082|Industry'),
                Patch(facecolor=([229/256, 196/256, 148/256]), edgecolor=([229/256, 196/256, 148/256]),label='CO\u2082|Transport'),
                Patch(facecolor=([179/256, 179/256, 179/256]), edgecolor=([179/256, 179/256, 179/256]),label='non-CO\u2082 - CH\u2084'),
                Patch(facecolor=([141/256, 160/256, 203/256]), edgecolor=([141/256, 160/256, 203/256]),label='non-CO\u2082 - F-gases'),
                Patch(facecolor=([10/256, 10/256, 10/256]), edgecolor=([10/256, 10/256, 10/256]),label='non-CO\u2082 - N\u2082O')]
axs['i'].legend(handles=custom_legend, loc='upper right',bbox_to_anchor=(1, 1),prop=dict(size=20),framealpha=0,ncol=1) 
axs['i'].set_title('', fontsize=16)
axs['i'].set_ylabel('', fontsize=16)
axs['i'].set_xlabel('') 
axs['i'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['i'].xaxis.set_tick_params(labelsize=22)
axs['i'].set_ylim((-7,25))
axs['i'].set_yticklabels([])
axs['i'].set_ylabel("non-OECD without net-zero target", fontsize=24,rotation=270)
axs['i'].yaxis.set_label_position("right")
axs['i'].yaxis.labelpad = 22

'''
Primary Energy
'''
fig, axs = plt.subplot_mosaic([['a', 'b','c'], ['d', 'e','f'], ['g','h','i']], constrained_layout=True, figsize=(24,20))
for label, ax in axs.items():
    #label physical distance in and down:
    trans = mtransforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=22, weight = 'bold', verticalalignment='top',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

df_nz.filter(scenario="CurPol", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100],variable=var_primary, region='OECD',keep=True).plot.stack(ax=axs['a'], legend=False,cmap=cmp)
axs['a'].set_title('CurPol', fontsize=24)
axs['a'].set_ylabel('EJ $\mathregular{yr^{-1}}$', fontsize=24)
axs['a'].set_xlabel('') 
axs['a'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['a'].yaxis.grid(False)
axs['a'].xaxis.set_tick_params(labelsize=20)
axs['a'].yaxis.set_tick_params(labelsize=20)
axs['a'].set_ylim((-10,700))
df_nz.filter(scenario="NZ", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='OECD',keep=True).plot.stack(ax=axs['b'], legend=False,cmap=cmp)
axs['b'].set_title('NZ', fontsize=24)
axs['b'].set_ylabel('', fontsize=16)
axs['b'].set_xlabel('') 
axs['b'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['b'].xaxis.set_tick_params(labelsize=22)
axs['b'].set_ylim((-10,700))
axs['b'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['b'].set_yticklabels([])
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100],variable=var_primary, region='OECD',keep=True).plot.stack(ax=axs['c'], legend=False,cmap=cmp)
axs['c'].set_title('NZ-Br', fontsize=24)
axs['c'].set_ylabel('', fontsize=20)
axs['c'].set_xlabel('') 
axs['c'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['c'].yaxis.grid(False)
axs['c'].xaxis.set_tick_params(labelsize=22)
axs['c'].set_ylabel("OECD", fontsize=24,rotation=270)
axs['c'].yaxis.set_label_position("right")
axs['c'].yaxis.labelpad = 22
axs['c'].set_ylim((-10,700))
axs['c'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['c'].set_yticklabels([])
df_nz.filter(scenario="CurPol", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_target',keep=True).plot.stack(ax=axs['d'], legend=False,cmap=cmp)
axs['d'].set_title('', fontsize=16)
axs['d'].set_ylabel('EJ $\mathregular{yr^{-1}}$', fontsize=24)
axs['d'].set_xlabel('') 
axs['d'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['d'].xaxis.set_tick_params(labelsize=22)
axs['d'].yaxis.set_tick_params(labelsize=22)
axs['d'].set_ylim((-10,700))
df_nz.filter(scenario="NZ", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_target',keep=True).plot.stack(ax=axs['e'], legend=False,cmap=cmp)
axs['e'].set_title('', fontsize=16)
axs['e'].set_ylabel('', fontsize=16)
axs['e'].set_xlabel('') 
axs['e'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['e'].xaxis.set_tick_params(labelsize=22)
axs['e'].set_ylim((-10,700))
axs['e'].set_yticklabels([])
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_target',keep=True).plot.stack(ax=axs['f'],legend=False,cmap=cmp)
axs['f'].set_title('', fontsize=16)
axs['f'].set_ylabel('', fontsize=16)
axs['f'].set_xlabel('') 
axs['f'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['f'].xaxis.set_tick_params(labelsize=22)
axs['f'].set_ylim((-10,700))
axs['f'].set_yticklabels([])
axs['f'].set_ylabel("non-OECD with net-zero target", fontsize=24,rotation=270)
axs['f'].yaxis.set_label_position("right")
axs['f'].yaxis.labelpad = 22
df_nz.filter(scenario="CurPol", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['g'], legend=False,cmap=cmp)
axs['g'].set_title('', fontsize=16)
axs['g'].set_ylabel('EJ $\mathregular{yr^{-1}}$', fontsize=24)
axs['g'].set_xlabel('') 
axs['g'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['g'].xaxis.set_tick_params(labelsize=22)
axs['g'].yaxis.set_tick_params(labelsize=22)
axs['g'].set_ylim((-10,700))
df_nz.filter(scenario="NZ", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['h'], legend=False,cmap=cmp)
axs['h'].set_title('', fontsize=16)
axs['h'].set_ylabel('', fontsize=20)
axs['h'].set_xlabel('') 
axs['h'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['h'].xaxis.set_tick_params(labelsize=22)
axs['h'].set_ylim((-10,700))
axs['h'].set_yticklabels([])
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['i'], legend=True,cmap=cmp)
handles, labels = axs['i'].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
axs['i'].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right', bbox_to_anchor=(1.52, -0.05), prop=dict(size=20.5), framealpha=0, ncol=1)
axs['i'].set_title('', fontsize=16)
axs['i'].set_ylabel('', fontsize=16)
axs['i'].set_xlabel('') 
axs['i'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['i'].xaxis.set_tick_params(labelsize=22)
axs['i'].yaxis.set_tick_params(labelsize=22)
axs['i'].set_ylabel("non-OECD without net-zero target", fontsize=24,rotation=270)
axs['i'].yaxis.set_label_position("right")
axs['i'].yaxis.labelpad = 22
axs['i'].set_ylim((-10,700))
axs['i'].set_yticklabels([])

'''
Supplementary Figures
'''
'''
Plot continuous graphs for emissions
'''
fig, axs = plt.subplot_mosaic([['a','b','c','d'],['e','f','g','h'],['i','j','k','l']], constrained_layout=True, figsize=(24,20),sharey=False)
for label, ax in axs.items():
    #label physical distance in and down:
    trans = mtransforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=22, weight="bold", verticalalignment='top',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
cmp2 = ListedColormap(['#66c2a5','#fc8d62','#e78ac3','#a6d854','#e5c494','#b3b3b3','#8da0cb','black'])

df_nz.filter(scenario="NZ-Al", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100],variable=var_sectors, region='OECD',keep=True).plot.stack(ax=axs['a'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['a'].set_title('NZ-Al', fontsize=26)
axs['a'].set_ylabel('Gt CO\u2082eq $\mathregular{yr^{-1}}$ (GWP-100 AR6)', fontsize=24)
axs['a'].set_xlabel('') 
axs['a'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['a'].yaxis.grid(False)
axs['a'].xaxis.set_tick_params(labelsize=22)
axs['a'].yaxis.set_tick_params(labelsize=22)
axs['a'].yaxis.labelpad = -5
axs['a'].set_ylim((-6,16))
axs['a'].yaxis.set_major_locator(MaxNLocator(integer=True))
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='OECD',keep=True).plot.stack(ax=axs['b'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['b'].set_title('NZ-Br', fontsize=26)
axs['b'].set_ylabel('', fontsize=16)
axs['b'].set_xlabel('') 
axs['b'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['b'].xaxis.set_tick_params(labelsize=22)
axs['b'].set_ylim((-6,16))
axs['b'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['b'].set_yticklabels([])
df_nz.filter(scenario="NZ-Str", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100],variable=var_sectors, region='OECD',keep=True).plot.stack(ax=axs['c'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['c'].set_title('NZ-Str', fontsize=26)
axs['c'].set_ylabel('', fontsize=20)
axs['c'].set_xlabel('') 
axs['c'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['c'].yaxis.grid(False)
axs['c'].xaxis.set_tick_params(labelsize=22)
axs['c'].set_ylabel('')
axs['c'].yaxis.labelpad = 22
axs['c'].set_ylim((-6,16))
axs['c'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['c'].set_yticklabels([])
df_nz.filter(scenario="1.5C", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100],variable=var_sectors, region='OECD',keep=True).plot.stack(ax=axs['d'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['d'].set_title('1.5C', fontsize=26)
axs['d'].set_ylabel('', fontsize=20)
axs['d'].set_xlabel('') 
axs['d'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['d'].yaxis.grid(False)
axs['d'].xaxis.set_tick_params(labelsize=22)
axs['d'].set_ylabel("OECD", fontsize=24,rotation=270)
axs['d'].yaxis.set_label_position("right")
axs['d'].yaxis.labelpad = 22
axs['d'].set_ylim((-6,16))
axs['d'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['d'].set_yticklabels([])
df_nz.filter(scenario="NZ-Al", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_target',keep=True).plot.stack(ax=axs['e'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['e'].set_title('', fontsize=16)
axs['e'].set_ylabel('Gt CO\u2082eq $\mathregular{yr^{-1}}$ (GWP-100 AR6)', fontsize=24)
axs['e'].set_xlabel('') 
axs['e'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['e'].xaxis.set_tick_params(labelsize=22)
axs['e'].yaxis.set_tick_params(labelsize=22)
axs['e'].set_ylim((-7,30))
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_target',keep=True).plot.stack(ax=axs['f'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['f'].set_title('', fontsize=16)
axs['f'].set_ylabel('', fontsize=16)
axs['f'].set_xlabel('') 
axs['f'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['f'].xaxis.set_tick_params(labelsize=22)
axs['f'].set_ylim((-7,30))
axs['f'].set_yticklabels([])
df_nz.filter(scenario="NZ-Str", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_target',keep=True).plot.stack(ax=axs['g'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['g'].set_title('', fontsize=16)
axs['g'].set_ylabel('', fontsize=16)
axs['g'].set_xlabel('') 
axs['g'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['g'].xaxis.set_tick_params(labelsize=22)
axs['g'].set_ylim((-7,30))
axs['g'].set_yticklabels([])
axs['g'].set_ylabel('')
df_nz.filter(scenario="1.5C", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_target',keep=True).plot.stack(ax=axs['h'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['h'].set_title('', fontsize=16)
axs['h'].set_ylabel('', fontsize=16)
axs['h'].set_xlabel('') 
axs['h'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['h'].xaxis.set_tick_params(labelsize=22)
axs['h'].set_ylim((-7,30))
axs['h'].set_yticklabels([])
axs['h'].set_ylabel("non-OECD with net-zero target", fontsize=24,rotation=270)
axs['h'].yaxis.set_label_position("right")
axs['h'].yaxis.labelpad = 22
df_nz.filter(scenario="NZ-Al", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['i'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['i'].set_title('', fontsize=16)
axs['i'].set_ylabel('Gt CO\u2082eq $\mathregular{yr^{-1}}$ (GWP-100 AR6)', fontsize=24)
axs['i'].set_xlabel('') 
axs['i'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['i'].xaxis.set_tick_params(labelsize=22)
axs['i'].yaxis.set_tick_params(labelsize=22)
axs['i'].set_ylim((-7,20))
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['j'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['j'].set_title('', fontsize=16)
axs['j'].set_ylabel('', fontsize=16)
axs['j'].set_xlabel('') 
axs['j'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['j'].xaxis.set_tick_params(labelsize=22)
axs['j'].set_ylim((-7,20))
axs['j'].set_yticklabels([])
axs['j'].set_ylabel('')
df_nz.filter(scenario="NZ-Str", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['k'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=False,cmap=cmp2)
axs['k'].set_title('', fontsize=16)
axs['k'].set_ylabel('', fontsize=16)
axs['k'].set_xlabel('') 
axs['k'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['k'].xaxis.set_tick_params(labelsize=22)
axs['k'].set_ylim((-7,20))
axs['k'].set_yticklabels([])
axs['k'].set_ylabel('')
df_nz.filter(scenario="1.5C", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_sectors, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['l'],total={"color": "black", "ls": "--", "lw": 1.0}, legend=True,cmap=cmp2)
custom_legend = [Line2D([0], [0], color='black', linestyle='--',label='Total'),
                Patch(facecolor=([102/256, 194/256, 165/256]), edgecolor=([102/256, 194/256, 165/256]),label='CO\u2082|AFOLU'),
                Patch(facecolor=([252/256, 141/256, 98/256]), edgecolor=([252/256, 141/256, 98/256]),label='CO\u2082|Buildings'),
                Patch(facecolor=([231/256, 138/256, 195/256]), edgecolor=([231/256, 138/256, 195/256]),label='CO\u2082|Electricity'),
                Patch(facecolor=([166/256, 216/256, 84/256]), edgecolor=([166/256, 216/256, 84/256]),label='CO\u2082|Industry'),
                Patch(facecolor=([229/256, 196/256, 148/256]), edgecolor=([229/256, 196/256, 148/256]),label='CO\u2082|Transport'),
                Patch(facecolor=([179/256, 179/256, 179/256]), edgecolor=([179/256, 179/256, 179/256]),label='non-CO\u2082 - CH\u2084'),
                Patch(facecolor=([141/256, 160/256, 203/256]), edgecolor=([141/256, 160/256, 203/256]),label='non-CO\u2082 - F-gases'),
                Patch(facecolor=([10/256, 10/256, 10/256]), edgecolor=([10/256, 10/256, 10/256]),label='non-CO\u2082 - N\u2082O')]
axs['l'].legend(handles=custom_legend, loc='upper right',bbox_to_anchor=(1, 1),prop=dict(size=18),framealpha=0,ncol=1) 
axs['l'].set_title('', fontsize=16)
axs['l'].set_ylabel('', fontsize=16)
axs['l'].set_xlabel('') 
axs['l'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['l'].xaxis.set_tick_params(labelsize=22)
axs['l'].set_ylim((-7,20))
axs['l'].set_yticklabels([])
axs['l'].set_ylabel("non-OECD without net-zero target", fontsize=24,rotation=270)
axs['l'].yaxis.set_label_position("right")
axs['l'].yaxis.labelpad = 22

'''
Primary Energy
'''
fig, axs = plt.subplot_mosaic([['a', 'b','c','d'], ['e','f','g','h'], ['i','j','k','l']], constrained_layout=True, figsize=(24,20))
for label, ax in axs.items():
    #label physical distance in and down:
    trans = mtransforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=22, weight = 'bold', verticalalignment='top',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    
df_nz.filter(scenario="NZ-Al", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100],variable=var_primary, region='OECD',keep=True).plot.stack(ax=axs['a'], legend=False,cmap=cmp)
axs['a'].set_title('NZ-Al', fontsize=24)
axs['a'].set_ylabel('EJ $\mathregular{yr^{-1}}$', fontsize=24)
axs['a'].set_xlabel('') 
axs['a'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['a'].yaxis.grid(False)
axs['a'].xaxis.set_tick_params(labelsize=20)
axs['a'].yaxis.set_tick_params(labelsize=20)
axs['a'].set_ylim((-10,700))
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='OECD',keep=True).plot.stack(ax=axs['b'], legend=False,cmap=cmp)
axs['b'].set_title('NZ-Br', fontsize=24)
axs['b'].set_ylabel('', fontsize=16)
axs['b'].set_xlabel('') 
axs['b'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['b'].xaxis.set_tick_params(labelsize=22)
axs['b'].set_ylim((-10,700))
axs['b'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['b'].set_yticklabels([])
df_nz.filter(scenario="NZ-Str", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='OECD',keep=True).plot.stack(ax=axs['c'], legend=False,cmap=cmp)
axs['c'].set_title('NZ-Str', fontsize=24)
axs['c'].set_ylabel('', fontsize=16)
axs['c'].set_xlabel('') 
axs['c'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['c'].xaxis.set_tick_params(labelsize=22)
axs['c'].set_ylim((-10,700))
axs['c'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['c'].set_yticklabels([])
df_nz.filter(scenario="1.5C", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100],variable=var_primary, region='OECD',keep=True).plot.stack(ax=axs['d'], legend=False,cmap=cmp)
axs['d'].set_title('1.5C', fontsize=24)
axs['d'].set_ylabel('', fontsize=20)
axs['d'].set_xlabel('') 
axs['d'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['d'].yaxis.grid(False)
axs['d'].xaxis.set_tick_params(labelsize=22)
axs['d'].set_ylabel("OECD", fontsize=24,rotation=270)
axs['d'].yaxis.set_label_position("right")
axs['d'].yaxis.labelpad = 22
axs['d'].set_ylim((-10,700))
axs['d'].yaxis.set_major_locator(MaxNLocator(integer=True))
axs['d'].set_yticklabels([])
df_nz.filter(scenario="NZ-Al", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_target',keep=True).plot.stack(ax=axs['e'], legend=False,cmap=cmp)
axs['e'].set_title('', fontsize=16)
axs['e'].set_ylabel('EJ $\mathregular{yr^{-1}}$', fontsize=24)
axs['e'].set_xlabel('') 
axs['e'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['e'].xaxis.set_tick_params(labelsize=22)
axs['e'].yaxis.set_tick_params(labelsize=22)
axs['e'].set_ylim((-10,700))
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_target',keep=True).plot.stack(ax=axs['f'], legend=False,cmap=cmp)
axs['f'].set_title('', fontsize=16)
axs['f'].set_ylabel('', fontsize=16)
axs['f'].set_xlabel('') 
axs['f'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['f'].xaxis.set_tick_params(labelsize=22)
axs['f'].set_ylim((-10,700))
axs['f'].set_yticklabels([])
df_nz.filter(scenario="NZ-Str", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_target',keep=True).plot.stack(ax=axs['g'], legend=False,cmap=cmp)
axs['g'].set_title('', fontsize=16)
axs['g'].set_ylabel('', fontsize=16)
axs['g'].set_xlabel('') 
axs['g'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['g'].xaxis.set_tick_params(labelsize=22)
axs['g'].set_ylim((-10,700))
axs['g'].set_yticklabels([])
df_nz.filter(scenario="1.5C", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_target',keep=True).plot.stack(ax=axs['h'],legend=False,cmap=cmp)
axs['h'].set_title('', fontsize=16)
axs['h'].set_ylabel('', fontsize=16)
axs['h'].set_xlabel('') 
axs['h'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['h'].xaxis.set_tick_params(labelsize=22)
axs['h'].set_ylim((-10,700))
axs['h'].set_yticklabels([])
axs['h'].set_ylabel("non-OECD with net-zero target", fontsize=24,rotation=270)
axs['h'].yaxis.set_label_position("right")
axs['h'].yaxis.labelpad = 22
df_nz.filter(scenario="NZ-Al", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['i'], legend=False,cmap=cmp)
axs['i'].set_title('', fontsize=16)
axs['i'].set_ylabel('EJ $\mathregular{yr^{-1}}$', fontsize=24)
axs['i'].set_xlabel('') 
axs['i'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['i'].xaxis.set_tick_params(labelsize=22)
axs['i'].yaxis.set_tick_params(labelsize=22)
axs['i'].set_ylim((-10,700))
df_nz.filter(scenario="NZ-Br", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['j'], legend=False,cmap=cmp)
axs['j'].set_title('', fontsize=16)
axs['j'].set_ylabel('', fontsize=20)
axs['j'].set_xlabel('') 
axs['j'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['j'].xaxis.set_tick_params(labelsize=22)
axs['j'].set_ylim((-10,700))
axs['j'].set_yticklabels([])
df_nz.filter(scenario="NZ-Str", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['k'], legend=False,cmap=cmp)
axs['k'].set_title('', fontsize=16)
axs['k'].set_ylabel('', fontsize=20)
axs['k'].set_xlabel('') 
axs['k'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['k'].xaxis.set_tick_params(labelsize=22)
axs['k'].set_ylim((-10,700))
axs['k'].set_yticklabels([])
df_nz.filter(scenario="1.5C", year=[2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065,2070,2075,2080,2085,2090,2095,2100], variable=var_primary, region='non-OECD_nontarget',keep=True).plot.stack(ax=axs['l'], legend=True,cmap=cmp)
handles, labels = axs['l'].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
axs['l'].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right', bbox_to_anchor=(1.75, -0.05), prop=dict(size=20.5), framealpha=0, ncol=1)
axs['l'].set_title('', fontsize=16)
axs['l'].set_ylabel('', fontsize=16)
axs['l'].set_xlabel('') 
axs['l'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['l'].xaxis.set_tick_params(labelsize=22)
axs['l'].yaxis.set_tick_params(labelsize=22)
axs['l'].set_ylabel("non-OECD without net-zero target", fontsize=24,rotation=270)
axs['l'].yaxis.set_label_position("right")
axs['l'].yaxis.labelpad = 22
axs['l'].set_ylim((-10,700))
axs['l'].set_yticklabels([])

'''
#Energy indicators
'''
fig, axs = plt.subplot_mosaic([['a', 'b','c'], ['d', 'e','f'], ['g','h','i']], constrained_layout=True, figsize=(24,20))
for label, ax in axs.items():
    #label physical distance in and down:
    trans = mtransforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=22, weight = 'bold', verticalalignment='top',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

df_nz.filter(variable='Unabated FF|Electricity [%]', region="OECD").plot(ax=axs['a'],color='scenario', legend=False)
axs['a'].set_title('Unabated fossil fuels in electricity',size=24)
axs['a'].set(xlabel="")
axs['a'].set_ylabel(ylabel='Share of total (%)',fontsize=24)
axs['a'].xaxis.set_tick_params(labelsize=22)
axs['a'].yaxis.set_tick_params(labelsize=22)
axs['a'].yaxis.labelpad = 0
axs['a'].yaxis.grid(True, zorder=0)
df_nz.filter(variable='Energy intensity|[MJ/2010 USD]', region="OECD").plot(ax=axs['b'],color='scenario', legend=False)
axs['b'].set_title('Energy intensity per unit GDP',size=24)
axs['b'].set(xlabel="")
axs['b'].set_ylabel(ylabel='MJ/2010 USD PPP',fontsize=24)
axs['b'].xaxis.set_tick_params(labelsize=22)
axs['b'].yaxis.set_tick_params(labelsize=22)
axs['b'].yaxis.labelpad = 0
axs['b'].yaxis.grid(True, zorder=0)
df_nz.filter(variable='Final Energy', region="OECD").plot(ax=axs['c'],color='scenario', legend=False)
axs['c'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['c'].set_title('Final energy consumption',size=24)
axs['c'].set(xlabel="")
axs['c'].set_ylabel(ylabel='EJ/yr$\mathregular{-^{1}}$',fontsize=24)
axs['c'].xaxis.set_tick_params(labelsize=22)
axs['c'].yaxis.set_tick_params(labelsize=22)
axs['c'].yaxis.labelpad = 0
axs['c'].yaxis.grid(True, zorder=0)
axs['c'].set_ylim((100,160))
axs['c'].annotate('OECD', xy=(2105, 125), xytext=(2105, 125),color='black',size=24,rotation=270, annotation_clip=False)
handles, labels = axs['c'].get_legend_handles_labels()
order = [1,2,3,4,5,6,0]
axs['c'].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right',bbox_to_anchor=(1,1),
           labelspacing = 0.3,prop={'size': 20},framealpha=0.7,edgecolor="none",frameon=True) 
df_nz.filter(variable='Unabated FF|Electricity [%]', region="non-OECD_target").plot(ax=axs['d'],color='scenario', legend=False)
axs['d'].set_title('',size=16)
axs['d'].set(xlabel="")
axs['d'].set_ylabel(ylabel='Share of total (%)',fontsize=24)
axs['d'].xaxis.set_tick_params(labelsize=22)
axs['d'].yaxis.set_tick_params(labelsize=22)
axs['d'].yaxis.labelpad = 0
axs['d'].yaxis.grid(True, zorder=0)
df_nz.filter(variable='Energy intensity|[MJ/2010 USD]', region="non-OECD_target").plot(ax=axs['e'],color='scenario', legend=False)
axs['e'].set_title('',size=16)
axs['e'].set(xlabel="")
axs['e'].set_ylabel(ylabel='MJ/2010 USD PPP',fontsize=24)
axs['e'].xaxis.set_tick_params(labelsize=22)
axs['e'].yaxis.set_tick_params(labelsize=22)
axs['e'].yaxis.labelpad = 0
axs['e'].yaxis.grid(True, zorder=0)
df_nz.filter(variable='Final Energy', region="non-OECD_target").plot(ax=axs['f'],color='scenario', legend=False)
axs['f'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['f'].set_title('',size=16)
axs['f'].set(xlabel="")
axs['f'].set_ylabel(ylabel='EJ/yr$\mathregular{-^{1}}$',fontsize=24)
axs['f'].xaxis.set_tick_params(labelsize=22)
axs['f'].yaxis.set_tick_params(labelsize=22)
axs['f'].yaxis.labelpad = 0
axs['f'].yaxis.grid(True, zorder=0)
axs['f'].set_ylim((100,350))
axs['f'].annotate('non-OECD with net-zero target', xy=(2105, 115), xytext=(2105, 115),color='black',size=24,rotation=270, annotation_clip=False)
df_nz.filter(variable='Unabated FF|Electricity [%]', region="non-OECD_nontarget").plot(ax=axs['g'],color='scenario', legend=False)
axs['g'].set_title('',size=16)
axs['g'].set(xlabel="")
axs['g'].set_ylabel(ylabel='Share of total (%)',fontsize=24)
axs['g'].xaxis.set_tick_params(labelsize=22)
axs['g'].yaxis.set_tick_params(labelsize=22)
axs['g'].yaxis.labelpad = 0
axs['g'].yaxis.grid(True, zorder=0)
df_nz.filter(variable='Energy intensity|[MJ/2010 USD]', region="non-OECD_nontarget").plot(ax=axs['h'],color='scenario', legend=False)
axs['h'].set_title('',size=16)
axs['h'].set(xlabel="")
axs['h'].set_ylabel(ylabel='MJ/2010 USD PPP',fontsize=24)
axs['h'].xaxis.set_tick_params(labelsize=22)
axs['h'].yaxis.set_tick_params(labelsize=22)
axs['h'].yaxis.labelpad = 0
axs['h'].yaxis.grid(True, zorder=0)
df_nz.filter(variable='Final Energy', region="non-OECD_nontarget").plot(ax=axs['i'],color='scenario', legend=False)
axs['i'].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs['i'].set_title('',size=16)
axs['i'].set(xlabel="")
axs['i'].set_ylabel(ylabel='EJ/yr$\mathregular{-^{1}}$',fontsize=24)
axs['i'].xaxis.set_tick_params(labelsize=22)
axs['i'].yaxis.set_tick_params(labelsize=22)
axs['i'].yaxis.labelpad = 0
axs['i'].yaxis.grid(True, zorder=0)
axs['i'].set_ylim((0,200))
axs['i'].annotate('non-OECD without net-zero target', xy=(2105, 5), xytext=(2105, 5),color='black',size=24,rotation=270, annotation_clip=False)
