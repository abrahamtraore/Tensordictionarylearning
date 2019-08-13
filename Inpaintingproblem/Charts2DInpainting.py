#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 17:45:27 2018

@author: Traoreabraham
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pdb

def Time_range(Time_per_epoch,nbpoints):
    Result=[]
    for i in range(len(Time_per_epoch)-1):
        Temp=np.linspace(Time_per_epoch[i],Time_per_epoch[i+1],nbpoints)
        for t in Temp:
            Result.append(t)
    return Result

def Samplearray(Objectivefunction,numberofpoints):
    result=[]
    Array=Objectivefunction
    period=int(np.size(Array)/numberofpoints)#We consider the objective function  the first iteration
    for i in range(numberofpoints):
        result.append(Array[i*period])
    return np.array(result)

def Transform_into_list(Array):
    Result=[]
    [W,H]=np.array(Array.shape,dtype=int)
    for w in range(W):
        for element in Array[w,:]:
            Result.append(element)
    return Result

def Sample_intermeadiary_points_one_list(listofpoints,numberofpoints):
    Result=[]
    #Result.append(listofpoints[0])
    listofpointscopy=list(listofpoints[1:len(listofpoints)-1])
    period=int(len(listofpointscopy)/numberofpoints)
    for i in range(numberofpoints):
        Result.append(listofpointscopy[i*period])
    Result.append(listofpoints[len(listofpoints)-1])
    return Result
#listofpoints=[1,2,3,4,5,6,7,8,9]
#numberofpoints=2
#Result=Sample_intermeadiary_points(listofpoints,numberofpoints)
#pdb.set_trace()     
   
def Sample_points_all_lists(Arrayofpoints,numberofpoints):
    Result=[]
    [nrows,ncols]=np.array(Arrayofpoints.shape,dtype=int)
    for n in range(nrows):
        Temp=Sample_intermeadiary_points_one_list(Arrayofpoints[n],numberofpoints)
        for elements in Temp:
            Result.append(elements)
    
    return Result



#PSNR over different epochs for OTL
PSNROTLsingleepochsone=np.array([14.5094,16.0967,19.1163,18.9457])
PSNROTLsinglesingleepochstwo=np.array([8.7732,15.4424,19.7941,19.9684])
PSNROTLsinglesingleepochsthree=np.array([9.6645,18.7105,21.0345,21.3795])
PSNROnlinesingleepochsfour=np.array([12.3197,16.1481,17.4180,17.7171])
PSNROTLsingleepochsfive=np.array([10.8579,19.8469,19.3713,18.6899])   
PSNROTLsingle=PSNROTLsingleepochsone
PSNROTLsingle=PSNROTLsingle+PSNROTLsinglesingleepochstwo
PSNROTLsingle=PSNROTLsingle+PSNROTLsinglesingleepochsthree
PSNROTLsingle=PSNROTLsingle+PSNROnlinesingleepochsfour
PSNROTLsingle=PSNROTLsingle+PSNROTLsingleepochsfive
PSNROTLsingle=PSNROTLsingle/5

#PSNR over different epochs for OTLminibatch
PSNROTLsingleminibatchepochsone=np.array([16.8955,17.4619,17.3437,18.2715])
PSNROTLsingleminibatchepochstwo=np.array([18.9209,19.3657,19.5262,18.4071])
PSNROTLsingleminibatchepochsthree=np.array([8.5928,15.3109,16.9659,15.3724])
PSNROTLsingleminibatchepochsfour=np.array([18.8274,4.5112,17.8352,19.5215])
PSNROTLsingleminibatchepochsfive=np.array([16.3061,15.3746,16.1683,16.0000])

PSNROTLsingleminibatch=PSNROTLsingleminibatchepochsone
PSNROTLsingleminibatch=PSNROTLsingleminibatch+PSNROTLsingleminibatchepochstwo
PSNROTLsingleminibatch=PSNROTLsingleminibatch+PSNROTLsingleminibatchepochsthree
PSNROTLsingleminibatch=PSNROTLsingleminibatch+PSNROnlinesingleepochsfour
PSNROTLsingleminibatch=PSNROTLsingleminibatch+PSNROTLsingleminibatchepochsfour
PSNROTLsingleminibatch=PSNROTLsingleminibatch/5

print("The PSNR over the number of epochs for OTLsingle is")
print(PSNROTLsingle)

print("The PSNR over the number of epochs for OTLminibatch is")
print(PSNROTLsingleminibatch)
Epochs=np.array([1,2,3,4])
f=plt.figure()
plt.plot(Epochs,PSNROTLsingle,label="OTLsingle-nonnegative",color='blue',markersize=10,ls='--',marker='*')
plt.xlabel("Epochs",fontsize=20)
plt.ylabel("PSNR",fontsize=20)
plt.legend(fontsize=15)
plt.show()
f.savefig("PSNRvsEpochsOTLm.pdf")

f=plt.figure()
Epochs=np.array([1,2,3,4])
plt.plot(Epochs,PSNROTLsingleminibatch,label="OTLminibatch-nonnegative",color='red',markersize=10,ls='--',marker='H')
plt.xlabel("Epochs",fontsize=20)
plt.ylabel("PSNR",fontsize=20)
plt.legend(fontsize=15)
plt.show()
f.savefig("PSNRvsEpochsOTLs.pdf")

##For the second plot, we take into account the fact that the two methods have differnt running time
#
#plt.xlabel("CPU running time (seconds)")
#plt.ylabel("Objective function on logscale")
#plt.legend()
#plt.show()
#f.savefig("ObjectivefunctionperepochOnlineVsMinibatchTimeVision.pdf")




#pdb.set_trace()



#These charts represent the results for the Cap paper
#Lag=np.array([4,5,6])
#RMSEOnlinesingle=np.array([0.2640443219502373, 0.35295563578281858,0.5532928323635411])
#RunningtimeOnlinesingle=np.array([61485.314256999998,64555.844251000002,65921.896438666663])
#RMSETuckerBatch=np.array([0.25397809130853394,0.33843666369259778,0.54205088966162285])
#RunningtimeTuckerBatch=np.array([86000,875113.09340433322,909563.28849999991])
#RMSEAlto=np.array([0.83546900947685465,1.168293769686956,1.5634654291911048])
#RMSEOnlineminibatch=np.array([0.26404432776560699,0.35295563812664793,0.551068425974095])
#RunningtimeOnlineminibatch=np.array([15397.890069999999,32944.352461000002,15775.247533])
#Data=np.zeros((3,4))
#Data[:,0]=RMSEOnlinesingle
#Data[:,1]=RMSETuckerBatch
#Data[:,2]=RMSEAlto
#Data[:,3]=RMSEOnlineminibatch
#df2 =pd.DataFrame(Data, columns=['DLTsingle', 'Tucker','Alto','DTLminibatch'])
#ax = df2.plot(kind='bar',color=['maroon','red','green','blue'])#.bar()
#ax.set_xlabel("Rank")
#ax.set_xticklabels([i+3 for i in range(3)])
#ax.set_ylabel("Root Mean Square Error",fontsize=15)
#fig=ax.get_figure()
#fig.savefig("RMSE.pdf")

#import numpy as np
#import matplotlib.pyplot as plt
#plt.style.use('seaborn-deep')
#Lag=np.array([4,5,6])
#Data=np.zeros((3,3))
#Data[:,0]=RunningtimeOnlinesingle
#Data[:,1]=RunningtimeTuckerBatch
#Data[:,2]=RunningtimeOnlineminibatch
#Data=Data/3600
#df2 =pd.DataFrame(Data, columns=['DLTsingle', 'Tucker','DTLminibatch'])
#ax = df2.plot(kind='bar',color=['maroon','red','blue'])#.bar()
#ax.set_xlabel("Rank")
#ax.set_xticklabels([i+3 for i in range(3)])
#
#ax.set_ylabel("CPU Running Time (in hours)",fontsize=15)
#fig=ax.get_figure()
#fig.savefig("Runningtime.pdf")



#RMSEOnline=np.array([0.1249,0.1249,0.1249])
#RMSEAlto=np.array([0.1253,0.1251,0.1250])
#RMSETucker=np.array([0.1248,0.1249,0.1249])
#Data=np.zeros((3,3))
#Data[:,0]=RMSEOnline
#Data[:,1]=RMSETucker
#Data[:,2]=RMSEAlto

#df2 =pd.DataFrame(Data, columns=['DLTsingle', 'Tucker','Alto'])
#ax = df2.plot(kind='bar',color=['maroon','red','blue'])#.bar()
#ax.set_xlabel("Lag parameter L")
#ax.set_xticklabels([i+1 for i in range(3)])

#ax.set_ylabel("Root Mean Square Error",fontsize=10)
#ax.set_ylim(0.1240,0.1255)
#fig=ax.get_figure()
#fig.savefig("RMSESaptiotemporal.pdf")




#Data=np.zeros((5,4))
#Data[:,0]=RMSETucker
#Data[:,1]=RMSEOnlinesingle
#Data[:,2]=RMSEOnlineMinibatch
#Data[:,3]=RMSEALto
#
#Std=np.zeros((5,4))
#Std[:,0]=StdOnlinesingle
#Std[:,1]=StdOnlineMinibatch
#Std[:,2]=StdTucker
#Std[:,3]=StdAlto
#
#df2=pd.DataFrame(Data, columns=['OTLsingle','OTLminibatch','TuckerBatch','Alto'])
#ax=df2.plot(kind='bar',color=['maroon','red','blue','green'])#,yerr=Std.T)#.bar()
#ax.set_xlabel("Rank R")
#ax.set_xticklabels([i+1 for i in range(5)])
#ax.set_ylabel("Root Mean Square Error",fontsize=10)

PSNROnlinesingleallrunsrank10=np.array([24.205233664650248,22.297233999281897,22.157737826874136])
Objectivefunctiononlinesinglepochrank10run1=np.array([24512.727391147284, 24499.021761336033, 24501.146126665408, 24460.492674061181, 24448.322986293166])
Objectivefunctiononlinesinglepochrank10run2=np.array([24495.997574514226, 24497.500559376727, 24441.498238504068, 24405.686429190202, 24443.326448758569])
Objectivefunctiononlinesinglepochrank10run3=np.array([24479.463112639747, 24488.655848091174, 24421.326444680344, 24453.019311339929, 24416.448908809722])
Objectivefunctiononlinesinglepochrank10=(Objectivefunctiononlinesinglepochrank10run1+Objectivefunctiononlinesinglepochrank10run2+Objectivefunctiononlinesinglepochrank10run3)/3

PSNROnlinesingleallrunsrank12=np.array([24.205233664650248,22.297233999281897,22.157737826874136])
Objectivefunctiononlinesinglepochrank12run1=np.array([24489.824145992006, 24363.082592746825, 24397.060742111691, 24385.332042448867, 24386.528717031673])
Objectivefunctiononlinesinglepochrank12run2=np.array([24506.183612676359, 24364.272364271143, 24379.876019304174, 24367.982000963144, 24391.929580978755])
Objectivefunctiononlinesinglepochrank12run3=np.array([24490.586777325327, 24467.405443768355, 24473.199393269701, 24389.115265591237, 24385.585479316593])
Objectivefunctiononlinesinglepochrank12=(Objectivefunctiononlinesinglepochrank12run1+Objectivefunctiononlinesinglepochrank12run2+Objectivefunctiononlinesinglepochrank12run3)/3

PSNROnlinesingleallrunsrank14=np.array([18.739808123584748,23.528969920554729,19.628829152577843])
Objectivefunctiononlinesinglepochrank14run1=np.array([24496.810414241416, 24463.558410093188, 24445.669371919452, 24350.386670176038, 24333.794093814631])
Objectivefunctiononlinesinglepochrank14run2=np.array([24500.98246063929, 24471.178766882767, 24439.44076391323, 24378.567467848407, 24369.769498639933])
Objectivefunctiononlinesinglepochrank14run3=np.array([24485.922762996564, 24480.535688608521, 24376.581678681625, 24402.487390663122, 24383.01882822219])
Objectivefunctiononlinesinglepochrank14=(Objectivefunctiononlinesinglepochrank14run1+Objectivefunctiononlinesinglepochrank14run2+Objectivefunctiononlinesinglepochrank14run3)/3


PSNROnlinesingleallrunsrank16=np.array([20.630745800977227,18.832363489378711,20.287761057173256])
Objectivefunctiononlinesinglepochrank16run1=np.array([24426.498009810501, 24415.578152233418, 24391.23107655127, 24345.060640645657, 24387.817998138591])
Objectivefunctiononlinesinglepochrank16run2=np.array([24453.035436200767, 24303.702783240191, 24397.453023810867, 24357.876161220898, 24375.061801536871])
Objectivefunctiononlinesinglepochrank16run3=np.array([24495.71356524104, 24379.599700637369, 24374.181173028028, 24369.981785401298, 24363.802263826361])
Objectivefunctiononlinesinglepochrank16=(Objectivefunctiononlinesinglepochrank16run1+Objectivefunctiononlinesinglepochrank16run2+Objectivefunctiononlinesinglepochrank16run3)/3

PSNROnlinesingleallrunsrank18=np.array([22.230572220674343,19.917177907604188,20.543230542166118])
Objectivefunctiononlinesinglepochrank18run1=np.array([24501.416544333897, 24436.927550578028,24419.655262214314, 24366.550725453919, 24360.061628254818])
Objectivefunctiononlinesinglepochrank18run2=np.array([24469.34670655485, 24362.197165366797,24344.67676365345, 24338.052064493757, 24373.60294202904])
Objectivefunctiononlinesinglepochrank18run3=np.array([24500.480397340874, 24472.776424190368, 24345.001823400689, 24360.610466314436, 24318.532345587162])
Objectivefunctiononlinesinglepochrank18=(Objectivefunctiononlinesinglepochrank18run1+Objectivefunctiononlinesinglepochrank18run2+Objectivefunctiononlinesinglepochrank18run3)/3

PSNROnlineminibatchallrunsrank10=np.array([18.058093491520953,21.353430804613012,19.046278568050564])
Objectivefunctiononlineminibatchpchrank10run1=np.array([24399.23620426972, 24382.586964847076, 24371.580329784232, 24387.12252507509, 24403.628135569226])
Objectivefunctiononlineminibatchpchrank10run2=np.array([24456.54148669069, 24430.533895987599, 24439.329017523134, 24396.507358126033, 24409.894235027357])
Objectivefunctiononlineminibatchpchrank10run3=np.array([24461.241299838177, 24407.657939273711, 24381.882393494845, 24394.591319350566, 24433.467932624266])
Objectivefunctiononlineminibatchpchrank10=(Objectivefunctiononlineminibatchpchrank10run1+Objectivefunctiononlineminibatchpchrank10run2+Objectivefunctiononlineminibatchpchrank10run3)/3

PSNROnlineminibatchallrunsrank12=np.array([19.97360522759687,18.482903548317406,18.650344232291864])
Objectivefunctiononlineminibatchpchrank12run1=np.array([24438.891362124836, 24377.781360863632, 24366.502168551484, 24355.275713430805, 24401.496297882495])
Objectivefunctiononlineminibatchpchrank12run2=np.array([24373.847570669765, 24410.882025206367, 24350.408564384368, 24407.186324877825, 24343.942215406638])
Objectivefunctiononlineminibatchpchrank12run3=np.array([24418.687290523689, 24421.914497423626, 24394.484585195307, 24402.004008417727, 24373.550888220183])
Objectivefunctiononlineminibatchpchrank12=(Objectivefunctiononlineminibatchpchrank12run1+Objectivefunctiononlineminibatchpchrank12run2+Objectivefunctiononlineminibatchpchrank12run3)/3

PSNROnlineminibatchallrunsrank14=np.array([18.967019535418036,18.192076294257884,22.772167863902286])
Objectivefunctiononlineminibatchpchrank14run1=np.array([24416.357120360175, 24345.970017783373, 24412.613113250311, 24354.316184723411, 24401.910387737902])
Objectivefunctiononlineminibatchpchrank14run2=np.array([24490.088842770972, 24406.180647576588, 24344.040136258925, 24372.346358914405, 24322.396793643507])
Objectivefunctiononlineminibatchpchrank14run3=np.array([24444.721946251415, 24404.189555237805, 24376.043696941626, 24386.471099236856, 24383.003605454214])
Objectivefunctiononlineminibatchpchrank14=(Objectivefunctiononlineminibatchpchrank14run1+Objectivefunctiononlineminibatchpchrank14run2+Objectivefunctiononlineminibatchpchrank14run3)/3

PSNROnlineminibatchallrunsrank16=np.array([18.494040964494303,20.810569986843028,18.055904182357416])
Objectivefunctiononlineminibatchpchrank16run1=np.array([24377.938900914593, 24371.60909405458, 24333.007270669725, 24305.484423753493, 24325.778850156297])
Objectivefunctiononlineminibatchpchrank16run2=np.array([24365.199994750052, 24388.823983947859, 24344.043702913688, 24365.55325707922, 24353.270175738424])
Objectivefunctiononlineminibatchpchrank16run3=np.array([24410.3091167889, 24370.859001329638, 24328.475399576237, 24336.325718037107, 24321.444056078933])
Objectivefunctiononlineminibatchpchrank16=(Objectivefunctiononlineminibatchpchrank16run1+Objectivefunctiononlineminibatchpchrank16run2+Objectivefunctiononlineminibatchpchrank16run3)/3

PSNROnlineminibatchallrunsrank18=np.array([17.935721993947926,19.212374236034776,17.940970171110305])
Objectivefunctiononlineminibatchpchrank18run1=np.array([24393.306811079223, 24294.475595557065, 24293.329150766083, 24290.220641429445, 24326.925321317642])
Objectivefunctiononlineminibatchpchrank18run2=np.array([24330.207880759059, 24308.856633714044, 24297.228464680386, 24263.565326335385, 24385.287435295861])
Objectivefunctiononlineminibatchpchrank18run3=np.array([24384.548926560859, 24292.308243479376, 24282.601354216989, 24280.594344790039, 24287.765535434475])
Objectivefunctiononlineminibatchpchrank18=(Objectivefunctiononlineminibatchpchrank18run1+Objectivefunctiononlineminibatchpchrank18run2+Objectivefunctiononlineminibatchpchrank18run3)/3

PSNROnlineAltoallrunsrank10=np.array([18.874138838696123,18.102859069593787,19.647533605641328])
PSNROnlineAltoallrunsrank12=np.array([16.470285132568087,19.102927411598014,14.405312475917993])
PSNROnlineAltoallrunsrank14=np.array([19.341114308675756,17.178034890570419,17.354111179281752])
PSNROnlineAltoallrunsrank16=np.array([14.806883373041341,17.4394319105416,18.117101112388259])
PSNROnlineAltoallrunsrank18=np.array([18.530903891973242,15.441426120335759,17.348732176834044])


PSNRTuckerunconstrainedrunsrank10=np.array([17.090185983973338,17.090185983973338,17.090185983973338])
PSNRTuckerunconstrainedrunsrank12=np.array([17.090179029191866,17.090179029191866,17.090179029191866])
PSNRTuckerunconstrainedrunsrank14=np.array([17.090194060818455,17.090194060818455,17.090194060818455])
PSNRTuckerunconstrainedrunsrank16=np.array([17.090203735397232,17.090203735397232,17.090203735397232])
PSNRTuckerunconstrainedrunsrank18=np.array([17.090204941498165,17.090204941498165,17.090204941498165])

PSNROnlineminibatchunconstrainedrunsrank10=np.array([18.6373596379,18.007078527,18.0332586116])
PSNROnlineminibatchunconstrainedrunsrank12=np.array([18.436890502,18.0828835046,18.3854426015])
PSNROnlineminibatchunconstrainedrunsrank14=np.array([18.3058198966,18.1536866811,18.322044132])
PSNROnlineminibatchunconstrainedrunsrank16=np.array([18.0887269226,18.0887269226,19.1924785618])
Objectivefunctiononlineminibatchunconstrainedpchrank16run1=np.array([24467.244592412062, 24493.560073421475, 24479.639353686169, 24425.745323543611, 24429.842426693405])
Objectivefunctiononlineminibatchunconstrainedpchrank16run2=np.array([24349.688053922771, 24387.525392539326, 24414.665931312065, 24397.23630330649, 24372.069776609878])
Objectivefunctiononlineminibatchunconstrainedpchrank16run3=np.array([24470.084497795109, 24417.932237552413, 24401.044442751914, 24406.908339973244, 24377.513137456004])
Objectivefunctiononlineminibatchunconstrainedpchrank16=(Objectivefunctiononlineminibatchunconstrainedpchrank16run1+Objectivefunctiononlineminibatchunconstrainedpchrank16run2+Objectivefunctiononlineminibatchunconstrainedpchrank16run3)/3


PSNROnlineminibatchunconstrainedrunsrank18=np.array([20.1891659328,19.3997298709,18.3571382936])

PSNROnlinesingleunconstrainedrunsrank10=np.array([13.722613093256557,12.773514645124528,13.096176225721148])
PSNROnlinesingleunconstrainedrunsrank12=np.array([17.801420883200461,18.185213645399649,18.13363459752884])
PSNROnlinesingleunconstrainedrunsrank14=np.array([9.1331691126396422,8.7767144431689346,8.7151363770265728])

PSNROnlinesingleunconstrainedrunsrank16=np.array([19.748257583465058,21.527612641848638,19.960914611496047])
Objectivefunctiononlinesingleunconstrainedpochrank16run1=np.array([24468.690594278862, 24476.141535933079, 24455.170617822478, 24461.723007323297, 24440.22930284838])
Objectivefunctiononlinesingleunconstrainedpochrank16run2=np.array([24476.344493744189, 24450.979005509424, 24456.735037115035, 24450.899794476976, 24443.228735457276])
Objectivefunctiononlinesingleunconstrainedpochrank16run3=np.array([24480.340687945085, 24472.823054760815, 24460.191526466126, 24436.544031314243, 24448.103350691552])
Objectivefunctiononlinesingleunconstrainedpochrank16=(Objectivefunctiononlinesingleunconstrainedpochrank16run1+Objectivefunctiononlinesingleunconstrainedpochrank16run2+Objectivefunctiononlinesingleunconstrainedpochrank16run3)/3

PSNROnlinesingleunconstrainedrunsrank18=np.array([16.516296778500344,16.397611001724535,16.556428275046123])


#f = plt.figure()
#PSNROnlinesingle=np.array([np.mean(PSNROnlinesingleallrunsrank10),np.mean(PSNROnlinesingleallrunsrank12),np.mean(PSNROnlinesingleallrunsrank14),np.mean(PSNROnlinesingleallrunsrank16),np.mean(PSNROnlinesingleallrunsrank18)])
#PSNROnlineminibatch=np.array([np.mean(PSNROnlineminibatchallrunsrank10),np.mean(PSNROnlineminibatchallrunsrank12),np.mean(PSNROnlineminibatchallrunsrank14),np.mean(PSNROnlineminibatchallrunsrank16),np.mean(PSNROnlineminibatchallrunsrank18)])
#PSNROnlineAlto=np.array([np.mean(PSNROnlineAltoallrunsrank10),np.mean(PSNROnlineAltoallrunsrank12),np.mean(PSNROnlineAltoallrunsrank14),np.mean(PSNROnlineAltoallrunsrank16),np.mean(PSNROnlineAltoallrunsrank18)])
#PSNRTuckerunconstrained=np.array([np.mean(PSNRTuckerunconstrainedrunsrank10),np.mean(PSNRTuckerunconstrainedrunsrank12),np.mean(PSNRTuckerunconstrainedrunsrank14),np.mean(PSNRTuckerunconstrainedrunsrank16),np.mean(PSNRTuckerunconstrainedrunsrank18)])
#PSNROnlineminibatchunconstrained=np.array([np.mean(PSNROnlineminibatchunconstrainedrunsrank10),np.mean(PSNROnlineminibatchunconstrainedrunsrank12),np.mean(PSNROnlineminibatchunconstrainedrunsrank14),np.mean(PSNROnlineminibatchunconstrainedrunsrank16),np.mean(PSNROnlineminibatchunconstrainedrunsrank18)])
#PSNROnlinesingleunconstrained=np.array([np.mean(PSNROnlinesingleunconstrainedrunsrank10),np.mean(PSNROnlinesingleunconstrainedrunsrank12),np.mean(PSNROnlinesingleunconstrainedrunsrank14),np.mean(PSNROnlinesingleunconstrainedrunsrank16),np.mean(PSNROnlinesingleunconstrainedrunsrank18)])
#
#Rank=np.array([10,12,14,16,18])
#plt.plot(Rank,PSNROnlinesingle,label="OTLsingle-nonnegative",color='red',markersize=10,ls='--',marker='o')
#plt.plot(Rank,PSNROnlineminibatch,label="OTLminibatch-nonnegative",color='green',markersize=10,ls='--',marker='*')
#plt.plot(Rank,PSNROnlineAlto,label="Alto",color='blue',markersize=10,ls='--',marker='>')
#plt.plot(Rank,PSNRTuckerunconstrained,label="Tuckerbatch",color='maroon',markersize=10,ls='--',marker='<')
#plt.plot(Rank,PSNROnlineminibatchunconstrained,label="OTLminibatch-unconstrained",color='black',markersize=10,ls='--',marker='+')
#plt.plot(Rank,PSNROnlinesingleunconstrained,label="OTLsingle-unconstrained",color='chocolate',markersize=10,ls='--',marker='H')
#
#
#plt.legend()
#plt.xlabel("Rank")
#plt.ylabel("PSNR in dB")
#plt.show()
#f.savefig("PSNR.pdf")
#
#f = plt.figure()
#Epoch=np.array([1,2,3,4,5])
#plt.plot(Epoch,np.log(Objectivefunctiononlinesinglepochrank14)/np.log(10),label="OTLsingle-nonnegative",color='magenta',markersize=10,ls='--',marker='o')
#plt.plot(Epoch,np.log(Objectivefunctiononlineminibatchpchrank14)/np.log(10),label="OTLminibatch-nonnegative",color='yellow',markersize=10,ls='--',marker='*')
#plt.legend()
#plt.xlabel("Epoch")
#plt.ylabel("Objective function (dB)")
#plt.show()
#f.savefig("ObjectivefunctiononlineInpainting.pdf")

#f = plt.figure()
#Epoch=np.array([1,2,3,4,5])
#plt.plot(Epoch,np.log(Objectivefunctiononlinesingleunconstrainedpochrank16)/np.log(10),label="OTLsingle-unconstrained",color='maroon',markersize=10,ls='--',marker='o')
#plt.plot(Epoch,np.log(Objectivefunctiononlineminibatchunconstrainedpchrank16)/np.log(10),label="OTLminibatch-unconstrained",color='red',markersize=10,ls='--',marker='*')
#plt.legend()
#plt.xlabel("Epoch")
#plt.ylabel("Objective function (dB)")
#plt.show()
#f.savefig("ObjectivefunctiononlineInpaintingunconstrained.pdf")

PSNROnlinesingle=np.array([np.mean(PSNROnlinesingleallrunsrank10),np.mean(PSNROnlinesingleallrunsrank12),np.mean(PSNROnlinesingleallrunsrank14),np.mean(PSNROnlinesingleallrunsrank16),np.mean(PSNROnlinesingleallrunsrank18)])
PSNROnlineminibatch=np.array([np.mean(PSNROnlineminibatchallrunsrank10),np.mean(PSNROnlineminibatchallrunsrank12),np.mean(PSNROnlineminibatchallrunsrank14),np.mean(PSNROnlineminibatchallrunsrank16),np.mean(PSNROnlineminibatchallrunsrank18)])
PSNROnlineAlto=np.array([np.mean(PSNROnlineAltoallrunsrank10),np.mean(PSNROnlineAltoallrunsrank12),np.mean(PSNROnlineAltoallrunsrank14),np.mean(PSNROnlineAltoallrunsrank16),np.mean(PSNROnlineAltoallrunsrank18)])
PSNRTuckerunconstrained=np.array([np.mean(PSNRTuckerunconstrainedrunsrank10),np.mean(PSNRTuckerunconstrainedrunsrank12),np.mean(PSNRTuckerunconstrainedrunsrank14),np.mean(PSNRTuckerunconstrainedrunsrank16),np.mean(PSNRTuckerunconstrainedrunsrank18)])
PSNROnlineminibatchunconstrained=np.array([np.mean(PSNROnlineminibatchunconstrainedrunsrank10),np.mean(PSNROnlineminibatchunconstrainedrunsrank12),np.mean(PSNROnlineminibatchunconstrainedrunsrank14),np.mean(PSNROnlineminibatchunconstrainedrunsrank16),np.mean(PSNROnlineminibatchunconstrainedrunsrank18)])
PSNROnlinesingleunconstrained=np.array([np.mean(PSNROnlinesingleunconstrainedrunsrank10),np.mean(PSNROnlinesingleunconstrainedrunsrank12),np.mean(PSNROnlinesingleunconstrainedrunsrank14),np.mean(PSNROnlinesingleunconstrainedrunsrank16),np.mean(PSNROnlinesingleunconstrainedrunsrank18)])
#
#Data=np.zeros((5,6))
#Data[:,0]=PSNRTuckerunconstrained
#Data[:,1]=PSNROnlinesingleunconstrained 
#Data[:,2]=PSNROnlineminibatchunconstrained
#Data[:,3]=PSNROnlineAlto
#Data[:,4]=PSNROnlinesingle
#Data[:,5]=PSNROnlineminibatch

Data=np.zeros((5,4))
Data[:,0]=PSNRTuckerunconstrained
Data[:,1]=PSNROnlinesingleunconstrained 
Data[:,2]=PSNROnlineAlto
Data[:,3]=PSNROnlinesingle

#df2=pd.DataFrame(Data, columns=['TuckerBatch-unconstrained','OTLsingle-unconstrained','OTLminibatch-unconstrained','ALTO','OTLsingle-nonnegative','OTLminibatch-nonnegative'])

df2=pd.DataFrame(Data, columns=['TuckerBatch-unconstrained','OTLsingle-unconstrained','ALTO','OTLsingle-nonnegative'])


#ax=df2.plot(kind='bar',color=['blue','maroon','red','green','magenta','yellow'],rot=0)#,yerr=Std.T)#.bar()

ax=df2.plot(kind='bar',color=['blue','maroon' ,'green','magenta'],rot=0)#,yerr=Std.T)#.bar()


ax.set_xlabel("Rank R")
ax.set_xticklabels([i+1 for i in range(5)])
ax.set_ylabel("Root Mean Square Error",fontsize=10)
ax.set_xlabel("Rank R")
ax.set_xticklabels([10+2*i for i in range(5)])
ax.set_ylim(0,40)
ax.set_ylabel('PSNR (dB)',fontsize=10)

##Comparison of objective functions
Rank=16
numberofpoints=5
adress='/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/Inpaintingproblem/ImageInpainting/Objectivefunctions/TuckerObjectivefunction'+str(Rank)+'.npz'
loadedTuckerunconstrained=np.load(adress)
ObjectivefunctionTuckerunconstrained=loadedTuckerunconstrained['Objectivefunction']
TimeperepochTuckerunconstrained=loadedTuckerunconstrained['Timeperepoch']
ObjectivefunctionperepochTuckerunconstrained=loadedTuckerunconstrained['Objectivefnctionperepoch']
ObjectivefunctionTuckerunconstrainedsampled=Sample_points_all_lists(ObjectivefunctionTuckerunconstrained,numberofpoints)

adress='/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/Inpaintingproblem/ImageInpainting/Objectivefunctions/OnlinesingleObjectivefunctionunconstrained'+str(Rank)+'.npz'
loadedOnlinesingleunconstrained=np.load(adress)
ObjectivefunctionOnlineunconstrained=loadedOnlinesingleunconstrained['Objectivefunction']
TimeperepochOnlinesingleunconstrained=loadedOnlinesingleunconstrained['Timeperepoch']
ObjectivefunctionperepochOnlinesingleunconstrained=loadedOnlinesingleunconstrained['Objectivefunctionperepoch']
ObjectivefunctionOnlineunconstrainedsampled=Sample_points_all_lists(ObjectivefunctionOnlineunconstrained,numberofpoints)


adress='/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/Inpaintingproblem/ImageInpainting/Objectivefunctions/OnlineminibatchObjectivefunctionunconstrained'+str(Rank)+'.npz'
loadedOnlineminibatchunconstrained=np.load(adress)
ObjectivefunctionOnlinminibatchunconstrained=loadedOnlineminibatchunconstrained['Objectivefunction']
TimeperepochOnlineminibatchunconstrained=loadedOnlineminibatchunconstrained['Timeperepoch']
ObjectivefnctionperepochOnlineminibatchunconstrained=loadedOnlineminibatchunconstrained['Objectivefunctionperepoch']
ObjectivefunctionOnlinminibatchunconstrainedsampled=Sample_points_all_lists(ObjectivefunctionOnlinminibatchunconstrained,numberofpoints)


f=plt.figure()
#plt.plot(np.array([0,1,2,3,4,5]),np.log(ObjectivefunctionperepochTuckerunconstrained[0:]),label="TuckerBatch-unconstrained",color='blue',markersize=10,ls='--',marker='o')
#plt.plot(np.array([0,1,2,3,4,5]),np.log(ObjectivefunctionperepochOnlinesingleunconstrained),label="OTLsingle-unconstrained",color='maroon',markersize=10,ls='--',marker='*')
#plt.plot(np.array([0,1,2,3,4,5]),np.log(ObjectivefnctionperepochOnlineminibatchunconstrained),label="OTLminibatch-unconstrained",color='red',markersize=10,ls='--',marker='H')

#plt.plot(np.array([1,2,3,4,5]),np.log(ObjectivefunctionperepochOnlinesingleunconstrained[1:]),label="OTLsingle-unconstrained",color='maroon',markersize=10,ls='--',marker='*')
#plt.plot(np.array([1,2,3,4,5]),np.log(ObjectivefnctionperepochOnlineminibatchunconstrained[1:]),label="OTLminibatch-unconstrained",color='red',markersize=10,ls='--',marker='H')

plt.plot(np.array([1,2,3,4,5]),np.log(ObjectivefunctionOnlineunconstrained[:,0]),label="OTLsingle-nonnegative",color='maroon',markersize=20,ls='--',marker='*')

plt.xlabel("Epoch",fontsize=20)
plt.ylabel("Fitting error",fontsize=18)
plt.legend(fontsize=20)
plt.show()
f.savefig("ObjectivefunctionperepochOnlineVsMinibatch.pdf")


pdb.set_trace()




