import numpy as np
import matplotlib as mpl
from math import sqrt
import datetime as dt
import pandas as pd


### Set input, output, filename and number of individuals here ###

date_and_time = dt.datetime.now().strftime('%y-%m-%d-%H.%M')
indir = '/Volumes/Expansion/Data/2023/Sep28/data/'
outdir = '/Volumes/Expansion/Data/2023/Sep28/data/python/'
file = '20fish'
ind = 20

### Functions ###

def deleteInf(A,B,C):
    num = len(A) - 1
    i = 0
    infs=[]
    while i < num:
        if np.isinf(A[i]):
            infs.append(i)
        i += 1
    A=np.delete(A,infs)
    B=np.delete(B,infs)
    C=np.delete(C,infs)
    return A,B,C

def setTemp(A,b,c): #call with (X[i],xmin,xmax)
    temps_continuous=[]
    temps_discreet=[]
    tmin = 18
    tmax = 26
    for a in A:
        tempLRgradient = ((a-b)/(c-b))*(tmax-tmin) + tmin
        #tempRLgradient = tmax - ((a-b)/(c-b))*(tmax-tmin)
        temps_continuous.append(tempLRgradient)
        temps_discreet.append(round(tempLRgradient))
    return temps_continuous,temps_discreet

def pathFilter3(a,b,c,d):
    #a,b,c are the x,y,speed lists
    #d is fishID

    #create empty lists for reconstructed path
    A = []
    B = []
    C = []
    i=0
    l = 20
    #iterate through entire length of a (a,b,c have same length)
    while i<len(a)-l:
        #createSegment returns continuous path segment starting at i
        A_seg,B_seg,C_seg = createSegment(a,b,c,i)
        #segment is added to path if long enough
        if len(A_seg)>=l:
            A = A + A_seg
            B = B + B_seg
            C = C + C_seg
        i=i+len(A_seg)+1
    print('fish '+str(d)+'|','original: '+str(len(a)),'new: '+str(len(A)),'removed: '+str((len(a)-len(A))))
    return A,B,C
        
def createSegment(a,b,c,i):
    #a,b,c are x,y,speed lists

    #initialize segment with data point at i
    x,y,z=a[i],b[i],c[i]
    x2,y2=a[i+1],b[i+1]
    A_seg=[x]
    B_seg=[y]
    C_seg=[z]
    #r defines maximum distance between two points while still being continuous
    r=10
    while calcDistance(x,y,x2,y2) < r and i<len(a)-2:
        i=i+1
        x,y,z=a[i],b[i],c[i]
        x2,y2=a[i+1],b[i+1]
        A_seg.append(x)
        B_seg.append(y)
        C_seg.append(z)
    return A_seg,B_seg,C_seg
    
def calcDistance(x,y,x2,y2):
    return sqrt((x-x2)**2 + (y-y2)**2)

def average_by_sec(X,f):
    l=len(X[0])
    i = 1
    Y=[[],[],[],[],[]]
    hours = np.trunc(l/f)
    #print(hours)
    while i < hours:
        #print((i-1)*f)
        #print(i*f)
        I=0
        for x in X:
            Y[I].append(np.mean(x[(i-1)*f:i*f]))
            I+=1
        i+=1
    I=0
    for x in X: 
        Y[I].append(np.mean(x[i*f:(l-1)]))
        I+=1
    return Y[0],Y[1],Y[2],Y[3],Y[4]
    

X=[]
Y=[]
Z=[]
T=[]

for i in range(ind):
    
    with np.load(indir+file+"_fish"+str(i)+".npz") as npz:
        x = npz["X#wcentroid"]
        y = npz["Y#wcentroid"]
        z = npz["SPEED#wcentroid"]
               
# Delete type:inf rows        
        x,y,z = deleteInf(x,y,z)
                
#Limit data to specified length
        sub=5000
        #x = np.resize(x,(sub,))
        #y = np.resize(y,(sub,))
        #z = np.resize(z,(sub,))     
        
# Apply continuous path filter        
        x,y,z = pathFilter3(x,y,z,i)        
                
# Change frequency to one sample per second
        framerate = 30
        #x,y,z = x[::framerate],y[::framerate],z[::framerate]
                
# Add time variable        
        t = range(len(x))
        
        X.append(x)
        Y.append(y)
        Z.append(z)
        T.append(t)
        
### Set X and Y limits ###
            
        xmin = min(min(X,key=min))
        xmax = max(max(X,key=max))       
        ymin = min(min(Y,key=min))
        ymax = max(max(Y,key=max)) 
    

### Plot + Export data ###


df_all = pd.DataFrame()

for i in range(ind):
    
    # Plot fish paths
    
    mpl.pyplot.figure(i,figsize=(20,10))
    mpl.pyplot.axis([xmin,xmax,ymax,ymin])
    mpl.pyplot.plot(X[i],Y[i], c=mpl.color_sequences['tab20'][i], label='fish'+str(i))
    mpl.pyplot.legend(loc='upper left')
    
    
    # Assign Temperature and create CSV
    
    ctemp,dtemp = setTemp(X[i],xmin, xmax)
    
    #Take average per n seconds
    avg=""
    sec = 1000
    
    #X[i],Y[i],Z[i],dtemp,ctemp = average_by_sec([X[i],Y[i],Z[i],dtemp,ctemp],sec)
    #T[i] = range(len(X[i]))
    #print(len(X[i]),len(Y[i]),len(Z[i]),len(T[i]))
    #avg=" ("+str(sec)+"s_avg)"
    
    
    
    d = {'Time':T[i], 'x_centroid':X[i], 'y_centroid':Y[i],
         'Speed':Z[i],'Discreet_Temp':dtemp, 'Continuous_Temp':ctemp}
    
    df = pd.DataFrame(data=d)
    
    df.insert(0,'FishID',str(i))
    #df.to_csv(outDir + folder_name + '/fish#' + str(i) + ' ' + date_and_time + '.csv', index=False)
    
    #Store all fish data into a single .csv file
    
    df_all = pd.concat([df_all, df], ignore_index = True)
    

# Export All data as single file

df_all.to_csv(outdir +file+'_'+date_and_time+'.csv', index=False)
print("Exported all")

mpl.pyplot.show()
