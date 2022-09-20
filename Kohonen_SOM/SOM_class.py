import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self,dat,net_shape):
        self.dat=np.array(dat)
        self.M=np.shape(self.dat)[0]
        self.N=np.shape(self.dat)[1]
        self.net_shape=net_shape
        self.C=tuple(list(self.net_shape)+[self.N])     

    def gen_w(self,**kwargs):
        defaultKwargs = { 'seed': 0, 'low': 0.0, 'high' : 1.0 }
        kwargs = { **defaultKwargs, **kwargs }
        np.random.seed(kwargs['seed'])
        self.w=np.random.uniform(size=self.C, low=kwargs['low'], high=kwargs['high'])

    def train(self, **kwargs):
        defaultKwargs = { 'H': 0.2, 'S': 18, 'VH' : 0.993, 'VS': 0.960, 'UMAX': 10000, 'VMAX': 1000, 'HMIN' :  1/800.0}
        kwargs = { **defaultKwargs, **kwargs }

        self.H = kwargs['H']
        self.S = kwargs['S']
        self.VH = kwargs['VH']
        self.VS = kwargs['VS']
        
        self.UMAX = kwargs['UMAX']
        self.VMAX = kwargs['VMAX']
        self.HMIN = kwargs['HMIN']   
        
        h=self.H
        s=self.S

        for u in range(self.UMAX):
            
            p = self.dat[np.random.randint(self.M),:]      #random example
            self.w = self.hebb(p,h,s)                      #Hebb's rule

            #EXIT CLAUSES & DATA WRITING
            if(u % (self.UMAX/self.VMAX) == 0):
                h=self.VH*h            
                s=self.VS*s            
                if(h < self.HMIN):
                    print("step size is almost nough:",h)  ;
                    print("network did",u,"cycles")          ;
                    break                                      ;

            if(u==self.UMAX-1):
                print("no. of cycles reached",self.UMAX)    ;

    def plot_w(self,**kwargs):

        if(len(self.net_shape)==2):
            if(self.N==2):
                self.SOMplot2D()
            elif(self.N==3):
                self.SOMplot3D()
            else:
                print('No default plotting style')
        elif(len(self.net_shape)==1 and self.N==2):
            self.SOMplotLin()
        else:
            print('No default plotting style')

    def save_w(self,**kwargs):
        defaultKwargs = {'fname': "weight.npy"}
        kwargs = { **defaultKwargs, **kwargs }

        self.fname = kwargs['file_name']+".npy"
        f = open(self.fname, "wb")
        np.save(f, self.w)
        f.close()
        
    ##TRAINING FUNCTIONS    

    def win(self,p):                       #decides which neuron in the winner
        dmin = float("inf")    
        for index in np.ndindex(self.net_shape):
            d=sum((p-self.w[index])**2)    #2-distance
            if (d<dmin):
                dmin=d               
                winner=index                    
        return winner                        #returns the winner coordinates on the net

    def gauss(self,i1,i2,s):                #gaussian weighting function  
        i1=np.array(i1)
        i2=np.array(i2)
        return np.exp(-sum((i1-i2)**2)/(2*s**2))

    def hebb(self, p, h, s):
        iw=self.win(p)
        for index in np.ndindex(self.net_shape):
            self.w[index]=self.w[index]+h*self.gauss(iw,index,s)*(p-self.w[index])
        return self.w
    
    ## PLOTTING FUNCTIONS
    
    def connectpoints2D(self):
        for (i,j) in np.ndindex(self.net_shape):
            wxp= self.w[i,j,0] 
            wyp= self.w[i,j,1] 

            if(i <= self.net_shape[0]-2):
                wxf0 = self.w[i+1,j,0]
                wyf0 = self.w[i+1,j,1]
                plt.plot([wxp,wxf0],[wyp,wyf0],'r-')

            if(j <= self.net_shape[1]-2):
                wxf1 = self.w[i,j+1,0]
                wyf1 = self.w[i,j+1,1]
                plt.plot([wxp,wxf1],[wyp,wyf1],'r-')

    def connectpoints3D(self,ax):
        for (i,j) in np.ndindex(self.net_shape):
            wxp = self.w[i,j,0] 
            wyp = self.w[i,j,1] 
            wzp = self.w[i,j,2]

            if(i <= self.net_shape[0]-2):
                wxf0 = self.w[i+1,j,0]
                wyf0 = self.w[i+1,j,1]
                wzf0 = self.w[i+1,j,2]
                ax.plot3D([wxp,wxf0],[wyp,wyf0],[wzp,wzf0],'r-')

            if(j <= self.net_shape[1]-2):
                wxf1 = self.w[i,j+1,0]
                wyf1 = self.w[i,j+1,1]
                wzf1 = self.w[i,j+1,2]
                ax.plot3D([wxp,wxf1],[wyp,wyf1],[wzp,wzf1],'r-')

    def SOMplotLin(self):    
        plt.plot(self.dat[:,0],self.dat[:,1],'.')
        plt.plot(self.w[:,0],self.w[:,1],'or-')
        plt.suptitle(r'Kohonen SOM', fontsize=14)
        plt.show()

    def SOMplot2D(self):    
        plt.plot(self.dat[:,0],self.dat[:,1],'.')
        plt.plot(self.w[:,:,0],self.w[:,:,1],'or')
        self.connectpoints2D()
        plt.suptitle(r'Kohonen SOM', fontsize=14)
        plt.show()

    def SOMplot3D(self):
        # Creating figure
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        lw = np.reshape(self.w,(self.net_shape[0]*self.net_shape[1],3))
        self.connectpoints3D(ax)
        ax.plot3D(self.dat[:,0],self.dat[:,1],self.dat[:,2],'.')
        ax.plot3D(lw[:,0],lw[:,1],lw[:,2],'or')
        plt.suptitle(r'Kohonen SOM', fontsize=14)
        plt.show()
