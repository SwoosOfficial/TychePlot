import numpy as np
from Plot import Plot

class OLEDSimPlot(Plot):

    

    e=1.602*10**-19 #As
    eps_0=8.85*10**-12 #As/Vm
    eps_r=3.5
    carrierType=1
    offsetCathode=1 #1=true -1=false
    V=0
    E=V/(200*10**-9)#V/m; 200nm organics
    wf_m=-4.2*e #J
    wf_o=-2.8*e #J
    Delta=wf_m-wf_o
    initpos=0
    x_scale=10**9
    x_min=-7
    metalContactAnode=False
    metalContactCathode=False
    
    @classmethod
    def points(cls,x,dim):
        return int(x//dim)

    @classmethod
    def V_0(cls, x, wf, b):
        if b*cls.offsetCathode==-3 or abs(b)==2:
            return [cls.carrierType*wf if a>=0 else 0.0 for a in x]
        if b*cls.offsetCathode==3:
            return [cls.carrierType*wf+cls.V*cls.e if a>=0 else 0.0 for a in x]

    @classmethod
    def V_1(cls, x, wf, points,sigma=0.1*e, mu=0):
        z=np.random.normal(mu, sigma, points)
        return [cls.carrierType*wf+c if a>=0 else 0.0 for a,c in zip(x,z)]

    @classmethod
    def V_2(cls, x, b):
        if b==0:
            return 0
        if b==-1:
            return [cls.carrierType*(cls.e**2)/(16*np.pi*cls.eps_0*cls.eps_r*(a-x[0])) if a>=0 else 0.0 for a in x]
        if b==1:
            return [cls.carrierType*(cls.e**2)/(16*np.pi*cls.eps_0*cls.eps_r*(a-x[len(x)-1])) if a>=0 else 0.0 for a in x]

    @classmethod
    def V_3(cls, x, offset, E=None):
        if E is None:
            E=OLEDSimPlot.E
        return [cls.carrierType*cls.e*E*(a-offset) if a>=0 else 0.0 for a in x]

    
    def __init__(self, name, materials,
                 showColAxType=["lin","lin","lin"],
                 showColAxLim=[None,None,None],
                 showColLabel= ["","Depth","Potential"],
                 showColLabelUnit=["",
                  "Depth (nm)",
                  "Potential (eV)"               
                 ],
                 **kwargs):
        Plot.__init__(self, name,[], dataImported=True, overrideFileList=True, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit,  **kwargs)
        self.materials=materials
        self.stack=Stack(self.materials)
        
        
    def doPlot(self):
        return self.stack.plotStack(self)

class Material:
    iD=0
    def __init__(self, thickness, CBLike, name="Unknown Material", x=0, y=0, vacY=0, dim=10**-10, metallic=False, height=1, outsourceDesc=None):
        self.name=name
        self.thickness = thickness
        self.CBLike = CBLike
        self.x = x
        self.dim = dim
        self.y = y
        self.vacY=vacY
        self.metallic=metallic
        self.height=height
        self.outsourceDesc=outsourceDesc
        self.id = Material.iD+1
        Material.iD=Material.iD+1
    def __repr__(self):
        return "{}: ({:3.0f} nm)".format(self.name,self.thickness*10**9)
    def __str__(self):
        return self.name
    def name():
        return self.name
    


class Organic(Material):
    def __init__(self, thickness, CBLike, VLLike, name="Unknown Material", x=0, y=0, VLDevia=0.05, CBDevia=0.05, eps_r=3.5, metallic=False, y_2=0, sigma=0.001, polarity=0, **kwargs):
        Material.__init__(self, thickness, CBLike, name=name, x=x, y=y, metallic=False, **kwargs)
        self.VLLike = VLLike
        self.VLDevia=VLDevia
        self.CBDevia=CBDevia
        self.eps_r=eps_r
        self.y_2= y_2
        self.sigma= sigma
        self.polarity=polarity
    


class DopedOrganic(Organic):
    def __init__(self, thickness, CBLike, VLLike, CB2Like, VL2Like, name2="UnknownMaterial", y_3=0, y_4=0, **kwargs):
        Organic.__init__(self, thickness, CBLike, VLLike, **kwargs)
        self.CB2Like= CB2Like #Host
        self.VL2Like=VL2Like #Host
        self.y_3=y_3
        self.y_4=y_4
        self.name2=name2
        self.id2=Material.iD+1
        Material.iD=Material.iD+1


class Metal(Material):
    def __init__(self, thickness, CBLike, name="Unknown Material", x=0, y=0, **kwargs):
        Material.__init__(self, thickness, CBLike, name=name, x=x, y=y, metallic=True, **kwargs)


class Stack:
    Cz=['#1f77b4','#2ca02c','#17becf','#f8e520','#d62728','#ff7f0e','#bcbd22','#9467bd','#8c564b','#e377c2','#7f7f7f']
    C=["red","blue","green","cyan","yellow","magenta"]
    def __init__(self, Materials):
        self.A=[m.metallic for m in Materials]
        self.hil=self.A.index(False)
        self.eil=len(self.A) - self.A[::-1].index(False) - 1
        self.B=[-3]+[-2]*(self.hil-1)+[-1]+(self.eil-self.hil-1)*[0]+[1]+(len(self.A)-2-self.eil)*[2]+[3]
        self.Materials=Materials
        
    
    def plotStack(self, plot):
        ax = plot._newFig()[1]
        import matplotlib.pyplot as plt
        ax.set_xlabel(plot.showColLabelUnit[plot.xCol])
        ax.set_ylabel(plot.showColLabelUnit[plot.showCol])
        curPos=plot.initpos
        e=plot.e
        for m in self.Materials:
            nexPos= curPos+m.thickness
            m.x=np.linspace(curPos,nexPos,plot.points(m.thickness,m.dim))
            curPos= nexPos
        for m,b in zip(self.Materials, self.B):
            if b*plot.offsetCathode==3:
                a=m.CBLike
            if b*-plot.offsetCathode==3:
                c=m.CBLike
            if not m.metallic:
                m.y=np.asarray(plot.V_1(m.x,m.CBLike,plot.points(m.thickness,m.dim),sigma=m.sigma))+np.asarray(plot.V_2(m.x, b))+np.asarray(plot.V_3(m.x, m.x[0]))+np.asarray(plot.V_3(m.x,m.x[0], E=m.polarity))
                m.y_2=np.asarray(plot.V_1(m.x,m.VLLike,plot.points(m.thickness,m.dim),sigma=m.sigma))+np.asarray(plot.V_2(m.x, b))+np.asarray(plot.V_3(m.x, m.x[0]))+np.asarray(plot.V_3(m.x,m.x[0], E=m.polarity))
                m.vacY=np.asarray(plot.V_2(m.x, b))+np.asarray(plot.V_3(m.x, m.x[0]))+np.asarray(plot.V_3(m.x,m.x[0], E=m.polarity))
                ax.plot(np.asarray(m.x)*plot.x_scale,m.y*e**-1, color=Stack.Cz[m.id], label=m.name)
                ax.plot(np.asarray(m.x)*plot.x_scale,m.y_2*e**-1, color=Stack.Cz[m.id])
                ax.plot(np.asarray(m.x)*plot.x_scale,m.vacY*e**-1, "k-")
                if type(m) is DopedOrganic:
                    m.y_3=np.asarray(plot.V_1(m.x,m.CB2Like,plot.points(m.thickness,m.dim),sigma=m.sigma))+np.asarray(plot.V_2(m.x, b))+np.asarray(plot.V_3(m.x, m.x[0]))+np.asarray(plot.V_3(m.x,m.x[0], E=m.polarity))
                    m.y_4=np.asarray(plot.V_1(m.x,m.VL2Like,plot.points(m.thickness,m.dim),sigma=m.sigma))+np.asarray(plot.V_2(m.x, b))+np.asarray(plot.V_3(m.x, m.x[0]))+np.asarray(plot.V_3(m.x,m.x[0], E=m.polarity))
                    ax.plot(np.asarray(m.x)*plot.x_scale,m.y_3*e**-1, color=Stack.Cz[m.id2], label=m.name2)
                    ax.plot(np.asarray(m.x)*plot.x_scale,m.y_4*e**-1, color=Stack.Cz[m.id2])
                    if m.outsourceDesc is None:
                        ax.text(m.x[(len(m.x)-1)//2]*plot.x_scale,(m.y_4[(len(m.x)-1)//2]+m.y_3[(len(m.x)-1)//2])/2*e**-1*m.height**-1,m.name+":"+m.name2, ha="center", va="center")
                    else:
                        ax.annotate(m.name+":"+m.name2,xy=(m.x[(len(m.x)-1)//2]*plot.x_scale,(m.y_4[(len(m.x)-1)//2]+m.y_3[(len(m.x)-1)//2])/2*e**-1*m.height**-1),xytext=(m.x[(len(m.x)-1)//2]*plot.x_scale,m.outsourceDesc), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), ha="center", va="center", )
                    ax.fill_between(np.asarray(m.x)*plot.x_scale,m.y_3*e**-1,m.y_4*e**-1, facecolor=Stack.Cz[m.id2],interpolate=False, alpha=0.1)
                    ax.fill_between(np.asarray(m.x)*plot.x_scale,m.y_2*e**-1,m.y_4*e**-1, facecolor=Stack.Cz[m.id],interpolate=False, alpha=0.1)
                    ax.fill_between(np.asarray(m.x)*plot.x_scale,m.y*e**-1,m.y_3*e**-1, facecolor=Stack.Cz[m.id],interpolate=False, alpha=0.1)
                else:
                    ax.text(m.x[(len(m.x)-1)//2]*plot.x_scale,(m.y_2[(len(m.x)-1)//2]*m.height+m.y[(len(m.x)-1)//2])/2*e**-1*m.height**-1,m.name, ha="center", va="center")
                    ax.fill_between(np.asarray(m.x)*plot.x_scale,m.y*e**-1,m.y_2*e**-1, facecolor=Stack.Cz[m.id],interpolate=False, alpha=0.1)
        for m,b in zip(self.Materials, self.B):
            if m.metallic:
                if b==-2 and plot.metalContactAnode:
                    m.y=plot.V_0(m.x,c, 3*-plot.offsetCathode)
                    m.vacY=[d-m.CBLike for d in m.y]
                    ax.plot(np.asarray(m.x)*plot.x_scale,np.asarray(m.y)*e**-1, color=Stack.C[b-1], label=m.name)
                    ax.fill_between(np.asarray(m.x)*plot.x_scale,x_min,np.asarray(m.y)*e**-1, interpolate=True, facecolor=Stack.C[b-1], alpha=0.3)
                    ax.text(m.x[(len(m.x)-1)//2]*plot.x_scale,(m.y[0]+plot.x_min*e)/2*e**-1*m.height,m.name, ha="center", va="center")
                    ax.plot(np.asarray(m.x)*plot.x_scale,np.asarray(m.vacY)*e**-1, "k-",label="Vacuum")
                elif b==2 and plot.metalContactCathode:
                    m.y=plot.V_0(m.x,a, 3*plot.offsetCathode)
                    m.vacY=[d-m.CBLike for d in m.y]
                    ax.plot(np.asarray(m.x)*plot.x_scale,np.asarray(m.y)*e**-1, color=Stack.C[b-1], label=m.name)
                    ax.fill_between(np.asarray(m.x)*plot.x_scale,plot.x_min,np.asarray(m.y)*e**-1, interpolate=True, facecolor=Stack.C[b-1], alpha=0.3)
                    ax.text(m.x[(len(m.x)-1)//2]*plot.x_scale,(m.y[0]+plot.x_min*e)/2*e**-1*m.height,m.name, ha="center", va="center")
                    ax.plot(np.asarray(m.x)*plot.x_scale,np.asarray(m.vacY)*e**-1, "k-",label="Vacuum")
                else:
                    m.y=plot.V_0(m.x,m.CBLike, b)
                    m.vacY=plot.V_0(m.x,0,b)
                    ax.plot(np.asarray(m.x)*plot.x_scale,np.asarray(m.y)*e**-1, color=Stack.C[b-1], label=m.name)
                    ax.fill_between(np.asarray(m.x)*plot.x_scale,plot.x_min,np.asarray(m.y)*plot.e**-1,interpolate=True, facecolor=Stack.C[b-1], alpha=0.3)
                    ax.text(m.x[(len(m.x)-1)//2]*plot.x_scale,(m.y[0]+plot.x_min*plot.e)/2*plot.e**-1*m.height,m.name, ha="center", va="center")
                    ax.plot(np.asarray(m.x)*plot.x_scale,np.asarray(m.vacY)*plot.e**-1, "k-")
        if plot.titleBool:
            ax.set_title(plot.title, fontsize=plot.titleFontSize)
        ax.set_ylim(plot.x_min, plot.V+1)
        ax.grid(which="major",ls=":",alpha=0.25, axis="y")
        plt.tight_layout()
        plt.savefig(plot.name+".pdf")
        plt.savefig(plot.name+".pgf")
        return [plot,plot.name+".pdf"] #filename
