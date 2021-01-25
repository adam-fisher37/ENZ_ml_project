#Interpolation of refractiveindex.info n,k data for various dielectrics
#also some oscillator models for dielectrics/glass
#arl92@case.edu

from scipy.interpolate import CubicSpline
from numpy import max,min
import numpy as np
import tables

#UPDATE THIS WHEN CHANGING CODE!!!
def version():
    print('20200623 21:30:00')
 

    
#################################
#special functions
#################################

def nk2eps(n):
    er = np.real(n)**2-np.imag(n)**2
    ei = 2*np.real(n)*np.imag(n)
    return (er+1j*ei)

def nk_birefringent(theta,no,ne):
    neff = np.divide(np.multiply(ne,no),np.sqrt((np.square(ne)*np.square(np.cos(theta)))+(np.square(no)*np.square(np.sin(theta)))))
    return neff

def eps2nk(eps):
    n = np.sqrt(0.5*(np.sqrt(np.real(eps)**2+np.imag(eps)**2)+np.real(eps)))
    k = np.sqrt(0.5*(np.sqrt(np.real(eps)**2+np.imag(eps)**2)-np.real(eps)))
    return (n+1j*k)


#################################
#interpolation materials
#################################


def nk_material(mat,wave):
    #Get ref info data from refractiveindex.info, retreived 20191017
    filename = 'dielectric_nk_data.h5'
    f = tables.open_file(filename, mode='r')
    table = f.get_node('/g/'+mat)
    w = [x['w'] for x in table.iterrows()]
    n = [x['n'] for x in table.iterrows()]
    k = [x['k'] for x in table.iterrows()]
    f.close()
    if min(wave) < min(w) or max(wave) > max(w):
        print('Allowed wavelength range: %1.2e - %1.2e'%(min(w),max(w)))
        print('Please modify wavelength input.')
        return
    ncs = CubicSpline(w,n,bc_type='not-a-knot')
    kcs = CubicSpline(w,k,bc_type='not-a-knot')
    return (ncs(wave)+1j*kcs(wave))



#################################
#Oscillator Models
#################################

#Lorentz : https://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf  
#parameters designed to work with meep, I don't know why sigma...
def eps_Lorentz(wave,eps,w1,sig,gamma):
    eps = np.array(eps)
    w1 = np.array(w1)
    sig = np.array(sig)
    gamma = np.array(gamma)
    
    nosc = w1.size
    n = np.zeros(wave.size,dtype=complex)
    if np.any(np.array([w1.size,sig.size,gamma.size])!=nosc):
        print('Inconsistent Oscillator number!')
        return n
    freq = np.divide(2*np.pi,wave)
    if nosc > 1:
        osum = np.zeros(wave.size,dtype=complex)
        for i in range(nosc):
            osum += np.divide(w1[i]**2*sig[i],(w1[i]**2-freq**2-(1j*freq*gamma[i]*0.5/(np.pi))))
        n = eps+osum
    else:
        n = eps+np.divide(w1**2*sig,(w1**2-freq**2-(1j*freq*gamma*0.5/(np.pi))))
            
    return n

def nk_Lorentz(waverange,eps,w1,sig,gamma):
    eps = eps_Lorentz(waverange,eps,w1,sig,gamma)
    nk = eps2nk(eps)
    return nk
    
    
    

#Wemple DiDiminico
def nk_WDD(waverange,e0,ed):
    pen = np.divide(1239.84193E-9,waverange)
    wd = np.divide((e0*ed),np.subtract(e0**2,np.power(np.multiply(1,pen),2)))
    n = np.sqrt(wd+1)
    k = np.zeros(pen.size)
    return (n+1j*k)

def eps_WDD(waverange,e0,ed):
    n = nk_WDD(waverange,e0,ed)
    eps = nk2eps(n)
    return eps



#Cauchy-Urbach
def nk_Cauchy_Urbach(waverange,a,b,c=0,α=0,β=0,γ=1):
    n = a+(b*1E-12/np.power(waverange,2))+(c/np.power(waverange,4))
    if γ <= 0:
        γ = 1
    k = α*np.exp(β*12400*(np.divide(1,waverange)-(1/(γ*1E-9))))
    return (n+1j*k)

def eps_Cauchy_Urbach(waverange,a,b,c=0,α=0,β=0,γ=1):
    n = nk_Cauchy_Urbach(waverange,a,b,c,α,β,γ)
    eps = nk2eps(n)
    return eps



#Sellmier oscillator
def nk_Sellmeier(waverange,a,b):    
    nosc = a.size
    n = np.zeros(waverange.size,dtype=complex)
    if b.size != nosc:
        print('Inconsistent Oscillator number!')
        return n
    
    osum = np.zeros(waverange.size,dtype=complex)
    for i in range(nosc):
        osum += np.divide(np.multiply(a[i],waverange**2),(waverange**2 - b[i]))
    n = np.sqrt(1+osum)
    return n
    
def eps_Sellmeier(waverange,a,b):
    n = nk_Sellmeier(waverange,a,b)
    eps = nk2eps(n)
    return eps


#tauc-lorentz oscillator model
def eps_Tauc_Lorentz(waverange,eg,e1inf,a,e0,c):
    pen = np.divide(1239.84193E-9,waverange)
    eps2 = np.zeros(pen.size,dtype=complex)
    eps1 = np.zeros(pen.size,dtype=complex)
    alpha = np.sqrt((4*e0**2)-c**2)
    gamma = np.sqrt(e0**2-((c**2)/2))
    for i in range(pen.size):
        e = pen[i]
        if pen[i]>eg:
            eps2[i] = np.divide((a*e0*c*(e-eg)**2),e*((e**2-e0**2)**2+(c**2*e**2)))
        else:
            eps2[i] = 0

        aln = ((eg**2-e0**2)*(e**2))+((eg**2)*(c**2))-((e0**2)*(e0**2+(3*eg**2)))
        aatan = ((e**2-e0**2)*(e0**2+eg**2))+((eg**2)*(c**2))
        z4 = (e**2-gamma**2)**2+((alpha**2*c**2)/4)
        term1=(( a*c*aln*np.log( (e0**2+eg**2+(alpha*eg)) / (e0**2+eg**2-(alpha*eg)) ) )/(2*np.pi*z4*alpha*e0))
        term2=((a*aatan*(np.pi-np.arctan(((2*eg)+alpha)/c)+np.arctan(((-2*eg)+alpha)/c)))/(np.pi*z4*e0))
        term3=(((4*a*e0*eg*(e**2-gamma**2))*(np.arctan((alpha+2*eg)/(c))+np.arctan((alpha-2*eg)/(c))))/(np.pi*z4*alpha))
        term4=((a*e0*c*(e**2+eg**2)*np.log(np.abs(e-eg)/(e+eg)))/(np.pi*z4*e))
        term5=((2*a*e0*c*eg*np.log((np.abs(e-eg)*(e+eg))/(np.sqrt((e0**2-eg**2)**2+(eg**2*c**2)))))/(np.pi*z4))
        eps1[i] = e1inf + term1 - term2 + term3 - term4 + term5
    
    return (eps1+1j*eps2)

def nk_Tauc_Lorentz(waverange,eg,e1inf,a,e0,c):
    eps = eps_Tauc_Lorentz(waverange,eg,e1inf,a,e0,c)
    n = eps2nk(eps)
    return n



#cody-lorentz oscillator model **********NOT WORKING JUST COPY FROM T_L*******
def eps_Cody_Lorentz(waverange,eg,e1inf,a,e0,c):
    pen = np.divide(1239.84193E-9,waverange)
    eps2 = np.zeros(pen.size,dtype=complex)
    eps1 = np.zeros(pen.size,dtype=complex)

    for i in range(pen.size):
        e = pen[i]
        
    
    return (eps1+1j*eps2)

def nk_Cody_Lorentz(waverange,eg,e1inf,a,e0,c):
    eps = eps_Cody_Lorentz(waverange,eg,e1inf,a,e0,c)
    n = eps2nk(eps)
    return n


#################################
#special methods
#################################


#EMA Approx for two materials    
def eps_EMA(eps_mb,eps_m2,frac_m2,method = 'bruggeman'):
    #eps_mb  = cvec base material
    #eps_m2  = cvec dilute material
    #frac_m2 = frac of material 2 in eff. medium. range(0 <--> 1)
    if eps_mb.size != eps_m2.size:
        print('Resizing epsilon_mat2 for EMA')
        eps_m2 = np.linspace(eps_m2[0],eps_m2[-1],esp_mb.size)
        
    eps = np.zeros(eps_mb.size,dtype=complex)
    
    if method == 'bruggeman':
        eps1 = 0.25*(-eps_m2+2*eps_mb+3*eps_m2*frac_m2-3*eps_mb*frac_m2 - np.sqrt(8*np.multiply(eps_m2,eps_mb)+(-eps_m2+2*eps_mb+3*eps_m2*frac_m2 - 3*eps_mb*frac_m2)**2))
        eps2 = 0.25*(-eps_m2+2*eps_mb+3*eps_m2*frac_m2-3*eps_mb*frac_m2 + np.sqrt(8*np.multiply(eps_m2,eps_mb)+(-eps_m2+2*eps_mb+3*eps_m2*frac_m2 - 3*eps_mb*frac_m2)**2))
        #sometimes eps1 will be physical vs eps2
        return eps2
        
    elif method == 'maxwell-garnett':
        if np.any(np.abs((2+frac_m2)*eps_mb + (1-frac_m2)*eps_m2)> 1E-7):
            eps = eps_mb*((2-2*frac_m2)*eps_mb + (1+2*frac_m2)*eps_m2)/((2+frac_m2)*eps_mb + (1-frac_m2)*eps_m2)
        else:
            print('Singularity in M-G EMA')
        return eps
    
    elif method == 'looyenga':
        eps = np.power(frac_m2*eps_m2**(1/3)+(1-frac_m2)*eps_mb**(1/3),3)
        return eps
    
    
    elif method == 'linear':
        eps = (frac_m2*eps_m2)+((1-frac_m2)*eps_mb)
        return eps
    
    else:
        print('Method not available.')
        return eps
   
    
def eps_srough(eps_mb):
    eps_m2 = 8.85E-12*np.ones(eps_mb.size)
    eps = eps_EMA(eps_mb,eps_m2,0.5,'bruggeman')
    return eps

def nk_EMA(n_mb,n_m2,frac_m2,method = 'bruggeman'):
    eps_mb = nk2eps(n_mb)
    eps_m2 = nk2eps(n_m2)
    eps = eps_EMA(eps_mb,eps_m2,frac_m2,method)
    n = eps2nk(eps)
    return n
    
#surface roughness on the surface material
def nk_srough(n_mb):
    n_m2 = np.ones(n_mb.size)
    n = nk_EMA(n_mb,n_m2,0.5,'bruggeman')
    return n
    

#EMA for uniaxial symmetric metal nanowire array embedded in dielectric
#Shekhar et al. Nano Convergence 2014, 1:14
def eps_nanowire(rho,ang,eps_rod,eps_host):
    #rho : fill fraction of nanowire (nanowire area / unit cell area)
    eps_para = np.divide((((1+rho)*np.multiply(eps_rod,eps_host))+((1-rho)*np.square(eps_host))),(((1+rho)*eps_host)+((1-rho)*eps_rod)))         
    eps_perp = (rho*eps_rod)+((1-rho)*eps_host)
    nk_para = eps2nk(eps_para)
    nk_perp = eps2nk(eps_perp)
    nk_eff = nk_birefringent(ang,nk_para,nk_perp)
    eps_eff = nk2eps(nk_eff)
    return eps_eff

def nk_nanowire(rho,ang,nk_rod,nk_host):
    eps_rod = nk2eps(nk_rod)
    eps_host = nk2eps(nk_host)
    eps_nr = eps_nanowire(rho,ang,eps_rod,eps_host)
    nk_nr = eps2nk(eps_nr)
    return nk_nr
    
    
    
#List the available materials
def materials():
    print('-'*50)
    print('Interpolation Materials: (mat,waverange)')
    print('-'*50)
    mats = ['azo','ito','al2o3','sio2','tio2']
    for i in mats:
        filename = 'dielectric_nk_data.h5'
        f = tables.open_file(filename, mode='r')
        table = f.get_node('/g/'+i)
        w = [x['w'] for x in table.iterrows()]
        f.close()
        print('%6s : allowed wavelength range:[%1.2e - %1.2e]'%(i,min(w),max(w)))
    print(' ')
    print('-'*50)
    print('Oscillator Models:')
    print('-'*50)
    print('Cauchy-Urbach:     (waverange,a,b,c=0,α=0,β=0,γ=1)')
    print('Sellmeier:         (waverange,[a],[b])')
    print('Tauc-Lorentz:      (waverange,eg,e1inf,a,e0,c)')
    print('Wemple-DiDominico: (waverange,e0,ed)')
    print(' ')
    print('-'*50)
    print('Others:')
    print('-'*50)
    print('EMA:               (mat,n_m2,frac_m2,method = \'bruggeman\')\n    bruggeman\n    maxwell-garnett\n    looyenga\n    linear')
    print('Surface Roughness: (mat)')
    print('Nanowire:          (fillFract,ang,mat_rod,mat_host)')

    