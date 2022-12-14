# Lars Zwaan, 12414069
# Dirk Kuiper, #
# Extreme Astrophyics final project
# plotting the spectrum of J1820, consising of self-absorbed synchrotron
# and SSC (inverse Compton upscattering of local synchrotron photons) emission

# imports used in original jet slices code
import math
import matplotlib.pyplot as plt
import numpy as np
import math

# imports used in IC tutorial
import scipy.integrate as integrate
from scipy.integrate import quad

# leave out sys before handing in
# used to exit before outputting rest of code, eg in testing
from sys import exit

# a bunch of constants needed during the course, in CGS
c = 2.998 * 10**10
b = 0.2898
k_B = 1.4 * 10**(-16)
h = 6.626 * 10**(-27)
wien_displacementlaw_freq_const = 5.8789 * 10**10
m_e = 10**(-27)
M_sun = 2 * 10**33
G = 7 * 10**(-8)
sigma_T = 7 * 10**(-25)
m_p = 1.67 * 10**(-24)
e = 5 * 10**(-10)
MsgrA = 4.1 * 10**6 * M_sun
DsgrA = 2.425 * 10**2

# SECTION: SUPPORTING FUNCTIONS
# these may or may not be used during the rest of the code

# volume of a sphere for a radius
def vol_sphere(R):
    vol = (4/3) * math.pi * R**3
    return vol

# gravitational radius for a mass
def Rg(M):
    Rg = 2 * G * M / c**2
    return Rg

# eddington luminosity for a mass
def L_edd(M):
    L_edd = 4*math.pi * G * M * m_p * c / sigma_T
    return L_edd

# the radius of the blob in this problem
def blob_radius(R0, v, t):
    R = R0*(1+v*t)
    return R

# calculate velocity from given Gamma factor
def gamma_to_velo(gamma):
    beta = (-(gamma**(-2) - 1))**2
    v = beta**(1/2) * c
    return v

# extinction coefficient used to calculate source function (R&L 6.53)
def extinction_coeff(q, m, const_C, B, pitch_angle, p, nu):
    alpha_nu = ((3**(1/2) * q**3) / (8*math.pi * m)) * (3*q / (2*math.pi* m**3 * c**5))**(p/2)
    alpha_nu *= const_C * (B*math.sin(pitch_angle))**((p+2)/2) * nu**(-(p+4)/2)
    alpha_nu *= math.gamma((3*p + 2) / 12) * math.gamma((3*p + 22) / 12)
    return alpha_nu

# total power used to calculate source function (R&L 6.36)
def power_nu(q, m, const_C, B, pitch_angle, p, nu):
    Pnu = (3**(1/2) * q**3 * const_C * B * math.sin(pitch_angle))
    Pnu *= (2*math.pi * m * c**2 * (p+1))**(-1)
    Pnu *= ((m*c*2*math.pi*nu) / (3*q*B*math.sin(pitch_angle)))**(-(p-1)/2)
    Pnu *= math.gamma((p/4) + 19/12) * math.gamma((p/4) - 1/12)
    return Pnu

# source function using ex_coeff and power (R&L 6.54)
def source_func(q, m, const_C, B, pitch_angle, p, nu):
    S_nu = power_nu(q, m, const_C, B, pitch_angle, p, nu)
    S_nu *= (4*math.pi * extinction_coeff(q, m, const_C, B, pitch_angle, p, nu))**(-1)
    return S_nu

# intensity from the homogeneous slab solution
def intensity(q, m, const_C, B, pitch_angle, p, nu, R_t):
    tau_nu = extinction_coeff(q, m, const_C, B, pitch_angle, p, nu) * R_t
    I = source_func(q, m, const_C, B, pitch_angle, p, nu) * (1 - math.exp(-tau_nu))
    return I

# SECTION: CONICAL JET
# creating the setup that is needed for the conical jet

# PROJECT DESCRIPTION:
# (later maybe move this to top of code but easier here now)

# Assume a conical jet divided into ~10-20 slices (or more if your code runs fast)
# Remember the jet is huge so divide into equal widths in log10(z) space!
# Assume plasma moving with constant Lorentz factor (play with values from gamma~1 to ~4 to see what works best),
# containing a power-law distribution of electrons in equipartition with the 
# internal magnetic energy density to start, later you can also play with this
# ratio (effectively changing the plasma beta).  For each slice of the jet calculate
# self-absorbed synchrotron and SSC (inverse Compton upscattering of local
# synchrotron photons) emission and add up into your final spectrum. The jet will
# have low optical depth, typically there will be < few scatters depending on your
# Compton Y, but play with different values to get best "match" to data.
# If you want a challenge, vary the input power to study the range seen over
# an outburst cycle (from very low tau to tau~1) , and make a plot of the resulting
# radio/X-ray luminosity correlation.

# START IC PART (PS4)

# note: parts unnecessary for the final N vs hv plot have been deleted
# constant names have been made consistent
# some comments from the original notebook have been deleted

def compton_y(pre,post):
    return(np.mean((post-pre)/pre))

def random_direction(number=None):
    """Returns randomly oriented unit vectors.

    Args: 
        None
        
    Parameters:
        number: number of random vectors to return

    Returns::
        (number,3)-element numpy array: Randomly oriented unit vectors
    """

    #
    # This is how you draw a random number for a uniform
    # distribution:
    #

    if number is None:
        number=1

    phi=2.*np.pi*np.random.rand(number)
    cos_phi=np.cos(phi)
    sin_phi=np.sin(phi)
    cos_theta=2.*np.random.rand(number)-1
    sin_theta=np.sqrt(1 - cos_theta**2)
    return((np.array([sin_theta*cos_phi,sin_theta*sin_phi,cos_theta])).transpose())

def photon_origin(number=None):
    """Returns emission location of a photon
    """
    if number is None:
        number=1
    return(np.zeros([number,3]))

def draw_seed_photons(mc_parms,number=None):
    """Returns a single seed photon
    
    Args:
        mc_parms (dictionary): MC parameters
    
    Parameters:
        number (integer): number of photons to return
        
    Returns:
        (number x 4)numpy array: Photon momentum 4-vectors
        (number x 3)numpy array: Initial photon positions
    """

    if number is None:
        number=1
    x_seed=photon_origin(number=number)
    n_seed=random_direction(number=number)
    hnu=mc_parms['hnu_dist'](mc_parms,number=number)
    p_seed=(np.array([hnu,hnu*n_seed[:,0],hnu*n_seed[:,1],hnu*np.abs(n_seed[:,2])])).transpose()/c
    return(p_seed,x_seed)

#
# Write a function that returns tau(P)
#

def tau_of_scatter():
    """Calculates optical depth a photon traveled to before interacting, given probability
    
    Args:
        None
        
    Returns:
        real: Optical depth as function of P

    """
    
    # First, draw your random probability P
    tau=-np.log(np.random.rand())
    return(tau)

#
# Calculate the distance s the photon travels after scattering
#

def distance_of_scatter(mc_parms):
    """Calculates the distance that corresponds to an optical depth tau   

    Args:
        tau (real): optical depth photon travels before scattering occurs
        mc_parsm (dictionary): MC parameters

    Returns:
        real: distance

    """

    tau=tau_of_scatter()
    electron_density=mc_parms['tau']/mc_parms['H']/sigma_T
    distance=tau/sigma_T/electron_density
    
    return(distance)

def scatter_location(x_old,p_photon,mc_parms):
    """This function goes through the steps of a single scattering

    Args:
        x_old.   (three-element numpy array): holds the position
        p_photon (four-element numpy array): the in-coming photon four-momentum
        mc_parms (dictionary): the simulation parameters
    
    Returns:
        three-element numpy array: scattering location
    """
    
    # ...path-length:
    distance = distance_of_scatter(mc_parms)
    
    # ...in direction:
    photon_direction=p_photon[1:]/p_photon[0]
    
    # Update photon position with the new location
    x_new = x_old + distance*photon_direction
    
    return(x_new)


def draw_electron_velocity(mc_parms,p_photon):
    """Returns a randomized electron velocity vector for inverse 
       Compton scattering, taking relativistic foreshortening of the
       photon flux in the electron frame into account
       
       Args:
           mc_parms (dictionary): Monte-Carlo parameters
           p_photon (4 dimentional np array): Photon 4-momentum
           
       Returns:
           3-element numpy array: electron velocity
    """
    v=mc_parms['v_dist'](mc_parms)
    n=draw_electron_direction(v,p_photon)
    return(v*n)

def draw_electron_direction(v,p_photon):
    """Draw a randomized electron direction, taking account of the
       increase in photons flux from the foward direction, which
       effectively increases the cross section for forward scattering.
       
       Args:
            v (real): electron speed
            p_photon (4 element numpy array): photon 4-momentum
            
       Returns:
           3-element numpy array: randomized electron velocity vector
    """
    phi=2.*np.pi*np.random.rand()
    cosp=np.cos(phi)
    sinp=np.sin(phi)
    cost=mu_of_p_electron(v/c,np.random.rand())
    sint=np.sqrt(1 - cost**2)
    
    n_1=p_photon[1:]/p_photon[0]
    if (np.sum(np.abs(n_1[1:2])) != 0):
        n_2=np.cross(n_1,np.array([1,0,0]))
    else:
        n_2=np.cross(n_1,np.array([0,1,0]))
    n_2/=np.sqrt(np.sum(n_2**2))
    n_3=np.cross(n_1,n_2)
    
    # express new vector in old base
    n_new=(n_2*cosp+n_3*sinp)*sint + n_1*cost
    return(n_new/np.sqrt(np.sum(n_new**2)))

def mu_of_p_electron(beta,p):
    """Invert probability for foreshortened effective
       Thomson scattering cross section, with
    
       P = 
       
       Args:
           beta (real): v/c for electron
           p: probability value between 0 and 1
           
       Returns:
           real: cos(theta) relative to photon direction
    """
    mu=1/beta-np.sqrt(1/beta**2 + 1 - 4*p/beta + 2/beta)
    return(mu)

#
# Functions for general Lorentz transform
#

def lorentz_transform(p,v):
    """Returns general Lorentz transform

    Args:
        p (four-element numpy array): input four-vector
        v (three-element numpy array): the 3-velocity of the frame we want to transform into

    Returns:
        four-element numpy array: the transformed four-vector
    """

    beta=np.sqrt(np.sum(v**2))/c
    beta_vec=v/c
    gamma=1./np.sqrt(1. - beta**2)
    matrix=np.zeros((4,4))
    matrix[0,0]=gamma
    matrix[1:,0]=-gamma*beta_vec
    matrix[0,1:]=-gamma*beta_vec
    matrix[1:,1:]=(gamma-1)*np.outer(beta_vec,beta_vec)/beta**2
    for i in range(1,4):
        matrix[i,i]+=1
    return(np.dot(matrix,p))

#
# Calculate the scattering angle for a probability between 0 and 1
#

def cos_theta_thomson(p):
    """Invert P(<\theta) to calculate cos(theta)
        
        Args:
            p (real): probability between 0 and 1
            
        Returns:
            real: scattering angle drawn from Thomson distribution
    """
    a=-4 + 8*p
    b=a**2 + 4
    return((np.power(2,1/3)*np.power(np.sqrt(b)-a,2/3)-2)/
           (np.power(2,2/3)*np.power(np.sqrt(b)-a,1/3)))

#
# Here we perform the Thomson scattering part of inverse Compton scatterin
#

def thomson_scatter(p_photon):
    """This function performs Thomson scattering on a photon
    
    Args:
        p_photon (4-element numpy array): Incoming photon four-vector
        
    Returns:
        4-element numpy array: Scattered photon four-vector
    """
    
    n_1=p_photon[1:]/p_photon[0]
    if (np.sum(np.abs(n_1[1:2])) != 0):
        n_2=np.cross(n_1,np.array([1,0,0]))
    else:
        n_2=np.cross(n_1,np.array([0,1,0]))
    n_2/=np.sqrt(np.sum(n_2**2))
    n_3=np.cross(n_1,n_2)

    # scattering is uniform in phi
    phi=2.*np.pi*np.random.rand()
    cosp=np.cos(phi)
    sinp=np.sin(phi)
    
    # draw cos_theta from proper distribution
    cost=cos_theta_thomson(np.random.rand())
    sint=np.sqrt(1 - cost**2)
    
    # express new vector in old base
    n_new=(n_2*cosp+n_3*sinp)*sint + n_1*cost
    n_new/=np.sqrt(np.sum(n_new**2))
    
    # return scatterd 4-momentum vector
    return(np.array(p_photon[0]*np.array([1,n_new[0],n_new[1],n_new[2]])))

#
# Compute the new photon four-momentum after inverse Compton scattering
#
# This takes three steps:
#
# 1. Lorentz transform to electron frame
#
# 2. Randomize the photon direction, but keep its energy unchanged (Thomson scattering)
#
# 3. Lorentz transform back to the observer's frame
#

def inverse_compton_scatter(p_photon,mc_parms):
    """This function performs an iteration of inverse Compton scattering off an electron of velocity v_vec.
    
    Args:
        p_photon (four element numpy array): input photon four-momentum
        v_vec (three element numpy array): 3-velocity vector of the scattering electron
        
    Returns:
        four-element numpy array: scattered photon four-momentum in observer's frame
    """
    
    # throw the dice one more time to draw a random electron velocity
    velocity=draw_electron_velocity(mc_parms,p_photon)
    # first, transform to electron frame
    p_photon_prime=lorentz_transform(p_photon,velocity)

    # Thomson scatter
    p_out_prime=thomson_scatter(p_photon_prime)
    
    # transform back to observer frame
    return(lorentz_transform(p_out_prime,-velocity))

def monte_carlo(mc_parms):
    """Perform a simple Monte-Carlo simulation

    Args:
       mc_parms (dictionary): Monte-Calro parameters
    
    Returns:
        numpy array: List of escaped photon energies
        numpy array: Lost of seed energies of all escaping photons
    """
    
    # arrays to store initial and final photon energies
    hnu_seed=np.zeros(mc_parms['n_photons'])
    hnu_scattered=hnu_seed.copy()

    # draw our seed-photon population. Much faster to do this once for all photons
    p_photons,x_photons=draw_seed_photons(mc_parms,number=mc_parms['n_photons'])
   
    # run the scattering code n_photons times
    for p_photon,x_photon,i in zip(p_photons,x_photons,range(mc_parms['n_photons'])):
        # initial photon four-momentum
        # store seed photon energy for future use (calculating Compton-y parameter)
        hnu_seed[i]=p_photon[0]*c

        # keep scattering until absorbed or escaped
        scattered=True
        while (scattered):
            # find next scattering location
            x_photon = scatter_location(x_photon,p_photon,mc_parms)
            # if it's inside the corona, perform inverse Compton scatter
            if (x_photon[2]>=0 and x_photon[2]<=mc_parms['H']):
                p_photon=inverse_compton_scatter(p_photon,mc_parms)
            else:
                scattered=False
                if (x_photon[2]<=0):
                    p_photon*=0

        # store the outgoing photon energy in the array
        hnu_scattered[i]=p_photon[0]*c

    # only return escaped photons and their seed energy
    return(hnu_scattered[hnu_scattered > 0],hnu_seed[hnu_scattered > 0])

def plot_mc(mc_parms,bins=None,xlims=None):
    """Run an MC simulation and plot a histogram of the output
    
    Args:
        mc_parms (dictionary): Monte-Carlo parameters
    
    Paramters:
        bins (numpy array): Optional spectral bins
        xlims (2-element list, real): plot-limits
    
    Returns:
        numpy array: The energies of all photons escaping the corona
        numpy array: The seed-energies of the escaping photons
    """
    
    
    # Now run simulation and normalize all outgoing photon energies 
    # so we can investigate energy gains and losses
    hnu_scattered,hnu_seeds=np.array(monte_carlo(mc_parms))/mc_parms['kt_seeds'] 
    
    if (xlims is None):
        xlims=[hnu_scattered.min(),hnu_scattered.max()]    
    if (bins is None):
        bins=np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),num=100)
    else:
        bins=np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),num=bins)

    fig=plt.figure()
    plt.hist(hnu_scattered,bins=bins,log=True,
             label=r'$\tau=${:4.1f}'.format(mc_parms['tau']))
    plt.xscale('log')
    plt.xlim(xlims[0],xlims[1])
    plt.xlabel(r'$h\nu/h\nu_{0}$',fontsize=20)
    plt.ylabel(r'$N(h\nu)$',fontsize=20)
    plt.legend()
    plt.show()
    print('Fraction of escaping photons: {0:5.3e}\n'.format(hnu_scattered.size/mc_parms['n_photons']))
    print('Compton y parameter: {0:5.3e}\n'.format(compton_y(hnu_seeds,hnu_scattered)))
    return(hnu_scattered,hnu_seeds)

#
# Run an example of a Monte Carlo simulation for optical depth 0.1
#

# mc_parms={'n_photons':100000,         # start somewhat small and go up
#           'kt_seeds':1.6e-9,         # 1 keV input photons
#           'H':1e7,                   # say H ~ R, and R ~ 100 R_g ~ 3e7 cm
#           'velocity':3e9,            # 10% speed of light--pretty hot
#           'tau':0.1,                 # tau ~ 0.1: Small-ish optical depth
#           'v_dist':f_of_v_mono,      # name of velocity distribution function
#           'hnu_dist':f_of_hnu_mono,  # name of photon distribution function
#          }
# hnu_scattered,hnu_seeds=plot_mc(mc_parms,bins=100)

#
# Proper Maxwellian velocity distribution
#

def f_of_v_maxwell(mc_parms):
    """Returns a single randomly drawn velocity from distribution function
    
    Args:
        mc_parms (dictionary): Monte-Carlo parameters

    Returns:
        real: electron velocity drawn from distribution
    """
    
    # we need to draw x, y, and z velocity from a Maxwell-Boltzmann distribution'
    # and calculate the resulting speed. This is a non-relativistic Maxwelleian,
    # so me must truncate it below c.
    
    v=3e10 # needed for clipping...
    while v >= c:
        v=np.sqrt(mc_parms['kt_electron']/(m_e))*np.sqrt(np.sum((np.random.normal(0,1,3))**2))
    
    return(v)

#
# For comparison, let's define a Maxwellian
#

def maxwellian(v,kT):
    """Calculate a non-relativistic Maxwellian
    
    Args:
        v (real): Velocity
        kT (real): temperature in energy units
    
    Returns:
        value of the Maxwellian at v
    """    
    return(v**2*np.sqrt(2/np.pi*(m_e/kT)**3)*np.exp(-(m_e*v**2/(2*kT))))

#
# In case you want to improve on the seed photon distribution, this will be helpful
# Note that the photon distribution function goes as nu^2, not nu^3
#

def f_planck(x):
    """Photon distribution function of a Planck distribution
    
    Args:
        x (real): photon energy e in untis of kT
        
    Returns:
        real: differential photon number dN/de at energy e
    """
    norm=2.4041138063192817
    return x**2/(np.exp(x)-1)/norm

def p_planck(hnu=None):
    """Numerical integration of cumulative planck PDF (to be inverted)
    
    Parameters:
        hnu (numpy array): bins of photon energy e in untis of kT
        
    Returns:
        numpy array: cumulative photon PDF as function of hnu
        numpy array: hnu values used for PDF
    """
    if (hnu is None):
        number=1000
        hnu=np.append(np.linspace(0,1-1./number,number),np.logspace(0,4,number))

    p=np.zeros(2*number)
    for i in range(1,2*number):
        p[i]=((quad(f_planck,0,hnu[i]))[0])
    return (p,hnu)

def hnu_of_p_planck(number=None,pdf=None,hnu=None):
    """Numerically invert Planck PDF
    
    Args:
        None
        
    Parameters:
        planck_pdf (numpy array): Previously calculated Planck PDF
        planck_hnu (numpy array): energy grid for PDF
        number (interger): Number of photon energies to generate
        
    Returns:
        numpy array: energies corresponding to p
        numpy array: cumulative PDF userd to calculate e
        numpy array: hnu grid used to calculate PDF
    """
    if number is None:
        number=1
    if (pdf is None):
        pdf,hnu=p_planck()

    e_phot=np.interp(np.random.rand(number),pdf,hnu)
    return(e_phot,pdf,hnu)

#
# Looks good?
#
# Here is a new seed photon distribution to replace the one we used above.
# All you have to do is compile this function and change
# mc_parms['hnu_prob']=f_og_hnu_planck
#

def f_of_hnu_planck(mc_parms,number=None,pdf=None,energies=None):
    """Returns randomly drawn photon energy from a Planck distribution
    
    Args:
        mc_parms (dictionary): Monte-Carlo parameters

    Parameters:
        pdf (numpy array): Previously calculated Planck PDF
        hnu (numpy array): energy grid for PDF

    Returns:
        numpy array: seed photon energies drawn from photon distribution
    """
    
    if number is None:
        number=1
    if (pdf is None):
        e,pdf,energies=hnu_of_p_planck(number=number)
    else:
        e,pdf,energies=hnu_of_p_planck(number=number,pdf=pdf,hnu=energies)        
    e*=mc_parms['kt_seeds']
    
    return(e)

#
# Looking good?
#
# Now run an exaple Monte Carlo simulation with a Maxwellian electron
# distribution.
#

mc_parms={'n_photons':10000,            # start somewhat small and go up
          'kt_seeds':1.6e-9,            # 1 keV input photons
          'H':1e7,                      # say H ~ R, and R ~ 100 R_g ~ 3e7 cm
          'velocity':3e9,               # 10% speed of light--pretty hot
          'tau':0.1,                    # tau ~ 0.1: Small-ish optical depth
          'kt_electron':3.2e-8,         # electron temperature: 20 keV
          'v_dist':f_of_v_maxwell,      # name of photon distribution function
          'hnu_dist':f_of_hnu_planck,   # name of photon distribution function
         }

# Let's make mc_parms consistent
mc_parms['velocity']=np.sqrt(mc_parms['kt_electron']/(m_e)) # thermal speed
print("thermal velocity: {:e}".format(np.sqrt(mc_parms['kt_electron']/(m_e))))

# hnu_scattered,hnu_seeds=plot_mc(mc_parms)

# NOTES ON WHAT TO CHANGE TO INTEGRATE
# velocity in mc_parms has to be the same as velocity in the jet: v gamma_to_velo
# in mc_parms, a tau is given as input, while in the jet, its calculated: make this consistent in some way
# in mc_parms, H~R~100R_g -> in jet, 10r_g is used. make consistent
# both n_photons and their input energy need to have some sort of value; idk what
# electron temperature unclear to me. possibly use original IC value
# for both velocity and frequency distribution, use synchrotron jet output (dirks idea)

# START CONICAL JET PART (PS3)

def conical_jet():

    # "divided into ~10-20 slices"
    # "divide huge jet in equal widths in log10(z) space"
    number_slices = 20
    cone_size = [s for s in np.logspace(0, 20, number_slices)]
    r0 = 10*Rg(MsgrA)

    # "play with values gamma=1 to gamma=4"
    # note: code doesnt run with gamma=1, so at least 1.01 or so
    gamma = 1.11
    v = gamma_to_velo(gamma)
    print("jet velocity for gamma={}: {:e}".format(gamma, v))

    # ANYTHING BELOW THIS IS FROM PROBLEM SET AND NEEDS TO BE ADJUSTED TO PROJECT

    Qj = 10**(40) # from naud (normalise thingie in question PS4) 
    m = m_e
    pitch_angle = math.pi / 2
    p = 2
    gamma_max = 1000

    nu_list = [nu for nu in np.logspace(8, 19, 1000)]
    fluxes_list = []

    old_r = r0

    for s in cone_size:
        fluxes = []
        r = r0 + s * math.tan(5*math.pi/180)
        for nu in nu_list:
            Ue0 = Qj / (math.pi * r0**2 * v)
            Ub0 = Ue0
            B0 = (8*math.pi * Ub0)**(1/2)
            B = B0 * (r/r0)**(-1)

            C = Ue0 * (r/r0)**-2 / math.log(gamma_max)
            power_jet = (10**(-22) * C * B / (p+1)) * (10**(-7) * nu / B)**(-(p-1)/2)
            source_func_jet = power_jet / (4*math.pi*extinction_coeff(e, m, C, B, pitch_angle, p, nu))
            tau = extinction_coeff(e, m, C, B, pitch_angle, p, nu) * r
            intensity_jet = source_func_jet * (1 - math.exp(-tau)) * 10**(23)

            # not sure about this tbh; has to do with emitting surface, but i think s should be involved then
            domega = 4*math.pi * (r-old_r) / DsgrA**2

            flux = intensity_jet * domega 
            fluxes.append(flux)
        
        old_r = r

        fluxes_list.append(fluxes)

        plt.plot(nu_list, fluxes, label='r={}'.format(r))

    plt.plot(nu_list, np.sum(np.array(fluxes_list), 0))
    plt.xlabel(r"$\nu [Hz]$")
    plt.ylabel("Intensity [$mJy$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Total intensity vs frequency from different slices of a conical jet")
    plt.show()

    # now its not exactly flat; this is not a problem now
    # due to a simplification; leon will go through 'geometrical factor' later in WC

    # also go through units!! (idk how)

    return

conical_jet()
