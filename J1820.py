# Lars Zwaan, 12414069
# Dirk Kuiper, 12416657
# Extreme Astrophyics final project
# plotting the spectrum of J1820, consising of self-absorbed synchrotron
# and SSC (inverse Compton upscattering of local synchrotron photons) emission

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

# imports used in original jet slices code
import math
import matplotlib.pyplot as plt
import numpy as np

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
pc = 3.08567758 * 10**18

# see https://arxiv.org/abs/2003.02360
M_J1820 = 8.48 * M_sun

# This distance implies that the source reached (15 ± 3) per cent of the 
# Eddington luminosity at the peak of its outburst. 
# https://www.researchgate.net/publication/338704321_A_radio_parallax_to_the_black_hole_X-ray_binary_MAXI_J1820070
D_J1820 = 2.96 * 10**3 * pc

# gravitational radius for a mass
def Rg(M):
    Rg = 2 * G * M / c**2
    return Rg

# eddington luminosity for a mass
def L_edd(M):
    L_edd = 4*math.pi * G * M * m_p * c / sigma_T
    return L_edd

# calculate velocity from given Gamma factor
def gamma_to_velo(gamma):
    beta = (-(gamma**(-2) - 1))**2
    v = beta**(1/2) * c
    return v

# extinction coefficient used to calculate source function (R&L 6.53)
def extinction_coeff(q, m, C, B, pitch_angle, p, nu):
    alpha_nu = ((3**(1/2) * q**3) / (8*math.pi * m)) * (3*q / (2*math.pi* m**3 * c**5))**(p/2)
    alpha_nu *= C * (B*math.sin(pitch_angle))**((p+2)/2) * nu**(-(p+4)/2)
    alpha_nu *= math.gamma((3*p + 2) / 12) * math.gamma((3*p + 22) / 12)
    return alpha_nu

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
def compton_y(pre,post):
    return(np.mean((post-pre)/pre))

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
def photon_origin(number=None):
    """Returns emission location of a photon
    """
    if number is None:
        number=1
    return(np.zeros([number,3]))

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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

# copied from IC-MC tutorial by Sebastian Heinz, University of Wisconsin-Madison
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
    return hnu_scattered[hnu_scattered > 0], hnu_seed[hnu_scattered > 0]

# a powerlaw distribution
def powerlaw(gamma, p):
    return gamma**-p

# normalise the powerlaw function to get a PDF
def powerlaw_PDF(gamma, p, norm):
    return(powerlaw(gamma, p)/norm)

# returns a single randomly drawn velocity from powerlaw_PDF
def f_of_v_powerlaw(mc_parms):

    number = 100
    N = np.zeros(number)
    gamma = np.logspace(np.log10(mc_parms['gamma_min']),np.log10(mc_parms['gamma_max']),number)

    p = mc_parms['p']
    norm = mc_parms['powerlaw_norm']

    for i in range(number):
        N[i] = quad(powerlaw_PDF, 1, gamma[i], args=(p, norm))[0]

    gamma_to_transform = np.interp(np.random.rand(1), N, gamma)

    return gamma_to_velo(gamma_to_transform)

def hnu_of_p_synchro(number=None,pdf=None,hnu=None):

    e_phot=np.interp(np.random.rand(number),pdf,hnu)

    return(e_phot,pdf,hnu)

def f_of_hnu_mono(mc_parms,number=None):
    """Returns randomly drawn velocity from distribution function
    
    Args:
        mc_parms (dictionary): Monte-Carlo parameters
    
    Parameters:
        number (integer): Number of photon energies to generate

    Returns:
        numpy array: seed photon energies drawn from photon distribution
    """
    if number is None:
        number=1
    return(np.ones(number)*mc_parms['kt_seeds'])

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

    # plt.plot(energies)
    # plt.yscale('log')
    # plt.show()

    # print(e.min())
    # print(sum(e)/len(e))
    # print(e.max())
    # exit()

    e*=mc_parms['kt_seeds']
    
    return(e)

def f_of_hnu_synchro(mc_parms,number=None,pdf=None,energies=None):

    # does what f_of_hnu_planck does but then with synchro input instead of planck
    # changes to this compared to planck:
    # 1) got rid of if number is none (is it ever none?)
    # 2) got rid of if pdf is none as i now define a pdf

    # its actually a cdf but being consistent with the original mistake
    pdf = mc_parms['pdf']
    nu_list = mc_parms['nu_list']

    # DO I NEED THIS h*?
    # i dont think so cause i wanna have nu on the x-axis, nothnu
    # energies = [h*nu for nu in nu_list]
    energies = [nu for nu in nu_list]

    e,pdf,energies=hnu_of_p_synchro(number=number,pdf=pdf,hnu=energies)

    # plt.plot(energies)
    # plt.yscale('log')
    # plt.show()

    # print(e)
    # print("\n\n")
    # print(energies)
    # exit()

    # DO I NEED THIS ADDITIONAL STEP?
    # i think so, mono and planck do it as well
    # but their e's are different, so this NEEDS TO CHANGE
    e*=mc_parms['kt_seeds']

    # print(e)
    # print("\n\n")
    # print(energies)
    # exit()
    
    return(e)

# this is literally copied from Rahuls picture so it better work :(
def running_mean_convolve(x, N):
    return np.convolve(x, np.ones(N) / float(N), 'valid')

# this is literally copied from Rahuls picture so it better work :(
def num_to_flux(hnu_scattered, mc_parms):
    N_hnu, bin_array = np.histogram(hnu_scattered, bins=100)
    avg_bin_array = running_mean_convolve(bin_array, 2) * mc_parms['kt_seeds'] / h
    flux = (N_hnu * h * avg_bin_array) / (2*np.pi*mc_parms['H']**2)

    plt.loglog(avg_bin_array, flux)
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'F_{\nu}$')
    plt.show()

    return flux, avg_bin_array


# NOTES ON WHAT TO CHANGE TO INTEGRATE
# both n_photons needs to have some sort of value; idk what
# kt_seeds: use energy of photon at specific (h)nu
# velocity in mc_parms has to be the same as velocity in the jet: v gamma_to_velo
# in mc_parms, a tau is given as input, while in the jet, its calculated: make this consistent in some way
# in mc_parms, H~R~100R_g -> in jet, 10r_g is used. make consistent
# kt_electron: "At high energies, the spectrum displays a characteristic cutoff that indicates the electron temperature."
# src: file:///C:/Users/lars1/Downloads/c9890edb-82da-4382-847e-d67b10a4a6eb.pdf
# for frequency distribution, use synchrotron jet output (dirks idea)

# START CONICAL JET PART (PS3)

def conical_jet():

    # "divided into ~10-20 slices"
    # "divide huge jet in equal widths in log10(z) space"
    number_slices = 10
    cone_size = [s for s in np.logspace(0, 3, number_slices+1)]
    cone_size_diff = np.diff(cone_size)
    r0 = 10*Rg(M_J1820)

    # "play with values gamma=1 to gamma=4"
    # Zdziarski et al 2022:
    gamma = 3
    v = gamma_to_velo(gamma)
    # print(gamma_to_velo(1000)/c)
    # exit()

    # Zdziarski et al 2022:
    jet_opening_angle = 1.5
    incl_angle = 64 * math.pi/180

    doppler_factor = 1/(gamma*(1-v*np.cos(incl_angle)/c))

    # adjust Qj to get same SSA peak as in data
    Qj = 10**58
    m = m_e
    pitch_angle = math.pi / 2

    # source?
    # p = 1.66
    # gamma_min = 1
    # gamma_max = 10**2

    # old, but keep using for now (PS3 Sols)
    p = 2.4
    gamma_min = 1
    gamma_max = 10**2

    nu_list = [nu for nu in np.logspace(11.5, 26, 1000)]
    fluxes_list = []
    number_photons_list = []
    intensities_list = []

    slice_counter = 0
    hnu_scattered_list = []
    IC_fluxes = []

    for s in cone_size[:-1]:
        fluxes = []
        intensities = []
        r = r0 + s * math.tan(jet_opening_angle*math.pi/180)

        # STUFF FROM PS3 SOLS:
        B_initial = 10**8.5
        phi_B = B_initial*r0*cone_size[:-1][0]
        B = phi_B / (r*cone_size_diff[slice_counter])
        Ub = B**2/(8*np.pi)
        Ue = Ub
        C = Ue/np.log(gamma_max)

        # OWN OLD STUFF:
        # Zdziarski et al 2022:
        # B0 = 10**4 G
        # Ue0 = Qj / (math.pi * r0**2 * v)
        # Ub0 = Ue0
        # B0 = (8*math.pi * Ub0)**(1/2)
        # B = B0 * (r*cone_size_diff[slice_counter])**(-1)
        # C = Ue0 * (r*cone_size_diff[slice_counter])**(-2) / math.log(gamma_max)

        for nu in nu_list:

            # power_jet units: erg cm^-3 s^-1 Hz^-1
            power_jet = (10**(-22) * C * B / (p+1)) * (10**(-7) * nu / B)**(-(p-1)/2)

            # extinction_coeff units: cm^-1
            # source_func_jet units: erg cm^-2 s^-1 Hz^-1
            source_func_jet = power_jet / (4*math.pi*extinction_coeff(e, m, C, B, pitch_angle, p, nu))
            # tau units: cm^-1 * cm = unitless
            tau = extinction_coeff(e, m, C, B, pitch_angle, p, nu) * r

            # intensity_jet units: erg cm^-2 s^-1 Hz^-1
            # for [Jy], multiply intensity_jet by 10**23
            intensity_jet = source_func_jet * (1 - math.exp(-tau))

            flux = intensity_jet * 4 * math.pi

            # we want the units to be in erg cm^-2 s^-1 for comparison
            # to spectrum in paper, so technically this is then nuFnu
            # flux /= nu??
            # flux *= nu
            # flux = flux??
            # THIS SHOULD BE DONE WAY LATER IN CODE TO PREVENT FUCKING WITH NUMBER PHOTONS

            # correct for distance and emitting surface (cylinder)
            flux_earth = flux * (2*math.pi* r * cone_size_diff[slice_counter] * np.sin(incl_angle)*doppler_factor) / (4*math.pi * D_J1820**2)
            fluxes.append(flux_earth)
            intensities.append(intensity_jet)

        # find cutoff energy of slice
        cut_off_found = False
        for flux in fluxes:
            if flux < fluxes[0] and cut_off_found == False:
                cut_off_found = True
                cut_off_energy = nu_list[fluxes.index(flux)] * h

        # calculate number photons per nu per slice
        number_photons = [fluxes[i] / (h*nu_list[i]) for i in range(len(fluxes))]
        number_photons_list.append(number_photons)

        # now, call for IC scattering with synchro as input
        # for this, i need to make a pdf (probability density function)
        # and then a cdf (cumulative density distribution)
        # (see explanation on Jeff's piece of paper)

        # total number of photons for the slice, to normalize:
        number_photons_total_slice = integrate.simps(number_photons)
        pdf = number_photons / number_photons_total_slice
        pdf = np.cumsum(pdf)

        # we need to input a number of photons and scale this per slice
        # the first slice has the most photons: normalize that to norm_goal_photons
        # by dividing each slice by norm_factor_photons and multiplying by norm_goal_photons
        if slice_counter == 0:
            norm_fac = 1000 / number_photons_total_slice
        n_photons = norm_fac * number_photons_total_slice
        n_photons = int(n_photons)

        # normalize to get PDF wqith surface of 1
        powerlaw_norm = quad(powerlaw, gamma_min, gamma_max, args=(p))[0]

        # FIX kt_seeds (was 1.6e-9 in example) (has very little effect)
        mc_parms = {'n_photons': n_photons,
                    'kt_seeds': 1.6e-9,
                    # might be s ipv r, ask/check later
                    'H':r,
                    'velocity':v,
                    # should use tau from calculations, but that one is tiny so doesnt give any scattering
                    'tau':0.1,
                    'kt_electron':cut_off_energy,
                    'v_dist':f_of_v_powerlaw,
                    'hnu_dist':f_of_hnu_synchro,
                    # i know the nomenclature is off, but im being consitent with the original mistake
                    'pdf': pdf,
                    'nu_list':nu_list,
                    'p': p,
                    'powerlaw_norm': powerlaw_norm,
                    'gamma_min':gamma_min,
                    'gamma_max':gamma_max,
                    }

        # do the monte carlo for each slice
        hnu_scattered, hnu_seeds=np.array(monte_carlo(mc_parms))/mc_parms['kt_seeds']

        # Rahuls stuff, doesnt seem to work here at least
        # flux, avg_bin_array = num_to_flux(hnu_scattered, mc_parms)
        # exit()

        # bins=None
        # xlims=None
        # if (xlims is None):
        #     xlims=[hnu_scattered.min(),hnu_scattered.max()]
        # if (bins is None):
        #     bins=np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),num=100)
        # else:
        #     bins=np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),num=bins)

        # plt.hist(hnu_scattered,bins=bins,log=True,
        #         label=r'$\tau=${:4.1f}'.format(mc_parms['tau']))
        # plt.xscale('log')
        # plt.xlim(xlims[0],xlims[1])
        # plt.xlabel(r'$\nu$')
        # plt.ylabel(r'$N$')
        # plt.legend()
        # plt.show()

        # convert this histogram type thing to a number_photons vs frequency
        N_list = [0 for nu in nu_list]
        for nu in hnu_scattered:
            # find closest value in nu_list
            # https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
            # most closest nu's are now the last one, this maybe has to do with a log scale? 
            # that they always fall into that bin?
            closest_nu = nu_list[min(range(len(nu_list)), key = lambda i: abs(nu_list[i]-nu))]
            N_list[nu_list.index(closest_nu)] += 1

        # plt.loglog(nu_list, N_list)
        # plt.show()
        # exit()

        # to get back to a flux, multiply by energy hnu and remove normalisation
        for i in range(len(N_list)):
            N_list[i] = (N_list[i]/norm_fac) * h*nu_list[i]**2
            # N_list[i] = (N_list[i]/norm_fac)
            # N_list[i] = N_list[i]

        # plt.loglog(nu_list, N_list)
        # plt.show()
        # exit()

        IC_fluxes.append(N_list)
        # for nu in nu_list:
        fluxes = [fluxes[i]*nu_list[i] for i in range(len(fluxes))]
        fluxes_list.append(fluxes)
        intensities_list.append(intensities)

        hnu_scattered_list.append(hnu_scattered)

        # keep track of which slice were at and let us know :)
        slice_counter += 1
        print("Slice {} made".format(slice_counter))
    
    # unpack list of lists into single list
    hnu_scattered_list = [hnu for hnu_scattered in hnu_scattered_list for hnu in hnu_scattered]
    hnu_scattered_list = np.array(hnu_scattered_list)

    # plot monte carlo total
    # bins=None
    # xlims=None
    # if (xlims is None):
    #     xlims=[hnu_scattered_list.min(),hnu_scattered_list.max()]
    # if (bins is None):
    #     bins=np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),num=100)
    # else:
    #     bins=np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),num=bins)

    # plt.hist(hnu_scattered_list,bins=bins,log=True,
    #         label=r'$\tau=${:4.1f}'.format(mc_parms['tau']))
    # plt.xscale('log')
    # plt.xlim(xlims[0],xlims[1])
    # plt.xlabel(r'$\nu$')
    # plt.ylabel(r'$N$')
    # plt.legend()
    # plt.show()

    # plot Fnu for each slice
    # for fluxes in fluxes_list:
    #     plt.plot(nu_list, fluxes, label='r={:e}'.format(r), linestyle='dashed')

    # for IC_flux in IC_fluxes:
    #     plt.scatter(nu_list, IC_flux)

    # plot total Fnu per nu
    plt.plot(nu_list, np.sum(np.array(fluxes_list), 0), color='black')

    # plot total IC_flux
    # plt.plot(nu_list, np.sum(np.array(IC_fluxes), 0), color='black')

    plt.xlabel(r"$\nu\ [Hz]$")
    # change units if i multiply flux by nu or something idk
    plt.ylabel(r"\nu\ $F_{\nu}\ [erg\ cm^{-2}\ s^{-1}\ Hz^{-1}]$")
    plt.title("Input photon spectrum (synchrotron")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    # plot Inu for each slice
    # for intensities in intensities_list:
    #     plt.plot(nu_list, intensities, label='r={:e}'.format(r), linestyle='dashed')

    # # plot total Inu
    # plt.plot(nu_list, np.sum(np.array(intensities_list), 0), color='black')
    # plt.xlabel(r"$\nu\ [Hz]$")
    # plt.ylabel(r"$I_{\nu}\ [erg\ cm^{-2}\ s^{-1}\ Hz^{-1}]$")
    # plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()

    # plot number of photons for each slice
    # for number_photons in number_photons_list:
    #     plt.plot(nu_list, number_photons, label='r={:e}'.format(r), linestyle='dashed')

    # for IC_flux in IC_fluxes:
    #     plt.scatter(nu_list, IC_flux)

    IC_fluxes_2 = IC_fluxes
    IC_fluxes = np.sum(np.array(IC_fluxes), 0)

    # print(len(IC_fluxes))
    # virgin = True
    # for i in range(len(IC_fluxes)):
    #     if IC_fluxes[i] == 0:
    #         if virgin == True:
    #             new_IC_fluxes = np.delete(IC_fluxes, i)
    #             new_nu_list = np.delete(nu_list, i)
    #             virgin = False
    #         else:
    #             new_IC_fluxes = np.delete(new_IC_fluxes, i)
    #             new_nu_list = np.delete(new_nu_list, i)

    # print(len(new_IC_fluxes))

    # exit()

    # IC_fluxes = scipy.ndimage.uniform_filter1d(IC_fluxes)

    plt.plot(nu_list, IC_fluxes, color='black')
    plt.xlabel(r"$\nu\ [Hz]$")
    plt.ylabel(r"\nu\ $F_{\nu}\ [erg\ cm^{-2}\ s^{-1}\ Hz^{-1}]$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Spectrum after IC scattering")
    plt.show()

    # divide by h*nu**2 to get to a number of photons, to see whats happening
    for IC_flux in IC_fluxes_2:
        for i in range(len(nu_list)):
            IC_flux[i] /= h*nu_list[i]**2

    IC_fluxes_2 = np.sum(np.array(IC_fluxes_2), 0)

    plt.scatter(nu_list, IC_fluxes_2, color='black')
    plt.xlabel(r"$\nu\ [Hz]$")
    plt.ylabel(r"$\frac{F_{\nu}}{\nu}\ [oonits]$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Spectrum after IC scattering")
    plt.show()

    return

conical_jet()
