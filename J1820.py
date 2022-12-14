# Lars Zwaan, 12414069
# Dirk Kuiper, 12416657
# Extreme Astrophyics final project
# plotting the spectrum of J1820, consising of self-absorbed synchrotron
# and SSC (inverse Compton upscattering of local synchrotron photons) emission

# PROJECT DESCRIPTION:

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
# Atri et al 2020:
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
    v = (1 - (1/gamma**2))**(1/2) * c
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
    v=mc_parms['v_dist'](mc_parms)[0]
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

    mc_parms['drawn_gammas'].append(gamma_to_transform[0])

    return gamma_to_velo(gamma_to_transform), gamma_to_transform

# returns a hnu value from synchrotron input spectrum
def hnu_of_p_synchro(number=None,pdf=None,hnu=None):

    e_phot=np.interp(np.random.rand(number),pdf,hnu)

    return(e_phot,pdf,hnu)

# returns an energy value from the synchrotron input spectrum
def f_of_hnu_synchro(mc_parms,number=None,pdf=None,energies=None):

    # its actually a cdf but being consistent with the original mistake
    pdf = mc_parms['pdf']
    nu_list = mc_parms['nu_list']

    energies = [nu for nu in nu_list]

    e,pdf,energies=hnu_of_p_synchro(number=number,pdf=pdf,hnu=energies)
    e*=mc_parms['kt_seeds']
    
    return(e)

def main():

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

    # Zdziarski et al 2022:
    jet_opening_angle = 1.5
    incl_angle = 64 * math.pi/180

    doppler_factor = 1/(gamma*(1-v*np.cos(incl_angle)/c))

    m = m_e
    pitch_angle = math.pi / 2

    # Abe et al 2022:
    p = 2.44
    gamma_min = 1
    gamma_max = 10**2

    nu_list = [nu for nu in np.logspace(11.5, 26, 1000)]
    fluxes_list = []
    number_photons_list = []
    intensities_list = []

    slice_counter = 0
    hnu_scattered_list = []
    IC_fluxes = []

    # list with all drawn gammas to compare with the analytical distribution
    drawn_gammas = []

    # for each slice in the cone, calculate the Synchrotron spectrum and
    # use that as input for the Inverse Compton Scattering
    for s in cone_size[:-1]:
        fluxes = []
        intensities = []
        r = r0 + s * math.tan(jet_opening_angle*math.pi/180)

        # Zdziarski et al 2022:
        # B_initial = 10**4 G
        B_initial = 10**8.5
        phi_B = B_initial*r0*cone_size[:-1][0]
        B = phi_B / (r*cone_size_diff[slice_counter])
        Ub = B**2/(8*np.pi)

        # Assume equipartition
        Ue = Ub
        C = Ue/np.log(gamma_max)

        for nu in nu_list:

            # power_jet units: erg cm^-3 s^-1 Hz^-1
            power_jet = (10**(-22) * C * B / (p+1)) * (10**(-7) * nu / B)**(-(p-1)/2)

            # extinction_coeff units: cm^-1
            # source_func_jet units: erg cm^-2 s^-1 Hz^-1
            source_func_jet = power_jet / (4*math.pi*extinction_coeff(e, m, C, B, pitch_angle, p, nu))
            # tau units: unitless [cm^-1 * cm]
            tau = extinction_coeff(e, m, C, B, pitch_angle, p, nu) * r

            # intensity_jet units: erg cm^-2 s^-1 Hz^-1
            # for [Jy], multiply intensity_jet by 10**23
            intensity_jet = source_func_jet * (1 - math.exp(-tau))
            flux = intensity_jet * 4 * math.pi

            # correct for distance and emitting surface (cylinder)
            flux_earth = flux * (2*math.pi* r * cone_size_diff[slice_counter] * np.sin(incl_angle)*doppler_factor) / (4*math.pi * D_J1820**2)

            fluxes.append(flux_earth)
            intensities.append(intensity_jet)

        # find cutoff energy of slice, for the electron energy
        cut_off_found = False
        for flux in fluxes:
            if flux < fluxes[0] and cut_off_found == False:
                cut_off_found = True
                cut_off_energy = nu_list[fluxes.index(flux)] * h

        # calculate number photons per nu per slice
        number_photons = [fluxes[i] / (h*nu_list[i]) for i in range(len(fluxes))]
        number_photons_list.append(number_photons)

        # now, call for IC scattering with this synchrotron radiation as input
        # for this, make a pdf (probability density function) and then a cdf
        # (cumulative density distribution)

        # total number of photons for the slice, to normalize:
        number_photons_total_slice = integrate.simps(number_photons)
        pdf = number_photons / number_photons_total_slice
        pdf = np.cumsum(pdf)

        # we need to input a number of photons and scale this per slice
        # the first slice has the most photons: normalize that to 10**3 or more
        # depending on desirable computing time, by dividing each slice by
        # norm_fac and multiplying by that 10**3 photons
        if slice_counter == 0:
            norm_fac = 10**5 / number_photons_total_slice
        n_photons = norm_fac * number_photons_total_slice
        n_photons = int(n_photons)

        # calculate normalisation factor for the electron PDF with surface of 1
        powerlaw_norm = quad(powerlaw, gamma_min, gamma_max, args=(p))[0]

        mc_parms = {'n_photons': n_photons,
                    'kt_seeds': 1.6e-9,
                    'H':r,
                    'tau':0.1,
                    'kt_electron':cut_off_energy,
                    'v_dist':f_of_v_powerlaw,
                    'hnu_dist':f_of_hnu_synchro,
                    'pdf': pdf,
                    'nu_list':nu_list,
                    'p': p,
                    'powerlaw_norm': powerlaw_norm,
                    'gamma_min':gamma_min,
                    'gamma_max':gamma_max,
                    'drawn_gammas':drawn_gammas,
                    }

        # do the monte carlo IC scattering for each slice
        hnu_scattered, hnu_seeds=np.array(monte_carlo(mc_parms))/mc_parms['kt_seeds']

        # convert this histogram type thing to a number_photons vs frequency
        N_list = [0 for nu in nu_list]
        for nu in hnu_scattered:
            # find closest value in nu_list
            # https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
            closest_nu = nu_list[min(range(len(nu_list)), key = lambda i: abs(nu_list[i]-nu))]
            N_list[nu_list.index(closest_nu)] += 1

        # to get back to a flux, multiply by energy hnu and remove normalisation
        # as we're plotting a nu Fnu vs nu spectrum, multiply by nu again
        for i in range(len(N_list)):
            N_list[i] = (N_list[i]/norm_fac) * h*nu_list[i]**2

        IC_fluxes.append(N_list)

        # we want the units to be in erg cm^-2 s^-1 for comparison
        # to spectrum in paper, so this is then nuFnu
        fluxes = [fluxes[i]*nu_list[i] for i in range(len(fluxes))]

        fluxes_list.append(fluxes)
        intensities_list.append(intensities)
        hnu_scattered_list.append(hnu_scattered)

        # keep track of which slice were at and let us know :)
        slice_counter += 1
        print("Slice {} made".format(slice_counter))
    
    # show analytical electron distribution as a power law function
    number = 1000
    N = np.zeros(number)
    gamma = np.logspace(np.log10(gamma_min),np.log10(gamma_max),number)

    # show drawn electron distribution as a power law function
    counts_drawn_gammas = [0 for gam in gamma]
    for gamma_drawn in drawn_gammas:
        closest_gamma = gamma[min(range(len(gamma)), key = lambda i: abs(gamma[i]-gamma_drawn))]
        counts_drawn_gammas[np.where(gamma == closest_gamma)[0][0]] += 1

    # bin the drawn electron distribution to plot nicer
    number_bins = 50
    len_bin = int(len(counts_drawn_gammas) / number_bins)
    counts_drawn_gammas_new = []
    gamma_binned = []
    start = 0
    for i in range(1, number_bins + 1):
        end = len_bin * i
        counts_drawn_gammas_new.append(sum(counts_drawn_gammas[start:end]))
        gamma_binned.append(0.5 * (gamma[start] + gamma[end-1]))
        start = end

    max_fac = max(counts_drawn_gammas_new)
    counts_drawn_gammas_new = [count / max_fac for count in counts_drawn_gammas_new]

    # plot electron powerlaw distribution
    for i in range(number):
        N[i] = powerlaw_PDF(gamma[i], mc_parms['p'], mc_parms['powerlaw_norm'])

    plt.scatter(gamma_binned, counts_drawn_gammas_new, label='Drawn')
    plt.loglog(gamma,N, label='Analytical')
    plt.xlabel('$\gamma$')
    plt.ylabel('N($\gamma$)')
    plt.title("Electron powerlaw distribution for p={}".format(p))
    # plt.savefig("plots/electron_powerlaw_distribution.png")
    # plt.close()
    plt.show()

    # unpack list of lists into single list
    hnu_scattered_list = [hnu for hnu_scattered in hnu_scattered_list for hnu in hnu_scattered]
    hnu_scattered_list = np.array(hnu_scattered_list)

    # plot nuFnu for each slice, thats just flux
    slice_num = 0
    for fluxes in fluxes_list:
        slice_num += 1
        plt.plot(nu_list, fluxes, label='Slice {}'.format(slice_num), linestyle='dashed')

    # plot total nuFnu per nu
    plt.plot(nu_list, np.sum(np.array(fluxes_list), 0), color='black', label='Total')
    plt.xlabel(r"$\nu\ [Hz]$")
    plt.ylabel(r"$\nu\ F_{\nu}\ [erg\ cm^{-2}\ s^{-1}]$")
    plt.title("Input photon spectrum (synchrotron)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plt.savefig("plots/input_synchrotron_nuFnu")
    # plt.close()
    plt.show()

    # fluxes divnu is actually the flux; before it was nuFnu
    fluxes_divnu_list = []
    for fluxes in fluxes_list:
        fluxes_divnu = [fluxes[i]/nu_list[i] for i in range(len(fluxes))]
        fluxes_divnu_list.append(fluxes_divnu)

    # plot Fnu for each slice
    slice_num = 0
    for fluxes_divnu in fluxes_divnu_list:
        slice_num += 1
        plt.plot(nu_list, fluxes_divnu, label='Slice {}'.format(slice_num), linestyle='dashed')

    # plot total Fnu per nu
    plt.plot(nu_list, np.sum(np.array(fluxes_divnu_list), 0), color='black', label='Total')
    plt.xlabel(r"$\nu\ [Hz]$")
    plt.ylabel(r"$F_{\nu}\ [erg\ cm^{-2}\ s^{-1}\ Hz^{-1}]$")
    plt.title("Input photon spectrum (synchrotron)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plt.savefig("plots/input_synchrotron_Fnu")
    # plt.close()
    plt.show()

    # fluxes divnunu is actually the flux over nu; before it was nuFnu
    fluxes_divnunu_list = []
    for fluxes in fluxes_list:
        fluxes_divnunu = [fluxes[i]/(h*nu_list[i]**2) for i in range(len(fluxes))]
        fluxes_divnunu_list.append(fluxes_divnunu)

    # plot Fnu/nu for each slice
    slice_num = 0
    for fluxes_divnunu in fluxes_divnunu_list:
        slice_num += 1
        plt.plot(nu_list, fluxes_divnunu, label='Slice {}'.format(slice_num), linestyle='dashed')

    # plot total Fnu/nu per nu
    plt.plot(nu_list, np.sum(np.array(fluxes_divnunu_list), 0), color='black', label='Total')
    plt.xlabel(r"$\nu\ [Hz]$")
    plt.ylabel(r"$\frac{F_{\nu}}{\nu}\ [erg\ cm^{-2}\ s^{-1}\ Hz^{-2}]$")
    plt.title("Input photon spectrum (synchrotron)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plt.savefig("plots/input_synchrotron_Fnu_over_nu")
    # plt.close()
    plt.show()
    
    IC_nu_Fnu = IC_fluxes
    IC_fluxes = np.sum(np.array(IC_fluxes), 0)

    # plot inverse compton scattered nuFnu spectrum
    plt.plot(nu_list, IC_fluxes, color='black')
    plt.xlabel(r"$\nu\ [Hz]$")
    plt.ylabel(r"$\nu\ F_{\nu}\ [erg\ cm^{-2}\ s^{-1}\ Hz]$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Spectrum after IC scattering")
    # plt.savefig("plots/comptoned_spectrum")
    # plt.close()
    plt.show()

    # divide by h*nu**2 to get to a number of photons, to see whats happening
    for IC_nufnu in IC_nu_Fnu:
        for i in range(len(nu_list)):
            IC_nufnu[i] /= h*nu_list[i]**2

    # plot inverse compton scattered number of photons spectrum
    IC_nu_Fnu = np.sum(np.array(IC_nu_Fnu), 0)
    plt.plot(nu_list, IC_nu_Fnu, color='black')
    plt.xlabel(r"$\nu\ [Hz]$")
    plt.ylabel(r"Number of photons: $\frac{F_{\nu}}{\nu}\ [erg\ cm^{-2}\ s^{-1}\ Hz^{-2}]$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Number of photons per frequency after IC scattering")
    # plt.savefig("plots/comptoned_number_photons")
    # plt.close()
    plt.show()

    # plot inverse compton scattered number of photons spectrum,
    # now including input synchrotron

    # plot Fnu/nu for each slice
    for fluxes_divnunu in fluxes_divnunu_list:
        plt.plot(nu_list, fluxes_divnunu, linestyle='dashed')

    # plot total Fnu/nu per nu
    plt.plot(nu_list, np.sum(np.array(fluxes_divnunu_list), 0), color='black', label='Input synchrotron')
    plt.scatter(nu_list, IC_nu_Fnu, color='black', s=0.5, label='IC scattered')
    plt.xlabel(r"$\nu\ [Hz]$")
    plt.ylabel(r"Number of photons: $\frac{F_{\nu}}{\nu}\ [erg\ cm^{-2}\ s^{-1}\ Hz^{-2}]$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Number of photons per frequency before/after IC scattering")
    plt.legend()
    # plt.savefig("plots/synchrotron_and_comptoned_number_photons")
    # plt.close()
    plt.show()

    return

main()
