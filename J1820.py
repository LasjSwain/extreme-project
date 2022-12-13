# Lars Zwaan, 12414069
# Dirk Kuiper, #
# Extreme Astrophyics final project
# plotting the spectrum of J1820, consising of self-absorbed synchrotron
# and SSC (inverse Compton upscattering of local synchrotron photons) emission

import math
import matplotlib.pyplot as plt
import numpy as np
import math

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

def conical_jet():

    # "divided into ~10-20 slices"
    # "divide huge jet in equal widths in log10(z) space"
    number_slices = 20
    cone_size = [s for s in np.logspace(0, 20, number_slices)]
    r0 = 10*Rg(MsgrA)

    # "play with values gamma=1 to gamma=4"
    # note: code doesnt run with gamma=1, so at least 1.01 or so
    gamma = 2
    v = gamma_to_velo(gamma)

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
    # plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Total intensity vs frequency from different slices of a conical jet")
    plt.show()

    # now its not exactly flat; this is not a problem now
    # due to a simplification; leon will go through 'geometrical factor' later in WC

    # also go through units!! (idk how)

    return

conical_jet()
