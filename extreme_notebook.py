# Lars Zwaan, 12414069
# Extreme Astrophyics problem sets 1, 2, 3, 4

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from scipy.special import kn

# a bunch of constants i'll need during the course, in CGS
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


# PS1 ex5: plot a planck function
def planck_func(nu, T):
    B_nu = ( (2*h*nu**3) / (c**2) ) / (math.exp((h*nu) / (k_B*T)) - 1)
    return B_nu

def plot_ex5():
    # plot it for a couple of temperatures:
    temps = [1, 1.5, 2, 2.725, 3]

    for T in temps:

        nu_list = [i for i in range(int(0.1 * 10**11), int(6 * 10**11), 10**9)]
        B_list = [planck_func(nu, T) for nu in nu_list]

        plt.plot(nu_list, B_list, label='{}K'.format(T))

    T = 2.725
    nu_max = T * wien_displacementlaw_freq_const
    nu_max_y = [i for i in np.arange(min(B_list), max(B_list), ((max(B_list) - min(B_list)) / 1000))]
    plt.plot([nu_max] * len(nu_max_y), nu_max_y, linestyle='dashed', color='red', label='expected peak for T=2.725K')
    plt.xlabel("Frequency [$Hz$]")
    plt.ylabel("Intensity [$erg$ $cm^{-2}$ $sr^{-1}$ $Hz^{-1}$]")
    plt.title('CMB Planck spectrum')
    plt.legend()
    plt.show()

    return

# PS2 ex5: an expanding blob of plasma emitting

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

# define some numbers and calculate the asked values
def setup():
    velocity = 0.3*c
    R0 = 10 * Rg(MsgrA)
    gamma_max = 1000
    m = m_e
    pitch_angle = math.pi / 2
    p = 2

    # frequencies and efficiencies are just picked numbers to make it look good
    # nu has to be somewhere in radio (high) and eta around 10-8 to 10-3
    nus = [10**10, 10**10.1, 10**10.2]
    etas = [10**(-6)]

    for eta in etas:
        for nu in nus:
            
            # times are just the frame in which u see the peak in I
            times = [t for t in np.logspace(-10.5, -8, 1000)]
            intensities = []

            # see notes for the derivation of formulas
            for t in times:

                R_t = blob_radius(R0, velocity, t)

                Ue = (eta * L_edd(MsgrA)) / (4*math.pi * R_t**2 * velocity)
                Ub = Ue
                const_C = Ue / math.log(gamma_max)
                const_K = 2*vol_sphere(R_t)*Ub

                B_t = ((3*const_K) / (R_t**3))**(1/2)

                I = intensity(e, m, const_C, B_t, pitch_angle, p, nu, R_t) * 10**23
                intensities.append(I)

            plt.plot(times, intensities, label='nu = {} * 10e10 $Hz$'.format(round(nu / 10**10, 2)))

    plt.ylabel("Intensity [$mJy$]")
    plt.xscale("log")
    plt.xlabel("Time [$sec$]")
    plt.title('Expanding blob lightcurve')
    plt.legend()
    plt.show()

    return

# setup()

# PS3 ex3: Conical jet

def conical_jet():

    cone_size = [s for s in np.logspace(0, 20, 100)]
    r0 = 10*Rg(MsgrA)
    v = 0.3*c
    Qj = 10**(40) # from naud (normalise thingie in question)
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

            domega = 4*math.pi * (r-old_r) / DsgrA**2 # not sure about this tbh
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

# conical_jet()

# PS3 ex4: Bremsstrahlung

def bremsspectrum():
    Z = 1

    # pick something like the bohr radius for b
    b = 10**(-11)
    m = m_e
    v = 0.09 * c
    omegas = [o for o in np.logspace(3, 20, 1000)]
    dW_domes = []

    for omega in omegas:
        dW_dome = (8 * Z**2 * e**6 * omega**2) / (3 * math.pi * c**3 * v**4 * m**2)
        dW_dome *= (kn(1, b*omega / v))**2

        dW_domes.append(dW_dome)

    plt.plot(omegas, dW_domes)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\omega [Hz]$")
    plt.ylabel(r"$\frac{dW}{d\omega} [erg^{-1} sec^{-1} Hz^{-1}]$")
    plt.title("Bremsstrahlung revisited")
    plt.show()

# bremsspectrum()

# git test

# git test dirk
