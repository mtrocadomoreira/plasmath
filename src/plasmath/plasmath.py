import os

import numpy as np  # noqa
import sympy as sm  # noqa
from pint import Quantity, UnitRegistry

ureg = UnitRegistry()
ureg.load_definitions(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "unit_defs.txt")
)


def get_ureg():
    return ureg


def symb_args(func):
    def wrapped(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, Quantity):
                new_args.append(arg.magnitude)
            else:
                new_args.append(arg)
        new_args = tuple(new_args)

        for k, v in kwargs.items():
            if isinstance(v, Quantity):
                kwargs[k] = v.magnitude

        return func(*new_args, **kwargs)

    return wrapped


# Variables and constants


@ureg.check("[number_density]")
def plasma_frequency(n0):
    """plasma frequency [rad/s]"""
    omp = np.sqrt(ureg.e**2 * n0 / (ureg.eps0 * ureg.m_e))
    return omp.to("Hz").to_compact()


@ureg.check("[number_density]")
def plasma_skin_depth(n0):
    skd = ureg.c / plasma_frequency(n0)
    return skd.to("cm").to_compact()


@ureg.check("[number_density]", "[mass]", "[charge]")
def beam_frequency(nb0, Mb, qb):
    """beam frequency [rad/s]"""
    omb = np.sqrt(qb**2 * nb0 / (ureg.eps0 * Mb))
    return omb.to("Hz").to_compact()


@ureg.check("[number_density]", "[mass]", "[charge]", "1")
def betatron_frequency(nb0, Mb, qb, gamma):
    """beam betatron frequency [rad/s]"""
    ombeta = np.sqrt(qb**2 * nb0 / (ureg.eps0 * Mb)) / np.sqrt(2 * gamma)
    return ombeta.to("Hz").to_compact()


@ureg.check("[number_density]", "[mass]", "[charge]", "1")
def betatron_skin_depth(nb0, Mb, qb, gamma):
    """beam betatron skin depth, 1/kbeta [cm]"""
    skd_beta = ureg.c / betatron_frequency(nb0, Mb, qb, gamma)
    return skd_beta.to_base_units().to_compact()


# TODO: implement flat profiles
@ureg.check("1", "[length]", "[length]", None, None)
def nb0_peak(
    Nb, sigz, sigr, longit_profile: str = "cos", transv_profile: str = "gauss"
):
    """peak density for a Gaussian (transv. and longit.) profile [1/cm^3]"""

    match longit_profile:
        case "cos":
            int_longit = np.sqrt(2 * np.pi) * sigz
        case "gauss":
            int_longit = np.sqrt(2 * np.pi) * sigz
        case "parab":
            int_longit = 4 / 3 * np.sqrt(5) * sigz
        case "flat":
            raise NotImplementedError(
                "Sorry, the flat longitudinal profile has not been implemented yet."
            )
        case _:
            raise ValueError(
                'Invalid longitudinal profile. Valid keywords for `longit_profile` are "cos", "gauss", "parab" or "flat"'
            )

    match transv_profile:
        case "gauss":
            int_transv = sigr**2
        case "gauss_r":
            int_transv = sigr**2 / 2
        case "flat":
            raise NotImplementedError(
                "Sorry, the flat transverse profile has not been implemented yet."
            )
        case _:
            raise ValueError(
                'Invalid transverse profile. Valid keywords for `transv_profile` are "gauss", "gauss_r", or "flat"'
            )

    nb0 = Nb / (2 * np.pi * int_longit * int_transv)

    return nb0.to(1 / ureg.cm**3)


@ureg.check("[number_density]", "[length]", "[charge]", "[number_density]", None)
def nb0_to_current(nb0, sigr, qb, n0, return_norm: bool = False):
    """convert peak density to peak current for a transversely Gaussian profile [A]"""

    result = 0.5 * norm(sigr, n0) ** 2 * qb / ureg.e * norm(nb0, n0)

    if return_norm:
        return result.to_base_units().to_compact()
    else:
        return (result * ureg.I_A).to_base_units().to_compact()


@ureg.check("[current]", "[length]", "[charge]", "[number_density]", None)
def current_to_nb0(I0, sigr, qb, n0, return_norm: bool = False):
    """convert peak current to peak density for a transversely Gaussian profile [1/cm^3]"""

    result = 2 * I0 / ureg.I_A / (qb / ureg.e * norm(sigr, n0))

    if return_norm:
        return result.to_base_units().to_compact()
    else:
        return (result * n0).to_base_units().to_compact()


# Normalization


# TODO: make n0 optional
def norm(
    quant,
    n0,
    angular: bool | None = None,
    species_momentum: bool | None = None,
    m_sp=None,
):
    n0.check("[number_density]")

    def f_momentum(quant):
        if species_momentum is False:
            return quant / (ureg.m_e * ureg.c)
        elif species_momentum is True:
            try:
                assert m_sp is not None
                assert m_sp.check("[mass]")
            except AssertionError:
                raise ValueError(
                    'The species mass must be provided with argument "m_sp" for the momentum to be normalized with respect to the species.'
                )
            return quant / (m_sp * ureg.c)
        else:
            raise ValueError(
                'Must specify the keyword "species_momentum" as True or False for momentum quantities.'
            )

    def f_frequency(quant):
        if angular is True:
            return quant / plasma_frequency(n0)
        elif angular is False:
            return quant / plasma_frequency(n0) / (2 * ureg.pi)
        else:
            raise ValueError(
                'Must specify the keyword "angular" as True or False for frequency quantities.'
            )

    normalizations = {
        "[length]": lambda quant: quant / plasma_skin_depth(n0),
        "[time]": lambda quant: quant * plasma_frequency(n0),
        "[velocity]": lambda quant: quant / ureg.c,
        "[momentum]": f_momentum,
        "[energy]": lambda quant: quant / (ureg.m_e * ureg.c**2),
        "[number_density]": lambda quant: quant / n0,
        "[emittance]": lambda quant: quant / plasma_skin_depth(n0),
        "[frequency]": f_frequency,
        "[current]": lambda quant: quant / ureg.I_A,
        "1": lambda quant: quant,
    }

    for k, v in normalizations.items():
        if quant.check(k) is True:
            return v(quant).to_base_units().to_compact()

    raise NotImplementedError(
        f"Don't know how to normalize the dimensionality {quant.dimensionality}"
    )


@ureg.check("[length]", "[length]", "[number_density]")
def emit_to_uth(emit, sigr, n0):
    uth = norm(emit, n0) / norm(sigr, n0)
    return uth


# Numerical


def courant_limit(dx):
    dxv = np.asarray(dx)
    return 1.0 / np.sqrt(np.dot(1 / dxv, 1 / dxv))


def get_ndumps(dt: float, tmax: float, totdumps: int = 50, report: bool = True):
    """returns a tuple (ndumps,tmax_new) with the factor ndumps for a desired number of total dumps, and with an updated tmax (in case last dump is too far from nominal tmax)"""

    totiter = np.floor(tmax / dt)
    ndumps_fl = totiter / totdumps

    last_tdump_min = np.floor(ndumps_fl) * dt * totdumps
    last_tdump_max = np.ceil(ndumps_fl) * dt * totdumps

    if abs(tmax - last_tdump_min) > abs(tmax - last_tdump_max):
        ndumps = np.ceil(ndumps_fl)
        tmax_new = last_tdump_max + 10 * dt
    else:
        ndumps = np.floor(ndumps_fl)
        tmax_new = tmax

    if report is True:
        print("Original tmax: ", tmax)
        print("Updated tmax: ", tmax_new)
        print("Last dump at t = ", (ndumps * totdumps * dt))
        print("ndumps = ", ndumps)

    return (ndumps, tmax_new)


# Profiles


@symb_args
def profile_gauss(sig: float, xc: float = 0.0, var: str = "x", printpars: bool = True):
    """normalized Gaussian profile along one dimension (function)"""

    var, sig_s, xc_s = sm.symbols(" ".join([var, "sig_s", "xc_s"]))
    sig_s = sm.S(sig)
    xc_s = sm.S(xc)

    expfac = 2 * sig_s**2
    if printpars is True:
        print(
            f"Written formula:\n exp(-(({var} - {sm.N(xc_s)})^2) / {sm.N(expfac):.6f})"
        )
    return (sm.exp(-((var - xc_s) ** 2) / expfac), var)


@symb_args
def profile_gauss_rcyl(sig: float, var: str = "r", printpars: bool = True):
    """normalized Gaussian profile along r in cylindrical coordinates (function)"""
    var, sig_s = sm.symbols(" ".join([var, "sig_s"]))
    sig_s = sm.S(sig)

    expfac = sig_s**2
    if printpars is True:
        print(f"Written formula:\n exp(-({var}^2) / {sm.N(expfac):.6f})")
    return (sm.exp(-(var**2) / expfac), var)


# TODO: write written formula without '<=' symbols (I think this is a problem for osiris, have to make it < or >)
@symb_args
def profile_cos(
    sig: float,
    xc: float = 0.0,
    xhead: float | None = None,
    var: str = "x",
    sigcut: float | None = None,
    printpars: bool = True,
):
    """normalized cosine profile, with the option to cut the leading part (sigcut: distance from center to cut position)"""

    # Sort out xc and xhead
    if xhead is not None:
        if sigcut is None:
            xc = xhead - np.sqrt(2 * np.pi) * sig
        else:
            xc = xhead - sigcut

    # Go into symbolic mode
    var, sig_s, xc_s, sigcut_s = sm.symbols(
        " ".join([var, "sig_s", "xc_s", "sigcut_s"])
    )
    sig_s = sm.S(sig)
    xc_s = sm.S(xc)

    if sigcut is None:
        frontlimit = sm.sqrt(2 * sm.pi) * sig_s + xc_s
    else:
        sigcut_s = sm.S(sigcut)
        frontlimit = sigcut_s + xc_s

    backlimit = xc_s - sm.sqrt(2 * sm.pi) * sig_s
    cosfactor = sm.sqrt(sm.pi / 2) / sig_s

    def fcos(xvar, sig, xc):
        return 0.5 * (1.0 + sm.cos(cosfactor * (xvar - xc_s)))

    if printpars is True:
        print(
            f"Written formula:\n 0.5*(1.0 + cos({sm.N(cosfactor):.6f} * ({var} - ({sm.N(xc_s):.6f})))) for {var} >= {sm.N(backlimit):.6f} and {var} <= {sm.N(frontlimit):.6f}"
        )

    return (
        sm.Piecewise(
            (0.0, var < backlimit),
            (0.0, var > frontlimit),
            (fcos(var, sig_s, xc_s), True),
        ),
        var,
    )


# TODO: write written formula without '<=' symbols (I think this is a problem for osiris, have to make it < or >)
@symb_args
def profile_parab(
    sig: float,
    xc: float = 0.0,
    xhead: float | None = None,
    var: str = "x",
    sigcut: float | None = None,
    printpars: bool = True,
):
    """normalized parabolic profile, with the option to cut the leading part (sigcut: distance from center to cut position)"""

    # Sort out xc and xhead
    if xhead is not None:
        if sigcut is None:
            xc = xhead - np.sqrt(5) * sig
        else:
            xc = xhead - sigcut

    # Go into symbolic mode
    var, sig_s, xc_s, sigcut_s = sm.symbols(
        " ".join([var, "sig_s", "xc_s", "sigcut_s"])
    )
    sig_s = sm.S(sig)
    xc_s = sm.S(xc)

    if sigcut is None:
        frontlimit = sm.sqrt(5) * sig_s + xc_s
    else:
        sigcut_s = sm.S(sigcut)
        frontlimit = sigcut_s + xc_s

    backlimit = xc_s - sm.sqrt(5) * sig_s
    sigquot = 5 * sig_s**2

    def fparab(xvar, sig, xc):
        return 1 - (xvar - xc_s) ** 2 / sigquot

    if printpars is True:
        print(
            f"Written formula:\n 1.0 - ({var} - ({sm.N(xc_s):.6f}))^2 / {sm.N(sigquot):.6f} for {var} >= {sm.N(backlimit):.6f} and {var} <= {sm.N(frontlimit):.6f}"
        )

    return (
        sm.Piecewise(
            (0.0, var < backlimit),
            (0.0, var > frontlimit),
            (fparab(var, sig_s, xc_s), True),
        ),
        var,
    )
