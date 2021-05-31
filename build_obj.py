#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:54:16 2020

@author: jlee
"""



import time, sys, os
import h5py
import numpy as np
import scipy
from matplotlib import pyplot as pyplot

import fsps
import sedpy
import prospect
import emcee



def build_obs(filternames, mags, ldist, phot_mask,
              e_mags=None, snr=10.0,    **extras):
    """Build a  dictionary of observational data.

    :param  filternames:
        The names   of the relevant filters,
        in the same order   as the photometric data.
        (data   type: list)

    :param  mags:
        The AB apparent magnitudes of   the object from the relevant filters.
        (data   type: numpy.array)

    :param  e_mags:
        The uncertainties   of mags. If None, SNR will be used to determine the uncertainties.
        (data   type: numpy.array)
        
    :param  ldist:
        The luminosity distance to assume   for translating apparent magnitudes 
        into absolute   magnitudes. [unit: Mpc]
        (data   type: scalar)

    :param  phot_mask:
        IMPORTANT: the mask is *True*   for values that you *want* to fit, 
        and *False* for values you want to ignore.
        (data   type: numpy.array boolean)
        
    :returns obs:
        A   dictionary of observational data to use in the fit.
    """
    from prospect.utils.obsutils import fix_obs
    import  sedpy

    # The obs dictionary, empty for now
    obs = {}

    # Filter objects in the "filters" key of the `obs`  dictionary
    obs["filters"]  = sedpy.observate.load_filters(filternames)

    # Now we store  the measured fluxes for a single object, **in the same order as "filters"**
    # The units of  the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
    obs["maggies"]  = 10**(-0.4*mags)

    # And now we store  the uncertainties (again in units of maggies)
    if  np.sum(e_mags == None):
        obs["maggies_unc"] = (1./snr)   * obs["maggies"]
    else:
        obs["maggies_unc"] = 0.4*np.log(10.0)*obs["maggies"]*e_mags

    # Now we need a mask, which says which  flux values to consider in the likelihood.
    obs["phot_mask"] =  phot_mask

    # This  is an array of effective wavelengths for each of the filters.  
    # It is not necessary,  but it can be useful for plotting so we store it here as a convenience
    obs["phot_wave"] =  np.array([f.wave_effective for f in obs["filters"]])

    # We do not have a  spectrum, so we set some required elements of the obs dictionary to None.
    # (this would be a  vector of vacuum wavelengths in angstroms)
    obs["wavelength"] = None
    # (this would be the spectrum in units  of maggies)
    obs["spectrum"] = None
    # (spectral uncertainties are given here)
    obs['unc']  = None
    # (again, to ignore a particular wavelength set the value of the 
    #   corresponding elemnt    of the mask to *False*)
    obs['mask'] = None

    # This  function ensures all required keys are present in the obs dictionary,
    # adding default values if  necessary
    obs = fix_obs(obs)

    return  obs



def build_model(object_redshift=None, luminosity_distance=0.0, fixed_metallicity=None,
                fixed_dust2=False, add_duste=False, add_neb=False,
                mass_0=1.0e+8, logzsol_0=-0.5, dust2_0=0.05, tage_0=13., tau_0=1.,
                mass_1=1.0e+7, logzsol_1=0.5, dust2_1=0.5, tage_1=5., tau_1=3.,
                mass_2=1.0e+6, logzsol_2=0.1, dust2_2=0.1, tage_2=2., tau_2=1.,
                **extras):
    """Build a  prospect.models.SedModel object
    
    :param  object_redshift: (optional, default: None)
        If given,   produce spectra and observed frame photometry appropriate 
        for this redshift. Otherwise,   the redshift will be zero.
        
    :param  luminosity_distance: (optional, default: 0.0)
        The luminosity distance (in Mpc) for the model.  Spectra and observed   
        frame   (apparent) photometry will be appropriate for this luminosity distance.
        
    :param  fixed_metallicity: (optional, default: None)
        If given,   fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.
    
    :param  fixed_dust2: (optional, default: False)
        If `True`, fix the diffuse dust parameter   to the initially given value.
        
    :param  add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to   the model.

    :param  add_neb: (optional, default: False)
        If `True`, add (fixed) parameters   relevant for nebular emission, and
        turn nebular emission   on.

    :returns model:
        An instance of prospect.models.SedModel
    """
    from prospect.models.sedmodel import SedModel
    from prospect.models.templates  import TemplateLibrary
    from prospect.models import priors

    # --- Get a basic delay-tau SFH parameter set.  ---
    # This  has 5 free parameters:
    #    "mass",    "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #    "zred"=0.1,    "sfh"=4
    # See the python-FSPS documentation for details about most  of these
    # parameters.   Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view  the parameters, their initial values, and the priors in detail.
    model_params =  TemplateLibrary["parametric_sfh"]
    
    # Add lumdist parameter.  If this is not added  then the distance is
    # controlled by the "zred"  parameter and a WMAP9 cosmology.
    if  luminosity_distance > 0:
        model_params["lumdist"] =   {"N": 1, "isfree": False,
                                   "init": luminosity_distance, "units":"Mpc"}

    # Adjust model  initial values (only important for optimization or emcee)
    model_params["zred"]["init"] =  0.0

    model_params["mass"]["init"] =  mass_0
    model_params["logzsol"]["init"] = logzsol_0
    model_params["dust2"]["init"] = dust2_0
    model_params["tage"]["init"] =  tage_0
    model_params["tau"]["init"] = tau_0

    # If we are going to be using emcee, it is  useful to provide an
    # initial scale for the cloud of walkers (the default is 0.1)
    # For dynesty these can be  skipped
    model_params["mass"]["init_disp"] = mass_1
    model_params["mass"]["disp_floor"]  = mass_2

    model_params["logzsol"]["init_disp"] =  logzsol_1
    model_params["logzsol"]["disp_floor"] = logzsol_2

    model_params["dust2"]["init_disp"]  = dust2_1
    model_params["dust2"]["disp_floor"] = dust2_2

    model_params["tage"]["init_disp"] = tage_1
    model_params["tage"]["disp_floor"]  = tage_2

    model_params["tau"]["init_disp"] =  tau_1
    model_params["tau"]["disp_floor"] = tau_2

    # adjust priors
    model_params["mass"]["prior"] = priors.LogUniform(mini=1.0e+7,  maxi=1.0e+12)    
    model_params["logzsol"]["prior"] =  priors.TopHat(mini=-2.0, maxi=0.19)    
    model_params["dust2"]["prior"]  = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tage"]["prior"] = priors.TopHat(mini=0.001, maxi=13.8)
    model_params["tau"]["prior"] =  priors.LogUniform(mini=0.1, maxi=30.0)

    # Change the model  parameter specifications based on some keyword arguments
    if  fixed_metallicity is not None:
        #   make it a fixed parameter
        model_params["logzsol"]["isfree"]   = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]["init"] =   fixed_metallicity

    if  fixed_dust2:
        #   make it a fixed parameter
        model_params["dust2"]["isfree"] =   False
        model_params["dust2"]["init"]   = 0.6    # initially given value

    if  object_redshift is not None:
        #   make sure zred is fixed
        model_params["zred"]["isfree"] = False
        #   And set the value to the object_redshift keyword
        model_params["zred"]["init"] = object_redshift

    if  add_duste:
        #   Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if  add_neb:
        #   Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this  new dictionary of parameter specifications
    model = SedModel(model_params)

    return  model



def build_sps(zcontinuous=1, **extras):
    """
    :param  zcontinuous: 
        A   value of 1 insures that we use interpolation between SSPs to 
        have a continuous   metallicity parameter (`logzsol`)
        See python-FSPS documentation   for details
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return  sps



def build_noise(**extras):
    return  None, None

