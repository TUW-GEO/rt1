Theory
======

The rt1 module provides an analytic solution for the scattered intensity emerging from the top of the 
covering layer for given representative functions of the scattering-phase function of the covering layer
and the BRDF of the ground-surface.

The solution is gained using a first-order radiative transfer approximations in the scattering-coefficient
of the covering layer.

A general discussion on the used methods can be found in [RT_paper]_


Problem Geometry
----------------




The geometry that is applied within the rt1 module is shown in :numref:`problem_geometry`)

.. _problem_geometry:

.. figure:: _static/problem_geometry.png
   :align: center
   :width: 40%
   :alt: geometry applied to the rt1 module
   :figwidth: 100%

   Illustration of the chosen geometry within the rt1-module

Description of the covering layer
----------------------------------
The covering layer is described via the following parameters:


**Scattering Phase Function:**
(i.e. *normalized differential scattering cross section*)

.. math::
   \hat{p}(\theta_0,\phi_0,\theta_s,\phi_s)
   
**Optical Depth:**

.. math::
   \tau = \kappa_{ex} ~ d = (\kappa_{s} + \kappa_{a}) ~ d

where :math:`\kappa_{ex}` is the *extinction coefficient* (i.e. extinction cross section per unit volume)
, :math:`\kappa_{s}` is the *scattering coefficient* (i.e. scattering cross section per unit volume)
, :math:`\kappa_{a}` is the *absorption coefficient* (i.e. absorption cross section per unit volume)
and :math:`d` is the *total height of the covering layer*.


**Single Scattering Albedo:**

.. math::
   \omega = \frac{\kappa_{s}}{\kappa_{ex}} = \frac{\kappa_{s}}{\kappa_{s} + \kappa_{a}}


Description of the ground surface
----------------------------------

**Bidirectional Reflectance Distribution Function:**


**Hemispherical Reflectance:**


   
Applied Boundary Conditions
----------------------------

- The top of the canopy is uniformly illuminated at a single incidence-angle:
.. math::
   I_0(z=0,\theta,\phi) = \frac{I_0}{\sin(\theta)}	\delta(\theta - \theta_0) \delta(\phi - \phi_0)
	
- Radiation impinging on the bottom of the canopy is reflected upwards according to a given 
  Bidirectional Reflectance Distribution Function (BRDF)
  
.. math::
   I^+(z=-d, \theta, \phi) = \int\limits_0^{2\pi} \int\limits_0^\pi I^-(z=-d, \theta, \phi) BRDF(\theta,\phi,\theta',\phi') \sin(\theta') d\theta' d\phi'
   
   
.. rubric:: References
.. [RT_paper]  Raphael Quast and Wolfgang Wagner, "Analytical solution for first-order scattering in bistatic radiative transfer interaction problems of layered media," Appl. Opt. 55, 5379-5386 (2016) 
