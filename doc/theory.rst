Theory
======

The rt1 module provides a method for calculating the scattered radiation from a 
uniformly illuminated rough surface covered by a homogeneous layer of tenuous media.
The following sections are intended to give a general overview of the underlying theory of the
rt1 module. A more general discussion on the derivation of the used results can be found in [QuWa16]_.
Details on how to define the scattering properties of the covering layer and the ground surface
within the rt1-module are given in :ref:`cha_model_specification`.



The utilized theoretical framework is based on applying the Radiative Transfer Equation (RTE) :eq:`RTE` to 
the geometry shown in :numref:`problem_geometry`.

.. math::
   \cos(\theta) \frac{\partial I_f(\textbf{r},\theta,\phi)}{\partial r} = -\kappa_{ex} I_f(\textbf{r},\theta,\phi)
   + \kappa_s \int\limits_{0}^{2\pi}\int\limits_{0}^{\pi/2} I_f(\textbf{r},\theta',\phi') \hat{p}(\theta,\phi,\theta',\phi') \sin(\theta')d\theta' d\phi'
   :label: RTE


The individual variables are hereby defined as follows:

- :math:`\theta` denotes the azimuth angle in a spherical coordinate system
- :math:`\phi` denotes the polar angle in a spherical coordinate system
- :math:`\mathbf{r}` denotes the distance-vector within the covering layer
- :math:`I_f(\textbf{r},\theta,\phi)` denotes the specific intensity at a distance :math:`\mathbf{r}` within the covering layer propagating in direction :math:`(\theta,\phi)`.
- :math:`\kappa_{ex}` denotes the extinction-coefficient (i.e. extinction cross section per unit volume)
- :math:`\hat{p}(\theta,\phi,\theta',\phi')` denotes the scattering phase-function of the constituents of the covering layer

Problem Geometry and Boundary Conditions
-----------------------------------------

.. _problem_geometry:

.. figure:: _static/problem_geometry.png
   :align: center
   :width: 40%
   :alt: geometry applied to the rt1 module
   :figwidth: 100%

   Illustration of the chosen geometry within the rt1-module (adapted from [QuWa16]_)


As shown in :numref:`problem_geometry`, the considered problem geometry is defined as a rough surface covered by a homogeneous layer of a scattering and absorbing medium.

In order to be able to solve the RTE :eq:`RTE`, the boundary-conditions are specified as follows:

- The top of the covering layer is uniformly illuminated at a single incidence-direction:

.. math::
      I_0(z=0,\theta,\phi) = \frac{I_0}{\sin(\theta)}	\delta(\theta - \theta_i) \delta(\phi - \phi_i)

- Radiation impinging at the ground surface is reflected upwards according to its associated Bidirectional Reflectance Distribution Function (BRDF)

.. math::
   I^+(z=-d, \theta, \phi) = \int_{0}^{2\pi} \int_{0}^{\pi} I^-(z=-d, \theta, \phi) BRDF(\theta,\phi,\theta',\phi') \sin(\theta') d\theta' d\phi'

The superscripts :math:`I^\pm` hereby indicate a separation between upwelling :math:`(+)` and downwelling :math:`(-)` intensity.

The additional specifications of the covering layer and the ground surface are summarized as follows:

   
Parameters used to describe the scattering properties of the covering layer
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

**Scattering Phase Function:**
(i.e. *normalized differential scattering cross section*)

.. math::
   \hat{p}(\theta,\phi,\theta',\phi') \qquad \textrm{with} \qquad   \int\limits_0^{2\pi} ~ \int\limits_{0}^{\pi} \hat{p}(\theta,\phi,\theta',\phi') \sin(\theta') d\theta' d\phi' = 1
   
**Optical Depth:**

.. math::
   \tau = \kappa_{ex} ~ d = (\kappa_{s} + \kappa_{a}) ~ d

where :math:`\kappa_{ex}` is the *extinction coefficient* (i.e. extinction cross section per unit volume)
, :math:`\kappa_{s}` is the *scattering coefficient* (i.e. scattering cross section per unit volume)
, :math:`\kappa_{a}` is the *absorption coefficient* (i.e. absorption cross section per unit volume)
and :math:`d` is the *total height of the covering layer*.


**Single Scattering Albedo:**

.. math::
   \omega = \frac{\kappa_{s}}{\kappa_{ex}} = \frac{\kappa_{s}}{\kappa_{s} + \kappa_{a}}   \leq 1


Parameters used to describe the scattering properties of the ground surface
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

**Bidirectional Reflectance Distribution Function:**

.. math::
   BRDF(\theta,\phi,\theta',\phi')  \qquad \textrm{with} \qquad   \int\limits_0^{2\pi} ~ \int\limits_{0}^{\pi/2} BRDF(\theta,\phi,\theta',\phi') \cos(\theta') \sin(\theta') d\theta' d\phi' = R(\theta,\phi) \leq 1

where :math:`R(\theta,\phi)` denotes the **Directional-Hemispherical Reflectance** of the ground surface.


   
First-order solution to the RTE
--------------------------------

Incorporating the above specifications, a solution to the RTE is obtained by assuming that the scattering coefficient :math:`\kappa_s` of the covering layer is small (i.e. :math:`\kappa_s\ll 1`)
Using this assumption, the RTE is expanded into a series with respect to powers of :math:`\kappa_s`, given by:

.. math::
   I^+ = I_{\textrm{surface}} + I_{\textrm{volume}} + I_{\textrm{interaction}} + \mathcal{O}(\kappa_s^2)

where the individual terms can be interpreted as follows:

- :math:`I_{\textrm{surface}}`: radiation scattered once by the ground surface
- :math:`I_{\textrm{volume}}`: radiation scattered once by the covering layer
- :math:`I_{\textrm{interaction}}`: radiation scattered once by the ground surface and once by the covering layer


.. math::
   I_{\textrm{surface}}(\theta_0, \theta_{ex}) = I_0 e^{-\frac{\tau}{\cos(\theta_0)}} ~ e^{-\frac{\tau}{\cos(\theta_{ex})}} \cos(\theta_0) BRDF(\pi-\theta_0, \phi_0, \theta_{ex}, \phi_{ex})

.. math::
   I_{\textrm{volume}}(\theta_0, \theta_{ex}) = I_0 \omega  \frac{\cos(\theta_0)}{\cos(\theta_0) + \cos(\theta_{ex})} \left( 1 - e^{-\frac{\tau}{\cos(\theta_0)}} ~ e^{-\frac{\tau}{\cos(\theta_{ex})}}  \right)    \hat{p}(\pi-\theta_0, \phi_0, \theta_{ex}, \phi_{ex})




   
   
.. rubric:: References
.. [QuWa16]  Raphael Quast and Wolfgang Wagner, "Analytical solution for first-order scattering in bistatic radiative transfer interaction problems of layered media," Appl. Opt. 55, 5379-5386 (2016) 
