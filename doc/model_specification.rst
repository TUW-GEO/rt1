.. _cha_model_specification:

Model Specification
====================

TBD: what I am missing here is the concept of linear combinations of phase functions

TBD: I also suggest to add a subsection on current limitations, where you could mention for instance that something like peak like delta functions are not represented yet.

TBD: some more cross references to the literature would be good as well, but has lower priority I would say.




Evaluation Geometries
----------------------
.. role:: latex(raw)
   :format: latex

From the general definition of the fn-coefficients :eq:`fn_coef_definition` it is apparent that they are in principle dependent on :math:`\theta_0,\phi_0,\theta_{ex}` and :math:`\phi_{ex}`.
If the series-expansions (:eq:`brdf_expansion` and :eq:`p_expansion`) contain a high number of Legendre-polynomials, the resulting fn-coefficients turn out to be rather lengthy and moreover their evaluation might consume a lot of time.
Since usually one is only interested in an evaluation with respect to a specific (a-priori known) geometry of the measurement-setup, the rt1-module incorporates a parameter that allows specifying the 
geometry at which the results are being evaluated. This generally results in a considerable speedup for the fn-coefficient generation.

The measurement-geometry is defined by the value of the :code:`geometry`-parameter of the RT1-class object:

.. code::

    R = RT1(... , geometry = '????')

The :code:`geometry`-parameter must be a **4-character string** that can take one of the following possibilities:

    - :code:`'mono'` for `Monostatic Evaluation`_
    - any combination of :code:`'f'` and :code:`'v'` for `Bistatic Evaluation`_



To clarify the definitions, the used angles are illustrated in :numref:`evaluation_angles`.

.. _evaluation_angles:

.. figure:: _static/evaluation_angles.png
   :align: center
   :width: 40%
   :alt: image illustrating the definitions of the used angles
   :figwidth: 100%

   Illustration of the used incidence- and exit-angles


Monostatic Evaluation
''''''''''''''''''''''

Monostatic evaluation refers to measurements where both the
transmitter and the receiver are at the same location.

In terms of spherical-coordinate description, this is equal to (see :numref:`evaluation_angles`):

.. math::
    \theta_{ex} &= \theta_0 \\
    \phi_{ex} &= \phi_0 + \pi


Since a monostatic setup drastically simplifies the evaluation of the fn-coefficients,
setting the module exclusively to monostatic evaluation results in a considerable speedup.


The module is set to be evaluated at monostatic geometry by setting:

.. code::

    R = RT1(... , geometry = 'mono')



.. note::
	- If :code:`geometry` is set to :code:`'mono'`, the values of :code:`t_ex` and :code:`p_ex` have no effect on the calculated results since they are automatically set to :code:`t_ex = t_0` and :code:`p_ex = p_0 + Pi`
	- For azimuthally symmetric phase-functions [#]_, the value of :code:`p_0` has no effect
	  on the calculated result and the best performance will be achieved by setting :code:`p_0 = 0.`


.. [#] This referrs to any phase-function whose generalized scattering angle parameters satisfy :code:`a[0] = ?, a[1] == a[2] = ?`. The reason for this simplification stems from the fact that the azimuthal dependency of a generalized scattering angle with :code:`a[1] == a[2]` can be expressed in terms of :math:`\cos(\phi_0 - \phi_{ex})^n`. For the monostatic geometry this reduces to :math:`\cos(\pi)^n = 1` independent of the choice of :math:`\phi_0`.


Bistatic Evaluation
''''''''''''''''''''

Any possible bistatic measurement geometry can be chosen by manually selecting the
angles that shall be treated symbolically (i.e. variable), and those that are treated as numerical constants (i.e. fixed).

The individual characters of the :code:`geometry`-string hereby represent
the properties of the incidence- and exit angles (see :numref:`evaluation_angles`) in the order of appearance within the RT1-class element, i.e.:

.. code::

	geometry[0] ...	t_0
	geometry[1] ... t_ex
	geometry[2] ... p_0
	geometry[3] ... p_ex


- The character :code:`'f'` indicates a **fixed** angle
	- The given numerical value of the angle will be used rather than it's
	  symbolic representation to speed up evaluation.
	- The resulting fn-coefficients are only valid for the chosen specific value of the angle.

- The character :code:`'v'` indicates a **variable** angle
	- The angle will be treated symbolically when evaluating the fn-coefficients
	  in order to provide an analytic representation of the interaction-term
	  where the considered angle is treated as a variable.
	- The resulting fn-coefficients can be used for any value of the angle.


As an example, the choice :code:`geometry = 'fvfv'` represents a measurement setup where the surface is illuminated at
constant (polar- and azimuth) incidence-angles and the location of the receiver is variable both in azimuth- and polar direction.

.. note::
	- Whenever a single angle is set *fixed*, the calculated fn-coefficients are only valid for this specific choice!
	- If the chosen scattering-distributions reqire an approximation with a high degree of Legendre-polynomials, evaluating
	  the interaction-contribution with :code:`geometry = 'vvvv'` might take considerable time since the resulting fn-coefficients
	  are very long symbolic expressions.


