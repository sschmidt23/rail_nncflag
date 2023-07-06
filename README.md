# rail_bpz

[![codecov](https://codecov.io/gh/LSSTDESC/rail_bpz/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/rail_bpz)
[![PyPI](https://img.shields.io/pypi/v/bpz?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/pz-rail-bpz/)


RAIL interface to BPZ algorithms via the [DESC_BPZ](https://github.com/LSSTDESC/DESC_BPZ) package implementation (also available via PyPI with `pip install desc-bpz`).  Anyone using BPZ via either rail_bpz or DESC_BPZ should cite both [Benitez (2000)](https://ui.adsabs.harvard.edu/abs/2000ApJ...536..571B/abstract) and [Coe et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006AJ....132..926C/abstract).

As the "lite" name implies, not all features of BPZ are implemented, the main product is the marginalized redshift PDF, which is output for a sample as a `qp` ensemble.  However, several other quantities are computed and stored as "ancillary" data and stored with the ensemble, these are:
- zmode (float): the mode of the marginalized posterior redshift PDF distribution.
- zmean (float): the mean of the marginalized posterior redshift PDF distribution.
- tb (int): the integer index for the "best-fit" SED template **at the redshift mode, `zmode`**.  Note that the best-fit template can be different at different redshifts as the SED observed colors change with redshift, so you **can not** assume this single SED for the full marginalized PDF, it should only be used for the "point estimate" redshift `zmode`.
- todds (float): relating to the comment above on tb, `todds` is a new quantity not included with the original BPZ implementation, it is the fraction of marginalized posterior probability assigned to `tb`.  So, high values of `todds` would mean that no other templates fit well, even at other redshifts, while a lower value of `todds` means that there are alternative fits, either at the same redshift or other redshifts.  If you are wanting to compute physical quantities based on `tb`, a lower value of `todds` would mean that such fits would be missing degenerate SED solutions, and should not be trusted.


