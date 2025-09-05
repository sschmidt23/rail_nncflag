import os
import pytest
from rail.core.stage import RailStage
from rail.core.data import DataStore, TableHandle, QPHandle
from rail.utils.path_utils import RAILDIR
from rail.estimation.algos import nnc_flag

import scipy.special
sci_ver_str = scipy.__version__.split('.')

parquetdata = "./tests/validation_10gal.pq"
fitsdata = "./tests/validation_10gal.fits"
traindata = os.path.join(RAILDIR, 'rail/examples_data/testdata/training_100gal.hdf5')
validdata = os.path.join(RAILDIR, 'rail/examples_data/testdata/validation_10gal.hdf5')
trainens = "./tests/training_100gal_pdfs.hdf5"
valens = "./tests/validation_10gal_pdfs.hdf5"

DS = RailStage.data_store
DS.__class__.allow_overwrite = True


@pytest.mark.parametrize(
    "include_odds, gname",
    [(True, "wodds"),
     (False, "noodds")])
def test_bpz_lite(include_odds, gname):

    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'trainfrac': 0.75,
                         'hdf5_groupname': 'photometry',
                         'include_odds': include_odds,
                         'zphot_name': 'zmode',
                         'model': f'testmodel_nnc_{gname}.pkl'}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'ref_band': 'mag_i_lsst',
                         'hdf5_groupname': 'photometry',
                         'zphot_name': 'zmode',
                         'model': f'testmodel_nnc_{gname}.pkl'}
    train_algo = nnc_flag.NNFlagInformer
    pz_algo = nnc_flag.NNFlagEstimator
    DS.clear()
    training_data = DS.read_file("training_data", TableHandle, traindata)
    validation_data = DS.read_file("validation_data", TableHandle, validdata)
    train_pdfs = DS.read_file("train_pdf", QPHandle, trainens)
    val_pdfs = DS.read_file("val_pdf", QPHandle, valens)
    train_pz = train_algo.make_stage(name=f"{gname}_train", **train_config_dict)
    train_pz.inform(training_data, train_pdfs)
    pz = pz_algo.make_stage(name=f"nnc_estimate_{gname}", **estim_config_dict)
    estim = pz.estimate(validation_data, val_pdfs)
    os.remove(pz.get_output(pz.get_aliased_tag("output"), final_name=True))
    model_file = estim_config_dict.get("model", "None")
    if model_file != "None":
        try:
            os.remove(model_file)
        except FileNotFoundError:  # pragma: no cover
            pass
