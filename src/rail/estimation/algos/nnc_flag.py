"""
Implementaiton of neural-network quality flag code
originally written by Adam Broussard, adapted by
Irene Moskowitz

The code trains a neural net on photometry, specz,
a measure of photo-z "width", and a measure of how
peaked the PDF is (zconf/ODDS-like), and returns a
flag valued between 0 (likely bad redshift) and 1
(likely good redshift)
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.core.data import DataHandle, ModelHandle, QPHandle, TableHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.utils.path_utils import RAILDIR
from rail.core.common_params import SHARED_PARAMS
from keras.models import Sequential
from keras.layers import Dense
# from keras.models import model_from_yaml
# from keras.models import model_from_json
# from keras.callbacks.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.losses import Huber

default_node_counts = [100, 200, 100, 50, 1]
default_activations = ['selu', 'selu', 'selu', 'selu', 'sigmoid']


class NNFlagInformer(CatInformer):
    """Train the neural network classifier
    """
    name = "NNFlagInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(hdf5_groupname=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          include_odds=Param(bool, False, msg="if True, compute ODDS and include in NN"),
                          nodecounts=Param(list, default_node_counts, msg="number of notes in NN"),
                          splitnum=Param(int, 5, msg="number of neural nets to train"),
                          activations=Param(list, default_activations, msg="list of activation functions for NN"),
                          epochs=Param(int, 1000, msg="max number of training epochs for NN"),
                          acc_cutoff=Param(float, 0.07, msg="boundary value of dz/1+z used for good/bad in NN"),
                          zphot_name=Param(str, "zmode", msg="name of point estimate to grab from PDF ancil data"),
                          trainfrac=Param(float, 0.75, msg="fraction of data to keep train vs validation in model creation"),
                          seed=Param(int, 1234, msg="seed for numpy"),
                          )
    inputs = [('pdfs', QPHandle),
              ('input', TableHandle)]

    outputs = [('model', ModelHandle)]

    def __init__(self, args, **kwargs):
        """Init function, init config stuff
        """
        super().__init__(args, **kwargs)
        self.nodecounts = self.config.nodecounts
        self.activations = self.config.activations
        self.epochs = self.config.epochs
        self.splitnum = self.config.splitnum
        # self.correct_pz = self.config.correct_pz
        # self.data_name = data_name
        self.acc_cutoff = self.config.acc_cutoff
        self.odds = None

    def _compute_colors(self, data, refband, bands):
        nbands = len(bands)
        if self.config.include_odds:
            print("including ODDS")
            npar = nbands + 1
        else:
            npar = nbands
        colordata = np.zeros([npar, self.ngal])
        colordata[0, :] = data[refband]
        for i in range(nbands - 1):
            colordata[i+1, :] = data[bands[i]] - data[bands[i+1]]
        if self.odds is not None:
            colordata[npar - 1, :] = self.odds
        return colordata.T

    def _split_data(self, data, szdata, pzdata, trainfrac, seed):
        """
        make a random partition of the training data into training and
        validation, validation data will be used to determine bump
        thresh and sharpen parameters.
        """
        nobs = data.shape[0]
        ntrain = round(nobs * trainfrac)
        # set a specific seed for reproducibility
        rng = np.random.default_rng(seed=seed)
        perm = rng.permutation(nobs)
        self.features_train = data[perm[:ntrain], :]
        self.trainsz = szdata[perm[:ntrain]]
        self.trainpz = pzdata[perm[:ntrain]]
        self.features_val = data[perm[ntrain:], :]
        self.valsz = szdata[perm[ntrain:]]
        self.valpz = pzdata[perm[ntrain:]]
        # return x_train, x_val, z_train, z_val, p_train, p_val

    def inform(self, input_data, pdf_data):
        """need a custom inform because this class has two inputs"""
        self.set_data("input", input_data)
        self.set_data("pdfs", pdf_data)
        self.run()
        self.finalize()
        return self.get_handle("model")

    def meanreprocess(self, X):
        return (X - self.feature_avgs) / np.sqrt(self.feature_vars)

    def run(self):
        """compute the best fit prior parameters
        """
        if self.config.hdf5_groupname:
            photom_data = self.get_data("input")[self.config.hdf5_groupname]
        else:  # pragma: no cover
            photom_data = self.get_data("input")
        self.ngal = len(photom_data[self.config.ref_band])

        # grab the photoz point estimate from PDF ancil data
        ens = self.get_data("pdfs")

        ancilkeys = ens.ancil.keys()
        if self.config.zphot_name not in ancilkeys:  # pragma: no cover
            raise KeyError(f"zphot_name {self.config.zphot_name} not present in qp ancil keys!")
        else:
            zphot = ens.ancil[self.config.zphot_name].flatten()

        if self.config.include_odds:
            lowz = zphot - .06 * (1. + zphot)
            hiz = zphot + 0.06 * (1 + zphot)
            self.odds = np.array([ens[i].cdf(hiz[i]) - ens[i].cdf(lowz[i]) for i in range(ens.npdf)])
        else:
            self.odds = None

        if self.config.ref_band not in self.config.bands or self.config.ref_band not in photom_data.keys():  # pragma: no cover
            raise KeyError(f"ref_band {self.config.ref_band} not in bands list!")
        else:
            color_data = self._compute_colors(photom_data, self.config.ref_band, self.config.bands)

        if self.config.redshift_col not in photom_data.keys():  # pragma: no cover
            raise KeyError(f"redshift column {self.config.redshift_col} not present in input data!")
        else:
            specz = photom_data[self.config.redshift_col]

        self._split_data(color_data, specz, zphot, self.config.trainfrac, self.config.seed)

        self.is_goodfit_train = (np.abs(self.trainpz - self.trainsz) < (1. + self.trainsz) * self.acc_cutoff)
        self.is_goodfit_val = (np.abs(self.valpz - self.valsz) < (1. + self.valsz) * self.acc_cutoff)

        # Now, actually train the models, stuff from the create_models function in the original code
        self.nnlist = []
        for i in range(self.splitnum):
            xmodel = Sequential()
            xmodel.add(Dense(self.nodecounts[0], activation=self.activations[0],
                             input_shape=(color_data.shape[1],)))
            for xnode, xact in zip(self.nodecounts[1:], self.activations[1:]):
                if xact == 'selu':
                    kernel_init = 'lecun_normal'
                else:
                    kernel_init = 'glorot_uniform'
                xmodel.add(Dense(xnode, activation=xact, kernel_initializer=kernel_init))
            initial_learning_rate = 0.005  # this is hardcoded in original code!
            lr_schedule = ExponentialDecay(initial_learning_rate,
                                           decay_steps=3500,
                                           decay_rate=0.1,
                                           staircase=False)
            xmodel.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])
            self.nnlist.append(xmodel)
        # back to get_models() stuff
        self.feature_avgs = np.average(self.features_train, axis=0)
        self.feature_vars = np.var(self.features_train, axis=0)
        for x, thismodel in enumerate(self.nnlist):
            x_train = self.meanreprocess(self.features_train)
            y_train = self.is_goodfit_train.reshape(-1, 1)
            x_val = self.meanreprocess(self.features_val)
            y_val = self.is_goodfit_val.reshape(-1, 1)

            es = EarlyStopping(patience=25, restore_best_weights=True)
            history = thismodel.fit(x_train, y_train, batch_size=1000, epochs=self.epochs, verbose=0,
                                    validation_data=(x_val, y_val), callbacks=[es])

        predictions = []
        for xmod in self.nnlist:
            predictions.append(xmod.predict(x_val, verbose=1))
        # singlepred = np.average(predictions, axis=0)

        self.model = dict(feature_avgs=self.feature_avgs,
                          feature_vars=self.feature_vars,
                          nnlist=self.nnlist,
                          include_odds=self.config.include_odds)

        self.add_data("model", self.model)


class NNFlagEstimator(CatEstimator):
    """Estimator stage, takes model file that includes a set of trained NNCs
    and calculates a flag for each input PDF to see how likely that PDF is to
    be a good fit.  I'm going to ignore a bunch of the options and just do the
    base computation where it predicts the flag and avearges them, which is
    the default behavior in the example script
    """
    name = "NNFlagEstimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update(bands=SHARED_PARAMS,
                          hdf5_groupname=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          zphot_name=Param(str, "zmode", msg="name for point estimate ancil data to use"),
                          )
    inputs = [('pdfs', QPHandle),
              ('input', TableHandle),
              ('model', ModelHandle)]
    outputs = [('output', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do BPZ specific setup
        """
        super().__init__(args, **kwargs)
        if self.config.ref_band not in self.config.bands:  # pragma: no cover
            raise ValueError("ref_band not present in bands list! ")
        self.odds = None

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        self.modeldict = self.model
        self.knnmodels = self.modeldict['nnlist']
        self.feature_avgs = self.modeldict['feature_avgs']
        self.feature_vars = self.modeldict['feature_vars']
        self.include_odds = self.modeldict['include_odds']
        if self.include_odds:
            print("include_odds set to True in model, will calculate ODDS")

    def meanreprocess(self, X):
        return (X - self.feature_avgs) / np.sqrt(self.feature_vars)

    def _compute_colors(self, data, refband, bands):
        nbands = len(bands)
        if self.odds is not None:
            npar = nbands + 1
        else:
            npar = nbands
        colordata = np.zeros([npar, self.ngal])
        colordata[0, :] = data[refband]
        for i in range(nbands - 1):
            colordata[i+1, :] = data[bands[i]] - data[bands[i+1]]
        if self.odds is not None:
            colordata[npar - 1, :] = self.odds
        return colordata.T

    def estimate(self, input_data, pdf_data):
        self.open_model(**self.config)
        self.set_data("input", input_data)
        self.set_data("pdfs", pdf_data)
        self.validate()
        self.run()
        self.finalize()
        results = self.get_handle("output")
        results.read(force=True)
        return results

    # def _process_chunk(self, start, end, data, first):
    def run(self):
        """
        Run flag calculation
        """

        if self.config.hdf5_groupname:
            photom_data = self.get_data("input")[self.config.hdf5_groupname]
        else:  # pragma: no cover
            photom_data = self.get_data("input")
        self.ngal = len(photom_data[self.config.ref_band])

        if self.include_odds:
            ens = self.get_data("pdfs")
            ancilkeys = ens.ancil.keys()
            if self.config.zphot_name not in ancilkeys:  # pragma: no cover
                raise KeyError(f"zphot_name {self.config.zphot_name} not present in qp ancil keys!")
            else:
                zphot = ens.ancil[self.config.zphot_name].flatten()

            lowz = zphot - .06 * (1. + zphot)
            hiz = zphot + 0.06 * (1 + zphot)
            self.odds = np.array([ens[i].cdf(hiz[i]) - ens[i].cdf(lowz[i]) for i in range(ens.npdf)])
        else:
            self.odds = None

        if self.config.ref_band not in self.config.bands or self.config.ref_band not in photom_data.keys():  # pragma: no cover
            raise KeyError(f"ref_band {self.config.ref_band} not in bands list!")
        else:
            color_data = self._compute_colors(photom_data, self.config.ref_band, self.config.bands)
        white_color_data = self.meanreprocess(color_data)

        predictions = []
        for xmod in self.knnmodels:
            predictions.append(xmod.predict(white_color_data, verbose=1))
        singlepred = np.average(predictions, axis=0)
        outdict = dict(nncflag=singlepred.flatten())
        if self.odds is not None:
            outdict['ODDS'] = np.array(self.odds)
        self.add_data("output", outdict)
