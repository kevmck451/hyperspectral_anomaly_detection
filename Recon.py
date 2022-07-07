#Recon is used for things specfically related to reconnasance:
    #Anomaly Detection Algorithms
    #Algorithm Test using real and synthesizing anomalies
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
from scipy.stats import chi2
from math import sqrt
import numpy as np
import spectral

class Recon:

    def __init__(self):
        print('init')

    def rx_seg(self, hdr_file_path):

        spectral.settings.envi_support_nonlowercase_params = True
        # self.img = spectral.io.envi.open(hdr_file_path)
        self.img = spectral.open_image(hdr_file_path + '.hdr')

        data = self.img.load()
        data = np.where(data < 0, 0, data)
        # wastage bands
        # Water Vapour Absorption Band, Water Vapour Absorption Band, Not Illuminated
        if 105 < self.img.nbands < 225:
            w_bands = list(range(105, 120)) + list(range(150, 171)) + list(range(215, 224))
            data = np.delete(data, w_bands, axis=2)

        inc = 500

        rxvals = None
        for i in range(0, data.shape[0], inc):
            rxrow = None

            for j in range(0, data.shape[1], inc):
                d = data[i:i + inc, j:j + inc, :]
                rxcol = spectral.rx(d)  # spectral.rx(data, window=(5,211))  Local RX
                rxrow = np.append(rxrow, rxcol, axis=1) if rxrow is not None else rxcol
            rxvals = np.vstack((rxvals, rxrow)) if rxvals is not None else rxrow


        nbands = data.shape[-1]
        P = chi2.ppf(0.999, nbands) #0.998

        v = spectral.imshow(self.img, bands=(30, 20, 10), figsize=(12, 6), classes=(1 * (rxvals > P)))
        v.set_display_mode('overlay')
        v.class_alpha = 0.7 #how transparent the overlaid portion is
        plt.title('RX-Seg Algorithm')
        plt.axis('off')
        plt.pause(500)

    # Computes RX anomaly detector scores
    def rx(self, image, ):
        pass

        '''
            The RX anomaly detector produces a detection statistic equal to the 
            squared Mahalanobis distance of a spectrum from a background distribution
            according to
        
            .. math::
        
                y=(x-\mu_b)^T\Sigma^{-1}(x-\mu_b)
        
            where `x` is the pixel spectrum, :math:`\mu_b` is the background
            mean, and :math:`\Sigma` is the background covariance
        
        '''



