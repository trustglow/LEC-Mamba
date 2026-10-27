import numpy as np
import os.path
from pathlib import Path
import scipy.io as sio


def load_data(name):
    if name == 'IP':
        path = 'Datasets/Indian_pines/Indian_pines_corrected.mat'
        T =sio.loadmat(path)['indian_pines_corrected']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']

##############################################################################
    elif name == 'PU':
        path = 'Datasets/Pavia_University/PaviaU.mat'
        T =sio.loadmat(path)['paviaU']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Pavia_University/PaviaU_gt.mat')['paviaU_gt'] 
##############################################################################

    elif name == 'HOU13':
        path = 'Datasets/Houston13/Houstondata.mat'
        T =sio.loadmat(path)['Houstondata']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Houston13/Houstonlabel.mat')['Houstonlabel'] 

##############################################################################
    elif name == 'KSC':
        path = 'Datasets/KSC/KSC.mat'
        T =sio.loadmat(path)['KSC']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/KSC/KSC_gt.mat')['KSC_gt'] 

##############################################################################
    elif name == 'ZY06':
        path = 'Datasets/ZY102D06/ZY102D06.mat'
        T =sio.loadmat(path)['ZY102D06']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/ZY102D06/ZY102D06_gt.mat')['gt'] 

##############################################################################
    elif name == 'AU':
        path = 'Datasets/Augsburg/data_HS_LR.mat'
        T =sio.loadmat(path)['data_HS_LR']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Augsburg/Augsburg_gt.mat')['gt'] 

##############################################################################
    elif name == 'AUsar':
        path = 'Datasets/Augsburg/data_SAR_HR.mat'
        T =sio.loadmat(path)['data_SAR_HR']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Augsburg/Augsburg_gt.mat')['gt'] 

##############################################################################
    elif name == 'FL_T':
        path = 'Datasets/Flevoland/T_Flevoland_14cls.mat'
        first_read =sio.loadmat(path)['T11']
        T = np.zeros(first_read.shape + (9,), dtype=np.float32)
        T[: ,:, 0]=first_read.real
        del first_read
        T[: ,:, 1]=sio.loadmat(path)['T22'].real
        T[: ,:, 2]=sio.loadmat(path)['T33'].real
        T[: ,:, 3]=sio.loadmat(path)['T12'].real
        T[: ,:, 4]=sio.loadmat(path)['T13'].real
        T[: ,:, 5]=sio.loadmat(path)['T23'].real

        T[: ,:, 6]=sio.loadmat(path)['T12'].imag
        T[: ,:, 7]=sio.loadmat(path)['T13'].imag
        T[: ,:, 8]=sio.loadmat(path)['T23'].imag

        labels = sio.loadmat('Datasets/Flevoland/Flevoland_gt.mat')['gt'] 
##############################################################################
    elif name == 'SF':
        first_read = sio.loadmat('Datasets/san_francisco/SanFrancisco_Coh.mat')['T']
        first_read=first_read.astype(np.complex64)
        T = np.zeros(first_read.shape[:2] + (9,), dtype=np.float32)
        T[: ,:, 0]=first_read[: ,:, 0].real
        T[: ,:, 1]=first_read[: ,:, 3].real
        T[: ,:, 2]=first_read[: ,:, 5].real 
        T[: ,:, 3]=first_read[: ,:, 1].real
        T[: ,:, 4]=first_read[: ,:, 2].real
        T[: ,:, 5]=first_read[: ,:, 4].real

        T[: ,:, 3]=first_read[: ,:, 1].imag
        T[: ,:, 4]=first_read[: ,:, 2].imag
        T[: ,:, 5]=first_read[: ,:, 4].imag
        del first_read 
        labels = sio.loadmat('Datasets/san_francisco/SanFrancisco_gt.mat')['gt'] 

##############################################################################
    elif name == 'ober':
        path = 'Datasets/Oberpfaffenhofen/T_Germany.mat'
        first_read =sio.loadmat(path)['T11']
        T = np.zeros(first_read.shape + (9,), dtype=np.float32)
        T[: ,:, 0]=first_read.real
        del first_read
        T[: ,:, 1]=sio.loadmat(path)['T22'].real
        T[: ,:, 2]=sio.loadmat(path)['T33'].real
        T[: ,:, 3]=sio.loadmat(path)['T12'].real
        T[: ,:, 4]=sio.loadmat(path)['T13'].real
        T[: ,:, 5]=sio.loadmat(path)['T23'].real

        T[: ,:, 6]=sio.loadmat(path)['T12'].imag
        T[: ,:, 7]=sio.loadmat(path)['T13'].imag
        T[: ,:, 8]=sio.loadmat(path)['T23'].imag

        labels = sio.loadmat('Datasets/Oberpfaffenhofen/Label_Germany.mat')['label']

##############################################################################
    elif name == 'BL':
        path = 'Datasets/Berlin/data_HS_LR.mat'
        T =sio.loadmat(path)['data_HS_LR']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Berlin/Berlin_gt.mat')['gt'] 

##############################################################################
    elif name == 'BLsar':
        path = 'Datasets/Berlin/data_SAR_HR.mat'
        T =sio.loadmat(path)['data_SAR_HR']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Berlin/Berlin_gt.mat')['gt'] 
##############################################################################
    elif name == 'MU':
        path = 'Datasets/MUUFL/HSI.mat'
        T =sio.loadmat(path)['HSI']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/MUUFL/muufl_gt.mat')['gt'] 

##############################################################################
    elif name == 'MUsar': ##lidar
        path = 'Datasets/MUUFL/LiDAR.mat'
        T =sio.loadmat(path)['LiDAR']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/MUUFL/muufl_gt.mat')['gt'] 

##############################################################################

    elif name == 'HOU':
        path = 'Datasets/hou/data_HS_LR.mat'
        T =sio.loadmat(path)['data_HS_LR']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/hou/Houstonlabel.mat')['Houstonlabel'].T

##############################################################################
    elif name == 'HOUsar':
        path = 'Datasets/hou/data_MS_HR.mat'
        T =sio.loadmat(path)['data_MS_HR']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/hou/Houstonlabel.mat')['Houstonlabel'].T
##############################################################################
    elif name == 'TR':
        path = 'Datasets/Trento/HSI.mat'
        T =sio.loadmat(path)['HSI']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Trento/GT_Trento.mat')['gt']
##############################################################################
    elif name == 'TRsar':
        path = 'Datasets/Trento/LiDAR.mat'
        T =sio.loadmat(path)['LiDAR']
        T = np.expand_dims(T, axis=-1)
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Trento/GT_Trento.mat')['gt']
##############################################################################
    elif name == 'HC':
        path = 'Datasets/WHU_HanChuan/WHU_HanChuandata.mat'
        T = sio.loadmat(path)['WHU_HanChuandata']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/WHU_HanChuan/WHU_HanChuanlabel.mat')['WHU_HanChuanlabel']
##############################################################################
    elif name == 'BO':
        path = 'Datasets/Botswana/Botswana.mat'
        T = sio.loadmat(path)['Botswana']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Botswana/Botswana_gt.mat')['Botswana_gt']
##############################################################################
    elif name == 'SF-RS2':
        path = 'Datasets/SF-RS2/T_SF.mat'
        first_read = sio.loadmat(path)['T11']
        T = np.zeros(first_read.shape + (9,), dtype=np.float32)
        T[:, :, 0] = first_read.real
        del first_read
        T[:, :, 1] = sio.loadmat(path)['T22'].real
        T[:, :, 2] = sio.loadmat(path)['T33'].real
        T[:, :, 3] = sio.loadmat(path)['T12'].real
        T[:, :, 4] = sio.loadmat(path)['T13'].real
        T[:, :, 5] = sio.loadmat(path)['T23'].real
        T[:, :, 6] = sio.loadmat(path)['T12'].imag
        T[:, :, 7] = sio.loadmat(path)['T13'].imag
        T[:, :, 8] = sio.loadmat(path)['T23'].imag

        labels = sio.loadmat('Datasets/SF-RS2/Label_SF.mat')['label']
##############################################################################
    else:
        print("Incorrect data name")
        
    return T, labels

