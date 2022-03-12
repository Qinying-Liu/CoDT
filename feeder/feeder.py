import numpy as np
import torch

from . import tools


class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6):
        self.data_path = data_path
        self.label_path = label_path
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.load_data()

    def load_data(self, ):
        self.label = np.load(self.label_path).squeeze()
        self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy


class Feeder_dual(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, label_path,
                 shear_amplitude=0.5, temperal_padding_ratio=6,
                 shear_amplitude1=0.5, temperal_padding_ratio1=6):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.shear_amplitude1 = shear_amplitude1
        self.temperal_padding_ratio1 = temperal_padding_ratio1
        self.load_data()

    def load_data(self, ):
        self.label = np.load(self.label_path).squeeze()
        self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        data1 = self._aug(data_numpy)
        data2 = self._aug1(data_numpy)
        return [data1, data2], label, index

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy

    def _aug1(self, data_numpy):
        if self.temperal_padding_ratio1 > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio1)

        if self.shear_amplitude1 > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude1)

        return data_numpy


class Feeder_quadruple(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self,
                 data_path,  label_path,
                 shear_amplitude=0.5, temperal_padding_ratio=6,
                 shear_amplitude1=0.5, temperal_padding_ratio1=6):
        self.data_path = data_path

        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.shear_amplitude1 = shear_amplitude1
        self.temperal_padding_ratio1 = temperal_padding_ratio1
        self.load_data()

    def load_data(self, ):
        self.label = np.load(self.label_path).squeeze()
        self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        data_q = self._aug(data_numpy)
        data_k = self._aug1(data_numpy)
        data_q1 = self._aug(data_numpy)
        data_k1 = self._aug1(data_numpy)
        return [data_q, data_k, data_q1, data_k1], label, index

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy

    def _aug1(self, data_numpy):
        if self.temperal_padding_ratio1 > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio1)

        if self.shear_amplitude1 > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude1)

        return data_numpy