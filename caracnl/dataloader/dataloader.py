import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 20:47:32 2021

@author: Aimé Labbé
"""


import numpy as np
import pathlib
import re

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import walk, path
from scipy.constants import pi
from scipy.io import loadmat
from typing import List

# font = {'size': 9}
# matplotlib.rc('font', **font)

class DataLoader(ABC):
    def __init__(self, base_directory):
        self.base_directory = base_directory

    def _get_files(self, keywords: List[str]) -> dict[str, str]:
        """Return the list of all files containing a specific token"""
        file_dict = {[(key, []) for key in keywords]}

        for (dir_path, dir_names, file_names) in walk(self.base_directory):
            files = file_names
            for f in files:
                # Remove the timestamp and the extension from the filename
                # File format : 'YYYY_MM_DD_HH_MM_ss_ms_<key>.ext'
                #                    1  2  3  4  5  6  7
                file = "_".join(f.split("_")[7:]).split(".")[0]
                for key in keywords:
                    if key in file:
                        file_dict[key].append(path.join(dir_path, f))
        return file_dict

    @staticmethod
    def _files_to_data(file_list, keyword: str) -> np.ndarray:
        """Read a data file and transform its content to complex numbers."""

        output = []
        for file in file_list:
            mat = loadmat(file)
            data = mat[keyword]
            output.append(10 ** (data[:, 1] / 20) * np.exp(1j * data[:, 2] * pi / 180))
        return output

    def _files_to_prms(self, file_list):
        """Read and extract parameters from parameter files"""
        P_VNA = []

        for prms_file in file_list:
            with open(prms_file, 'rb') as f:
                for line in f:
                    if b'Puissance :' in line:
                        P_VNA.append(10 ** (1 / 10 * float(re.findall("[0-9]+.[0-9]+"))))  # dBm -> mW


class WithAmpDataReader(DataLoader):
    data_keywords = ['S21_smi_short', 'S31_smi_short',
                     'S21_smi_50Ohm', 'S31_smi_50Ohm',
                     'S21_smi_open', 'S31_smi_open',
                     'S21_smi', 'S31_smi',
                     ]
    prms_keywords = ['Param_ex']

    def read(self) -> np.ndarray:
        """
        Read and compute the s11 data from an experiment directory.
        The amplifier was used to acquire the data.
        """
        datafile_dict = self._get_files(self.data_keywords)
        prmsfile_dict = self._get_files(self.prms_keywords)

        data = {(key, self._files_to_data(datafile_dict[key], key)) for key in datafile_dict}
        prms = self._files_to_prms(prmsfile_dict[self.prms_keywords[0]])

        # calibration du pont de reflectométrie
        rho1 = data['S31_smi_50Ohm'][0] / data['S21_smi_50Ohm'][0]
        rho2 = data['S31_smi_short'][0] / data['S21_smi_short'][0]
        rho3 = data['S31_smi_open'][0] / data['S21_smi_open'][0]

        Zref = (rho2 + rho3 - 2 * rho1) / (rho2 - rho3)
        G = rho1 + rho2 * (Zref - 1)
        Zinc = rho1 / G

        Zref_moyen = np.mean(Zref)
        Zinc_moyen = np.mean(Zinc)
        # G_moyen=np.mean(G) # Pas une constante (?)
        G_moyen = G

        # Calcul S11
        mat = loadmat(S21FilesOI[0])
        ω = 2 * pi * mat['S21_smi'][:, 0]  # Pulsation [rad. s**-1]

        P_VNA = []
        S11 = []
        for S21_file, S31_file, exp_param_file in zip(S21FilesOI, S31FilesOI, ExpParam_OI):
            mat = loadmat(S21_file)  # incident
            S21 = 10 ** (mat['S21_smi'][:, 1] / 20) * np.exp(1j * mat['S21_smi'][:, 2] * np.pi / 180.)

            mat = loadmat(S31_file)  # reflechi
            S31 = 10 ** (mat['S31_smi'][:, 1] / 20) * np.exp(1j * mat['S31_smi'][:, 2] * np.pi / 180.)

            S11.append((S31 / S21 - G_moyen * Zinc_moyen) / (G_moyen - S31 / S21 * Zref_moyen))

            with open(exp_param_file, 'rb') as f:
                for line in f:
                    if line[:11] == b'Puissance :':
                        P_VNA.append(float(line[12:-5]) + 40)

        P_VNA = 10 ** (0.1 * np.array(P_VNA))
        S11 = np.array(S11)

        return ω, S11, P_VNA


class WithoutAmpDataReader(DataReader):
    def read_from_file(self, directory: pathlib.Directory) -> np.dnarray:
        """
        Read and compute the s11 from an experiment directory.
        No amplifier was used for this experiment.
        """

        # Liste des fichiers
        SubdirOI, FilesOI, ExpParam_OI = [], [], []
        for (dirpath, dirnames, filenames) in walk(directory):
            for dirname in dirnames:
                SubdirOI.append(dirname)
            SubdirOI.sort()

        # Obtenir les paramètres expérimentaux
        for subdir in SubdirOI:
            for (sub_dirpath, sub_dirnames, sub_filenames) in walk(path.join(DirOI, subdir)):
                files = sub_filenames
                for f in files:
                    if f == subdir + '_S11_comp_smi.mat':
                        FilesOI.append(path.join(sub_dirpath, f))
                    if f[-14:] == '_Param_exp.txt':
                        ExpParam_OI.append(path.join(sub_dirpath, f))

        mat = loadmat(FilesOI[0])
        freq = mat['S11_comp_smi'][:, 0]
        ω = freq * 2 * pi

        # Lecture data
        P_VNA = []
        S11 = []
        for data_file, exp_param_file in zip(FilesOI, ExpParam_OI):
            mat = loadmat(data_file)
            r = 10 ** (0.05 * mat['S11_comp_smi'][:, 1])  # Norme [un]
            φ = mat['S11_comp_smi'][:, 2] / 180 * π  # Phase [rad]
            S11.append(r * np.exp(1j * φ))
            with open(exp_param_file, 'rb') as f:
                for line in f:
                    if line[:11] == b'Puissance :':
                        P_VNA.append(10 ** (1 / 10 * float(line[12:-5])))  # dBm -> mW

        S11 = np.array(S11)
        P_VNA = np.array(P_VNA)

        return ω, S11, P_VNA


def load_data(directory: pathlib.Directory) -> np.array:
    pass
