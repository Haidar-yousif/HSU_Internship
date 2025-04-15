import os
import shutil
import zipfile
import rarfile as rf
from tqdm import tqdm
import warnings

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

import numpy as np
from utils.data.util_fn import open_file


DATASETS_CONFIG = {
    "Cuprite": {
            "urls": [
                "https://rslab.ut.ac.ir/documents/81960329/82034936/CupriteS1_R188.mat",
                "https://rslab.ut.ac.ir/documents/81960329/82034936/groundTruth_Cuprite_end12.zip"
            ],
            "img": "CupriteS1_R188.mat",
            "gt": "groundTruth_Cuprite_end12.mat",
            "zip": "groundTruth_Cuprite_end12.zip"
        },
    "Samson": {
            "urls": [
                "https://rslab.ut.ac.ir/documents/81960329/82034930/Data_Matlab.rar",
                "https://rslab.ut.ac.ir/documents/81960329/82034930/GroundTruth.zip"
                ],
            "rar": "Data_Matlab.rar",
            "zip": "GroundTruth.zip",
            "img": "samson_1.mat",
            "gt": "end3.mat"
        },
    "JasperRidge": {
            "urls": [
                "https://rslab.ut.ac.ir/documents/81960329/82034928/jasperRidge2_R198.mat",
                "https://rslab.ut.ac.ir/documents/81960329/82034928/GroundTruth.zip"
            ],
            "zip": "GroundTruth.zip", 
            "img": "jasperRidge2_R198.mat",
            "gt": "end4.mat",
        },
    "Urban4": {
        "urls": [
            "https://rslab.ut.ac.ir/documents/437291/1493656/Urban_R162.mat",
            "https://rslab.ut.ac.ir/documents/437291/1493656/groundTruth_4_end.zip",
             #"https://rslab.ut.ac.ir/documents/81960329/82034926/Urban_R162.mat",
            #"https://rslab.ut.ac.ir/documents/81960329/82034926/groundTruth_4_end.zip"
        ],
        "zip": "groundTruth_4_end.zip",
        "img": "Urban_R162.mat",
        "gt":  "end4_groundTruth.mat",
    },
    "Urban5": {
        "urls": [
            "https://rslab.ut.ac.ir/documents/437291/1493656/Urban_R162.mat",
            "https://rslab.ut.ac.ir/documents/437291/1493656/groundTruth_Urban_end5.zip",
            #"https://rslab.ut.ac.ir/documents/81960329/82034926/Urban_R162.mat",
            #"https://rslab.ut.ac.ir/documents/81960329/82034926/groundTruth_5_end.zip"
        ],
        "zip": "groundTruth_5_end.zip",
        "img": "Urban_R162.mat",
        "gt":  "end5_groundTruth.mat",
    },
    "Urban6": {
        "urls": [
            "https://rslab.ut.ac.ir/documents/437291/1493656/Urban_R162.mat",
            "https://rslab.ut.ac.ir/documents/437291/1493656/groundTruth_Urban_end6.zip",
            #"https://rslab.ut.ac.ir/documents/81960329/82034926/Urban_R162.mat",
            #"https://rslab.ut.ac.ir/documents/81960329/82034926/groundTruth_6_end.zip"
        ],
        "zip": "groundTruth_6_end.zip",
        "img": "Urban_R162.mat",
        "gt":  "end6_groundTruth.mat",
    },

}

def unzip(folder, filename, remove_zip=False):
    zip_file_name = os.path.join(folder, filename)
    with zipfile.ZipFile(zip_file_name) as file:
        file.extractall(path=folder)
    if remove_zip:
        os.remove(zip_file_name)

def unrar(folder, filename, remove_rar=False):
    rar_file_name = os.path.join(folder, filename)
    with rf.RarFile(rar_file_name) as file:
        file.extractall(path=folder)
    if remove_rar:
            os.remove(rar_file_name)
    else:
        print(f"File {filename} already exists. Skipping.")

class TqdmUpTo(tqdm):
    """
        Provides update_to(n) which uses tqdm.update(delta_n)
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def download_data(dataset, folder):

    print(folder)

    if dataset.get("download", True):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.makedirs(folder)
            for url in dataset["urls"]:
                # download the files
                filename = url.split("/")[-1]
                #print(os.path.join(folder + filename))
                if not os.path.exists(folder + filename):
                    with TqdmUpTo(
                        unit="B",
                        unit_scale=True,
                        miniters=1,
                        desc="Downloading {}".format(filename),
                    ) as t:
                        urlretrieve(url, filename=folder + filename, reporthook=t.update_to)
        elif not os.path.isdir(folder):
            print("WARNING: {} is not downloadable.".format(dataset))


def get_samson(dataset, folder):
    """
        unzip/unrar Samson dataset files and collect 
        all information contained in it.
    """
    rar_dir = dataset["rar"]
    zip_dir = dataset["zip"]

    img_filename = dataset["img"]
    gt_filename = dataset["gt"]

    # unrar and move rar content to folder
    if not os.path.exists(folder + img_filename):
        unrar(folder, rar_dir)
        path_to_rar_content = os.path.join(folder, rar_dir.split('.')[0], dataset["img"])
        shutil.move(path_to_rar_content, folder)
    else:
        print(f"File {img_filename} already exists. Skipping.")

    if not os.path.exists(folder + gt_filename): 
        unzip(folder, zip_dir)
        path_to_zip_content = os.path.join(folder, zip_dir.split('.')[0], dataset["gt"])
        shutil.move(path_to_zip_content, folder)
    else:
        print(f"File {gt_filename} already exists. Skipping.")

    img = open_file(folder + img_filename)
    gt = open_file(folder + gt_filename)

    # spectral library
    SNR = None
    sigma = None
    eps = None
    wavelenghts = None
    # TODO: can we find it?
    max_value   = None

    n_rows      = int(img["nRow"][0][0])
    n_cols      = int(img["nCol"][0][0])
    n_samples   = n_rows * n_cols
    n_bands     = int(img["nBand"][0][0])
    n_sources   = gt["A"].shape[0]

    # print(img.keys())
    # print(gt.keys())

    # Spectra matrix
    Y = np.reshape(img["V"].T, (n_rows, n_cols, n_bands))
    # Abundance matrix
    A = gt["A"] 
    # Endmembers matrix
    M = gt["M"]

    return n_bands, wavelenghts, n_sources, \
             n_samples, SNR, sigma,   \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols


def get_urban(dataset, folder):

    img_filename = dataset["img"]
    gt_filename  = dataset["gt"]

    zip_dir = dataset["zip"]

    if not os.path.exists(folder + gt_filename):
        unzip(folder, zip_dir)
        path_to_zip_content = os.path.join(folder, zip_dir.split('.')[0], dataset["gt"])
        print(path_to_zip_content)
        shutil.move(path_to_zip_content, folder)
    else:
        print(f"File {img_filename} already exists. Skipping.")
        print(f"File {gt_filename} already exists. Skipping.")

    img = open_file(folder + img_filename)
    gt = open_file(folder + gt_filename)
    

    # spectral library
    SNR         = None
    sigma       = None
    eps         = None

    n_rows      = int(img["nRow"][0][0])
    n_cols      = int(img["nCol"][0][0])
    n_bands     = int(img["Y"].shape[0])
    wavelenghts = img["SlectBands"]

    max_value   = img["maxValue"][0]
    n_samples   = n_rows * n_cols
    n_sources   = int(gt["nEnd"][0][0])
    
    # Spectra matrix
    Y = img["Y"].T.reshape((n_rows, n_cols, n_bands))  / max_value
    # Abundance matrix
    A = gt["A"]
    # Endmembers matrix
    M = gt["M"]

    return n_bands, wavelenghts, n_sources, \
             n_samples, SNR, sigma,   \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols


def get_jasper_ridge(dataset, folder):
    img_filename = dataset["img"]
    gt_filename  = dataset["gt"]

    zip_dir = dataset["zip"]

    if not os.path.exists(folder + gt_filename):
        unzip(folder, zip_dir)
        path_to_zip_content = os.path.join(folder, zip_dir.split('.')[0], dataset["gt"])
        shutil.move(path_to_zip_content, folder)
    else:
        print(f"File {img_filename} already exists. Skipping.")
        print(f"File {gt_filename} already exists. Skipping.")

    img = open_file(folder + img_filename)
    gt = open_file(folder + gt_filename)

    # spectral library
    SNR         = None
    sigma       = None
    eps         = None

    n_rows      = int(img["nRow"][0][0])
    n_cols      = int(img["nCol"][0][0])
    wavelenghts = img["SlectBands"]
    max_value   = img["maxValue"]
    n_samples   = n_rows * n_cols
    n_sources   = gt["M"].shape[1]
    n_bands     = gt["M"].shape[0]
    
    # Spectra matrix
    Y = img["Y"].T.reshape((n_rows, n_cols, n_bands))
    # Abundance matrix
    A = gt["A"].T
    # Endmembers matrix
    M = gt["M"]
    
    warnings.warn(f"Abundance and endmembers matrix are missing for JasperRidge.")
    return n_bands, wavelenghts, n_sources, \
             n_samples, SNR, sigma,   \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols


def get_cuprite(dataset, folder):

    img_filename = dataset["img"]
    gt_filename  = dataset["gt"]

    if not os.path.exists(folder + gt_filename):
        zip_dir = dataset["zip"]
        unzip(folder, zip_dir)
        path_to_zip_content = os.path.join(folder, zip_dir.split('.')[0], dataset["gt"])
        print(path_to_zip_content)
        shutil.move(path_to_zip_content, folder)
    else:
        print(f"File {gt_filename} already exists. Skipping.")

    if not os.path.exists(folder + img_filename):
        unzip(folder, img_filename)
    else:
        print(f"File {img_filename} already exists. Skipping.")
        
    data = open_file(folder + img_filename)

    gt = open_file(folder + gt_filename)
    n_sources = gt["nEnd"]
    n_bands = gt["slctBnds"].shape[-1]
    wavelenghts = gt["waveLength"]

    SNR = None
    sigma = None
    eps = None
    max_value = None

    n_samples = 190 * 250
    A = None
    Y = data["Y"].reshape(n_bands, 190, 250)
    M = None
    
    n_rows = Y.shape[1]
    n_cols = Y.shape[2]

    warnings.warn(f"Abundance and endmembers matrix are missing for Cuprite.")

    return n_bands, wavelenghts, n_sources, \
             n_samples, SNR, sigma,   \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """
    This function is an adaptation of the function from 
    https://github.com/nshaud/DeepHyperX/tree/master
    Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        data: numpy array (size n_bands by n_materials)
            library of materials used to sample the endmembers
        n_bands: int
            spectral dimension
        n_materials: int
            number of materials spectra in the library
        wavelengths: numpy array (size n_bands)
            wavelengths
        n_sources: int
            number of endmembers
        distr: Instance of the class Dirichlet
            abundance distribution (mixture of Dirichlet distributions)
        n_samples: int
            number of samples
        n_outliers: int
            number of outliers
        spread: float
            constraint violation for the outliers (default: 0.1)
        SNR: float
            signal-to-noise ratio (default: 40)
        sigma: float
            standard deviation of the Gaussian noise
        eps: numpy array (size n_bands by n_samples)
            noise matrix
        M: numpy array (size n_bands by n_sources)
            endmembers matrix
        A: numpy array (size n_samples by n_sources)
            abundances matrix
        Y: numpy array (size n_rows, n_cols, n_bands)
            spectra matrix
    """
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    else:
        dataset = datasets[dataset_name]
        folder = target_folder + dataset.get("folder", dataset_name + "/")

    # retrieve from the internet
    download_data(dataset, folder)

    if dataset_name == "Samson":
        n_bands, wavelenghts, n_sources, \
             n_samples, SNR, sigma,   \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols             = get_samson(dataset, folder) 

    if dataset_name == "Urban4" or dataset_name == "Urban5" or dataset_name == "Urban6":
        n_bands, wavelenghts, n_sources, \
             n_samples, SNR, sigma,   \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols             = get_urban(dataset, folder)

    if dataset_name == "JasperRidge":
        n_bands, wavelenghts, n_sources, \
             n_samples, SNR, sigma,   \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols             = get_jasper_ridge(dataset, folder)

    if dataset_name == "Cuprite":
        n_bands, wavelenghts, n_sources, \
             n_samples, SNR, sigma,   \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols             = get_cuprite(dataset, folder)

    # Building dictionary
    return n_bands, wavelenghts, n_sources, \
             n_samples, SNR, sigma,   \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols


if __name__ == "__main__":
    # unit tests
    print(get_dataset(dataset_name="Samson", target_folder="./Datasets/"))
    print(get_dataset(dataset_name="Urban6", target_folder="./Datasets/"))
    print(get_dataset(dataset_name="Urban4", target_folder="./Datasets/"))
    print(get_dataset(dataset_name="Urban5", target_folder="./Datasets/"))
    print(get_dataset(dataset_name="Cuprite", target_folder="./Datasets/"))

    