# Exploring Dirichlet priors in $\beta$-VAE Inverse Problem - submission repository

This repository contains code, data, and supplementary materials for the paper:
- Exploring Dirichlet priors in $\beta$-VAE Inverse Problem
- Anonymous authors
---

## Getting Started 

### Installation of the environment

#### Using `conda`

We recommend using a `conda` virtual Python environment for installation.
```
conda create --name bhuenv python=3.12.0
```

Activate the new `conda` environment to install the Python packages.

```
conda activate bhuenv
```

Clone the repository
```
git clone https://github.com/anonymous1828/Beta-Dirichlet-VAE-for-BHU.git
```

Change directory and install the required Python packages.
```
cd Beta-Dirichlet-VAE-for-BHU && pip install -r requirements.txt
```

### Download the datasets
### Download Datasets

For this submission, we worked with a generated a Synthetic dataset with various SNR, and 2 real datasets Samson and Urban datasets. The synthetic dataset and Samson is already available in the ```Datasets``` directory.   
Urban is too heavy to be stored on Github, so when running the model for the first time on it, it should download and unzip automatically.   
If it doesn't work because of a broken link, you'll need to follow those steps to obtain it.
1) download the dataset 
   - observations https://rslab.ut.ac.ir/documents/437291/1493656/Urban_R162.mat/24b3640f-ea17-a8cc-6e09-9b2ff22bf8c3?t=1710110006859&download=true
   - 4 endmembers ground truth : https://rslab.ut.ac.ir/documents/437291/1493656/groundTruth_4_end.zip/c03d6f3a-cb26-865d-6d4b-04c4480bdc57?t=1710109903885&download=true
   - 5 endmembers ground truth : https://rslab.ut.ac.ir/documents/437291/1493656/groundTruth_Urban_end5.zip/fe72c2db-d724-b848-92f7-c5fb0d6d1e79?t=1710109956268&download=true
   - 6 endmembers : https://rslab.ut.ac.ir/documents/437291/1493656/groundTruth_Urban_end6.zip/4ae2518a-298b-bdd2-0e28-8ee60c8e3ef1?t=1710109871882&download=true
   If the provided links are also broken, you can download them directly from the website of the Remote Sensing lab of University of Teheran https://rslab.ut.ac.ir/data (in section (9) Urban). You'll have to download the matlab file Urban_R162.mat.
2) Create a directory in ./Datasets/ named Urban**k** where **k** is the number of endmembers (Urban4 for instance) and place the Urban_R162.mat and the groundTruth_Urban_end**k**.zip (for Urban4, your directory must contain Urban_R162.mat and groundTruth_Urban_end4.zip). 
3) The files will be automatically unzip at the first run.

Finally, if the automatic download interrupt when running the program with Urban for the first, you'll have to manually delete the file called Urban**k**, to try again.


### Running the code
#### Quick start: reproducing the results of the paper
On the synthetic dataset, run
```
python main.py --dataset Synth_alpha_4.0_snr_40_n_ems_3 --n_ems 3 --concentration_sets 4 --reg_factor 1e-3 --lr 1e-3 --rec_loss mse --n_epochs 400 
```
Despite the results being obtained within the HySUPP libray (see other repository), you can run:
- for Samson
```
python main.py --dataset Samson --n_ems 3 --concentration_sets 1 --reg_factor 1e-3 --lr 1e-3 --rec_loss sad --n_epochs 400 
```
- for Urban4
```
python main.py --dataset Urban4 --n_ems 4 --concentration_sets 1 --reg_factor 1e-4 --lr 1e-4 --rec_loss sad --n_epochs 300 
```
- for Urban5
```
python main.py --dataset Urban5 --n_ems 5 --concentration_sets 1 --reg_factor 1e-5 --lr 1e-4 --rec_loss sad --n_epochs 100
```
- for Urban6
```
python main.py --dataset Urban6 --n_ems 6 --concentration_sets 1 --reg_factor 1e-6 --lr 1e-4 --rec_loss sad --n_epochs 200 
```


#### Available options
Several options are available and can be displayed by running
```
python main.py --help
```
When running the script, you can specify various options. Below is a breakdown of each available option:

| **Option**       | **Type**  |  **Description** | **Available options** |
|------------------|-----------|------------------|-----------------------|
| `--dataset`      | `str`     | Dataset use to train the model. | `Samson`, `Urban4`, `Urban5`, `Urban6`, `Synth_alpha_4.0_snr_40_n_ems_3`, `Synth_alpha_4.0_snr_30_n_ems_3`, `Synth_alpha_4.0_snr_20_n_ems_3`| 
| `--concentration_sets` | `float` | Value of target parameters of the Dirichlet distribution in the $D_{KL}$. For the experiment, `1` were selected for real datasets, and `4` for synthetic ones. | $\mathbb R_{\geq 0}$ (ex: `1` or `4`) |
| `--reg_factor`  | `float`    | Beta regularization factor applied to regularization loss in total loss computation. | $\mathbb R_{\geq 0}$ (ex: `1e-3`)   |
| `--n_ems`  | `int`     | Number of endmembers. Always required and must correspond to the datasets, therefore set it to 3 for Samson, and the synthetic datasets, and 4, 5, or 6 for Urban4, Urban5, and Urban6 respectively  | 3, 4, 5, 6  |
| `--rec_loss`    | `str`      | Data fidelity loss / Reconstruction loss. To reproduce the experiment on the synthetic dataset and obtain the simplex figure please select the MSE loss. | `mse`, `sad`|
| `--reg_loss`    | `str`      | Regularization loss. The $D_{KL}$ only is available. | `gamma_div` |
| `--epochs`      | `int`      | Number of training epochs       | $[1, ...]$  |
| `--optimizer`   | `str`      | Optimizer. To reproduce experiments, select `custom_rms` | `custom_rms`, `adam`, `sgd` |
| `--lr`          | `float`    | Learning rate for the optimizer | $\mathbb R$ (ex:`1e-3`) |
| `--momentum`    | `float`    | Momentum in the optimizer. Select 0.99 to reproduce the results | $[0, 1]$  |
| `--batch-size`  | `int`      | Batch size                      | `200`       |
| `--M_init`  | `str`      | `he`, `nfindr`, `random`. For experiments, `he` was used. |
| `--data_sampling_random_seed`  | `int`     | Random seed chosen for generating ew datasets. Default is `1234` | `200`       |
| `--random_seed_for_endmembers_selections` | `int` | Random seed chosen for selection of endmembers when generating a new dataset. Default is 26. | `42` |

