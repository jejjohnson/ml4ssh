# Implicit Neural Representations for Interpolation

**Author**: J. Emmanuel Johnson

---
**Collaborators:**
* [Redouane Lguensat](https://redouanelg.github.io)
* [Julien Le Sommer](https://lesommer.github.io)
* [Ronan Fablet](https://rfablet.github.io)



---
## Overview



**Baseline Methods**: Optimal Interpolation

$$
\mathbf{x}^a = \mathbf{x}^b + \mathbf{K}\left(\mathbf{y}_{obs} - \mathbf{x}^b \right)
$$

**Standard Methods**: Kriging

$$
\mathbf{y} = \boldsymbol{\mu}(\mathbf{x}_\phi) + \mathbf{K}_\phi \left(\mathbf{y}_{obs} -  \boldsymbol{\mu}(\mathbf{X}_\phi)\right)
$$

**Modern Methods**: Implicit Neural Representations

$$
\mathbf{y}_{obs} = \boldsymbol{f}(\mathbf{x}_{\phi};\boldsymbol{\theta})
$$

---
## Experiments

---
### [QG Simulations](experiments/qg/README.md)


#### Challenge

|      Simulated Altimetry Tracks      |       Simulated SSH Field        |
|:------------------------------------:|:--------------------------------:|
| ![Animation](experiments/qg/assets/obs_p_movie.gif) | ![Animation](experiments/qg/assets/p_movie.gif) |

For more information, see the [experiment page](experiments/qg/README.md).


---
### [OSE (Data Challenge 2021a)](experiments/dc21a/README.md)

#### Challenge

|          Altimetry Tracks           |                   SSH Field                   |
|:-----------------------------------:|:---------------------------------------------:|
| ![Animation](experiments/dc21a/assets/movie_obs.gif)  | ![Animation](experiments/dc21a/assets/movie_field_duacs.gif) |

For more information, see the [experiment page](experiments/dc21a/README.md).

---
## Demos


**Image Regression** (Jupyter Notebook)

> A standard image regression problem on a fox image. This is the [same experiment](https://bmild.github.io/fourfeat/) as the demo in [Tancik et al (2020)](https://bmild.github.io/fourfeat/)

**Image Regression + Physics Loss** (Jupyter Notebook) (**TODO**)

> The standard image regression problem with the physics informed loss function, i.e. the Poisson constraint (gradient, laplacian).
> This is the [same experiment]() as the Siren paper [Sitzmann et al (2020)](https://www.vincentsitzmann.com/siren/)

**QG Simulation** (Jupyter Notebook)

> This uses a **subset** of the QG simulations to demonstrate how each of the networks perform. This application is
> useful for training INR as potential mesh-free surrogates.

**QG Simulation + Physics Loss** (Jupyter Notebook)

> This uses a **subset** of the QG simulations to demonstrate how each of the networks perform along with the
> physics-informed QG loss function.


---
## Installation Instructions


### Conda Environment (Preferred)

```bash
conda env create -f environments/torch_linux.yaml
```

### Pip Install (TODO)

[//]: # (```bash)

[//]: # (pip install git+https://)

[//]: # (```)


### Download (TODO)

[//]: # (```bash)

[//]: # (git clone https://)

[//]: # (cd inr4ssh)

[//]: # (pip install .)

[//]: # (```)




---
## Data Download

### Datasets

* 1.5 Layer QG Simulations
  * `94MB`
* SSH Data Challenge 2021a
  * Train/Test Data - `116MB`
  * Results: BASELINE - ~`15MB`; DUACS - ~`4.5MB`
* SSH Data Challenge 2020b (TODO)
* SSH 5 Year Altimetry Tracks (TODO)

---
### Instructions

**Step 1**: Go into data folder

```bash
cd data
```

**Step 2**: Give permissions

```bash
chmod +x dl_dc21a.sh
```

**Step 3**: Download data (bash or python)

See the detailed steps below.

---
#### Option 1: Bash Script

**Run the bash script directly from the command line**

```bash
bash dl_dc21a.sh username password path/to/save/dir
```

---
#### Option 2: Python script + `credentials.yaml` (Preferred)

**Create a `.yaml` file**. You can even append it to your already lon `.yaml` file.

```yaml
aviso:
  username: username
  password: password
```

**Download with the python script**

```bash
python dl_dc21a.py --credentials-file credentials.yaml --save-dir path/to/save/dir
```

---
### Bonus: M1 MacOS Compatible

I have included some environment files for the new M1 MacOS. This is because I personally use an M1 Macbook and I wanted to test out the new [PyTorch M1 compatability](https://pytorch.org/blog/pytorch-1.12-released/#prototype-introducing-accelerated-pytorch-training-on-mac) which makes use of the M1 GPU. I personally found that the training and inference time for using PyTorch are much faster. This coincides with other users experiences (e.g. [here](https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html)) In addition, Anaconda claims that other packages potentially get a [20 % speedup](https://www.anaconda.com/blog/apple-silicon-transition). To install, use the requirement file:

```bash
mamba env create -f environments/torch_macos.yaml
```

**Differences**:
* The training scripts use the `skorch` distribution. This is because it takes advantage of the `M1` GPU and I have seen a substantial speed-up.
* A different environment file, i.e. `torch_macos.yaml`.

---
### Known Bugs

#### Datashader (Mac OS)

I cannot get datashader to work for the M1. But using the new [Anaconda distribution](https://www.anaconda.com/blog/new-release-anaconda-distribution-now-supporting-m1) works fine.

```bash
mamba create -n anaconda
mamba install anaconda=2022.05
```

---
## Acknowledgements

#### Data

* [ocean-data-challenges/2021a_SSH_mapping_OSE](https://github.com/ocean-data-challenges/2021a_SSH_mapping_OSE) - Altimetry SSH datasets

---
#### Discussions

* [Jordi Bolibar](https://jordibolibar.wordpress.com)
* Quentin Favre
* [Jean-Michel Brankart](https://www.ige-grenoble.fr/-Jean-Michel-Brankart-451-)
* Pierre Brasseur

---
#### Code

* [hrkz/torchqg](https://github.com/hrkz/torchqg) - Quasi-geostrophic spectral solver in PyTorch
* [lucidrains/siren-pytorch](https://github.com/lucidrains/siren-pytorch) - Siren PyTorch Model
* [kklemon/gon-pytorch](https://github.com/kklemon/gon-pytorch/blob/master/gon_pytorch/modules.py) - Fourier Features Network Model
* [didriknielsen/survae_flows](https://github.com/didriknielsen/survae_flows/tree/master/survae/nn/layers) - Activation Functions & Conditional Distributions
* [boschresearch/multiplicative-filter-networks](https://github.com/boschresearch/multiplicative-filter-networks) - Multiplicative Filter Networks (Fourier, Gabor) Models
* [vsitzmann/siren](https://github.com/vsitzmann/siren) - simple differential operators
* [boschresearch/torchphysics](https://github.com/boschresearch/torchphysics) - Advanced differential operators
