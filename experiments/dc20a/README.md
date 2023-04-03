
# Experiments



## Scripts

---
### Data

This includes all data related scripts including download, preprocessing and cleaning to be ML ready.

---
---
#### Data: Download



> This script will download the simulated NATL60 data and the simulated altimetry observations.

**Example Python Script**:

```bash
python experiments/dc20a/main.py \
    --stage="download" \
    --my_config=experiments/dc20a/configs/local_config.py \
    --dldir=/gpfswork/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2020a/raw
```

**Example Bash Script**:

```bash
bash dl_meom.sh /gpfswork/rech/yrf/commun/data_challenges/dc20a_osse/raw
```


---
#### Data: Preprocess

> This script will preprocess the downloaded data (lightly) to make the altimetry dataset alongtrack.
> It will also clean up some of the labels to ensure everything is "concatenation compatible".

Example script:

```bash
python experiments/dc20a/main.py \
    --stage="preprocess" \
    --my_config=experiments/dc20a/configs/config.py
```

---
#### Data: ML Ready

> This script will grab the preprocessed data and compose them into 'experiments'. These experiments
> are different combinations of which nadir tracks to include. The experiments are
> "nadir1", "nadir4", "swot1nadir1", "swot1nadir5".

Example Script:

```bash
python experiments/dc20a/main.py \
    --stage="ml_ready" \
    --my_config=experiments/dc20a/configs/config.py
```

---
### Training

**Train**:

> This script will do the entire training procedure and logging to weights and biases.


---
**Train More**:

> This script grabs a pretrained model and performs the training procedure. One could continuing training
> or start a new training procedure with a pretrained module.

---
### Inference

> This script will grab an already trained model and predict a field.

**Example script**:

```bash
python experiments/dc20a/main.py \
    --stage="inference" \
    --results_name="/Users/eman/code_projects/logs/saved_data/test_res.nc" \
    --my_config=experiments/dc20a/configs/config_local.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
    --my_config.model.pretrain=True \
    --my_config.model.pretrain_reference="experiment-ckpts:v20" \
    --my_config.model.pretrain_checkpoint="last.ckpt" \
    --my_config.model.pretrain_id="2t4asxkq" \
    --my_config.model.pretrain_entity="ige" \
    --my_config.model.pretrain_project="inr4ssh" \
    --my_config.log.mode="disabled"
```

---
### Metrics

> This script will grab an xarray dataset and do the evaluation procedure. So routines include data consistency,
> cleaning, converting to the same bin. Some metrics include statistical, e.g. RMSE, and spectral, e.g.
> Spatial-Temporal PSD and PSD-Score.

**Example Script**:

In this script, we will load an xarray dataset that was previously used for predictions. We will calculate all of
the metrics necessary based on the predicted dataset and the NATL60 simulations.

```bash
python experiments/dc20a/main.py \
    --stage="metrics" \
    --results_name="/Users/eman/code_projects/logs/saved_data/test_res.nc" \
    --my_config=experiments/dc20a/configs/config_local.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
    --my_config.model.pretrain=True \
    --my_config.model.pretrain_reference="experiment-ckpts:v20" \
    --my_config.model.pretrain_checkpoint="last.ckpt" \
    --my_config.model.pretrain_id="2t4asxkq" \
    --my_config.model.pretrain_entity="ige" \
    --my_config.model.pretrain_project="inr4ssh" \
    --my_config.log.mode="disabled"
```

---
### Viz

**Analysis**:

**Results**:

---
### Local Machine


### JeanZay

```bash
python -u experiments/dc20a/main.py
```

**SRun Job**

```bash
bash experiments/dc20a/scripts/jeanzay/train_v100_swot1nadir5.sh
```

**SBatch Job** (CPU)

```bash
sbatch experiments/dc20a/scripts/jeanzay/jz_cpu_100h.sh
```

## Results


---
### 4 NADIRs

| Method   | µ(RMSE) | σ(RMSE) | λx (degree) | λt (days) | λr (degree) | Notes              | Reference        |
| :------- | ------: | ------: | ----------: | --------: | ----------: | :----------------- | :--------------- |
| DUACS    |    0.92 |    0.01 |        1.42 |     12.13 |             | Covariances DUACS  |                  |
| MIOST    |    0.93 |    0.01 |        1.34 |     10.41 |             | Multiscale Mapping |                  |
| 4DVarNet |    0.94 |    0.01 |        0.83 |      8.03 |             | 4DVarNet Mapping   |                  |
| NerF     |    0.42 |    0.02 |           4 |      2.33 |        1.80 | SIREN              | eval_siren.ipynb |

**Updated**:

| Method |   µ(RMSE)  |   σ(RMSE) |   λx (degree) |   λt (days) |   λr (degree) | Notes   | Reference        |
|:-------|-----------:|----------:|--------------:|------------:|--------------:|:--------|:-----------------|
| DUACS  |   0.915166 | 0.00789674 |        1.5709 |     10.6279 |       1.33194 | GF/GF   | eval_siren.ipynb |
| MIOST  |   0.926897 | 0.00745803 |       1.41409 |     10.0393 |       1.19281 | GF/GF   | eval_siren.ipynb |
| SIREN |   0.781432 |  0.044154 |       2.43671 |     13.7338 |       1.76746 | GF/GF   | eval_siren.ipynb |





---
### 5 NADIRS + SWOT

| Method   | µ(RMSE) | σ(RMSE) | λx (degree) | λt (days) | λr (degree) | Notes              | Reference        |
| :------- | ------: | ------: | ----------: | --------: | ----------: | :----------------- | :--------------- |
| DUACS    |    0.92 |    0.02 |        1.22 |     11.37 |             | Covariances DUACS  |                  |
| MIOST    |    0.94 |    0.01 |        1.18 |     10.33 |             | Multiscale Mapping |                  |
| 4DVarNet |    0.96 |    0.01 |         0.7 |      4.35 |             | 4DVarNet Mapping   |                  |
| SIREN    |    0.42 |    0.02 |           4 |      2.33 |        1.80 | SUBSET             | eval_siren.ipynb |

**Updated**:

| Method |   µ(RMSE)  |   σ(RMSE) |   λx (degree) |   λt (days) |   λr (degree) | Notes   | Reference        |
|:-------|-----------:|----------:|--------------:|------------:|--------------:|:--------|:-----------------|
| DUACS  |   0.920991 | 0.0165399 |       1.24725 |     11.6148 |       1.21587 | GF/GF   | eval_siren.ipynb |



---
## Saved Models

| Model | Dataset | Test |
|

### SIREN Models


| Method  | Layers | Hidden | Dataset | Experiment |     `id`     |       `reference`       | `checkpoint` | `entity` | `project` |
|:-------:|:------:|:------:|:-------:|:----------:|:------------:|:-----------------------:|:------------:|:-------:|:---------:|
|  SIREN  |   6    |  256   | NATL60  |  `nadir1`  |  `299njfhp`  | `experiment-ckpts:v17`  | `last.ckpt`  |  `ige`  | `inr4ssh` |
|  SIREN  |   6    |  256   | NATL60  | `nadir4`   |  `299njfhp`  | `experiment-ckpts:v17`  | `last.ckpt`  | `ige`   | `inr4ssh` |

**Example Config**:

```bash
model.pretrain = True
model.pretrain_reference = "experiment-ckpts:v17"
model.pretrain_checkpoint = "last.ckpt"
model.pretrain_id = "299njfhp"  # ige/inr4ssh/299njfhp
model.pretrain_entity = "ige"
model.pretrain_project = "inr4ssh"
```


---
## Tips n Tricks


### Debugging

There is a lot of data. So to make sure everything works, there are some things we can modify to reduce the size
of the dataset. These include:

* Less Time Steps
* Coarser Evaluation Grid
* Smaller number of epochs

Here is an example script:

```bash
python experiments/dc20a/main.py \
    --stage="train" \
    --my_config=experiments/dc20a/configs/config_local.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
    --my_config.experiment="nadir4" \
    --my_config.trainer.num_epochs=10 \
    --my_config.lr_scheduler.warmup_epochs=1 \
    --my_config.lr_scheduler.max_epochs=100 \
    --my_config.lr_scheduler.eta_min=1e-5 \
    --my_config.preprocess.subset_time.time_max="2012-11-01" \
    --my_config.evaluation.time_max="2012-11-01" \
    --my_config.log.mode="disabled" \
    --my_config.model.hidden_dim=256 \
    --my_config.evaluation.lon_coarsen=5 \
    --my_config.evaluation.lat_coarsen=5
```
