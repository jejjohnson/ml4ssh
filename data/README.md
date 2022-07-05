# Data Download

## Datasets

* 1.5 Layer QG Simulations (TODO)
* SSH Data Challenge 2021a
* SSH Data Challenge 2020b (TODO)
* SSH 5 Year Altimetry Tracks (TODO)

---
## Instructions

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
### Option 1: Bash Script

**Run the bash script directly from the command line**

```bash
bash dl_dc21a.sh username password path/to/save/dir
```

---
### Option 2: Python script + `credentials.yaml` (Preferred)

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
