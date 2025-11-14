## Submodules and Weights Setup

This project uses git submodules for external models:
- `models/DeblurGANv2` (custom fork, model runs on cpu; figure out the env yourself)
- `models/DeblurDiff` (custom fork, with env YAML)

Follow these steps after cloning the main repo.

### 1) Initialize and update submodules

```bash
# From the repo root (ECE549_deblur_project)
git submodule update --init --recursive
```

If you pulled changes and the submodules didn’t update:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

To pull the latest commits in a submodule (when the team has advanced its fork):

```bash
cd models/DeblurGANv2
git pull origin main
cd ../..
git add models/DeblurGANv2
git commit -m "Bump DeblurGANv2 submodule"

cd models/DeblurDiff
git pull origin main
cd ../..
git add models/DeblurDiff
git commit -m "Bump DeblurDiff submodule"
```

### 2) DeblurDiff environment

The DeblurDiff submodule includes an environment file. Create/activate it:

```bash
cd models/DeblurDiff
conda env create -f environment.deblurdiff.yml
conda activate deblurdiff38
```

If you already have `deblurdiff38`, you can skip the create step.

### 3) Download weights (model.pth)

DeblurDiff weights are not committed. Download the checkpoint referenced in the DeblurDiff README and place it at:
DeblurGANv2 weights (fpn_inception.h5) are also not committed; download them and place them in the folder models/DeblurGANv2
```
ECE549_deblur_project/models/DeblurDiff/model.pth
```


### 4) Quick verification

Run DeblurDiff inference (adjust paths and device as needed):

```bash
cd models/DeblurDiff
python inference.py \
  --model ./model.pth \
  --input ../../datasets/<SET>/blur \
  --output ../../results/DeblurDiff/<SET> \
  --device cuda
# use --device cpu if no GPU
```

Run DeblurGANv2:
# Example batch script in predict.py processes images in .\dataset1\blur and stores results in .\submit
Make sure to create a conda env named "deblurdiff38" because interfaces.py assumes that the conda env exists.

```bash
cd models/DeblurGANv2
python predict.py
```
