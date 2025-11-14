# ECE549 Deblur Project

A simple framework for evaluating and comparing image deblurring models.

## Structure

```
ECE549_deblur_project/
├── models/              # Model repositories (gitignored)
│   └── DeblurGANv2/    # Clone model repos here
├── src/                # Main code
│   ├── interfaces.py   # Model interfaces
│   ├── metrics.py      # Evaluation metrics (PSNR, SSIM, Entropy)
│   └── evaluate.py     # Evaluation script
├── results/            # Output results (gitignored)
├── run_eval.sh         # Example run script
├── requirements.txt    # Dependencies
└── README.md
```

## Setup

```bash
# Install dependencies
conda env create -f environment.yml 

# Models are already in models/ directory
# Add more models by cloning into models/
```

## Usage

Basic evaluation:

```bash
python src/evaluate.py \
    --model DeblurGANv2 \
    --weights models/DeblurGANv2/fpn_inception.h5 \
    --blur_dir path/to/blur \
    --sharp_dir path/to/sharp \
    --output_dir results
```

Or use the example script:

```bash
./run_eval.sh
```

## Quick run with submodules

After setting up and installing both submodules (see `SUBMODULES.md`), place your test images in the `test/` folder and run:

```bash
python src/interfaces.py --model both --conda_env deblurdiff38 --device cuda
# or use CPU:
# python src/interfaces.py --model both --conda_env deblurdiff38 --device cpu
```

Results will be saved under `results/DeblurGANv2/` and `results/DeblurDiff/`.

## Output

Results saved to `results/<model_name>/`:
- `results.json` - Metrics per image and averages
- `triplets/` - Visual comparisons (blur | deblur | sharp)

## Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (higher = better)
- **SSIM**: Structural Similarity (higher = better)
- **Entropy Diff**: Information recovered (higher = better)

## Adding New Models

1. Clone model repo into `models/`
2. Add interface class in [src/interfaces.py](src/interfaces.py)
3. Run evaluation

Example interface:

```python
class YourModel(DeblurModel):
    def load_model(self, weights_path: str):
        # Load your model
        pass

    def predict(self, blurred_image: np.ndarray) -> np.ndarray:
        # Return deblurred image
        pass
```
