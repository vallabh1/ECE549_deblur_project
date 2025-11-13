#!/bin/bash
# Simple evaluation script

# Example usage for DeblurGANv2
python src/evaluate.py \
    --model DeblurGANv2 \
    --weights models/DeblurGANv2/fpn_inception.h5 \
    --blur_dir models/DeblurGANv2/dataset/blur \
    --sharp_dir models/DeblurGANv2/dataset/sharp \
    --output_dir results
