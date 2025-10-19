CFD Score
=========

CFD is a reference-free metric for evaluating object removal quality, capable of measuring unwanted object hallucination.
This implementation segments contextual regions around the removed area (via SAM), uses DINOv2 features,
and measures disruption between boundary and center regions.

Environment
-----------

```bash
cd cfd_score
bash environment.sh
```

Model assets expected (downloaded automatically) :
- SAM checkpoint at `./checkpoints/sam_vit_h_4b8939.pth`
- DINOv2 model

Inputs
------

- `--input_path`: directory of generated result images (after removal)
- `--mask_path`: directory of corresponding binary masks for removed objects
  - Filenames must match between the two directories


Run
---

```bash
python main.py \
  --input_path <your_input_image_folder_path> \
  --mask_path <your_mask_image_folder_path>
```

Prints the average CFD Score across images. Lower is better.


