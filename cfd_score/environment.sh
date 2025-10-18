conda create -n cfdscore python=3.10 -y
conda activate cfdscore
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python matplotlib numpy accelerate transformers
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/facebookresearch/dinov2.git
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
