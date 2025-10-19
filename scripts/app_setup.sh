# Check if current working directory is OmniPaint
if [ "$(basename $(pwd))" != "OmniPaint" ]; then
    echo "Error: This script must be run from the OmniPaint directory"
    echo "Current directory: $(pwd)"
    echo "Expected directory: OmniPaint"
    exit 1
fi
echo "✓ Running from OmniPaint directory. Setting up SAM2 (Take about 15 minutes to complete)..."

mkdir -p sam2_temp
git clone https://github.com/facebookresearch/sam2.git sam2_temp && cd sam2_temp
pip install -e .
echo "✓ Installed SAM2. Downloading sam2 weights..."
cd ..

mkdir -p checkpoints
cd checkpoints
for model in tiny small base_plus large; do
    if [ ! -f "sam2_hiera_${model}.pt" ]; then
        if ! wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_${model}.pt; then
            echo "Failed to download sam2_hiera_${model}.pt"
            echo "Please download manually from: https://github.com/facebookresearch/sam2"
        fi
    else
        echo "sam2_hiera_${model}.pt already exists, skipping download"
    fi
done

# Download config files if not already present
for model in t s b+ l; do
    if [ ! -f "sam2_hiera_${model}.yaml" ]; then
        if ! wget https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2/sam2_hiera_${model}.yaml; then
            echo "Failed to download sam2_hiera_${model}.yaml"
            echo "Please download manually from: https://github.com/facebookresearch/sam2"
        fi
    else
        echo "sam2_hiera_${model}.yaml already exists, skipping download"
    fi
done

pip install sentencepiece
pip install peft==0.10.0
pip install gradio==5.23.0
pip install gradio_image_annotation
pip install carvekit
pip install opencv-python==4.8.0.74
pip install numpy==1.26.4
pip install diffusers==0.31.0
pip install hydra-core
pip install matplotli

cd ..
python app.py


