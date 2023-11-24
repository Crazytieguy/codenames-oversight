# Download latest miniconda.
wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install. -b is used to skip prompt
bash Miniconda3-latest-Linux-x86_64.sh -b

# Activate.
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"

# (optional) Add activation cmd to bashrc so you don't have to run the above every time.
printf '\neval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc

conda create -n debate-env python=3.11 -y
conda activate debate-env

pip install -r requirements.txt
