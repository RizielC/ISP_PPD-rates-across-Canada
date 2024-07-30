#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512MB
#SBATCH --output=ISP_Output_HPC.txt

# Create the output directory if it doesn't exist
mkdir -p HPC_Output

# Load modules
module load python/3.11.5 scipy-stack

#Create and activate virtual environment
virtualenv --no-download ENV
source ENV/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install --no-index seaborn==0.13.2
pip install pandas matplotlib scikit-learn numpy

# Run the Python script
python ppd_prediction_script.py
