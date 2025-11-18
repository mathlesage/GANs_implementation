#!/bin/bash
name="metrics_evaluation"
outdir="outputs"
n_gpu=1
export DATA="/projects/m25146/data/"

    echo "Launching test for $name"
    
    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:${n_gpu}
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --account=m25146        # Your project account 
#SBATCH --job-name=gan_train      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log
source venv/bin/activate
# Run your training script
python metrics.py --epochs 10 --lr 0.0002 --batch_size 64 --gpus ${n_gpu} 
EOT

