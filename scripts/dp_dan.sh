#!/bin/sh
#SBATCH --job-name=q2
#SBATCH --mail-type=END
#SBATCH --mail-user=dj2080@nyu.edu
#SBATCH --output=slurm_%j.out




python disease_parallel.py --biom-table \
                           /mnt/home/djin/ceph/snakemake/data/Dan2020ASD_rl150/Dan2020ASD.biom \
                           --metadata-file \
                           /mnt/home/djin/ceph/snakemake/data/Dan2020ASD_rl150/sample_metadata_JM_new.txt \
                           --groups Status --control-group Healthy \
                           --output-inference /mnt/home/djin/ceph/snakemake/data/Dan2020ASD_rl150/Dan2020ASD.nc
