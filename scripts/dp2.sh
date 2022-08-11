#!/bin/sh
#SBATCH --job-name=q2
#SBATCH --mail-type=END
#SBATCH --mail-user=dj2080@nyu.edu
#SBATCH --output=slurm_%j.out




python disease_parallel.py --biom-table \
                           /mnt/home/djin/ceph/git/q2-differential/q2_differential/tests/data/table36new.biom \
                           --metadata-file \
                           /mnt/home/djin/ceph/git/q2-differential/q2_differential/tests/data/sample_metadata_6.txt \
                           --groups Status --control-group Healthy \
                           --output-inference test2.nc
