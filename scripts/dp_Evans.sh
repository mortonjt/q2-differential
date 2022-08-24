#!/bin/sh
#SBATCH --job-name=q2
#SBATCH --mail-type=END
#SBATCH --mail-user=dj2080@nyu.edu
#SBATCH --output=slurm_%j.out




python deseq2_parallel.py --biom-table \
                            /mnt/home/djin/ceph/ASD/Evans2022/vsearch/exported-feature-table/feature-table-filtered.biom \
                           --metadata-file \
                            /mnt/home/djin/ceph/ASD/Evans2022/vsearch/exported-feature-table/Metadata_group1_matched.txt \
                           --processes 128 \
                           --groups Genotype --control-group ch16_control \
                           --output-inference /mnt/home/djin/ceph/ASD/Evans2022/vsearch/exported-feature-table/test.nc
