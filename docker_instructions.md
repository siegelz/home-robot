1. Build the Docker image locally on your computer:
```
docker build . \
    -f docker/ovmm_baseline.Dockerfile \
    -t ovmm_baseline_submission
    --network host
```

2. Save the Docker image to a tar file (get image ID with `docker images`):
```
docker save 9c27e219663c -o ovmm_baseline_submission.tar
```

3. Upload the tar file to the cluster:
```
scp ovmm_baseline_submission.tar zs0608@adroit.princeton.edu:/scratch/network/ovmm_baseline_submission.tar
```

4. Change Apptainer cache in your ~/.bashrc
```
export APPTAINER_CACHEDIR=/scratch/gpfs/$USER/APPTAINER_CACHE
export APPTAINER_TMPDIR=/tmp
```

5. Build the Apptainer on the cluster:
```
mkdir -p /scratch/network/zs0608/APPTAINER_TMP
apptainer build --tmpdir /scratch/network/zs0608/APPTAINER_TMP ovmm_baseline_submission.sif docker-archive://ovmm_baseline_submission.tar
```

6. Run the Apptainer via Slurm:
```
sbatch run_ovmm_eval.slurm
```