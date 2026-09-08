#!/bin/bash
# Submit one session to a SLURM cluster. A template, like the session configs:
# copy it, edit the marked block for your cluster, keep the rest.
#
#   sbatch submit_pipeline.sh configs/Athos_2026_08_13.yaml
#   sbatch submit_pipeline.sh configs/Athos_2026_08_13.yaml export
#
# The second argument picks the pipeline: "sort" (default, needs the GPU) or
# "export" (needs neither GPU nor sorter -- see "Which half to run" below).
#
# It lives beside the two drivers because it is what you run -- the same reason
# they are not buried in tools/. It adds no capability: only what a batch node
# needs around them, which is a GPU allocation, an environment that activates in
# a non-interactive shell, and a pre-flight check.

# ---------- edit for your cluster ----------
#SBATCH --job-name=kilosort
#SBATCH --partition=gpu             # your GPU partition
#SBATCH --gres=gpu:1                # one GPU is what Kilosort uses
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                   # Kilosort streams the file; this is headroom
#SBATCH --time=12:00:00             # a long session sorts for hours
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
# #SBATCH --account=your_account    # uncomment if your cluster requires one

CONDA_ENV="${SPIKESORTING_ENV:-kilosort4}"   # or a full prefix path
# module load cuda/12.8                      # uncomment if your cluster uses modules
# ---------- end of the edit block ----------

set -euo pipefail

CONFIG="${1:?usage: sbatch submit_pipeline.sh <session.yaml> [sort|export]}"
STAGE="${2:-sort}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Argument checks first: they are free, and a typo should not cost the seconds
# that activating an environment and checking a share take.
case "${STAGE}" in
    sort)   DRIVER="run_sorting_pipeline.py" ;;
    export) DRIVER="run_exporting_pipeline.py" ;;
    *)      echo "unknown stage '${STAGE}': expected 'sort' or 'export'" >&2; exit 2 ;;
esac
if [[ ! -f "${CONFIG}" ]]; then
    echo "session config not found: ${CONFIG}" >&2
    exit 2
fi

# Kilosort's scratch comes from cache_dir in configs/machines/hpc.yaml, which is
# /tmp/kilosort_cache -- node-local on a compute node, which is what it must be.
# There is no environment override for it: if this cluster hands each job a
# private directory instead (/scratch/$SLURM_JOB_ID and the like), point that key
# at a path under it rather than trying to set it from here.

# Activate without assuming `conda activate` works in a non-interactive shell:
# on many clusters it is a function that only exists after conda's shell hook.
if [[ "${CONDA_ENV}" == /* ]]; then
    export PATH="${CONDA_ENV}/bin:${PATH}"
else
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
    else
        echo "conda not on PATH and CONDA_ENV is not a prefix path: ${CONDA_ENV}" >&2
        exit 2
    fi
fi

echo "host      : $(hostname)"
echo "python    : $(command -v python)"
echo "config    : ${CONFIG}"
echo "stage     : ${STAGE}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

# Pre-flight. Exits non-zero on anything required, so a broken environment or a
# missing input costs seconds here instead of failing hours into the sort.
python "${REPO}/tools/check_env.py" --config "${CONFIG}" --machine hpc

exec python "${REPO}/${DRIVER}" --config "${CONFIG}" --machine hpc

# ===========================================================================
# Notes
# ===========================================================================
#
# Which half to run
#   Sorting is the only stage that needs a GPU, and it is the reason to be here.
#   The exporting half needs neither GPU nor sorter, so it is usually faster to
#   run on the rig -- and it *should* run there when CatGT/TPrime are wanted,
#   since this cluster has neither and the pipeline silently falls back to its
#   own detectors and a least-squares fit.
#
#   If you do run the export here, drop the GPU line above (--gres) so the job
#   schedules on a normal node, and consider `export_figures: false` in the
#   session: the per-unit figures dominate that stage's runtime.
#
# Paths
#   The session file carries every data path, so it is machine-specific by
#   design. A session sorted here needs a copy whose `roots:` point at the
#   cluster's mount rather than the rig's Z: drive.
#
# The Blackrock binary
#   A .ns6 is transformed into a flat binary *beside the recording*, not in
#   scratch, because params.py points at it and re-sorts reuse it. That means
#   this job writes several GB back to the share, and fails outright if the
#   share is mounted read-only on the compute nodes. Check that before the first
#   run.
