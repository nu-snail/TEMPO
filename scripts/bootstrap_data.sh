#!/usr/bin/env bash
# Link the tracked dataset splits into each experiment's runtime/data/.
# Run once after cloning, from the repository root:
#   bash scripts/bootstrap_data.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

link_split() {  # link_split <data_dir> <name> <target>
    local path="$1/$2"
    if [[ -e "${path}" && ! -L "${path}" ]]; then
        echo "  ${path} already exists (real file/dir), leaving as is"
    else
        ln -sfn "$3" "${path}"
    fi
}

for EXP in QualityGatedExperiment CoverageGatedExperiment FullExperiment TRLExperiment; do
    [[ -d "${EXP}" ]] || continue
    DATA_DIR="${EXP}/runtime/data"
    if [[ -L "${DATA_DIR}" ]]; then
        echo "${DATA_DIR} is itself a symlink, skipping"
        continue
    fi
    mkdir -p "${DATA_DIR}"
    echo "${EXP}:"
    link_split "${DATA_DIR}" train   ../../../dataset/train
    link_split "${DATA_DIR}" eval    ../../../dataset/evaluation
    link_split "${DATA_DIR}" control ../../../dataset/control
done
