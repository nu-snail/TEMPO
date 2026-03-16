#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/../dataset"
TARGET_DIR="${ROOT_DIR}/runtime/data"

mkdir -p \
  "${TARGET_DIR}/train/stock_rankings" \
  "${TARGET_DIR}/train/salary_prediction" \
  "${TARGET_DIR}/train/legal_cases" \
  "${TARGET_DIR}/eval" \
  "${TARGET_DIR}/control"

cp "${SOURCE_DIR}/train/stock_rankings/stock_rankings_train_combined.json" \
  "${TARGET_DIR}/train/stock_rankings/"
cp "${SOURCE_DIR}/train/salary_prediction/salary_prediction_train_combined.csv" \
  "${TARGET_DIR}/train/salary_prediction/"
cp "${SOURCE_DIR}/train/legal_cases/legal_cases_train.json" \
  "${TARGET_DIR}/train/legal_cases/"

cp "${SOURCE_DIR}/evaluation/stock_rankings.json" \
  "${TARGET_DIR}/eval/stock_rankings.json"
cp "${SOURCE_DIR}/evaluation/nba_salaries.csv" \
  "${TARGET_DIR}/eval/nba_salaries.csv"
cp "${SOURCE_DIR}/evaluation/legal_cases.json" \
  "${TARGET_DIR}/eval/legal_cases.json"

# Control sets
for f in stock_rankings.json salary_transactions.csv legal_cases.json; do
  src="${SOURCE_DIR}/control/${f}"
  if [ -f "$src" ]; then
    cp "$src" "${TARGET_DIR}/control/"
  fi
done

echo "Data copied into ${TARGET_DIR} (train + eval + control)"
