#!/usr/bin/env bash
# Downloads the two public datasets into data/ and links every CSV into data/sheets/.
# QRData is CC BY-NC 4.0, so it is fetched here rather than committed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw data/sheets

if [ ! -d data/raw/QRData ]; then
  git clone --depth 1 https://github.com/xxxiaol/QRData data/raw/QRData
fi
if [ ! -d data/raw/QRData/benchmark/sheets ]; then
  unzip -q -o data/raw/QRData/benchmark/data.zip -d data/raw/QRData/benchmark/sheets
fi
if [ ! -f data/raw/hillstrom.csv ]; then
  curl -sL -o data/raw/hillstrom.csv \
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
fi

for f in data/raw/QRData/benchmark/sheets/data/*.csv; do
  ln -sf "$(pwd)/$f" data/sheets/
done
ln -sf "$(pwd)/data/raw/hillstrom.csv" data/sheets/hillstrom.csv
echo "rows in hillstrom.csv: $(($(wc -l < data/raw/hillstrom.csv) - 1))"
echo "csv files in data/sheets: $(ls data/sheets | wc -l)"
