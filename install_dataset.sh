#!/bin/sh

mkdir -p dataset && cd dataset || exit
if [[ ! -e dataDLMI-main ]]; then
    echo "Attempting to download now..."
    curl -O -L "https://github.com/chris-mrn/dataDLMI/archive/refs/heads/main.zip"
    unzip main.zip && rm -f main.zip
fi
echo "the data has been loaded"