#!/usr/bin/env sh
# Scarica un corpus di esempio (Tiny Shakespeare, ~1 MB).
# Qualsiasi file .txt in UTF-8 va bene: Dante, i tuoi appunti, codice sorgente...
set -e
cd "$(dirname "$0")"
curl -sSL -o shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "data/shakespeare.txt pronto ($(wc -c < shakespeare.txt) byte)"
