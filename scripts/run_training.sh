# /bin/bash


CHARA=$1
CHARA_STEM="${CHARA##*/}"
CHARA_STEM="${CHARA_STEM%%.*}"

for f in "$2/*"; do
  F_STEM="${f##*/}"
  F_STEM="${F_STEM%%.*}"
  echo "Running config: $f"
  python train.py -o "$3/$F_STEM-$CHARA_STEM.json" "$CHARA" "$f" &
done