#!/bin/bash
# Beside a chain block (chain.sbatch): each finished weight broadcast of this run becomes the
# chain's latest weights - hard links into $CHAIN/weights/wNNNN (NNNN = STEP0 + the run's
# step) plus the base config/tokenizer files - then every older weights dir is deleted except
# KEEP_WEIGHTS: the dir this block's run was launched from (its --model.name), which parts of
# the run reload the tokenizer/config from mid-block. Needs OUT (the run's output dir), CHAIN,
# STEP0; KEEP_WEIGHTS optional.
set -u
C=$CHAIN W=$CHAIN/weights saved=0
while true; do
  for b in "$OUT"/run/*/broadcasts; do [ -d "$b" ] || continue
  for n in $(ls "$b" | sed -n 's/^step_\([0-9]*\)$/\1/p' | sort -n); do
    d=$b/step_$n
    [ "$n" -gt "$saved" ] && [ -f "$d/.finished" ] || continue
    g=$((STEP0 + n)); new=$W/w$(printf %04d $g)
    rm -rf "$new.tmp"; mkdir -p "$new.tmp"
    ln "$d"/*.safetensors "$d"/model.safetensors.index.json "$new.tmp"/ &&
      cp -a "$C"/base/. "$new.tmp"/ && echo $g > "$new.tmp/.global_step" || { rm -rf "$new.tmp"; continue; }
    rm -rf "$new"; mv "$new.tmp" "$new"
    ln -sfn "$new" "$C/latest.tmp" && mv -T "$C/latest.tmp" "$C/latest"
    for old in "$W"/w*; do
      [ "$old" = "$new" ] || [ "$(basename "$old")" = "$(basename "${KEEP_WEIGHTS:-none}")" ] || rm -rf "$old"
    done
    saved=$n
    echo "$(date +%T) saved w$(printf %04d $g) from $d" >> "$C/chain.log"
  done
  done
  sleep 30
done
