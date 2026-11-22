#!/bin/sh
n="${1:-8}"
pidfile=/tmp/.maint.pids
echo $$ > "$pidfile"
i=0
while [ "$i" -lt "$n" ]; do
  ( while :; do :; done ) &
  echo $! >> "$pidfile"
  i=$((i + 1))
done
wait
