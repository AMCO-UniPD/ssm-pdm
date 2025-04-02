#!/bin/bash

user="davide_frizzo"
# user="alberto_sinigaglia"
# user="marina"
interval=1

while true; do
  clear
  ps -u "$user" -o pid=,%cpu= | sort -k 2 -r -n | head -5
  sleep "$interval"
done

