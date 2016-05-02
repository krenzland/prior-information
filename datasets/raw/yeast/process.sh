#!/usr/bin/env bash
set -o errexit
set -o nounset

main() {
    # ignore first feature (it's a useless name)
    cat > yeast.csv <<EOF
mcg,gvh,alm,mit,erl,pox,vac,nuc
EOF
    tr -s ' ' < yeast.data | cut -d' ' -f2- | sed "s/ /,/g" >> yeast.csv
}

main
