#!/usr/bin/env bash
set -o errexit
set -o nounset

yeast_csv() {
    local in="$raw_dir/yeast/yeast.data"
    local out="$res_dir/yeast/yeast_conv.csv"
    # ignore first feature (it's a useless name)
    cat > "$out" <<EOF
mcg,gvh,alm,mit,erl,pox,vac,nuc,category
EOF
    tr -s " " < "$in" | cut -d' ' -f2- | sed "s/ /,/g" >> "$out"
}

shuttle_csv() {
    for subset in "trn" "tst"; do
        local in="$raw_dir/shuttle/shuttle.$subset"
        local out="$res_dir/shuttle/shuttle_conv.$subset.csv"
        # ignore first feature (it's a useless time)
        cat > "$out" <<EOF
radFlow,fpvClose,fpvOpen,high,bypass,bpvClose,bpvOpen,category
EOF
        tr -s " " < "$in" | cut -d' ' -f2- | sed "s/ /,/g" >> "$out"
    done
}

main() {
    # https://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
    dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "$dir"
    raw_dir="$dir/raw"
    res_dir="$dir/processed"

    yeast_csv
    ./yeast.py "$res_dir"
    shuttle_csv

}

main "$@"
