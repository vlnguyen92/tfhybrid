#! /usr/bin/env bash
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TOPLEVEL=$(dirname $MYDIR)

pushd "$TOPLEVEL" >/dev/null
python -m hybridalex.datasets.download_and_convert_flowers "$@"
