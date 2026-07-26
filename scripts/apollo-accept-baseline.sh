#!/bin/sh
set -eu

if [ "$#" -ne 0 ]; then
    echo "apollo-accept-baseline does not accept arguments" >&2
    exit 2
fi

exec /usr/local/libexec/apollo-accept-gate
