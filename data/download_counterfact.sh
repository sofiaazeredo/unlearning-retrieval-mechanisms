#!/usr/bin/env bash
# Downloads CounterFact (Meng et al., 2022), 43 MB, into data/counterfact.json
set -euo pipefail
cd "$(dirname "$0")"
curl -L -o counterfact.json https://rome.baulab.info/data/dsets/counterfact.json
echo "saved $(pwd)/counterfact.json"
