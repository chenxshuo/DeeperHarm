#!/bin/bash
# setup venv 
# make sure using python 3.9 or 3.10, seems like 3.12 not working
which python 
python --version
if [[ "$(python --version | cut -d'.' -f2)" -gt 10 ]]; then
    echo "python version too high, we only support 3.9 or 3.10"
    exit 1
fi
uv venv .venv --prompt='red-team-deep-research'
# uv venv .venv -p 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
# moder serving
# web parser client
echo "setup done"