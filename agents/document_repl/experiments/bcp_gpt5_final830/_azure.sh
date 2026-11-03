# Sourced by the launchers in this directory: Azure credentials (AZURE2_*) from AZ_ENV.
set -a; . "${AZ_ENV:?set AZ_ENV}"; set +a
export AZURE2_ENDPOINT="$AZ_EP/" AZURE2_API_KEY="$AZ_KEY" \
       AZURE2_API_VERSION="${AZURE2_API_VERSION:?set AZURE2_API_VERSION}"
export PYTHONPATH=$RLMLIB_PATH${PYTHONPATH:+:$PYTHONPATH}
DEP=${AZURE_DEPLOYMENT_GPT5:?set AZURE_DEPLOYMENT_GPT5}
