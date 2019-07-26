#!/bin/bash
# execute bash command to run gunicorn workers
gunicorn rest_api.api_code:app \
    --bind 0.0.0.0:1020 \
    --workers ${API_WORKERS-2} \
    --timeout 30 \
    --capture-output
