#!/usr/bin/bash

source .venv/bin/activate
poetry run streamlit run ./src/utils/helpers/streamlit/eval_dog_vs_rover/main.py
