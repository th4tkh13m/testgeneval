#!/bin/bash
echo "Starting evaluate_instance.py..."
pip install coverage
python -m swebench_docker.evaluate_instance
