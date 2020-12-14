#!/bin/bash

python3 L1.py &

python3 L2.py &

python3 R3.py &

python3 R4.py &

sleep 120

sudo killall python3