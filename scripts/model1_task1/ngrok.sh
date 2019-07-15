#!/bin/bash
nohup ./ngrok http 8082 > ngrok.out 2>&1&
