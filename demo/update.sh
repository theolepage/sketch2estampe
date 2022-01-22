#!/bin/sh

sudo docker stop sketch2estampe
sudo docker rm sketch2estampe
sudo docker build -t sketch2estampe .
sudo docker run -d -p 5000:5000 -v "$(pwd)"/outputs:/app/outputs --name sketch2estampe sketch2estampe