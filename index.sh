#!/bin/bash

echo "Verifying Docker version in the machine"
docker version

docker build .
