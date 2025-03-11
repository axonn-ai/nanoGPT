#!/bin/bash

WRKSPC="/lustre/orion/csc547/scratch/ssingh37/communication"

cd $WRKSPC
ENV_NAME="comm-venv"

rm ${ENV_NAME}.tar.gz
tar -cvzf ${ENV_NAME}.tar.gz ${ENV_NAME}/
mv ${ENV_NAME} ${ENV_NAME}-backup