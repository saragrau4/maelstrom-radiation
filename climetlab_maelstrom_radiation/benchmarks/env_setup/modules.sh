#!/bin/bash

ML_SCR=modules_train.sh

ML_COLL=downscaling_unet_v1_0

# check if module collection has been set-up once
#RES=$( { ml restore ${ML_COLL}; } 2<&1 )

#if [[ "${RES}" == *"error"* ]]; then
if [[ 0 == 0 ]]; then  # Restoring from model collection currently throws MPI-setting errors. Thus, this approach is disabled for now.
  
#  echo "%${ML_SCR}: Module collection ${ML_COLL} does not exist and will be set up."

  ml --force purge
  ml Stages/2022

  ml GCCcore/.11.2.0
  ml TensorFlow/2.6.0-CUDA-11.5
  ml GCC/11.2.0  
  ml OpenMPI/4.1.1 
  ml Horovod/0.24.3

#  ml save ${ML_COLL}
#  echo "%${ML_SCR}: Module collection ${ML_COLL} created successfully."
else
  echo "%${ML_SCR}: Module collection ${ML_COLL} already exists and is restored."
  ml restore ${ML_COLL}
fi
