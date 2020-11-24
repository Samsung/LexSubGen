#!/bin/bash

export FILENAME=${BASH_SOURCE[0]}
export ABS_SCRIPT_PATH=$(readlink -f "$FILENAME")
export ENTRY_PATH=$(dirname "$ABS_SCRIPT_PATH")

export APP_DIR_PATH=$ENTRY_PATH/lexsubgen/analyzer/
export FRONTEND_DIR_PATH=$APP_DIR_PATH/frontend
export BACKEND_DIR_PATH=$APP_DIR_PATH/backend
rm -rf $BACKEND_DIR_PATH/static || exit
cd $FRONTEND_DIR_PATH || exit
npm install
npm run build
mv $FRONTEND_DIR_PATH/build/* $BACKEND_DIR_PATH

