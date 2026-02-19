#!/bin/bash
#
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

WRKSPC=$SCRATCH
# everything will be installed in $WRKSPC

ENV_NAME="axonn_nanogpt"
# this is the name of your python venv, change if needed

cd $WRKSPC
echo -e "${RED}Creating Python Environment in $WRKSPC:${GREEN}"
module load pytorch/2.0.1 
# note: there are several pytorch modules on perlmutter
# this command loads pytorch/2.3.1 (default)
# if you want to checkout other versions of the pytorch module:
# module avail pytorch

python -m venv $WRKSPC/$ENV_NAME --system-site-packages
# the --system-site-packages flag ensures that your python venv
# can use the system pytorch.

#Step 1 - activate your venv
source $WRKSPC/$ENV_NAME/bin/activate


echo -e "${RED}Installing Dependencies:${GREEN}"
#Step 2 - install axonn from source
git clone https://github.com/axonn-ai/axonn.git
cd axonn
pip install -e .

#Step 3 - install other packages
pip install numpy transformers datasets tiktoken wandb tqdm


echo -e "${RED}Your Python Environment is ready. To activate it run the following commands in the SAME order:${NC}"
echo -e "${GREEN}module load pytorch/2.0.1${NC}"
echo -e "${GREEN}source $WRKSPC/$ENV_NAME/bin/activate${NC}"
echo ""
echo -e "${NC}"
