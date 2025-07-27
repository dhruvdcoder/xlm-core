#!/bin/bash
# download the omegaconf/hydra sources and the antlr 4.11.1 binary:
mkdir temp_hydra
cd temp_hydra

# Install java
# uncomment if you need compatible java
#wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.tar.gz
#mkdir java
#tar -xvzf jdk-21_linux-x64_bin.tar.gz -C java
#CD=`pwd`
#export JAVA_HOME=$CD/java/jdk-21.0.7/
#export PATH=$JAVA_HOME/bin:$PATH

git clone https://github.com/facebookresearch/hydra
#wget https://www.antlr.org/download/antlr-4.11.1-complete.jar
wget https://www.antlr.org/download/antlr-4.13.2-complete.jar
# replace the old antlr binaries:
rm hydra/build_helpers/bin/antlr-4.9.3-complete.jar
cp antlr-4.13.2-complete.jar hydra/build_helpers/bin/
# edit text files to replace the strings "4.9.3" and "4.9.*" with "4.11.1":
grep -ErlI '4\.9\.[\*3]' | xargs sed -E -i 's/4\.9\.[\*3]/4.13.2/'
# build an sdist / wheel:
pip install build
(cd hydra; python -m build;)
# install the wheels and verify antlr version:
pip install hydra/dist/hydra_core-1.4.0.dev1-py3-none-any.whl
pip list 2>/dev/null | grep antlr4  # check that installed version of antlr4-python3-runtime is 4.13.2

# clean up
cd ..
echo "confirm clean up"
rm -r temp_hydra