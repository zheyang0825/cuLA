set -ex
cwd=`pwd`
cd third_party  && rm -rf cutlass  && git clone https://github.com/NVIDIA/cutlass.git && cd cutlass && git reset --hard 52ae719e
cd $cwd
cd third_party  && rm -rf flash-linear-attention  && git clone https://github.com/fla-org/flash-linear-attention.git && cd flash-linear-attention && git checkout v0.4.2
cd $cwd
#bash install.sh
#python setup.py bdist_wheel
