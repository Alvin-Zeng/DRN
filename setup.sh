conda install ipython --yes

pip install ninja yacs cython matplotlib tqdm

# give the instructions for CUDA 10.0
conda install -c pytorch torchvision cudatoolkit=10.0 --yes

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/tianzhi0549/FCOS.git
cd FCOS
python setup.py build develop --no-deps

cd $INSTALL_DIR
cp -r $INSTALL_DIR/FCOS/fcos_core $INSTALL_DIR
rm -rf $INSTALL_DIR/FCOS
rm -rf $INSTALL_DIR/cocoapi

unset INSTALL_DIR

