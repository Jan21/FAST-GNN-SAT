# Installs
```
conda create --prefix ./cenv python=3.7
pip install pytorch-lightning==1.9.0
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
````
if you want to download satcomp problems use this database tool
```
pip install gbd-tools
```
 
if you run download_satcomp.py it will start downloading (it assumes however that you have a local conda env in ./cenv, as in the instructions above)

For the selsam problems run this first
```
git clone https://github.com/liffiton/PyMiniSolvers.git
cd PyMiniSolvers
make
```
# Generate data
```
sh generate_problems.sh
```
# Train or finetune
```
python IGNN_sat.py
```
hyperparameters of training are defined in the bottom of the file

