# Installs
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install pytorch-lightning
cd FAST-GNN-SAT/PyMiniSolvers/
!make
cd ..

### Generate data
```
To generate random dataset run:

1_generate_problems.sh 

This will create random train, validation and test data.

```
### Train or finetune
```
Model is trained with:

2_train.sh

This will train the model and save he checkpoint. It is the way result in Fig 2 in the paper was obtained. The script runs the experiment with curriculum. To run the experiment without curriculum, uncomment the second line and comment out the first.

```
### Compute cluster centers and test
```
After model is trained run:

3_copmute_cluster_centers.sh

Computes cluster centers for true and false

and

4_test.sh

Will use the trained model to evaluate the result (only for SR(40) for other datasets uncomment the other lines in the script).

```

