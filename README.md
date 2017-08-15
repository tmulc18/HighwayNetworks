# Highway Networks

TensorFlow implementation of [Highway Networks](https://arxiv.org/abs/1505.00387). These networks shown to allow training with deeper architectures than traditional networks. 

## Data
All evaluations were done using MNIST which can be downloaded from (here)[https://pjreddie.com/projects/mnist-in-csv/]

The data should be structured like

`
--HighwayNetworks
    --Data
        --MNIST
            --train.csv
            --test.csv
`

## Repository Information
*  Highway Networks Demo.ipynb is notebook that shows how the networks is constructed.
*  utils.py includes data processing functions
*  modules.py has helper functions for build networks
*  models.py builds networks
