# Customizing datasets for Cox experiments

You can customize your dataset to use LassoNet on survival analysis. 

## Usage
- Put the input matrix in `my_dataset_x.csv` and outcome matrix in `my_dataset_y.csv` under the `data/` folder. 
- `my_dataset_x.csv`: For N samples and T features, the input matrix should be of size [N, T]. 
- `my_dataset_y.csv`: The outcome matrix should be of size [N, 2] with the first column as time and the second column as event (1 for uncensored and 0 otherwise). 
- You can optionally include the feature name as the header of csv file by setting `skip_header=1` in `cox_regression.py`.
- To run LassoNet using customized datasets:
```python cox_regression.py```

