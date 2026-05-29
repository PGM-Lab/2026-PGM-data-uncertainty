# Accounting for Data Uncertainty in Counterfactual Reasoning

This bundle contains the manuscript submitted at _PGM-2026_ and entitled  "Accounting for Data Uncertainty in Counterfactual Reasoning".
The organisation  is the following:

- _examples_: a toy example for running the method proposed in the paper.
- _ctfzeros_: python sources implementing the method.
- _models_: set of structural causal models in UAI format considered in the experimentation.
- _requirements.txt_: code dependencies.


## Setup
First of all, check the Python version. This sources have been coded with the following Python version:


```python
!python --version
```

    Python 3.11.13


Then, install the dependencies in the `requirement.txt` file. The main dependency is the python packege `bcause` (https://github.com/PGM-Lab/bcause).


```python
!pip install --upgrade pip setuptools wheel
!pip install -r ./requirements.txt
!pip install polytope~=0.2.5 --no-deps
```

## Model and data

In this repository, we provide functionality for preprocessing the model and data so they could work we our inference algorithm:


```python
from ctfzeros.prepro import load_and_preprocess
```


```python
filepath = "./models/simple_nparents1_nzr10_zdr00_0.uai"
datapath = "./models/simple_nparents1_nzr10_zdr00_0.csv"

model, data, _, _ = load_and_preprocess(filepath, datapath)
model
```




    <StructuralCausalModel (Y:2,X1:2|Uy:4,Ux1:2), dag=[Uy][Y|Uy:X1][X1|Ux1][Ux1]>




```python
model.draw()
```


    
![png](inference_files/inference_8_0.png)
    



```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 2 columns</p>
</div>



## Counterfactual inference

First, load corresponding modules for using LPID and DC3ID:


```python
from ctfzeros.imprecise_empirical import LPCC_imprecise_empirical, DCCC_imprecise_empirical

```

Set up the LPID inference engine with a perturbation $\epsilon=0.05$. Then calculate the probability of sufficiency $PS(X_1,Y)$:


```python
infLPID = LPCC_imprecise_empirical(model, data, perturbation=0.05)
infLPID.prob_sufficiency("X1", "Y")
```

    WARNING:polytope.solvers:`polytope` failed to import `cvxopt.glpk`.
    WARNING:polytope.solvers:will use `scipy.optimize.linprog`





    [7.724454227344187e-20, 0.4317658471578703]



Similarly, with the divide and conquer approach (DC3ID):


```python
infDC3ID = DCCC_imprecise_empirical(model, data, perturbation=0.05)
infDC3ID.prob_sufficiency("X1", "Y")
```




    [0.0, 0.4317658471578494]



Instead of the interval, we can obtain the list of individual queries:


```python
infDC3ID.set_interval_result(False)
infDC3ID.prob_sufficiency("X1", "Y")
```




    [0.3702430867734918,
     0.2277520565325892,
     0.4317658471578494,
     0.2655972876838508,
     0.0,
     0.0,
     0.0,
     0.0]



Finally, we can do the inference with a reduced number of solutions (`num_runs=5`) which can lead to an approximation:


```python
infDC3ID = DCCC_imprecise_empirical(model, data, perturbation=0.05, num_runs=5)
infDC3ID.set_interval_result(False)
infDC3ID.prob_sufficiency("X1", "Y")
```




    [0.22775205653259195,
     0.2655972876838502,
     0.3702430867734911,
     0.4317658471578494,
     0.0]

