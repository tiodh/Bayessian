# Bayessian Classifier


```python
import pandas as pd
import numpy as np
from pprint import pprint
import math
```


```python
data = pd.DataFrame({
    'refund':['yes','no','no','yes','no','no','yes','no','no','no'],
    'marital_status':['single','married','single','married','divorced','married','divorced','single','married','single'],
    'taxable_income':[125,100,70,120,95,60,220,85,75,90],
    'evade':['no','no','no','no','yes','no','no','yes','no','yes']
})
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>refund</th>
      <th>marital_status</th>
      <th>taxable_income</th>
      <th>evade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>yes</td>
      <td>single</td>
      <td>125</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>no</td>
      <td>married</td>
      <td>100</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>no</td>
      <td>single</td>
      <td>70</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>yes</td>
      <td>married</td>
      <td>120</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>no</td>
      <td>divorced</td>
      <td>95</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>no</td>
      <td>married</td>
      <td>60</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>yes</td>
      <td>divorced</td>
      <td>220</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>no</td>
      <td>single</td>
      <td>85</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>no</td>
      <td>married</td>
      <td>75</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>no</td>
      <td>single</td>
      <td>90</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
def norm_pdf(series, value):
    mean = series.mean()
    std = series.std()
    variance = series.var()
    return (1/math.sqrt(variance*2*math.pi))*math.e**(-1*(value-mean)**2/(2*variance))

norm_pdf(data['taxable_income'].where(data['evade']=='yes'), 110)
```




    2.6766045152977078e-05




```python
cols = list(data.columns[:-2])
label_col = data.columns[-1]
labels = data[label_col].unique()

p_label = data.groupby([label_col], as_index=True)[label_col].count().to_dict()
data_number = len(data)
for index, value in p_label.items():
    p_labels[index] = value/data_number
p_label
```




    {'no': 7, 'yes': 3}




```python
posterior = {}
for col in cols:
    uq_values = data[col].unique()
    posterior[col] = {}
    for uq_value in uq_values:
        posterior[col]
        filter1 = data[col]==uq_value
        tmp = {}
        for label in labels:
            filter2 = data[label_col]==label
            p_label_col = (data.where(filter1 & filter2).groupby([col,label_col], as_index=True)[col].count().sum()+1)/(p_label[label]+len(labels))
            tmp[label] = p_label_col*p_label[label]
        posterior[col][uq_value] = tmp
pprint(posterior, width=10)
```

    {'marital_status': {'divorced': {'no': 1.5555555555555554,
                                     'yes': 1.2000000000000002},
                        'married': {'no': 3.8888888888888893,
                                    'yes': 0.6000000000000001},
                        'single': {'no': 2.333333333333333,
                                   'yes': 1.7999999999999998}},
     'refund': {'no': {'no': 3.8888888888888893,
                       'yes': 2.4000000000000004},
                'yes': {'no': 3.1111111111111107,
                        'yes': 0.6000000000000001}}}



```python
col = 'taxable_income'
uq_values = data[col].unique()
posterior[col] = {}
for uq_value in uq_values:
    tmp = {}
    for label in labels:
        series = data
        tmp[label] = norm_pdf(data['taxable_income'].where(data[label_col]==label), uq_value)
    posterior[col][uq_value] = tmp
pprint(posterior, width=10)
```

    {'marital_status': {'divorced': {'no': 1.5555555555555554,
                                     'yes': 1.2000000000000002},
                        'married': {'no': 3.8888888888888893,
                                    'yes': 0.6000000000000001},
                        'single': {'no': 2.333333333333333,
                                   'yes': 1.7999999999999998}},
     'refund': {'no': {'no': 3.8888888888888893,
                       'yes': 2.4000000000000004},
                'yes': {'no': 3.1111111111111107,
                        'yes': 0.6000000000000001}},
     'taxable_income': {60: {'no': 0.004804961450304905,
                             'yes': 1.2151765699646583e-09},
                        70: {'no': 0.005589610036170942,
                             'yes': 2.6766045152977078e-05},
                        75: {'no': 0.005953234789016256,
                             'yes': 0.0008863696823876016},
                        85: {'no': 0.006584873139785386,
                             'yes': 0.04839414490382867},
                        90: {'no': 0.006838648989915059,
                             'yes': 0.07978845608028654},
                        95: {'no': 0.0070427728315149,
                             'yes': 0.04839414490382867},
                        100: {'no': 0.007192295359419549,
                              'yes': 0.01079819330263761},
                        120: {'no': 0.007192295359419549,
                              'yes': 1.2151765699646583e-09},
                        125: {'no': 0.0070427728315149,
                              'yes': 1.826944081672921e-12},
                        220: {'no': 0.0009571488527694265,
                              'yes': 1.2894519942795936e-148}}}



```python
predict_class = []
cols = list(data.columns[:-1])
for i in range(len(data)):
    predict={}
    for label in labels:
        predict[label] = 1
        for col in cols:
            predict[label] *= posterior[col][data[col].iloc[i]][label]
        predict[label] *= p_label[label]
    predict_class.append(sorted(predict, key=predict.get, reverse=True)[0])
predict_class
```




    ['no', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes']




```python
actual_class = data['evade'].to_list()
cross_tab = []
for i in range(len(actual_class)):
    if actual_class[i]==predict_class[i] and actual_class[i]=='yes':
        cross_tab.append('TP')
    elif actual_class[i]==predict_class[i] and actual_class[i]=='no':
        cross_tab.append('TN')
    elif actual_class[i]!=predict_class[i] and actual_class[i]=='yes':
        cross_tab.append('FP')
    elif actual_class[i]!=predict_class[i] and actual_class[i]=='no':
        cross_tab.append('FN')
        
print(cross_tab)
```

    ['TN', 'TN', 'TN', 'TN', 'TP', 'TN', 'TN', 'TP', 'TN', 'TP']



```python
regions = ['TP','TN','FP','FN']
result = {}
for region in regions:
    result[region] = cross_tab.count(region)
result
```




    {'TP': 3, 'TN': 7, 'FP': 0, 'FN': 0}




```python
accuracy = (result['TP']+result['TN'])/len(data)*100
print('accuracy: ', accuracy,'%')
```

    accuracy:  100.0 %



```python

```
