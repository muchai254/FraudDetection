## **Credit Card Fraud Detection**
It is important that credit card companies are able to detect fraudulent transactions so that customers do not end up paying for something that they did not pay for. This project aims at creating models that are able to detect potential fraudulent transactions and mark them as fraudulent.




**About dataset**

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly imbalanced; the positive class (frauds) account for 0.172% of all transactions.

Due to privacy and safety concerns, the credit card information is PCA transformed so as to keep the data safe, but still usable for the purpose of training our models. The dataset can be downloaded [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The data contains columns **V1** to **V28**, which are PCA transformed, **Time**, **Amount** and the dependent column, **Class** which contains "0" for non-fraudulent transactions and "1" for fraudulent transactions.

## Exploratory data analysis findings
Upon plotting a histogram on the Amount column, we observe that it is heavily imbalanced, with most of the transaction amounts ranging between $1 - $2000 with a few going up to $25000

![imbalanced data](./images/imbalanced%20amount.JPG)

This could really affect the models' performance, making them bias towards certain transaction amounts.
To solve this, we shall rescale the data by applying log transformation so as to give the data a normal distribution and then applying Robust Scaling so as to set limits on the data. This is done as follows:

```python
import numpy as np
data_df = df.copy()
data_df['Amount'] = np.log(data_df['Amount'] + 1)
data_df['Amount'].hist()

from sklearn.preprocessing import RobustScaler
new_df = data_df.copy()
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].to_numpy().reshape(-1,1))
time = new_df['Time']
new_df['Time'] = (time - time.min()) / (time.max())
new_df['Amount'].hist()
```

The results are as shown below:

![balanced data](./images/balanced%20amount.JPG)
## Training models (part 1)
Due to the imbalanced nature of our training data, only 492 frauds out of 284,807 transactions, our models will not perform very well when fed the testing data.
We shall first train the models on the imbalanced data, and then train them again on the data after balancing it out. This will be so as to show that models trained on balanced data perform better as compared to those trained on imbalanced data.

We shall be gauging the models according to **precision** and **recall** reports in detecting fraud cases. Precision is the ability of the model to **accurately flag transactions that are indeed fraudulent**, while recall is the **ability of the model to not report false negatives** (not reporting a transaction as non-fraudulent when it's actually fraudulent). The best model is one that scores high in both cases.
> A model with low recall will be flagging fraudulent transactions as non-fraudulent, therefore missing out on the fraudulent transactions.

> A model with low precision will be flagging non-fraudulent transactions as fraudulent.

The results of the first training are shown as follows:

![model with imbalanced data](./images/Model%20Performance%20(with%20imbalanced%20data).png)

From the results, the **SVC model** was the best performer.
## Training models (part 2)
In the second training, we balanced out the data by making the number of fraudulent transactions equal to that of non-fraudulent transactions. This was done by dropping the excess non-fraudulent rows. 
The results after training our models with this balanced data and testing it is as shown below:

![balanced data](./images/Model%20performance%20(with%20balanced%20data).png)

From these results, we can see the **SVC model** performed the best and even better than when trained with the imbalanced data.

From the results above, we can therefore recommend the SVC model to a financial institution for the purpose of flagging fraudulent transactions in real time.

![overall model performance](./images/Overall%20model%20performance%20.png)



