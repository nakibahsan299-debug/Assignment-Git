# Literature Review: Credit Card Fraud Detection

Topic: detecting fraudulent card transactions when fraud is extremely rare, with a
focus on how the class imbalance is handled and which metrics are used to judge models.

> Note: these summaries are based on how the sources are described in the
> course report on "credit card fraud detection using different CNN-based
> algorithms" (Roll 1907086). Skim the originals before submitting to confirm details.

## 1. Gambo, Zainal & Kassim (2022). *A Convolutional Neural Network Model for Credit Card Fraud Detection.* ICoDSA.
The authors tackle the imbalance in the ULB dataset (492 frauds out of about 284,800
transactions) by synthesising extra fraud examples with ADASYN, which creates more
samples in neighbourhoods where the minority class is hardest to separate. A small
1D CNN (two convolution blocks with 32 and 64 filters, batch norm, dropout) is then
trained on the balanced data and reports accuracy around 99.8% with recall near 1.0.
Those numbers look very strong, but it is not clear the resampling was kept away from
the test data, so they may be optimistic.

## 2. Ali, Kabir, Raka, Toma, Rahman & Ferdaus (2022). *SMOTE Based Credit Card Fraud Detection Using Convolutional Neural Network.* ICCIT.
This paper balances the data with SMOTE, which interpolates between a minority sample
and its nearest minority neighbours, and compares an ANN, a CNN and an LSTM on the result.
The CNN version reaches roughly 98.9% accuracy but its recall on fraud is noticeably
lower (about 0.91), meaning around one in ten frauds slips through. It shows that
accuracy alone hides this weakness.

## 3. Mizher & Nassif (2023). *Deep CNN Approach for Unbalanced Credit Card Fraud Detection Data.* ASET.
Instead of resampling, this work trains a deep CNN directly on the imbalanced data
and compares it against an SVM and a Random Forest. Its scores (accuracy about 93.5%,
precision 0.93, recall 0.92) are lower than the resampling papers, but they come from
a setup closer to real deployment, where nobody can rebalance the incoming stream.
The comparison with classical models is useful because it tests whether the deep
network is actually necessary.

## 4. Sulaiman, Schetinin & Sant (2022). *Review of Machine Learning Approach on Credit Card Fraud Detection.* Human-Centric Intelligent Systems.
A survey of machine-learning approaches to card fraud. As cited in the report, it
motivates the shift from hand-written rules to learned models and shows that classical
algorithms remain strong baselines. I use it to justify including non-deep models in
my own experiment.

## 5. Modi & Dayma (2017). *Review on Fraud Detection Methods in Credit Card Transactions.* I2C2.
An earlier overview of fraud detection techniques. It frames the core difficulty that
fraud is rare, changes over time, and that rules-based systems struggle to keep up.
It gives the background for why imbalance handling and evaluation choice matter.

## Takeaways for this project
- All three experimental papers lean on accuracy, which is almost meaningless when
  99.8% of transactions are legitimate. I will use precision-recall based metrics.
- Resampling (ADASYN/SMOTE) must only be applied to training data, otherwise
  synthetic points leak into the test set and inflate scores.
- Reported results differ a lot between papers, partly because of different splits,
  so a fair comparison needs the same split and metrics for every model.
