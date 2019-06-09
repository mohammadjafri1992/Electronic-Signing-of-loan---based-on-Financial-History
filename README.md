# Electronic-Signing-of-loan---based-on-Financial-History

In this project, we are going to look at the data and identify who will apply for the loan. This is based off of actual user data collected from different consumer data selling agencies. This is NOT however, actual data. This is generated data as per the actual data distributions.

### Project outcomes

The final accuracy metric of our model is around 65%. This means that we can predict with someone's financial data if he/she will apply for a loan.

This has an excellent use case in business. The companies using this model can target specific users with a unique APR and fair loan terms based on their previous financial habbits.

### Methodology

This model is very simple. We only used 3 models, LogisticRegression, Support Vector Machines and Random Foreset. We used some iterations of these models including k-fold cross validation and parameter tuning using grid search to identify the most useful parameters for our model.

Simple models are good enough to give us at least something to work with. We can fine tune our model later on.

As we gather more data, we can swith to deep learning instead of simple ML models. But we should always start with basic models if the amount of data that we have is a grave concern.

