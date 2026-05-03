# Cancer Drug Sensitivty Predictor 

In this project, I create machine learning models from scratch to predict how sensitive cancer cell lines are to different drugs, using gene expression data from nearly 1,000 cell lines screened against 266 compounds from the Genomics of Drug Sensitivity in Cancer (GDSC) dataset.

## background 
Given a cancer cell line's gene expression profile, the models predict drug sensitivity measured by IC50 (the concentration of a drug needed to inhibit cell growth by 50%). Lower IC50 means the cancer is more sensitive to that drug. The project covers everything from simple linear models to neural networks, all implemented from scratch in PyTorch.

The GDSC dataset is one of the largest publicly available cancer pharmacogenomics resources. Gene expression values are RMA-processed and Z-normalized per gene. IC50 values are Z-normalized per drug. About 18% of IC50 values are missing and are masked out during training rather than imputed.

## models implemented
- Ridge Regression (L2) --> Solved two ways: 1) analytically using the closed-form matrix solution, and 2) iteratively using gradient descent. Both approaches converge to the same answer. A range of regularization values are tested on the validation set to find the optimal lambda.
- LASSO Regression (L1) --> Fitted using subgradient descent since L1 regularization is not differentiable at zero. Produces sparser coefficient vectors than ridge regression, with more weights pushed exactly to zero, making it potentially more interpretable for identifying which genes matter most.
- Logistic Regression --> Binary classification version of the problem; cell lines are labeled sensitive or resistant based on a published IC50 threshold. Includes manual backpropagation implementation verified against PyTorch's autograd. Evaluated using ROC curves and AUC scores.
- Single-Task Dense Neural Network --> a fully connected neural network with ReLU activations, trained to predict drug sensitivity for one drug at a time. Backpropagation is implemented manually layer by layer. Includes early stopping based on validation loss and optional L2 regularization.
- Multitask Dense Neural Network --> One network trained simultaneously across all 217 drugs, outputting predictions for every compound in a single forward pass. Missing IC50 values are masked out during training using observation weights so they don't affect the loss. Network depth and width are systematically varied to find the best architecture.

## project structure 
```
cancer-drug-sensitivity/
      cancer-drug-sensitivity.ipynb          # Full notebook with all models, plots, and writeups
```

## main findings: 
- Ridge vs LASSO: Both achieve similar test R² on the held-out set, but LASSO coefficients are far more concentrated near zero, suggesting only a small number of genes drive most of the predicted sensitivity for any given drug.
- Linear vs Neural Network: Ridge and LASSO outperform the single-task neural network on this dataset. With only ~900 cell lines and nearly 18,000 gene features, linear models with strong regularization handle the high-dimensional low-sample regime better than a neural network with many free parameters.
- Multitask vs Single-Task: Training one network across all 217 drugs simultaneously outperforms the single-drug network. Shared hidden layers allow the model to learn gene expression patterns that are predictive across multiple drugs at once.
- Depth vs Width: Deeper networks (3-4 hidden layers) with moderate width (32 units) outperform wider but shallower networks on the multitask problem. Adding L2 regularization to the neural network reduces validation loss slightly compared to the unregularized baseline.

## tools used: 
- PyTorch (all model training runs on GPU via CUDA)
- pandas / numpy (data loading and preprocessing)
- matplotlib / seaborn (plotting loss curves, ROC curves, scatter plots)
- scipy (Pearson correlation for regression evaluation)
- Google Colab (GPU environment)

## How to Run
For detailed infomration on how to run this project, please reach out to me at sr3986@columbia.edu 
