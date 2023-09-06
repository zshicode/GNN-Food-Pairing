# Food pairing recommendation based on food-compound matrix and graph neural networks

Food pairing recommends various food combinations based on compound interactions among foods (Ahn et al., 2011; Spence, 2020). We adopt graph neural networks (GNN) with novel graph attention for detecting food interactions from food-compound matrix.

## Requirements

The code has been tested running under Python 3.9.12, with the following packages and their dependencies installed:

```bash
numpy
pandas
pytorch
sklearn
matplotlib
seaborn
rdkit
```

## Usage

1. Run `dataprepare.py` to preprocess the data.

2. Run `main.py` to run the model.

```bash
python main.py --data 1
```

After running `main.py`, the script outputs the classification accuracy of the model (see Method). Function `query()` requires target food string (e.g. `'Green tea'`) and outputs CSV containing the scores of other foods while pairing with target food.

## Datasets

In each dataset, `food.csv` includes names and categories of foods. `compound.csv` includes names and SMILES codes of compounds. `food-compound.csv` includes food-compound associations.

||Data source|Paper|GitHub link|
|:--:|:--:|:--:|:--:|
|Data1|[FooDB](https://foodb.ca/)|(Su et al., 2023)|https://github.com/TinaCausality/NutriFD_Dataset|
|Data2|(Ahn et al., 2011)|(Ahn et al., 2011)|https://github.com/lingcheng99/Flavor-Network|
|Data3|[FooDB](https://foodb.ca/)|(Rahman et al., 2021)|https://github.com/mostafiz67/FDMine_Framework|

Food-compound matrix is with elements as the score of food-compound pairs. 

In Data1, suppose the content of compound k in food i is a mg/100g (+- 1e-3 mg), then `score[i,k] = 0.1*(log10(a+1e-5)+5)` can scale the amount into [0,1] (since 100g = 1e5 mg).

In Data2, if there is compound k in food i, then `score[i,k] = 1`, otherwise `score[i,k] = 0`.

Data3 is a refined version of Data1. However, the score is defined as contribution rate of compound content (Rahman et al., 2021), scaling into [0,1].

## Method

Food-compound matrix is adopted as feature matrix to construct GNN for food classification evaluating representation learning (Shi et al., 2021). A novel graph attention (Jin et al., 2022) is adopted for fusing food-compound matrix and compound Morgan fingerprint matrix.

The model optimizes focal loss (Lin et al., 2020) to tackle imbalanced classification. Finally, food pairing recommendation is computed from food interactions modeled by food representation similarities.

## Food-drug interactions

Food pairing can be modeled using drug-drug interactions through comparing similarities among drugs and food compounds (Rahman et al., 2021). Food-drug interactions can be intergrated with other drug-based interactions beyond molecular fingerprint similarity (e.g. Tanimoto similarity) when using GNN.

## References

Ahn et al., Flavor network and the principles of food pairing, Sci Rep, 2011

Jin et al., Predicting miRNA-Disease Association Based on Neural Inductive Matrix Completion with Graph Autoencoders and Self-Attention Mechanism, Biomolecules, 2022

Lin et al., Focal Loss for Dense Object Detection, IEEE TPAMI, 2020

Rahman et al., A novel graph mining approach to predict and evaluate food-drug interactions, Sci Rep, 2021

Shi et al., A representation learning model based on variational inference and graph autoencoder for predicting lncRNA-disease associations, BMC Bioinform, 2021

Spence C. Food and beverage flavour pairing: A critical review of the literature, Food Res Int, 2020

Su et al., NutriFD: Proving the medicinal value of food nutrition based on food-disease association and treatment networks, arXiv, 2023