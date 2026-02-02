# Forest Cover Type Classification

## Overview
This project applies machine learning techniques to automatically classify forest cover types based on cartographic variables. The analysis is designed to support national forest service decision-making for conservation and land-use planning.

## Dataset
- **Source**: UCI Machine Learning Repository — Covertype Dataset
- **Records**: ~5,000 samples (downsampled from full dataset)
- **Features**: 54 (10 quantitative, 4 wilderness areas, 40 soil types)
- **Target**: 7 forest cover types (Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, Krummholz)

## Project Structure
```
Forest_Cover_type/
├── README.md
└── forest_cover_type_analysis.ipynb
```

## Key Analyses Performed

### 1. Exploratory Data Analysis (EDA)
- Class distribution analysis
- Correlation matrix of quantitative features
- Feature distribution histograms
- Identification of skewed and non-normal features

### 2. Model Training & Cross-Validation
Trained and evaluated 4 classifiers using stratified 5-fold cross-validation:
- **Logistic Regression** (baseline): 68.86% accuracy
- **K-Nearest Neighbors (KNN)**: 70.52% accuracy
- **Decision Tree**: 71.92% accuracy
- **Random Forest** (best): **80.78% accuracy**, F1-macro = 0.8055

### 3. Model Interpretation
- **Confusion matrices** for all models showing error patterns
- **Feature importances** from tree-based models
- Random Forest identified Elevation and Wilderness Area 4 as top predictive features

### 4. Feature Reduction
- **Recursive Feature Elimination (RFE)**: Selected 15 most important features
- **Performance comparison**: 
  - Full (54 features): 80.78% accuracy
  - Reduced (15 features): ~80.6% accuracy (minimal loss)
  - **Conclusion**: Reduced feature set maintains predictive power while reducing data collection burden

### 5. Dimensionality Reduction
- **PCA (2 components)**: Projects 54 features to 2D space
- Shows partial class separation, useful for visual interpretation
- Not a replacement for supervised classification but helps intuitive understanding

## Findings & Recommendations

### Recommended Model: Random Forest
✓ **Highest accuracy**: 80.78%  
✓ **Robust performance**: Low std dev (±1.4%)  
✓ **Interpretable**: Feature importances available  
✓ **Scalable**: Suitable for deployment  

### Key Insights
1. **Feature importance**: Elevation, Wilderness Area 4, and Soil types dominate predictions
2. **Class separation**: Some cover types naturally separate in feature space; others overlap
3. **Feature efficiency**: 15 key features capture ~99.7% of predictive power vs. full 54-feature set

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Execute the Notebook
```bash
jupyter notebook forest_cover_type_analysis.ipynb
```

Or use nbconvert to run headless:
```bash
jupyter nbconvert --to notebook --execute forest_cover_type_analysis.ipynb --output forest_cover_type_analysis-executed.ipynb
```

## Results Summary

| Model | Accuracy | F1-Macro | Std Dev |
|-------|----------|----------|---------|
| Random Forest | **80.78%** | **0.8055** | 0.0143 |
| Decision Tree | 71.92% | 0.7183 | 0.0110 |
| KNN (k=7) | 70.52% | 0.7014 | 0.0096 |
| Logistic Regression | 68.86% | 0.6861 | 0.0147 |

## Deployment Recommendations
1. **Use Random Forest** with top 15 RFE-selected features for production
2. **Preprocessing**: Standardize numeric features before inference
3. **Monitoring**: Track accuracy on new data; retrain quarterly
4. **Features to collect**: Focus on Elevation, Wilderness Area, and Soil type indicators

## Future Work
- Hyperparameter tuning (RandomForest depth, KNN neighbors)
- Ensemble methods (Stacking, Voting)
- Additional feature engineering (interaction terms, domain-specific indices)
- Class imbalance handling (SMOTE) if deploying on unbalanced datasets
- Geographic validation (test on holdout regions)

## Author
Data Science & ML Portfolio — Forest Cover Type Classification Project

---
**Last Updated**: February 2, 2026
