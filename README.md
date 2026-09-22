# 📈 Prédiction de l'indice Boursier MASI

Cette application Streamlit permet de visualiser l'historique et de prédire l'évolution future de l'indice boursier MASI (Moroccan All Shares Index) ou de toute autre valeur boursière.

## 🚀 Fonctionnalités
- **Chargement de données** via l'API Yahoo Finance ou import direct de fichier CSV (export Bourse de Casablanca / LeBoursier).
- **Visualisation interactive** des cours de clôture avec Plotly.
- **Prédiction des cours** à court/moyen terme via :
  - **Prophet** (Meta) avec calcul des intervalles de confiance.
  - **Random Forest Regressor** (Scikit-Learn).

## 🛠️ Installation en local

1. Cloner le dépôt :
```bash
git clone [https://github.com/votre-utilisateur/masi-prediction.git](https://github.com/votre-utilisateur/masi-prediction.git)
cd masi-prediction
