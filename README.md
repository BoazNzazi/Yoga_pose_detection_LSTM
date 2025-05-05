# 🧘‍♀️ Yoga Pose Detection

## 📌 Description

Ce projet utilise des techniques d'apprentissage profond pour reconnaître automatiquement différentes postures de yoga à partir d’images.  
Il repose sur une combinaison de réseaux de neurones convolutionnels (**CNN**) pour extraire les caractéristiques visuelles et de réseaux de neurones récurrents (**LSTM**) pour analyser la dynamique des mouvements.

L’application peut être utilisée pour :
- le suivi de séances de yoga en temps réel,
- la correction automatique des postures,
- l’aide à la rééducation ou à l’exercice physique guidé.

---

## 📁 Structure du projet

yoga_pose_detection/
│
├── data/ # Données d'entraînement et de test
│ └── TRAIN/TEST/VAL/ # Réparties en dossiers par catégorie (Downdog, Tree, etc.)
│
├── model/ # Fichiers du modèle CNN + LSTM
│ ├── model.py
│ └── train_model.py
│
├── utils/ # Fonctions utilitaires (prétraitement, métriques, etc.)
│
├── app/ # Interface (facultative) avec Streamlit ou autre
│
├── requirements.txt # Dépendances Python
├── README.md # Ce fichier
└── main.py # Script principal d’entraînement ou d’inférence


---

## 🚀 Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/BoazNzazi/yoga_pose_detection.git
   cd yoga_pose_detection
   
2. Créez un environnement virtuel (optionnel mais recommandé)
   
   ```bash
     python -m venv env
     source env/bin/activate  # sous Linux/Mac
     env\Scripts\activate     # sous Windows

3. Installez les dépendances :
  ```bash 
  pip install -r requirements.txt
```
---

## 🧠 Modèle utilisé

  - CNN (Convolutional Neural Network) : pour l'extraction des caractéristiques spatiales (formes, textures).
  - LSTM (Long Short-Term Memory) : pour modéliser les séquences temporelles entre les postures.

---

## 📊 Résultats

  - Précision globale : ~92%
  - Classes détectées : Downdog, Plank, Tree, Warrior, Goddess, Side Plank

---

## 🧑‍💻 Auteur

  Boaz Nzazi boaz.nzazi@unikin.ac.cd
