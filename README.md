# Prédiction de la consommation de carburant avec des SVM (Auto MPG)

Ce devoir approxime la consommation de carburant de véhicules (`mpg`, miles par gallon) à partir de leurs caractéristiques techniques. Le problème est traité de deux façons avec des SVM :

1. **Classification (SVC)** : `mpg` est discrétisé en 3 classes de consommation.
2. **Régression (SVR)** : `mpg` est prédit directement, avec la perte ε-insensible.

Pour chaque approche, les modèles sont **implémentés depuis zéro (numpy)** puis **comparés aux implémentations de scikit-learn** (version 1.9.1).

> Remarque : `mpg` = miles par gallon, donc **mpg élevé = faible consommation**.

## 1. Contenu du dépôt

| Fichier / dossier | Rôle |
| `devoir_svm.ipynb` | Notebook complet : données, prétraitement, classification, régression |
| `requirements.txt` | Dépendances (numpy, pandas, seaborn, matplotlib, scikit-learn 1.9.1) |

Classes écrites depuis zéro (dans le notebook) :
| Classe | Rôle |
| `LinearSVMScratch` | SVM linéaire, primal, sous-gradient sur la perte hinge |
| `KernelSVMScratch` | SVM à noyau (RBF, polynomial), dual, algorithme SMO |
| `OneVsRestScratch` | Stratégie multiclasse un-contre-tous |
| `LinearSVRScratch` | SVR linéaire, primal, sous-gradient sur la perte ε-insensible |
**Lancer le projet**
```bash
pip install -r requirements.txt
jupyter notebook devoir_svm.ipynb   # puis Kernel > Restart & Run All
```
La graine aléatoire est fixée (`SEED = 42`) : les résultats sont reproductibles.

---

## 2. Données et prétraitement

**Données** : `seaborn.load_dataset('mpg')`, 398 véhicules. Après suppression de 6 lignes sans `horsepower`, il reste **392 échantillons**, 7 variables explicatives :

- continues : `displacement`, `horsepower`, `weight`, `acceleration`, `model_year` ;
- discrètes / catégorielles : `cylinders` (3, 4, 5, 6, 8), `origin` (usa 245, japan 79, europe 68).

**Observations de l'exploration**

- `mpg` est asymétrique à droite (moyenne 23,4, médiane 22,75, de 9 à 46,6).
- `mpg` est fortement corrélé négativement à `weight` (−0,83), `displacement` (−0,81), `horsepower` (−0,78) et `cylinders` (−0,78).
- Ces variables sont très corrélées entre elles (par exemple `displacement` et `cylinders` : 0,95).
- Les voitures d'origine usa ont le `mpg` le plus bas, japan le plus élevé.

**Pipeline et justifications**

| Choix | Justification |
|---|---|
| Suppression des 6 lignes sans `horsepower` (1,5 %) | Le SVM n'accepte pas les valeurs manquantes. La perte est minime. |
| Suppression de `name` | Texte quasi unique par ligne, inutile pour généraliser. |
| Split train/test 80/20 (313 / 79) **avant** tout traitement | Évite la fuite de données (data leakage). |
| Standardisation des variables numériques (moyenne et écart-type appris sur le train) | Le SVM repose sur des produits scalaires et des distances : sans cela, `weight` (~3000) écraserait `acceleration` (~15). |
| One-hot pour `origin` | Variable nominale : un codage 0/1/2 imposerait un ordre artificiel. |
| `cylinders` gardée numérique | Seulement 5 valeurs, dont deux très rares (3 et 5 cylindres : 4 et 3 voitures). Un one-hot ajouterait des colonnes quasi vides. |
| Standardisation de `mpg` (régression uniquement) | ε est alors exprimé en écarts-types de `mpg` (1 écart-type = 7,94 mpg). |

Résultat : une matrice de **9 colonnes** (6 numériques standardisées + 3 colonnes one-hot). L'implémentation manuelle du prétraitement (numpy) donne exactement le même résultat que scikit-learn (`np.allclose` = `True`).
---
## 3. Classification (SVC)
### 3.1 Discrétisation de la cible

Deux stratégies comparées. Les seuils sont calculés **sur le train uniquement**.
| Stratégie | Seuils | Répartition train (conso élevée / moyenne / faible) | Répartition test |
|---|---|---|---|
| **A : tertiles** | 18,5 et 27,0 mpg | 104 / 99 / 110 (33 % / 32 % / 35 %) | 23 / 35 / 21 |
| **B : seuils fixes** | 20 et 30 mpg | 122 / 111 / 80 (39 % / 35 % / 26 %) | 29 / 40 / 10 |

- **Équilibre** : A est équilibrée dans le train. B est déséquilibrée, avec la classe « conso faible » minoritaire (seulement 10 exemples dans le test).
- **Difficulté** : les erreurs se concentrent aux frontières entre classes voisines, et la classe « moyenne » est la plus difficile car elle est située entre deux autres.
- **Choix retenu : A**, pour comparer les modèles sur des classes équilibrées. B sert à étudier l'effet du déséquilibre.

### 3.2 Modèles
- **SVC linéaire** : primal, descente de sous-gradient sur la perte hinge (pas décroissant en 1/√t, meilleur itéré conservé).
- **SVC à noyau** : dual, algorithme SMO, noyaux **RBF** et **polynomial (degré 3)**.
- **Multiclasse** : un-contre-tous (OvR) pour les modèles scratch. Côté scikit-learn : `LinearSVC` (OvR), `OneVsRestClassifier(SVC)` (même stratégie que le scratch) et `SVC` natif (un-contre-un).
- **Hyperparamètres du RBF** : validation croisée stratifiée à 5 plis sur le train, critère F1 macro, ce qui donne `C = 1`, `gamma = 0.1` (F1 CV = 0,846). Le linéaire (`C = 1`) et le polynomial (`C = 1`, `gamma` = « scale ») n'ont pas été réglés.

### 3.3 Résultats (jeu de test, 79 échantillons, stratégie A)
| Modèle | Accuracy | F1 macro | Nb de vecteurs de support |
|---|---|---|---|
| Linéaire, scratch (primal) | 0,582 | 0,572 | n.a. |
| Linéaire, `LinearSVC` (hinge) | 0,633 | 0,629 | n.a. |
| RBF, scratch (SMO, OvR) | 0,797 | 0,803 | 184 |
| RBF, `SVC` en OvR | 0,797 | 0,803 | 180 |
| RBF, `SVC` natif (OvO) | 0,772 | 0,778 | 164 |
| Polynomial d=3, scratch | 0,797 | 0,802 | 151 |
| Polynomial d=3, `SVC` en OvR | 0,797 | 0,801 | 145 |

**Accord scratch / scikit-learn** (part de prédictions identiques) : linéaire 0,949 ; RBF 1,000 ; polynomial 0,987.

**Matrice de confusion, RBF scratch** (lignes = vrai, colonnes = prédit ; ordre : élevée, moyenne, faible) :

```
[[20  3  0]
 [ 4 25  6]
 [ 0  3 18]]
```

### 3.4 Interprétation

- **Les noyaux non linéaires dépassent nettement le linéaire** (accuracy ≈ 0,80 contre 0,58 à 0,63). La classe « moyenne » est très mal reconnue par le linéaire (8 bonnes prédictions sur 35 pour le scratch, 11 sur 35 pour `LinearSVC`), la plupart de ses erreurs allant vers « conso faible ». Un seul hyperplan « moyenne contre les autres » ne peut pas isoler une classe située entre deux autres.
- **Le résultat linéaire n'est pas dû à un bug de l'implémentation.** Les objectifs primaux atteints par le scratch et par `LinearSVC` sont quasi identiques pour les 3 classifieurs binaires (52,54 contre 52,34 ; 198,07 contre 198,09 ; 60,93 contre 61,13). L'écart restant sur les prédictions (accord de 0,949) s'explique par la convergence approchée du sous-gradient (la courbe de convergence n'est pas totalement stabilisée à 20 000 itérations) et par le fait que `LinearSVC` régularise aussi le biais.
- **RBF et polynomial** : le scratch reproduit scikit-learn à presque 100 % (accord de 1,000 et 0,987). Les petites différences de nombre de vecteurs de support (184 contre 180) viennent de la tolérance de SMO et du choix aléatoire de la seconde variable (SMO simplifié).
- **Nombre de vecteurs de support élevé** (≈ 150 à 184 sur 313 points) : les classes se chevauchent, notamment autour des seuils.

### 3.5 Impact de la discrétisation (SVC RBF, `C = 1`, `gamma = 0.1`)
| Discrétisation | `class_weight` | Accuracy | F1 macro |
|---|---|---|---|
| A (tertiles) | aucun | 0,772 | 0,778 |
| A (tertiles) | balanced | 0,797 | 0,803 |
| B (seuils 20/30) | aucun | 0,810 | 0,799 |
| B (seuils 20/30) | balanced | 0,772 | 0,764 |

- Avec A, `class_weight='balanced'` améliore légèrement les scores.
- Avec B, il les **dégrade**, ce qui est contre-intuitif. Une explication plausible, non testée ici : la classe minoritaire n'a que 10 exemples dans le test, donc quelques prédictions changent fortement le F1. Un seul split ne permet pas de conclure.
- Les accuracies de A et B ne sont pas directement comparables : les classes ne sont pas les mêmes, donc le problème posé non plus.
---
## 4. Régression (SVR linéaire)
## 4.1 Modèle
SVR linéaire avec la perte ε-insensible : on ne pénalise une erreur que si elle dépasse ε.
```
minimiser  ½‖w‖² + C · Σ max(0, |yᵢ − (w·xᵢ + b)| − ε)
```
Implémenté en primal par sous-gradient : seuls les points hors du tube contribuent au gradient. La cible est standardisée, donc ε est en écarts-types de `mpg` (ε = 0,1 correspond à 0,79 mpg).
**Hyperparamètres** : validation croisée à 5 plis sur le train avec `SVR` linéaire : `C = 0.1`, `ε = 0.3` (RMSE CV = 3,37 mpg). `C = 0.1` est utilisé pour toutes les comparaisons.
### 4.2 Validation de l'implémentation (ε = 0,1, `C = 0.1`)
| Modèle | RMSE test (mpg) | R² test | Nb de SV (train) |
|---|---|---|---|
| Scratch (sous-gradient) | 3,368 | 0,778 | 256 |
| `SVR(kernel='linear')` | 3,368 | 0,778 | 245 |
| `LinearSVR` | 3,373 | 0,777 | 258 |

Écart maximal entre les coefficients du scratch et ceux de `SVR` : **0,0045**. Le coefficient dominant est `weight` (−0,43, cible standardisée), suivi de `model_year` (+0,29) et de l'origine (japan +0,22, usa −0,22).

### 4.3 Influence de ε sur les vecteurs de support et l'erreur

| ε (écarts-types) | ε (mpg) | SV sklearn | SV scratch | RMSE train | RMSE test | ‖w‖ |
|---|---|---|---|---|---|---|
| 0 | 0,00 | 313 | 313 | 3,433 | 3,360 | 0,622 |
| 0,05 | 0,40 | 280 | 294 | 3,435 | 3,346 | 0,624 |
| 0,1 | 0,79 | 245 | 256 | 3,435 | 3,368 | 0,617 |
| 0,2 | 1,59 | 188 | 196 | 3,364 | 3,278 | 0,654 |
| **0,3** | 2,38 | 138 | 146 | 3,331 | **3,229** | 0,713 |
| 0,5 | 3,97 | 64 | 71 | 3,343 | 3,265 | 0,707 |
| 0,75 | 5,96 | 32 | 34 | 3,458 | 3,486 | 0,643 |
| 1,0 | 7,94 | 14 | 15 | 3,751 | 3,892 | 0,586 |
| 1,5 | 11,92 | 6 | 4 | 6,718 | 6,685 | 0,279 |

**Lecture**

- **Vecteurs de support** : décroissance monotone de 313 (tous les points pour ε = 0) à 6. Un point strictement dans le tube a un multiplicateur nul : il n'influence pas le modèle.
- **Erreur** : stable autour de 3,3 mpg jusqu'à ε = 0,5, minimale pour ε = 0,3 (3,229, valeur aussi choisie par la validation croisée), puis elle augmente : le tube devient trop large et le modèle sous-apprend.
- **‖w‖** : courbe non monotone (légère hausse jusqu'à ε ≈ 0,3 à 0,5, puis effondrement pour les grands ε).
- **Parcimonie** : de ε = 0 à ε = 0,3, le nombre de SV passe de 313 à 138 pour un RMSE test équivalent, voire légèrement meilleur.
- **Nombre de SV du scratch** : estimé par tolérance (points sur ou hors du tube à `sv_tol = 0.02` près), alors que scikit-learn le tire du dual. D'où un léger écart.
- **ε = 1,5** : le scratch (RMSE 5,49) s'écarte de scikit-learn (6,69). Les deux sont en sous-apprentissage ; l'écart vient de la convergence approchée du sous-gradient.

### 4.4 Robustesse au bruit
La cible d'entraînement est contaminée (bruit gaussien d'écart-type 0,3 + 5 % d'outliers à ±3 écarts-types), puis évaluée sur un test **propre** (`C = 0.1`).

| Modèle | Nb de SV | RMSE test (mpg) |
|---|---|---|
| Moindres carrés, cible propre | n.a. | 3,256 |
| Moindres carrés, cible bruitée | n.a. | 3,662 |
| SVR ε = 0 | 313 | 3,305 |
| SVR ε = 0,1 | 267 | 3,370 |
| SVR ε = 0,3 | 159 | 3,337 |
| SVR ε = 0,5 | 110 | 3,392 |
| SVR ε = 1,0 | 38 | 3,844 |

- Le SVR (3,30 à 3,39 pour ε ≤ 0,5) résiste bien mieux au bruit que les moindres carrés (3,66).
- Le meilleur SVR est celui avec **ε = 0**, pas un ε plus grand. La robustesse vient donc surtout de la **perte linéaire au-delà du tube** (les gros outliers ont une influence limitée, contrairement à la perte quadratique), et non de ε.
- ε sert avant tout à rendre le modèle **parcimonieux** : 159 vecteurs de support au lieu de 313 pour un RMSE quasi identique (3,337 contre 3,305).
- Un ε trop grand (1,0) dégrade les résultats.

## 5. Réponses aux questions de l'énoncé

**Quelle stratégie de discrétisation, et quel impact ?**
Tertiles (A), pour avoir des classes équilibrées (33 % / 32 % / 35 % dans le train). Les seuils fixes (B) sont plus interprétables mais déséquilibrés (26 % pour la classe « conso faible » dans le train, 13 % dans le test). Dans les deux cas, les erreurs se concentrent sur la classe « moyenne », située entre deux frontières.

**SVC linéaire ou à noyau ?**
Le noyau gagne nettement : accuracy ≈ 0,80 (RBF et polynomial) contre 0,58 à 0,63 pour le linéaire. La classe « moyenne » ne peut pas être séparée du reste par un seul hyperplan en OvR.

**L'implémentation depuis zéro est-elle fidèle à scikit-learn ?**
Oui : accord de prédictions de 1,000 (RBF), 0,987 (polynomial), 0,949 (linéaire) ; pour le SVR, RMSE test identique (3,368) et écart maximal sur les coefficients de 0,0045.

**Influence de ε sur les vecteurs de support et le bruit ?**
Quand ε augmente, le tube s'élargit et le nombre de vecteurs de support décroît (313 → 6). Un ε modéré (0,3) donne le meilleur RMSE test avec 138 vecteurs de support. Un ε trop grand fait sous-apprendre le modèle. Face au bruit, c'est surtout la perte linéaire qui protège des outliers ; ε apporte la parcimonie.

---

## 6. Limites

- **Un seul split train/test** (79 exemples de test) : les scores ont une variance non négligeable (par exemple pour le tableau de la section 3.5). Une validation croisée répétée donnerait des intervalles de confiance.
- **Prétraitement hors validation croisée** : le scaler est appris sur tout le train avant la validation croisée, d'où une très légère fuite. Un `Pipeline` scikit-learn la supprimerait.
- **SMO simplifié** (seconde variable choisie au hasard) : plus lent et moins précis que l'implémentation de libsvm utilisée par scikit-learn.
- **Sous-gradient** : convergence approchée, avec un écart visible à ε = 1,5 et sur le SVC linéaire.
- **Hyperparamètres** : seuls ceux du RBF (C, gamma) et du SVR (C, ε) sont réglés par validation croisée. Le linéaire et le polynomial utilisent `C = 1`.
- **Bruit simulé** : l'expérience de robustesse utilise du bruit artificiel, pas un vrai bruit de mesure.
- **`cylinders`** n'est pas parfaitement ordonné par rapport à `mpg` (les 3 cylindres, 4 voitures, ont un `mpg` médian plus bas que les 5 cylindres). Ces classes sont très rares et le conserver en numérique reste raisonnable.
