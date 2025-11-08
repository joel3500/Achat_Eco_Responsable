Achat Responsable — Comparateur écolo + Recherche par image

Un mini-site qui t’aide à choisir des produits en ligne avec l’aide d’un LLM, en mettant l’accent sur l’impact environnemental (matériaux, eau, recyclabilité).
Une seconde page permet d’uploader une image d’un objet et de retrouver des URLs où l’on voit le même objet (ou très similaire), avec miniature, site et prix quand disponible.

*** Fonctionnalités ***

1) Comparateur écolo (/)** :

Ajoute 2+ URLs de pages produit (Amazon, Alibaba, etc.)
Extraction des caractéristiques (matériaux, eau, recyclabilité, etc.)
Classement décroissant (du plus responsable au moins bon)
Short-list de sites avec exemples d’URL et insertion rapide

2) Recherche par image (/images) :

Upload 1 image → résultats d’URLs similaires avec miniatures
Affiche site et tente d’extraire le prix (si présent sur la page)
Moteur Google Cloud Vision – Web Detection

----
Python 3.11 (ou ≥ 3.10)
Un projet Google Cloud avec Cloud Vision API activée
(clé Service Account JSON requise)

----

(Optionnel/futur) Bing Visual Search en priorité, Google en fallback
UX soignée : aperçu instantané de l’image (300px), toasts, lazy-loading