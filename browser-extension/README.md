# Extension Achat Éco-Responsable

Petite extension Chrome/Edge (Manifest V3) qui permet d'analyser
l'impact écologique de la page produit que tu regardes, en un clic sur
l'icône de la barre d'outils — sans avoir à copier-coller l'URL sur le
site.

## Tester en local (extension "non empaquetée")

1. Ouvre `chrome://extensions` (ou `edge://extensions`).
2. Active le **Mode développeur** (interrupteur en haut à droite).
3. Clique **Charger l'extension non empaquetée** et sélectionne ce dossier
   (`browser-extension/`).
4. Va sur une page produit (ex. Amazon), clique sur l'icône de
   l'extension, puis **Analyser cette page**.

## Si tu changes de domaine (Railway → Render, domaine personnalisé...)

Deux fichiers à mettre à jour avec la nouvelle URL :
- `manifest.json` → le tableau `host_permissions`
- `popup.js` → la constante `API_BASE`

## Publier sur le Chrome Web Store

1. Compte développeur Chrome Web Store (frais unique ~5 USD) :
   https://chrome.google.com/webstore/devconsole
2. Compresse le contenu de ce dossier en `.zip` (le zip doit contenir
   `manifest.json` à la racine, pas un sous-dossier).
3. Dans le Developer Dashboard : **New item** → uploader le zip → remplir
   la fiche (description, catégorie, captures d'écran, politique de
   confidentialité — même minimale, elle est exigée).
4. Soumettre pour révision (le délai peut aller de quelques heures à
   quelques jours).

## Pourquoi "activeTab" et pas "tabs" dans les permissions ?

`activeTab` ne donne accès à l'onglet actif que lorsque l'utilisateur
clique explicitement sur l'icône de l'extension — c'est suffisant pour
notre cas d'usage, et ça évite l'avertissement d'installation "peut lire
et modifier tout ton historique de navigation" qu'impose la permission
plus large `tabs`. Une permission plus discrète = plus de gens qui
acceptent d'installer l'extension.
