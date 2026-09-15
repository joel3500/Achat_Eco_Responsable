"""
Achat Éco-Responsable — application web (Flask)
==================================================
Ce fichier contient TOUT le "cerveau" du site :
  - les routes (les adresses comme / ou /images que le navigateur visite)
  - la logique qui va chercher le contenu d'une page produit sur Internet
  - l'appel à l'intelligence artificielle (OpenAI) pour analyser ce contenu
  - le calcul du score écologique
  - le petit tableau de bord de statistiques (/stats)

Si tu débutes en programmation : ce fichier est volontairement commenté
en détail. Chaque section explique POURQUOI le code existe, pas
seulement CE QU'IL FAIT (le code lui-même montre déjà ce qu'il fait).
"""

import os          # pour lire des variables d'environnement (mots de passe, clés API...)
import re          # "regex" = pour reconnaître des motifs de texte (ex: un prix "12,99 $")
import json         # pour lire/écrire du JSON (le format d'échange avec l'IA et le navigateur)
import math         # (non utilisé activement, gardé pour compatibilité future)
import time         # (non utilisé activement, gardé pour compatibilité future)
import base64       # pour encoder une image en texte (nécessaire pour l'envoyer à l'IA)
import tempfile     # pour créer un fichier temporaire (le certificat Google Cloud)
import sqlite3      # petite base de données locale utilisée par le tableau de bord /stats
import hashlib      # pour transformer une IP en empreinte (hash) irréversible
import hmac         # comparaison "sécurisée" de mots de passe (évite les attaques par timing)
import ipaddress    # pour reconnaître les adresses IP privées (ex: 127.0.0.1, réseau local)
import secrets      # pour générer un identifiant court et imprévisible (lien de partage)

from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv       # charge le fichier .env (variables secrètes en local)
import requests                       # pour faire des requêtes HTTP (aller chercher une page web)
from bs4 import BeautifulSoup         # pour "lire" du HTML et en extraire le texte
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from werkzeug.middleware.proxy_fix import ProxyFix

#---------  Imports pour couvrir le SCRAPPING (aller lire des pages web) -------------------#

from urllib.parse import urlparse   # pour découper une URL et en extraire le nom de domaine
from playwright.sync_api import sync_playwright   # navigateur invisible (headless) pour le JS
import extruct, w3lib.html          # pour lire les données structurées (JSON-LD) d'une page

#----------------------------------------------------------------#
# Page 2 : recherche par images
# -----------------------------------------------------------------------------#
from io import BytesIO
from flask import send_from_directory
#------------------------------------------------------------------------------#
# Pour extraire le prix sur une URL d'un produit donné
import html
from w3lib.html import get_base_url
#------------------------------------------------------------------------------
# Pour aider l'indexation de la page sur les moteurs de recherches
from datetime import datetime
from flask import url_for, Response
#--------------------------------------------------------------------------------

load_dotenv()  # lit le fichier .env s'il existe et remplit os.environ avec son contenu

#-------------------------------#
# Config de base                #
#-------------------------------#
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change si tu veux
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# En prod (Railway), les credentials Google arrivent en base64 (pas de fichier disponible sur le filesystem).
# En local, GOOGLE_APPLICATION_CREDENTIALS peut déjà pointer vers un fichier .json existant.
_gcreds_b64 = os.getenv("GOOGLE_CREDS_B64")
if _gcreds_b64:
    _creds_path = os.path.join(tempfile.gettempdir(), "gcp-credentials.json")
    with open(_creds_path, "wb") as fichier_credentials:
        fichier_credentials.write(base64.b64decode(_gcreds_b64))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds_path

client = OpenAI(api_key=OPENAI_API_KEY)  # le "client" est l'objet qui parle à l'API d'OpenAI

# ---- Paramètre pour activer/désactiver le rendu headless ---------------#
# "headless" = un vrai navigateur Chrome qui tourne sans fenêtre visible.
# On ne l'utilise que pour certains sites, car c'est plus lent qu'un simple téléchargement.
ALLOW_HEADLESS = os.getenv("ALLOW_HEADLESS_FETCH", "false").lower() in ("1", "true", "yes")

# Domaines sur lesquels on autorise le rendu headless (légal & utile)
HEADLESS_DOMAINS = {
    # Marketplaces demandées
    "amazon.ca", "amazon.com",
    "alibaba.com",
    "aliexpress.com",
    "wish.com",
    # Autres que tu as ajoutés
    "ebay.ca", "ebay.com",
    "etsy.com",
    "bestbuy.ca",
    "canadiantire.ca",
    "newegg.ca",
    "simons.ca",
    "mec.ca",
    "ikea.com",          # pages Canada sont sur ikea.com (/ca/en)
    "homedepot.ca",
    "rona.ca",
    "backmarket.ca", "backmarket.com",
    "poshmark.ca", "poshmark.com",
    "earthhero.com",
}

GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY")   # clé de l'API "Google Custom Search"
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")    # identifiant du moteur de recherche configuré

# ---- Admin / stats -------------------------------------------------------#
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
STATS_DB_PATH = os.getenv("STATS_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.db"))

# -----------------------------------------------------------------------------#

app = Flask(__name__)  # crée l'application web. C'est l'objet central de Flask.
app.secret_key = os.getenv("SECRET_KEY") or os.urandom(24)  # nécessaire pour les "sessions" (ex: rester connecté à /stats)
# Railway (et la plupart des PaaS) mettent l'app derrière un proxy : sans ça,
# request.remote_addr renverrait l'IP du proxy plutôt que celle du visiteur.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

#------------------------------------------------------------------------------#
# 4 variantes courantes de prix (CAD/$/USD/€)
# Une "regex" (expression régulière) est un motif qui décrit à quoi doit
# ressembler un bout de texte. Ici, chaque motif reconnaît un prix écrit
# de façon un peu différente (avant ou après le symbole de devise, etc.)
PRICE_REGEXES = [
    re.compile(r'(?:(?P<cur>\$|CAD|C\$)\s?(?P<val>\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d{2})?))', re.I),
    re.compile(r'(?:(?P<val>\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d{2})?)\s?(?P<cur>CAD|\$|C\$))', re.I),
    re.compile(r'(?:(?P<cur>€)\s?(?P<val>\d{1,3}(?:[ .]\d{3})*(?:[,]\d{2})?))', re.I),
    re.compile(r'(?:(?P<cur>USD)\s?(?P<val>\d{1,3}(?:[ ,]\d{3})*(?:[.]\d{2})?))', re.I),
]

# On s'identifie comme un vrai navigateur pour éviter que certains sites
# bloquent nos requêtes automatiquement (beaucoup de sites refusent les
# visiteurs qui n'ont pas de "User-Agent" reconnu).
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

#------------------------------------------------------------------------------#
# (facultatif) quelques réglages ( une Stratégie de référencements )
#------------------------------------------------------------------------------#
app.config.update({
    "COMPRESS_ALGORITHM": "gzip",   # ou "brotli" si tu ajoutes 'brotli'
    "COMPRESS_LEVEL": 6,            # 1-9 (6 est un bon compromis)
    "COMPRESS_MIN_SIZE": 1024,      # ne compresse que > 1 Ko
})

from flask_compress import Compress
Compress(app)  # compresse automatiquement les réponses HTML/JSON pour un site plus rapide

#--------------- (Fin de stratégies de référencement) -------------------------#

# ============================================
#  MODULE : STATS / ANALYTICS (page /stats)
# ============================================
# Ce petit module enregistre, de façon anonyme, qui visite le site et quels
# types d'articles sont analysés. Ça sert à savoir ce qui intéresse vraiment
# les visiteurs (utile pour prioriser les prochaines améliorations).
LIBELLES_CATEGORIES = {
    "vetements": "Vêtements & textile",
    "maison_meubles": "Maison, literie & meubles",
    "vehicules": "Véhicules",
    "electronique": "Électronique & accessoires",
    "electromenagers": "Électroménagers",
    "sport_plein_air": "Sport & plein air",
    "produits_menagers": "Produits ménagers & soins personnels",
    "jouets": "Jouets & articles pour enfants",
    "bagagerie": "Bagagerie & accessoires",
    "bricolage": "Bricolage & rénovation",
    "autre": "Autre",
}


def initialiser_base_de_donnees_stats():
    """Crée les tables SQLite si elles n'existent pas encore (ne fait rien si elles existent déjà)."""
    connexion = sqlite3.connect(STATS_DB_PATH)
    connexion.execute("""CREATE TABLE IF NOT EXISTS visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ip_hash TEXT, city TEXT, region TEXT, country TEXT, page TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    connexion.execute("""CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ip_hash TEXT, city TEXT, region TEXT, country TEXT, url TEXT, category TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    # Table utilisée par la fonctionnalité de partage : chaque comparaison
    # qu'un visiteur choisit de partager est sauvegardée ici sous un
    # identifiant court, pour pouvoir être réaffichée via /r/<id>.
    connexion.execute("""CREATE TABLE IF NOT EXISTS shared_results (
        id TEXT PRIMARY KEY,
        payload_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    # Migration douce : ajoute la colonne 'region' si la DB existait déjà sans elle.
    for nom_table in ("visits", "submissions"):
        colonnes_existantes = [ligne[1] for ligne in connexion.execute(f"PRAGMA table_info({nom_table})")]
        if "region" not in colonnes_existantes:
            connexion.execute(f"ALTER TABLE {nom_table} ADD COLUMN region TEXT")
    connexion.commit()
    connexion.close()


initialiser_base_de_donnees_stats()


def obtenir_ip_du_visiteur() -> str:
    """Renvoie l'adresse IP de la personne qui a fait la requête actuelle."""
    return request.remote_addr or "0.0.0.0"


def transformer_ip_en_empreinte(ip: str) -> str:
    """
    Transforme une IP en une empreinte (hash) qu'on ne peut pas retransformer
    en IP d'origine. On garde ainsi une notion de "visiteur unique" sans
    jamais stocker sa vraie adresse IP en clair dans la base de données.
    """
    return hashlib.sha256(f"achat_eco_salt:{ip}".encode()).hexdigest()[:16]


_cache_geolocalisation: Dict[str, tuple] = {}  # évite de refaire le même appel réseau deux fois


def geolocaliser_ip(ip: str) -> tuple:
    """Retourne (ville, région, pays) pour une IP. Ne fait jamais planter l'appelant.

    La ville est une estimation "meilleur effort" : pour certains FAI régionaux
    (ex: Altima Telecom au Saguenay–Lac-Saint-Jean), le bloc d'IP est enregistré
    administrativement dans une autre ville (souvent Montréal) même si l'abonné
    est ailleurs. La région (province/état) est nettement plus fiable.
    """
    if ip in _cache_geolocalisation:
        return _cache_geolocalisation[ip]

    try:
        if ipaddress.ip_address(ip).is_private:
            _cache_geolocalisation[ip] = ("Local", "Local", "Local")
            return _cache_geolocalisation[ip]
    except ValueError:
        _cache_geolocalisation[ip] = ("Inconnu", "Inconnu", "Inconnu")
        return _cache_geolocalisation[ip]

    resultat = ("Inconnu", "Inconnu", "Inconnu")
    try:
        reponse_geo = requests.get(
            f"http://ip-api.com/json/{ip}",
            params={"fields": "status,country,regionName,city"}, timeout=3
        )
        donnees_geo = reponse_geo.json()
        if donnees_geo.get("status") == "success":
            resultat = (
                donnees_geo.get("city") or "Inconnu",
                donnees_geo.get("regionName") or "Inconnu",
                donnees_geo.get("country") or "Inconnu",
            )
    except Exception:
        pass  # si le service de géolocalisation est indisponible, on garde "Inconnu"
    _cache_geolocalisation[ip] = resultat
    return resultat


def enregistrer_visite(nom_page: str) -> None:
    """Ajoute une ligne dans la table 'visits' à chaque chargement d'une page suivie."""
    try:
        ip = obtenir_ip_du_visiteur()
        ville, region, pays = geolocaliser_ip(ip)
        connexion = sqlite3.connect(STATS_DB_PATH)
        connexion.execute(
            "INSERT INTO visits (ip_hash, city, region, country, page) VALUES (?,?,?,?,?)",
            (transformer_ip_en_empreinte(ip), ville, region, pays, nom_page)
        )
        connexion.commit()
        connexion.close()
    except Exception as erreur:
        print(f"[stats] enregistrer_visite a échoué : {erreur}")  # ne doit jamais casser une page


def enregistrer_soumission(url: str, categorie: str) -> None:
    """Ajoute une ligne dans la table 'submissions' à chaque analyse d'article lancée."""
    try:
        ip = obtenir_ip_du_visiteur()
        ville, region, pays = geolocaliser_ip(ip)
        connexion = sqlite3.connect(STATS_DB_PATH)
        connexion.execute(
            "INSERT INTO submissions (ip_hash, city, region, country, url, category) VALUES (?,?,?,?,?,?)",
            (transformer_ip_en_empreinte(ip), ville, region, pays, url, categorie or "autre")
        )
        connexion.commit()
        connexion.close()
    except Exception as erreur:
        print(f"[stats] enregistrer_soumission a échoué : {erreur}")


# ============================================
#  MODULE : RECHERCHE PAR IMAGE (façon Google Lens)
# ============================================

def deviner_type_image(donnees_image: bytes) -> str:
    """
    Regarde les tout premiers octets du fichier (sa "signature") pour deviner
    son format (jpeg/png/gif/webp), sans avoir besoin de connaître son nom.
    """
    if donnees_image[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if donnees_image[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if donnees_image[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    if donnees_image[:4] == b'RIFF' and donnees_image[8:12] == b'WEBP':
        return "image/webp"
    return "image/jpeg"


def analyser_image_avec_google_vision(donnees_image):
    """Utilise Google Cloud Vision pour trouver des mots-clés (entités) liés à l'image."""
    try:
        from google.cloud import vision
        client_vision = vision.ImageAnnotatorClient()
        reponse = client_vision.web_detection(image=vision.Image(content=donnees_image))
        detection_web = reponse.web_detection
    except Exception:
        return {"entities": [], "pages": []}

    resultats = {
        "entities": [],
        "pages": [],
        "full_images": [],
        "partial_images": [],
    }

    if not detection_web:
        return resultats

    if detection_web.web_entities:
        resultats["entities"] = [entite.description for entite in detection_web.web_entities if entite.description]

    if detection_web.pages_with_matching_images:
        resultats["pages"] = [page.url for page in detection_web.pages_with_matching_images if page.url]

    if detection_web.full_matching_images:
        resultats["full_images"] = [image.url for image in detection_web.full_matching_images if image.url]

    if detection_web.partial_matching_images:
        resultats["partial_images"] = [image.url for image in detection_web.partial_matching_images if image.url]

    return resultats


def decrire_image_avec_ia(donnees_image):
    """Demande au modèle de langage (GPT) de décrire en une phrase le produit visible sur l'image."""
    client_openai_local = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        type_mime = deviner_type_image(donnees_image)
        image_en_base64 = base64.b64encode(donnees_image).decode("utf-8")
        reponse = client_openai_local.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Décris précisément le produit visible sur l’image."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{type_mime};base64,{image_en_base64}"}
                        },
                        {"type": "text", "text": "En une phrase courte : marque, type, couleur, modèle si possible."}
                    ]
                }
            ]
        )
        return reponse.choices[0].message.content.strip()
    except Exception:
        return ""


def rechercher_texte_sur_google(requete_texte):
    """Utilise l'API Google Custom Search pour trouver des pages correspondant à une description texte."""
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_ID:
        return []

    parametres_requete = {
        "q": requete_texte,
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_ID,
        "searchType": "image",
        "num": 10,
    }
    try:
        reponse = requests.get("https://www.googleapis.com/customsearch/v1", params=parametres_requete, timeout=20)
        reponse.raise_for_status()
    except Exception:
        return []

    resultats = []
    for resultat_brut in reponse.json().get("items", []):
        page_source = resultat_brut.get("image", {}).get("contextLink") or resultat_brut.get("link")
        if not page_source:
            continue
        try:
            nom_site = urlparse(page_source).netloc
        except Exception:
            nom_site = None
        resultats.append({
            "thumb": resultat_brut.get("link"),
            "url": page_source,
            "site": nom_site,
            "price": None,
        })
    return resultats


def fusionner_resultats(*listes_resultats):
    """Combine plusieurs listes de résultats en une seule, en retirant les doublons (même URL)."""
    resultats_fusionnes, urls_deja_vues = [], set()
    for liste in listes_resultats:
        if not liste:
            continue
        for article in liste:
            url_article = article.get("url")
            if not url_article or url_article in urls_deja_vues:
                continue
            urls_deja_vues.add(url_article)
            resultats_fusionnes.append(article)
    return resultats_fusionnes


def enrichir_avec_les_prix(articles, nombre_max=10):
    """Essaie de trouver le prix affiché sur chacune des premières pages trouvées."""
    for article in articles[:nombre_max]:
        prix = try_extract_price(article["url"])
        if prix:
            article["price"] = prix
    return articles


def recherche_image_complete(donnees_image):
    """
    Orchestre toute la recherche par image, étape par étape :
      1) on demande à Google Vision quels mots-clés il reconnaît sur l'image
      2) on demande en plus à l'IA de décrire le produit en une phrase
      3) on cherche les pages qui contiennent visuellement la même image
      4) on cherche en complément via une recherche texte classique
      5) on fusionne le tout et on tente de récupérer le prix
    """
    # 1) Google Vision → mots-clés (entités) reconnus sur l'image
    infos_vision = analyser_image_avec_google_vision(donnees_image)

    # 2) IA → description lisible, utilisée ensuite comme requête de recherche
    requete_texte = decrire_image_avec_ia(donnees_image)
    if not requete_texte and infos_vision.get("entities"):
        requete_texte = " ".join(infos_vision["entities"][:5])

    # 3) Google Cloud Vision Web Detection (image → pages visuellement similaires)
    resultats_visuels = gcv_web_detection(donnees_image)

    # 4) Google Custom Search (description → résultats complémentaires)
    resultats_texte = rechercher_texte_sur_google(requete_texte) if requete_texte else []

    # 5) Fusion (les résultats visuels passent en premier, le texte complète)
    resultats_combines = fusionner_resultats(resultats_visuels, resultats_texte)

    # 6) Enrichissement des prix
    resultats_combines = enrichir_avec_les_prix(resultats_combines)

    return {
        "description": requete_texte,
        "entities": infos_vision["entities"],
        "items": resultats_combines[:20]
    }

#================= ( FIN DE RECHERCHE de Correspondance )  ===========================#


def _extraire_prix_depuis_jsonld(objet_json):
    """Explore récursivement un bloc JSON-LD/Microdata à la recherche d'une 'Offer' (offre de prix)."""

    def nettoyer_valeur(valeur):
        if isinstance(valeur, (int, float)):
            return f"{valeur}"
        if isinstance(valeur, str):
            return valeur.strip()
        return None

    def chercher_offre(objet):
        if isinstance(objet, dict):
            type_objet = nettoyer_valeur(objet.get("@type")) or nettoyer_valeur(objet.get("type"))
            if type_objet and type_objet.lower() in {"offer", "aggregateoffer"}:
                prix = nettoyer_valeur(objet.get("price") or objet.get("lowPrice") or objet.get("highPrice"))
                devise = nettoyer_valeur(objet.get("priceCurrency"))
                if prix:
                    return f"{prix} {devise}".strip()
            for valeur in objet.values():
                trouve = chercher_offre(valeur)
                if trouve:
                    return trouve
        elif isinstance(objet, list):
            for element in objet:
                trouve = chercher_offre(element)
                if trouve:
                    return trouve
        return None

    return chercher_offre(objet_json)


def try_extract_price(url: str, timeout: float = 6.0) -> str | None:
    """Retourne une chaîne de prix si trouvée sur la page, sinon None (rapide & robuste)."""
    try:
        reponse = requests.get(url, headers=HEADERS, timeout=timeout)
        if reponse.status_code >= 400 or not reponse.headers.get("content-type", "").startswith("text/html"):
            return None

        texte_html = reponse.text
        url_de_base = get_base_url(texte_html, reponse.url)

        # 1) Métadonnées structurées (JSON-LD, microdata, RDFa) — la façon la plus fiable
        donnees_structurees = extruct.extract(texte_html, base_url=url_de_base, syntaxes=["json-ld", "microdata", "opengraph", "rdfa"])
        # JSON-LD en priorité
        for bloc in (donnees_structurees.get("json-ld") or []):
            prix = _extraire_prix_depuis_jsonld(bloc)
            if prix:
                return prix
        # Microdata/RDFa (solution de repli)
        for bloc in (donnees_structurees.get("microdata") or []) + (donnees_structurees.get("rdfa") or []):
            prix = _extraire_prix_depuis_jsonld(bloc)
            if prix:
                return prix

        # 2) OpenGraph (parfois og:price:amount / og:price:currency)
        balises_opengraph = {
            (balise.get("property") or balise.get("name") or "").lower(): balise.get("content")
            for balise in (donnees_structurees.get("opengraph") or []) if isinstance(balise, dict)
        }
        if balises_opengraph.get("og:price:amount"):
            montant = balises_opengraph.get("og:price:amount")
            devise = balises_opengraph.get("og:price:currency") or ""
            return f"{montant} {devise}".strip() if montant else None

        # 3) Dernier recours : on cherche un prix directement dans le texte avec nos regex
        for motif_regex in PRICE_REGEXES:
            trouve = motif_regex.search(html.unescape(texte_html))
            if trouve:
                groupes = trouve.groupdict()
                valeur = (groupes.get("val") or "").strip()
                devise = (groupes.get("cur") or "").strip()
                if valeur:
                    return f"{valeur} {devise}".strip()
    except Exception:
        return None

# ------------------------------------------------------------#
# Outils: téléchargement & nettoyage                          #
# ------------------------------------------------------------#
def fetch_article_text(url: str, timeout: int = 20) -> Dict[str, str]:
    """
    Récupère le HTML d'une page puis en extrait le texte brut lisible
    (sans les balises, scripts, styles, etc.).
    Retourne {"url": url, "title": "...", "text": "..."}.
    """
    entetes_requete = {
        "User-Agent": "Mozilla/5.0 (compatible; AchatResponsableBot/1.0; +https://example.local)"
    }
    reponse = requests.get(url, headers=entetes_requete, timeout=timeout)
    reponse.raise_for_status()
    code_html = reponse.text

    soupe_html = BeautifulSoup(code_html, "html.parser")
    # titre de la page
    titre = (soupe_html.title.string.strip() if soupe_html.title and soupe_html.title.string else url)
    # on supprime scripts/styles (ils ne contiennent pas de texte utile pour l'analyse)
    for balise in soupe_html(["script", "style", "noscript"]):
        balise.decompose()
    texte = soupe_html.get_text("\n")
    # on nettoie les espaces et sauts de ligne en trop
    texte = re.sub(r"\n{2,}", "\n", texte)
    texte = re.sub(r"[ \t]{2,}", " ", texte).strip()

    # on limite la taille (évite d'envoyer un texte énorme à l'IA, ce qui coûterait cher et serait lent)
    if len(texte) > 15000:
        texte = texte[:15000]

    return {"url": url, "title": titre, "text": texte}

# ------------------------------------------------------------#
# Helpers d'extraction (à coller sous fetch_article_text)     #
# ------------------------------------------------------------#
def extract_text_from_html(code_html: str) -> tuple[str, str]:
    """Titre + texte lisible à partir d'un HTML déjà téléchargé (ou déjà rendu par un navigateur)."""
    soupe_html = BeautifulSoup(code_html, "html.parser")
    titre = soupe_html.title.string.strip() if soupe_html.title and soupe_html.title.string else ""
    for balise in soupe_html(["script", "style", "noscript"]):
        balise.decompose()
    texte = soupe_html.get_text("\n")
    texte = re.sub(r"\n{2,}", "\n", texte)
    texte = re.sub(r"[ \t]{2,}", " ", texte).strip()
    if len(texte) > 15000:
        texte = texte[:15000]
    return titre, texte


def parse_jsonld_product(code_html: str, url: str) -> dict | None:
    """Essaie d'extraire un bloc 'Product' depuis le JSON-LD (schema.org/Product) d'une page."""
    try:
        donnees_json_ld = extruct.extract(code_html, base_url=url, syntaxes=["json-ld"])
        blocs = donnees_json_ld.get("json-ld", []) or []
        produit_trouve = None
        for bloc in blocs:
            if isinstance(bloc, dict):
                types_declares = bloc.get("@type")
                if isinstance(types_declares, str):
                    types_declares = [types_declares]
                if types_declares and "Product" in [t if isinstance(t, str) else "" for t in types_declares]:
                    produit_trouve = bloc
                    break
        if not produit_trouve:
            return None

        # Construit un petit texte utile pour l'IA à partir des champs trouvés
        lignes_texte = []
        nom = produit_trouve.get("name")
        marque = produit_trouve.get("brand")
        materiau = produit_trouve.get("material")
        couleur = produit_trouve.get("color")
        description = produit_trouve.get("description")
        code_gtin = produit_trouve.get("gtin13") or produit_trouve.get("gtin12") or produit_trouve.get("gtin")

        if nom:
            lignes_texte.append(f"Nom: {nom}")
        if marque:
            if isinstance(marque, dict):
                marque = marque.get("name", "")
            if marque:
                lignes_texte.append(f"Marque: {marque}")
        if materiau:
            lignes_texte.append(f"Matériaux: {materiau}")
        if couleur:
            lignes_texte.append(f"Couleur: {couleur}")
        if code_gtin:
            lignes_texte.append(f"GTIN: {code_gtin}")
        if description:
            lignes_texte.append(f"Description: {description}")

        offres = produit_trouve.get("offers")

        def ligne_prix(offre):
            prix = (offre or {}).get("price")
            devise = (offre or {}).get("priceCurrency")
            return f"Prix: {prix} {devise or ''}".strip() if prix else None

        if isinstance(offres, dict):
            ligne = ligne_prix(offres)
            if ligne:
                lignes_texte.append(ligne)
        elif isinstance(offres, list) and offres:
            ligne = ligne_prix(offres[0])
            if ligne:
                lignes_texte.append(ligne)

        texte_final = "\n".join(lignes_texte)
        return {"title": nom or "", "text": texte_final}
    except Exception:
        return None


def fetch_rendered(url: str, timeout_ms: int = 35000) -> dict:
    """
    Charge la page via un navigateur invisible (headless) qui exécute le JavaScript
    — nécessaire pour les sites qui affichent leur contenu dynamiquement.
    Tente d'abord d'utiliser le JSON-LD 'Product', puis complète avec le texte de la page.
    """
    with sync_playwright() as playwright:
        navigateur = playwright.chromium.launch(headless=True)
        contexte_navigateur = navigateur.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36")
        )
        page_navigateur = contexte_navigateur.new_page()
        page_navigateur.goto(url, timeout=timeout_ms, wait_until="networkidle")
        code_html = page_navigateur.content()
        titre_rendu = page_navigateur.title()
        contexte_navigateur.close()
        navigateur.close()

    # 1) JSON-LD si possible (c'est la source la plus fiable)
    bloc_produit = parse_jsonld_product(code_html, url)
    titre_json_ld = bloc_produit.get("title") if bloc_produit else ""
    texte_json_ld = bloc_produit.get("text") if bloc_produit else ""

    # 2) Texte brut de secours (au cas où le JSON-LD serait absent ou incomplet)
    titre_texte_brut, texte_brut = extract_text_from_html(code_html)

    titre_final = titre_json_ld or titre_rendu or titre_texte_brut or url
    texte_complet = "\n\n".join([partie for partie in [texte_json_ld, texte_brut] if partie]).strip()
    if len(texte_complet) > 15000:
        texte_complet = texte_complet[:15000]

    return {"url": url, "title": titre_final, "text": texte_complet}


def _match_allowed_domain(host: str, domaines_autorises: set[str]) -> bool:
    """Vérifie que 'host' correspond à un domaine autorisé (ou à un de ses sous-domaines).
    Exemple : m.amazon.ca et pages.ebay.com doivent être reconnus comme amazon.ca / ebay.com."""
    nom_hote = host.lower().split(":", 1)[0]   # enlève un éventuel :port
    if nom_hote.startswith("www."):
        nom_hote = nom_hote[4:]
    return any(nom_hote == domaine or nom_hote.endswith("." + domaine) for domaine in domaines_autorises)


def smart_fetch(url: str) -> dict:
    """
    Choisit automatiquement la bonne façon de récupérer une page :
      - un navigateur headless (plus lent, mais nécessaire) pour les sites connus
        où le contenu est chargé en JavaScript,
      - un simple téléchargement HTML sinon (beaucoup plus rapide).
    """
    nom_hote = urlparse(url).netloc
    if ALLOW_HEADLESS and _match_allowed_domain(nom_hote, HEADLESS_DOMAINS):
        return fetch_rendered(url)
    return fetch_article_text(url)

#------------------------------------#
#   Le Prompt LLM                    #
#------------------------------------#
# C'est le message "système" qui explique à l'IA quel rôle elle doit jouer.
INSTRUCTIONS_SYSTEME_IA = (
    "Tu es un expert en analyse du cycle de vie (ACV) et en durabilité. "
    "Tu lis un article produit/annonce/blog et tu extrais des faits concrets "
    "pour dresser un portrait écologique. Ne fabrique pas de chiffres si le texte "
    "n'en contient pas; dans ce cas, mets null et explique dans 'other_notes'. "
    "Réponds STRICTEMENT en JSON valide, sans texte autour."
)


def build_user_prompt(contenu_page: Dict[str, str]) -> str:
    """
    Construit le message envoyé à l'IA : on lui donne un "schéma" (la forme
    exacte du JSON qu'on veut recevoir en retour) ainsi que le texte de la page.
    """
    structure_attendue = {
        "url": contenu_page["url"],
        "title": contenu_page["title"],
        "category": "one of ['vetements','maison_meubles','electronique','electromenagers','sport_plein_air','produits_menagers','jouets','bagagerie','bricolage','vehicules','autre'] — catégorie générale du produit",
        "features": {
            "materials": "string: matériaux mentionnés (ex: coton bio, polyester recyclé...)",
            "water_use_liters": "number|null: litres (si mentionné, sinon null)",
            "energy_use_kwh": "number|null",
            "co2e_kg": "number|null: CO2e kg (si mentionné)",
            "biodegradability": "one of ['biodegradable','partially','non','unknown']",
            "recyclability": "one of ['high','medium','low','unknown']",
            "durability_repairability": "one of ['high','medium','low','unknown']",
            "certifications": ["array of strings (ex: B Corp, OEKO-TEX, FSC, GOTS, EPEAT, Energy Star)"],
            "packaging": "string: infos sur l’emballage si présent",
            "transport": "string: infos logistique (local, import, etc.)",
            "other_notes": "string: précisions/citations",
            "confidence": "number 0-1: confiance de l’extraction"
        },
        "subscores": {
            "materials": "0-100 (mieux=score élevé; ex: matières recyclées/bio = plus haut)",
            "water": "0-100 (moins d'eau = plus haut)",
            "energy": "0-100 (moins d'énergie = plus haut)",
            "co2e": "0-100 (moins d'émissions = plus haut)",
            "biodegradability_recyclability": "0-100",
            "durability": "0-100",
            "certifications": "0-100 (plus de labels pertinents = plus haut)",
            "packaging_transport": "0-100 (emballage recyclable/minimal + transport court = plus haut)"
        }
    }

    consignes = (
        "Lis le CONTENU ci-dessous et remplis le SCHEMA. "
        "Utilise uniquement les informations disponibles (ou 'unknown/null'). "
        "Si des nombres sont fournis dans le texte (ex: litres d'eau, kg CO2e), capture-les.\n\n"
        f"SCHEMA (exemple de clés attendues):\n{json.dumps(structure_attendue, ensure_ascii=False, indent=2)}\n\n"
        "CONTENU:\n"
        f"URL: {contenu_page['url']}\n"
        f"TITRE: {contenu_page['title']}\n"
        f"TEXTE:\n{contenu_page['text']}\n\n"
        "RÉPONDS UNIQUEMENT AVEC UN JSON VALIDE."
    )
    return consignes


def call_llm(contenu_page: Dict[str, str]) -> Dict[str, Any]:
    """
    Envoie le contenu de la page à l'IA (OpenAI) et renvoie sa réponse sous
    forme de dictionnaire Python. Tolère que la réponse soit entourée de
    balises ```...``` (les IA ajoutent parfois ça par habitude).
    """
    message_utilisateur = build_user_prompt(contenu_page)

    reponse_api = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": INSTRUCTIONS_SYSTEME_IA},
            {"role": "user", "content": message_utilisateur},
        ],
        temperature=0.2,
    )
    contenu_reponse = reponse_api.choices[0].message.content

    # Nettoie les éventuelles balises ```json ... ``` autour de la réponse
    contenu_reponse = contenu_reponse.strip()
    contenu_reponse = re.sub(r"^```(json)?", "", contenu_reponse).strip()
    contenu_reponse = re.sub(r"```$", "", contenu_reponse).strip()

    try:
        donnees_json = json.loads(contenu_reponse)
    except Exception:
        # Dernier recours : on extrait le premier bloc {...} trouvé dans le texte
        bloc_trouve = re.search(r"\{.*\}", contenu_reponse, flags=re.DOTALL)
        if not bloc_trouve:
            raise ValueError("Réponse LLM non JSON.")
        donnees_json = json.loads(bloc_trouve.group(0))
    return donnees_json

#------------------------------------#
# Scoring: pondérations (mieux = score plus haut)
#------------------------------------#
# Chaque critère a un "poids" (son importance relative dans la note finale).
# La somme de tous les poids fait 1.0 (= 100%).
POIDS_CRITERES_SCORE = {
    "materials": 0.18,
    "water": 0.14,
    "energy": 0.14,
    "co2e": 0.18,
    "biodegradability_recyclability": 0.14,
    "durability": 0.10,
    "certifications": 0.07,
    "packaging_transport": 0.05,
}


def compute_eco_score(sous_scores: Dict[str, Any]) -> float:
    """
    Calcule le score écologique global (entre 0 et 100) à partir des
    sous-scores renvoyés par l'IA, en appliquant la pondération de chaque
    critère. Si un critère manque, on lui donne la valeur neutre 50.
    """
    score_total = 0.0
    for nom_critere, poids in POIDS_CRITERES_SCORE.items():
        valeur = sous_scores.get(nom_critere, 50)
        try:
            valeur = float(valeur)
        except Exception:
            valeur = 50.0
        score_total += poids * valeur
    return round(score_total, 2)


def _extract_domain(url: str) -> str:
    """Renvoie juste le nom de domaine d'une URL (ex: 'amazon.ca' pour 'https://www.amazon.ca/xyz')."""
    try:
        nom_hote = urlparse(url).netloc.lower()
        return nom_hote[4:] if nom_hote.startswith("www.") else nom_hote
    except Exception:
        return ""


# --- Google Cloud Vision: Web Detection (image -> pages similaires) ---
def gcv_web_detection(donnees_image: bytes, nom_fichier: str = "upload.jpg") -> list[dict]:
    """
    Utilise Google Cloud Vision (Web Detection) pour trouver des pages web qui
    contiennent cette image (ou une variante très proche). Retourne une liste
    d'articles: {thumb, url, site, price(None au départ)}.
    """
    from urllib.parse import urlparse

    try:
        from google.cloud import vision
        client_vision = vision.ImageAnnotatorClient()
        reponse = client_vision.web_detection(image=vision.Image(content=donnees_image))
        detection_web = reponse.web_detection
    except Exception:
        return []

    resultats: list[dict] = []
    if detection_web and detection_web.pages_with_matching_images:
        for page_trouvee in detection_web.pages_with_matching_images:
            url_page = (page_trouvee.url or "").strip()
            if not url_page:
                continue
            # miniature si disponible (une correspondance "complète" est préférée à "partielle")
            premiere_image = (page_trouvee.full_matching_images or page_trouvee.partial_matching_images or [None])[0]
            miniature = getattr(premiere_image, "url", None) if premiere_image else None

            nom_hote = urlparse(url_page).netloc.lower()
            if nom_hote.startswith("www."):
                nom_hote = nom_hote[4:]

            resultats.append({
                "thumb": miniature,
                "url": url_page,
                "site": nom_hote,
                "price": None,  # Google Vision ne renvoie pas de prix
            })

    # dédoublonnage par URL (on garde l'ordre d'apparition)
    urls_vues, resultats_uniques = set(), []
    for resultat in resultats:
        if resultat["url"] in urls_vues:
            continue
        urls_vues.add(resultat["url"])
        resultats_uniques.append(resultat)

    # --- Enrichissement du prix sur les N premiers résultats (pour éviter d'être trop lent) ---
    nombre_max_a_enrichir = 8  # ajuste si tu veux plus/moins de scraping
    for article in resultats_uniques[:nombre_max_a_enrichir]:
        prix = try_extract_price(article["url"])
        if prix:
            article["price"] = prix

    return resultats_uniques

# ------------------------------
# Routes
# ------------------------------
# Une "route" est une adresse (URL) que le site sait gérer. Flask exécute
# la fonction juste en-dessous chaque fois qu'un visiteur ouvre cette adresse.

@app.get("/")
def index():
    enregistrer_visite("index")
    return render_template("index.html")


@app.post("/api/analyze")
def api_analyze():
    """
    Reçoit une liste d'URLs (envoyée par le JavaScript de la page d'accueil),
    analyse chaque page une par une, puis renvoie un classement écologique
    au format JSON.
    """
    donnees_requete = request.get_json(force=True)
    urls: List[str] = [u.strip() for u in donnees_requete.get("urls", []) if u.strip()]
    if len(urls) < 2:
        return jsonify({"error": "Veuillez fournir au moins 2 URL."}), 400

    resultats = []
    erreurs = []

    for url in urls:
        try:
            contenu_page = smart_fetch(url)
            analyse_ia = call_llm(contenu_page)
            sous_scores = analyse_ia.get("subscores", {}) or {}
            score_ecologique = compute_eco_score(sous_scores)
            categorie = analyse_ia.get("category") or "autre"
            enregistrer_soumission(contenu_page["url"], categorie)
            resultats.append({
                "url": contenu_page["url"],
                "title": analyse_ia.get("title") or contenu_page["title"],
                "features": analyse_ia.get("features", {}),
                "subscores": sous_scores,
                "eco_score": score_ecologique
            })
        except Exception as erreur:
            erreurs.append({"url": url, "error": str(erreur)})

    # Classement décroissant (le meilleur score en premier)
    classement = sorted(resultats, key=lambda article: article["eco_score"], reverse=True)
    # Ajoute le rang (1er, 2e, 3e...) à chaque article
    for rang, article in enumerate(classement, start=1):
        article["rank"] = rang

    return jsonify({"results": classement, "errors": erreurs})


@app.route("/api/analyze-one", methods=["POST", "OPTIONS"])
def api_analyze_one():
    """
    Analyse une seule URL — pensé pour l'extension de navigateur : quand tu es
    sur une page produit, l'extension appelle cette route pour obtenir tout
    de suite un score écologique, sans avoir besoin d'un deuxième article à comparer.

    Contrairement à /api/analyze, cette route accepte les requêtes venant de
    n'importe quel site (CORS ouvert) puisqu'elle est appelée directement
    depuis la page du détaillant que tu visites (ex: amazon.ca), pas depuis
    notre propre site.
    """
    if request.method == "OPTIONS":
        # Réponse "vide" au preflight CORS que le navigateur envoie avant le vrai POST.
        return ("", 204)

    donnees_requete = request.get_json(force=True)
    url = (donnees_requete.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL manquante."}), 400

    try:
        contenu_page = smart_fetch(url)
        analyse_ia = call_llm(contenu_page)
        sous_scores = analyse_ia.get("subscores", {}) or {}
        score_ecologique = compute_eco_score(sous_scores)
        categorie = analyse_ia.get("category") or "autre"
        enregistrer_soumission(contenu_page["url"], categorie)
        return jsonify({
            "url": contenu_page["url"],
            "title": analyse_ia.get("title") or contenu_page["title"],
            "category": categorie,
            "features": analyse_ia.get("features", {}),
            "eco_score": score_ecologique,
        })
    except Exception as erreur:
        return jsonify({"error": str(erreur)}), 500


@app.after_request
def autoriser_cors_pour_extension(reponse):
    """
    Ajoute les en-têtes CORS uniquement sur la route utilisée par l'extension
    de navigateur — le reste du site n'a pas besoin d'être appelable depuis
    un autre domaine, donc on ne l'ouvre pas partout.
    """
    if request.path == "/api/analyze-one":
        reponse.headers["Access-Control-Allow-Origin"] = "*"
        reponse.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        reponse.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return reponse


@app.post("/api/share")
def api_share():
    """
    Sauvegarde le résultat d'une comparaison déjà calculée, pour qu'on puisse
    la partager via un lien court et permanent (utile pour les réseaux sociaux :
    chaque comparaison devient une petite page avec son propre aperçu).
    """
    donnees_requete = request.get_json(force=True)
    resultats = donnees_requete.get("results", [])
    if not resultats or len(resultats) < 2:
        return jsonify({"error": "Rien à partager pour l'instant."}), 400

    # On ne garde que les champs utiles à l'affichage public (pas besoin
    # de tout le détail des sous-scores pour cette page de partage).
    resultats_a_sauvegarder = [{
        "title": article.get("title"),
        "url": article.get("url"),
        "eco_score": article.get("eco_score"),
        "rank": article.get("rank"),
        "features": article.get("features", {}),
    } for article in resultats[:10]]  # limite raisonnable

    identifiant = secrets.token_urlsafe(6)  # ex: "aZ3xQk1" — assez court pour un lien, assez long pour éviter les collisions
    connexion = sqlite3.connect(STATS_DB_PATH)
    connexion.execute(
        "INSERT INTO shared_results (id, payload_json) VALUES (?, ?)",
        (identifiant, json.dumps(resultats_a_sauvegarder, ensure_ascii=False))
    )
    connexion.commit()
    connexion.close()

    return jsonify({"share_url": url_for("page_resultat_partage", identifiant=identifiant, _external=True)})


@app.get("/r/<identifiant>")
def page_resultat_partage(identifiant):
    """Page publique et permanente montrant un résultat de comparaison déjà calculé."""
    connexion = sqlite3.connect(STATS_DB_PATH)
    connexion.row_factory = sqlite3.Row
    ligne = connexion.execute(
        "SELECT payload_json FROM shared_results WHERE id = ?", (identifiant,)
    ).fetchone()
    connexion.close()

    if not ligne:
        return render_template("partage.html", resultats=None), 404

    resultats = json.loads(ligne["payload_json"])
    gagnant = resultats[0] if resultats else None
    return render_template("partage.html", resultats=resultats, gagnant=gagnant)

# ---------------------------------------------------------------------------------------------------------
# Page 2 : recherche par images / Simple, clair : on upload, on interroge Google Vision + Google Search,
# on formate la réponse en (thumb, url, site, price), et on affiche.
# ------------------------------------------------------------------------------------------------------------------
@app.get("/images")
def images_page():
    enregistrer_visite("images")
    return render_template("images.html")


@app.get("/exemples")
def exemples_page():
    enregistrer_visite("exemples")
    return render_template("exemples.html")


@app.get("/don")
def don_page():
    return render_template("don.html")


@app.route("/stats", methods=["GET", "POST"])
def stats_page():
    """
    Tableau de bord réservé à l'administrateur du site.
    - En GET, si l'admin n'est pas encore connecté, on affiche le formulaire de mot de passe.
    - En POST, on vérifie le mot de passe envoyé; s'il est bon, on "connecte" l'admin
      via la session (il n'aura plus à le retaper tant que sa session est valide).
    """
    if request.method == "POST":
        mot_de_passe_saisi = request.form.get("password", "")
        # hmac.compare_digest compare deux chaînes de façon "à temps constant" :
        # ça évite qu'un attaquant devine le mot de passe caractère par caractère
        # en mesurant le temps de réponse du serveur.
        if ADMIN_PASSWORD and hmac.compare_digest(mot_de_passe_saisi, ADMIN_PASSWORD):
            session["is_admin"] = True
        else:
            return render_template("stats.html", authed=False, error="Mot de passe incorrect."), 401

    if not session.get("is_admin"):
        return render_template("stats.html", authed=False)

    from collections import Counter

    connexion = sqlite3.connect(STATS_DB_PATH)
    connexion.row_factory = sqlite3.Row  # permet d'accéder aux colonnes par leur nom (ligne["city"])

    lignes_categories = connexion.execute(
        "SELECT category, COUNT(*) as n FROM submissions GROUP BY category ORDER BY n DESC"
    ).fetchall()
    categories = [
        {"label": LIBELLES_CATEGORIES.get(ligne["category"], ligne["category"] or "Autre"), "count": ligne["n"]}
        for ligne in lignes_categories
    ]

    visiteurs_par_lieu = {
        (ligne["city"], ligne["region"], ligne["country"]): ligne["n"]
        for ligne in connexion.execute("SELECT city, region, country, COUNT(DISTINCT ip_hash) as n FROM visits GROUP BY city, region, country")
    }
    soumissions_par_lieu = {
        (ligne["city"], ligne["region"], ligne["country"]): ligne["n"]
        for ligne in connexion.execute("SELECT city, region, country, COUNT(*) as n FROM submissions GROUP BY city, region, country")
    }

    # Vue d'ensemble : volumes bruts + taux de conversion + sites les plus demandés
    # (indicateurs utiles pour démontrer la traction du projet)
    visiteurs_uniques = connexion.execute("SELECT COUNT(DISTINCT ip_hash) as n FROM visits").fetchone()["n"]
    total_pages_vues = connexion.execute("SELECT COUNT(*) as n FROM visits").fetchone()["n"]
    total_soumissions = connexion.execute("SELECT COUNT(*) as n FROM submissions").fetchone()["n"]
    visiteurs_ayant_soumis = connexion.execute("SELECT COUNT(DISTINCT ip_hash) as n FROM submissions").fetchone()["n"]
    nombre_pays_atteints = connexion.execute(
        "SELECT COUNT(DISTINCT country) as n FROM visits WHERE country NOT IN ('Local','Inconnu')"
    ).fetchone()["n"]

    urls_soumises = [ligne["url"] for ligne in connexion.execute("SELECT url FROM submissions")]
    connexion.close()

    # % de visiteurs ayant utilisé la fonctionnalité principale au moins une fois
    taux_de_conversion = round((visiteurs_ayant_soumis / visiteurs_uniques * 100), 1) if visiteurs_uniques else 0.0

    compteur_domaines = Counter(_extract_domain(url) or "(inconnu)" for url in urls_soumises)
    domaines_les_plus_demandes = [{"domain": domaine, "count": nombre} for domaine, nombre in compteur_domaines.most_common(10)]

    vue_ensemble = {
        "unique_visitors": visiteurs_uniques,
        "total_pageviews": total_pages_vues,
        "total_submissions": total_soumissions,
        "conversion_rate": taux_de_conversion,
        "countries_reached": nombre_pays_atteints,
    }

    lieux = [
        {
            "city": ville, "region": region, "country": pays,
            "visitors": visiteurs_par_lieu.get((ville, region, pays), 0),
            "submissions": soumissions_par_lieu.get((ville, region, pays), 0),
        }
        for ville, region, pays in (set(visiteurs_par_lieu) | set(soumissions_par_lieu))
    ]
    lieux.sort(key=lambda lieu: (lieu["submissions"], lieu["visitors"]), reverse=True)

    return render_template(
        "stats.html", authed=True,
        overview=vue_ensemble, categories=categories, locations=lieux, top_domains=domaines_les_plus_demandes
    )


@app.post("/images")
def images_search():
    fichier_recu = request.files.get("file")
    if not fichier_recu or not fichier_recu.filename:
        return render_template("images.html", error="Aucun fichier reçu.")

    donnees_image = fichier_recu.read()

    try:
        resultat_recherche = recherche_image_complete(donnees_image)
    except Exception as erreur:
        return render_template("images.html", error=f"Erreur lors de la recherche : {erreur}")

    type_mime = deviner_type_image(donnees_image)
    resultat = {
        "query_filename": fichier_recu.filename,
        "query_preview_b64": (
            f"data:{type_mime};base64,"
            + base64.b64encode(donnees_image).decode("utf-8")
        ),
        "description": resultat_recherche.get("description", ""),
        "items": resultat_recherche["items"]
    }

    return render_template("images.html", result=resultat)


# -----------------------------------------------------------------#
# Routes pour aider l'indexation sur les moteurs de recherche
# -----------------------------------------------------------------#

@app.get("/robots.txt")
def robots():
    """Fichier standard qui indique aux moteurs de recherche ce qu'ils peuvent explorer."""
    contenu_reponse = "User-agent: *\nAllow: /\nSitemap: " + url_for('sitemap', _external=True) + "\n"
    return Response(contenu_reponse, mimetype="text/plain")


@app.get("/sitemap.xml")
def sitemap():
    """Liste des pages du site, pour aider Google/Bing à toutes les découvrir."""
    liste_urls = [
        url_for("index", _external=True),
        url_for("images_page", _external=True),
        url_for("exemples_page", _external=True),
    ]
    date_du_jour = datetime.utcnow().date().isoformat()
    lignes_xml = ['<?xml version="1.0" encoding="UTF-8"?>',
                  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for url_page in liste_urls:
        lignes_xml += [f"<url><loc>{url_page}</loc><lastmod>{date_du_jour}</lastmod><changefreq>weekly</changefreq></url>"]
    lignes_xml.append("</urlset>")
    return Response("\n".join(lignes_xml), mimetype="application/xml")

# ------------------------------
# Main (local)
# ------------------------------
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("   Ouvre un terminal et exporte ta clé :")
        print("   Windows PowerShell: $env:OPENAI_API_KEY='sk-...'\n"
              "   macOS/Linux: export OPENAI_API_KEY='sk-...'")

    app.run(debug=True)
