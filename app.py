import os, re, json, math, time
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

#---------  Imports pour couvrir le SCRAPPING -------------------#

from urllib.parse import urlparse   # librairie pour utiliser le smart_fetch # Recherche visuelle (Bing Visual Search)
from playwright.sync_api import sync_playwright
import extruct, w3lib.html

# ---------------------------------------------------------------#
# Recherche visuelle (Bing Visual Search)
# ---------------------------------------------------------------#
import mimetypes
#----------------------------------------------------------------#

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
load_dotenv()
#-------------------------------#
# Config de base                #
#-------------------------------#
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change si tu veux
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Paramètre pour activer/désactiver le rendu headless ---------------#
ALLOW_HEADLESS = os.getenv("ALLOW_HEADLESS_FETCH", "false").lower() in ("1","true","yes")

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

BING_VISION_KEY = os.getenv("BING_VISION_KEY")
BING_VISION_ENDPOINT = os.getenv("BING_VISION_ENDPOINT", "https://api.bing.microsoft.com/v7.0/images/visualsearch")

# -----------------------------------------------------------------------------#

app = Flask(__name__)

#------------------------------------------------------------------------------#
# 4 variantes courantes de prix (CAD/$/USD/€)
PRICE_REGEXES = [
    re.compile(r'(?:(?P<cur>\$|CAD|C\$)\s?(?P<val>\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d{2})?))', re.I),
    re.compile(r'(?:(?P<val>\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d{2})?)\s?(?P<cur>CAD|\$|C\$))', re.I),
    re.compile(r'(?:(?P<cur>€)\s?(?P<val>\d{1,3}(?:[ .]\d{3})*(?:[,]\d{2})?))', re.I),
    re.compile(r'(?:(?P<cur>USD)\s?(?P<val>\d{1,3}(?:[ ,]\d{3})*(?:[.]\d{2})?))', re.I),
]

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
Compress(app)  # <-- ici, juste après la création de app

#--------------- (Fin de stratégies de référencement) -------------------------#

def _extract_price_from_jsonld(data):
    """Explore JSON-LD/Microdata pour Offer/AggregateOffer."""
    def norm(v):
        if isinstance(v, (int, float)): return f"{v}"
        if isinstance(v, str): return v.strip()
        return None

    def scan(obj):
        if isinstance(obj, dict):
            t = norm(obj.get("@type")) or norm(obj.get("type"))
            if t and t.lower() in {"offer", "aggregateoffer"}:
                price = norm(obj.get("price") or obj.get("lowPrice") or obj.get("highPrice"))
                cur   = norm(obj.get("priceCurrency"))
                if price:
                    return f"{price} {cur}".strip()
            for v in obj.values():
                out = scan(v)
                if out: return out
        elif isinstance(obj, list):
            for v in obj:
                out = scan(v)
                if out: return out
        return None

    return scan(data)

def try_extract_price(url: str, timeout: float = 6.0) -> str | None:
    """Retourne une chaîne de prix si trouvée, sinon None (rapide & robuste)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code >= 400 or not resp.headers.get("content-type", "").startswith("text/html"):
            return None

        html_text = resp.text
        base_url = get_base_url(html_text, resp.url)

        # 1) Métadonnées structurées (JSON-LD, microdata, RDFa)
        data = extruct.extract(html_text, base_url=base_url, syntaxes=["json-ld", "microdata", "opengraph", "rdfa"])
        # JSON-LD en priorité
        for block in (data.get("json-ld") or []):
            price = _extract_price_from_jsonld(block)
            if price: return price
        # Microdata/RDFa (fallback)
        for block in (data.get("microdata") or []) + (data.get("rdfa") or []):
            price = _extract_price_from_jsonld(block)
            if price: return price

        # 2) OpenGraph (parfois og:price:amount / og:price:currency)
        og = { (p.get("property") or p.get("name") or "").lower(): p.get("content") 
               for p in (data.get("opengraph") or []) if isinstance(p, dict) }
        if og.get("og:price:amount"):
            amount = og.get("og:price:amount")
            currency = og.get("og:price:currency") or ""
            return f"{amount} {currency}".strip() if amount else None

        # 3) Regex sur le texte
        for RX in PRICE_REGEXES:
            m = RX.search(html.unescape(html_text))
            if m:
                g = m.groupdict()
                val = (g.get("val") or "").strip()
                cur = (g.get("cur") or "").strip()
                if val:
                    return f"{val} {cur}".strip()
    except Exception:
        return None
    
# ------------------------------------------------------------#
# Outils: téléchargement & nettoyage                          #
# ------------------------------------------------------------#
def fetch_article_text(url: str, timeout: int = 20) -> Dict[str, str]:
    """
    Récupère HTML puis texte brut lisible d’une page.
    Retourne {"url": url, "title": "...", "text": "..."}.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AchatResponsableBot/1.0; +https://example.local)"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text

    soup = BeautifulSoup(html, "html.parser")
    # titre
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    # supprime scripts/styles/nav/footer
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    # nettoie
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    # limite (évite des prompts énormes)
    if len(text) > 15000:
        text = text[:15000]

    return {"url": url, "title": title, "text": text}

# ------------------------------------------------------------#
# Helpers d’extraction (à coller sous fetch_article_text)     #
# ------------------------------------------------------------#
def extract_text_from_html(html: str) -> tuple[str, str]:
    """Titre + texte lisible à partir d'un HTML déjà rendu."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if len(text) > 15000:
        text = text[:15000]
    return title, text


def parse_jsonld_product(html: str, url: str) -> dict | None:
    """Essaie d'extraire un bloc Product depuis JSON-LD (schema.org/Product)."""
    try:
        data = extruct.extract(html, base_url=url, syntaxes=["json-ld"])
        blocks = data.get("json-ld", []) or []
        product = None
        for item in blocks:
            if isinstance(item, dict):
                types = item.get("@type")
                if isinstance(types, str):
                    types = [types]
                if types and "Product" in [t if isinstance(t, str) else "" for t in types]:
                    product = item
                    break
        if not product:
            return None

        # Construit un petit texte utile pour le LLM
        parts = []
        name = product.get("name")
        brand = product.get("brand")
        material = product.get("material")
        color = product.get("color")
        description = product.get("description")
        gtin = product.get("gtin13") or product.get("gtin12") or product.get("gtin")

        if name: parts.append(f"Nom: {name}")
        if brand:
            if isinstance(brand, dict): brand = brand.get("name", "")
            if brand: parts.append(f"Marque: {brand}")
        if material: parts.append(f"Matériaux: {material}")
        if color: parts.append(f"Couleur: {color}")
        if gtin: parts.append(f"GTIN: {gtin}")
        if description: parts.append(f"Description: {description}")

        offers = product.get("offers")
        def offer_line(off):
            price = (off or {}).get("price")
            currency = (off or {}).get("priceCurrency")
            return f"Prix: {price} {currency or ''}".strip() if price else None

        if isinstance(offers, dict):
            line = offer_line(offers)
            if line: parts.append(line)
        elif isinstance(offers, list) and offers:
            line = offer_line(offers[0])
            if line: parts.append(line)

        text = "\n".join(parts)
        return {"title": name or "", "text": text}
    except Exception:
        return None


def fetch_rendered(url: str, timeout_ms: int = 35000) -> dict:
    """
    Charge la page via un navigateur headless (JS exécuté),
    tente d'utiliser JSON-LD Product, puis complète avec le texte de la page.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36")
        )
        page = context.new_page()
        page.goto(url, timeout=timeout_ms, wait_until="networkidle")
        html = page.content()
        title_rendered = page.title()
        context.close()
        browser.close()

    # 1) JSON-LD si possible
    jl = parse_jsonld_product(html, url)
    jl_title = jl.get("title") if jl else ""
    jl_text = jl.get("text") if jl else ""

    # 2) Texte brut de secours
    bs_title, bs_text = extract_text_from_html(html)

    final_title = jl_title or title_rendered or bs_title or url
    combined_text = "\n\n".join([part for part in [jl_text, bs_text] if part]).strip()
    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    return {"url": url, "title": final_title, "text": combined_text}


def _match_allowed_domain(host: str, allowed: set[str]) -> bool:
    """Match exact domain ou sous-domaine. Ex: m.amazon.ca, pages.ebay.com."""
    h = host.lower().split(":", 1)[0]   # enlève un éventuel :port
    if h.startswith("www."):
        h = h[4:]
    return any(h == d or h.endswith("." + d) for d in allowed)


def smart_fetch(url: str) -> dict:
    host = urlparse(url).netloc
    if ALLOW_HEADLESS and _match_allowed_domain(host, HEADLESS_DOMAINS):
        return fetch_rendered(url)
    return fetch_article_text(url)

#------------------------------------#
#   Le Prompt LLM                    #
#------------------------------------#
SYSTEM = (
    "Tu es un expert en analyse du cycle de vie (ACV) et en durabilité. "
    "Tu lis un article produit/annonce/blog et tu extrais des faits concrets "
    "pour dresser un portrait écologique. Ne fabrique pas de chiffres si le texte "
    "n'en contient pas; dans ce cas, mets null et explique dans 'other_notes'. "
    "Réponds STRICTEMENT en JSON valide, sans texte autour."
)

def build_user_prompt(doc: Dict[str, str]) -> str:
    """
    Construit les consignes pour obtenir un JSON standardisé.
    """
    schema = {
        "url": doc["url"],
        "title": doc["title"],
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

    instructions = (
        "Lis le CONTENU ci-dessous et remplis le SCHEMA. "
        "Utilise uniquement les informations disponibles (ou 'unknown/null'). "
        "Si des nombres sont fournis dans le texte (ex: litres d'eau, kg CO2e), capture-les.\n\n"
        f"SCHEMA (exemple de clés attendues):\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "CONTENU:\n"
        f"URL: {doc['url']}\n"
        f"TITRE: {doc['title']}\n"
        f"TEXTE:\n{doc['text']}\n\n"
        "RÉPONDS UNIQUEMENT AVEC UN JSON VALIDE."
    )
    return instructions

def call_llm(doc: Dict[str, str]) -> Dict[str, Any]:
    """
    Appelle le LLM et renvoie un dict Python.
    Tolère un JSON entouré de ```...```.
    """
    msg = build_user_prompt(doc)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": msg},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content

    # Nettoie les éventuels fences
    content = content.strip()
    content = re.sub(r"^```(json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()

    try:
        data = json.loads(content)
    except Exception:
        # Dernier recours : extrait le premier bloc {...}
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise ValueError("Réponse LLM non JSON.")
        data = json.loads(m.group(0))
    return data

#------------------------------------#
# Scoring: pondérations (mieux = score plus haut)
#------------------------------------#
WEIGHTS = {
    "materials": 0.18,
    "water": 0.14,
    "energy": 0.14,
    "co2e": 0.18,
    "biodegradability_recyclability": 0.14,
    "durability": 0.10,
    "certifications": 0.07,
    "packaging_transport": 0.05,
}

#------------------------------------#

def compute_eco_score(sub: Dict[str, Any]) -> float:
    """
    Calcule un score global [0..100] à partir des sous-scores.
    Si une clé manque, on considère 50 (neutre).
    """
    total = 0.0
    for k, w in WEIGHTS.items():
        s = sub.get(k, 50)
        try:
            s = float(s)
        except Exception:
            s = 50.0
        total += w * s
    return round(total, 2)

def tests():
    tests = [
        "https://www.amazon.ca/HyperX-Cloud-Alpha-Wireless-Headphone/dp/B09TRW57WB?th=1",
        "https://www.aliexpress.com/item/1005010013644974.html?spm=a2g0o.productlist.main.2.3b68CoevCoevj9&aem_p4p_detail=20251107170621635875746455130001499353&algo_pvid=8d10440d-4936-4f55-ab36-c704c8ad74c2&algo_exp_id=8d10440d-4936-4f55-ab36-c704c8ad74c2-1&pdp_ext_f=%7B%22order%22%3A%22230%22%2C%22eval%22%3A%221%22%2C%22fromPage%22%3A%22search%22%7D&pdp_npi=6%40dis%21CAD%21125.07%2151.69%21%21%21616.43%21254.79%21%40210328db17625639817935419eb863%2112000050844510458%21sea%21CA%210%21ABX%211%210%21n_tag%3A-29910%3Bd%3Aa7c3cc63%3Bm03_new_user%3A-29895%3BpisId%3A5000000187429864&curPageLogUid=w2glZASoKnHM&utparam-url=scene%3Asearch%7Cquery_from%3A%7Cx_object_id%3A1005010013644974%7C_p_origin_prod%3A&search_p4p_id=20251107170621635875746455130001499353_1",
        "https://www.alibaba.com/product-detail/Modern-X1-PRO-AI-Smart-Sports_1601600396621.html?selectedCarrierCode=SEMI_MANAGED_STANDARD%40%40STANDARD&priceId=811beff19fbb47feb6cc664408b7aa6f",
        "https://www.wish.com/search/ecouteurs/product/685ed09ee289e1554cb3f17f?source=search&position=13",
        "https://www.ebay.ca/p/14059427736?iid=314505929469"
    ]
    for u in tests:
        print(urlparse(u).netloc, "→", _match_allowed_domain(urlparse(u).netloc, HEADLESS_DOMAINS))

def _extract_domain(u: str) -> str:
    try:
        h = urlparse(u).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""

def bing_visual_search(image_bytes: bytes, filename: str = "upload.jpg") -> list[dict]:
    """
    Appelle l'API Bing Visual Search avec une image et renvoie
    une liste d'items: [{thumb, url, site, price}] (price facultatif).
    """
    if not BING_VISION_KEY:
        raise RuntimeError("BING_VISION_KEY manquant. Ajoute-le dans .env")

    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    files = {
        'image': (filename, image_bytes, content_type)
    }
    headers = {
        "Ocp-Apim-Subscription-Key": BING_VISION_KEY
    }
    resp = requests.post(BING_VISION_ENDPOINT, headers=headers, files=files, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Le JSON renvoie des 'tags' → 'actions'
    results = []
    for tag in data.get("tags", []):
        for action in tag.get("actions", []):
            # On prend ce qui pointe vers des pages produit/images similaires
            if action.get("actionType") in ("VisualSearch", "PagesIncluding", "ProductVisualSearch", "ShoppingSources", "ImageResults"):
                value = action.get("data", {}).get("value", [])
                for v in value:
                    url = v.get("hostPageUrl") or v.get("webSearchUrl") or v.get("contentUrl")
                    if not url:
                        continue
                    thumb = v.get("thumbnailUrl") or v.get("thumbnail", {}).get("url") or v.get("image", {}).get("thumbnailUrl")
                    name  = v.get("name") or v.get("hostPageDisplayUrl") or _extract_domain(url)

                    # Prix si présent (rare mais possible)
                    price = None
                    offer = v.get("offer") or v.get("offers") or {}
                    if isinstance(offer, dict):
                        price = offer.get("price") or offer.get("priceDisplay")
                    # fallback si le prix est imbriqué ailleurs (certains vendors)
                    if not price:
                        for k in ("aggregateRating", "insightsMetadata", "insightsSourcesSummary"):
                            # rien à faire, on ne force pas

                            pass

                    results.append({
                        "thumb": thumb,
                        "url": url,
                        "site": _extract_domain(url) or name,
                        "price": price
                    })
    # Déduplique par URL
    seen = set()
    unique = []
    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        unique.append(r)
    return unique

# --- Google Cloud Vision: Web Detection (image -> pages similaires) ---
def gcv_web_detection(image_bytes: bytes, filename: str = "upload.jpg") -> list[dict]:
    """
    Utilise Google Cloud Vision (Web Detection) pour trouver des pages
    qui contiennent cette image (ou une variante). Retourne une liste
    d'items: {thumb, url, site, price(None)}.
    """
    from google.cloud import vision
    from urllib.parse import urlparse

    try:
        client = vision.ImageAnnotatorClient()
        resp = client.web_detection(image=vision.Image(content=image_bytes))
        web = resp.web_detection
    except Exception:
        return []

    results: list[dict] = []
    if web and web.pages_with_matching_images:
        for p in web.pages_with_matching_images:
            url = (p.url or "").strip()
            if not url:
                continue
            # miniature si dispo (full match > partial match)
            img0 = (p.full_matching_images or p.partial_matching_images or [None])[0]
            thumb = getattr(img0, "url", None) if img0 else None

            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]

            results.append({
                "thumb": thumb,
                "url": url,
                "site": host,
                "price": None,  # GCV ne renvoie pas de prix
            })

    # dédoublonnage par URL (préserve l'ordre)
    seen, uniq = set(), []
    for r in results:
        if r["url"] in seen: continue
        seen.add(r["url"]); uniq.append(r)

    # --- Enrichissement prix sur les N premiers (évite d'être lent) ---
    N = 8  # ajuste si tu veux plus/moins de scraping
    for i, item in enumerate(uniq[:N]):
        price = try_extract_price(item["url"])
        if price:
            item["price"] = price

    return uniq

# ------------------------------
# Routes
# ------------------------------
@app.get("/")
def index():
    tests()
    return render_template("index.html")


@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(force=True)
    urls: List[str] = [u.strip() for u in data.get("urls", []) if u.strip()]
    if len(urls) < 2:
        return jsonify({"error": "Veuillez fournir au moins 2 URL."}), 400

    results = []
    errors = []

    for url in urls:
        try:
            doc = smart_fetch(url)
            llm = call_llm(doc)
            subs = llm.get("subscores", {}) or {}
            eco_score = compute_eco_score(subs)
            results.append({
                "url": doc["url"],
                "title": llm.get("title") or doc["title"],
                "features": llm.get("features", {}),
                "subscores": subs,
                "eco_score": eco_score
            })
        except Exception as e:
            errors.append({"url": url, "error": str(e)})

    # Classement décroissant (meilleur en premier)
    ranking = sorted(results, key=lambda x: x["eco_score"], reverse=True)
    # Ajoute le rang
    for i, item in enumerate(ranking, start=1):
        item["rank"] = i

    return jsonify({"results": ranking, "errors": errors})

# ---------------------------------------------------------------------------------------------------------
# Page 2 : recherche par images / Simple, clair : on upload, on interroge Bing Visual Search, on formate la réponse en (thumb, url, site, price), et on affiche.
# ------------------------------------------------------------------------------------------------------------------
@app.get("/images")
def images_page():
    return render_template("images.html")


@app.post("/images")
def images_search():
    """
    Traite 1 image uploadée, renvoie une page résultat avec :
    - l'image source (aperçu)
    - une liste d'URLs similaires avec miniatures
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return render_template("images.html", error="Aucun fichier reçu.")

    img_bytes = f.read()

    # Essayez d'abord Bing s'il est configuré, sinon fallback Google (si tu l'as ajouté)
    items = []
    try:
        if BING_VISION_KEY:  # la var d'env
            items = bing_visual_search(img_bytes, f.filename)
        if not items:  # vide ? on bascule GCV
            items = gcv_web_detection(img_bytes, f.filename)
    except Exception:
        items = gcv_web_detection(img_bytes, f.filename)

    result = {
        "query_filename": f.filename,
        "query_preview_b64": "data:" + (mimetypes.guess_type(f.filename)[0] or "image/jpeg") + ";base64," +
                             __import__("base64").b64encode(img_bytes).decode("utf-8"),
        "items": items[:20],
    }
    return render_template("images.html", result=result)

#---------------------------------------#
# Google Cloud Vision seul              #
#---------------------------------------# 
# 
# @app.post("/images")
#def images_search():
#    f = request.files.get("file")
#    if not f or not f.filename:
#        return render_template("images.html", error="Aucun fichier reçu.")
#    img_bytes = f.read()
#
#    try:
#        items = gcv_web_detection(img_bytes, f.filename)
#    except Exception as e:
#        return render_template("images.html", error=f"Google Vision a échoué: {e}")
#
#    result = {
#        "query_filename": f.filename,
#        "query_preview_b64": "data:" + (mimetypes.guess_type(f.filename)[0] or "image/jpeg") + ";base64," +
#                             __import__("base64").b64encode(img_bytes).decode("utf-8"),
#        "items": items[:20],
#    }
#    return render_template("images.html", result=result)

# -----------------------------------------------------------------#
# Routes pour aider l'indexation sur les moteurs de recherche
# -----------------------------------------------------------------#

@app.get("/robots.txt")
def robots():
    body = "User-agent: *\nAllow: /\nSitemap: " + url_for('sitemap', _external=True) + "\n"
    return Response(body, mimetype="text/plain")

@app.get("/sitemap.xml")
def sitemap():
    pages = [
        url_for("index", _external=True),
        url_for("images_page", _external=True),
    ]
    today = datetime.utcnow().date().isoformat()
    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for loc in pages:
        xml += [f"<url><loc>{loc}</loc><lastmod>{today}</lastmod><changefreq>weekly</changefreq></url>"]
    xml.append("</urlset>")
    return Response("\n".join(xml), mimetype="application/xml")

# ------------------------------
# Main (local)
# ------------------------------
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("   Ouvre un terminal et exporte ta clé :")
        print("   Windows PowerShell: $env:OPENAI_API_KEY='sk-...'\n"
              "   macOS/Linux: export OPENAI_API_KEY='sk-...'")
    app.run(debug=True)
