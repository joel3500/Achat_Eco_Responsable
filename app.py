import os, re, json, math, time, base64, tempfile
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

# En prod (Railway), les credentials Google arrivent en base64 (pas de fichier disponible sur le filesystem).
# En local, GOOGLE_APPLICATION_CREDENTIALS peut déjà pointer vers un fichier .json existant.
_gcreds_b64 = os.getenv("GOOGLE_CREDS_B64")
if _gcreds_b64:
    _creds_path = os.path.join(tempfile.gettempdir(), "gcp-credentials.json")
    with open(_creds_path, "wb") as f:
        f.write(base64.b64decode(_gcreds_b64))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds_path

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

GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")

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

# ============================================
#  MODULE : ULTIMATE IMAGE SEARCH (Lens‑like)
# ============================================

def _detect_mime(image_bytes: bytes) -> str:
    if image_bytes[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if image_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    if image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    return "image/jpeg"

def gcv_extract_info(image_bytes):
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        resp = client.web_detection(image=vision.Image(content=image_bytes))
        web = resp.web_detection
    except Exception:
        return {"entities": [], "pages": []}

    out = {
        "entities": [],
        "pages": [],
        "full_images": [],
        "partial_images": [],
    }

    if not web:
        return out

    if web.web_entities:
        out["entities"] = [e.description for e in web.web_entities if e.description]

    if web.pages_with_matching_images:
        out["pages"] = [p.url for p in web.pages_with_matching_images if p.url]

    if web.full_matching_images:
        out["full_images"] = [i.url for i in web.full_matching_images if i.url]

    if web.partial_matching_images:
        out["partial_images"] = [i.url for i in web.partial_matching_images if i.url]

    return out


def describe_image_with_llm(image_bytes):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        mime = _detect_mime(image_bytes)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Décris précisément le produit visible sur l’image."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}
                        },
                        {"type": "text", "text": "En une phrase courte : marque, type, couleur, modèle si possible."}
                    ]
                }
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def google_text_search(query):
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_ID:
        return []

    params = {
        "q": query,
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_ID,
        "searchType": "image",
        "num": 10,
    }
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=20)
        r.raise_for_status()
    except Exception:
        return []

    out = []
    for item in r.json().get("items", []):
        page = item.get("image", {}).get("contextLink") or item.get("link")
        if not page:
            continue
        try:
            site = urlparse(page).netloc
        except Exception:
            site = None
        out.append({
            "thumb": item.get("link"),
            "url": page,
            "site": site,
            "price": None,
        })
    return out


def merge_results(*lists):
    merged, seen = [], set()
    for lst in lists:
        if not lst:
            continue
        for item in lst:
            u = item.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            merged.append(item)
    return merged


def enrich_prices(items, max_n=10):
    for it in items[:max_n]:
        price = try_extract_price(it["url"])
        if price:
            it["price"] = price
    return items


def ultimate_search(image_bytes):
    # 1) Google Vision → entités textuelles
    gcv = gcv_extract_info(image_bytes)

    # 2) LLM → description lisible pour la recherche texte
    query = describe_image_with_llm(image_bytes)
    if not query and gcv.get("entities"):
        query = " ".join(gcv["entities"][:5])

    # 3a) Google Cloud Vision Web Detection (image → pages visuellement similaires)
    visual_items = gcv_web_detection(image_bytes)

    # 3b) Google Custom Search (description → résultats complémentaires)
    text_items = google_text_search(query) if query else []

    # 4) Fusion (visual en priorité, texte en complément)
    merged = merge_results(visual_items, text_items)

    # 5) Enrichissement prix
    merged = enrich_prices(merged)

    return {
        "description": query,
        "entities": gcv["entities"],
        "items": merged[:20]
    }

#================= ( FIN DE RESCHERCHE de Correspondance )  ===========================#

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

# --- Google Cloud Vision: Web Detection (image -> pages similaires) ---
def gcv_web_detection(image_bytes: bytes, filename: str = "upload.jpg") -> list[dict]:
    """
    Utilise Google Cloud Vision (Web Detection) pour trouver des pages
    qui contiennent cette image (ou une variante). Retourne une liste
    d'items: {thumb, url, site, price(None)}.
    """
    from urllib.parse import urlparse

    try:
        from google.cloud import vision
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


@app.get("/exemples")
def exemples_page():
    return render_template("exemples.html")


@app.post("/images")
def images_search():
    f = request.files.get("file")
    if not f or not f.filename:
        return render_template("images.html", error="Aucun fichier reçu.")

    img_bytes = f.read()

    try:
        result_search = ultimate_search(img_bytes)
    except Exception as e:
        return render_template("images.html", error=f"Erreur lors de la recherche : {e}")

    mime = _detect_mime(img_bytes)
    result = {
        "query_filename": f.filename,
        "query_preview_b64": (
            f"data:{mime};base64,"
            + base64.b64encode(img_bytes).decode("utf-8")
        ),
        "description": result_search.get("description", ""),
        "items": result_search["items"]
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
        url_for("exemples_page", _external=True),
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
    