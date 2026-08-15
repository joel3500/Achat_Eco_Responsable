/* ============================================================
   TRADUCTIONS — Achat Éco-Responsable (FR/EN/ES/DE/IT/AR/ZH)
   ============================================================ */
const ECO_LANGS = ['fr', 'en', 'es', 'de', 'it', 'ar', 'zh'];
const ECO_RTL_LANGS = ['ar'];

const ECO_TRANSLATIONS = {
  fr: {
    common: {
      notice: "💻 Ce site fonctionne beaucoup mieux sur un ordinateur ou une tablette (plus facile pour copier-coller les URLs) que sur un téléphone. L'utilisation mobile n'est pas encore optimisée, désolé pour l'inconvénient.",
      donateLink: "💚 Soutenir ce projet (don)",
      linkHome: "← Retour à la page de l'Achat Eco-Responsable",
      linkImages: "← j'ai une image dont je cherche les points de vente en ligne",
      linkExemples: "← je ne sais pas quel genre d'articles je peux comparer ici (voir des exemples)",
      footerMissionHtml: "© Tous droits réservés • Automne 2025 • Projet à intérêt Écologique, incitant les lobis industriels à faire de meilleurs choix pour la préservation de notre environnement à tous. <br>\nRéalisé par JOEL SANDÉ, du Saguenay Lac Saint-Jean. <br><br>\nL'argent d'accord, mais notre environnement d'abord. Certe on ne peut pas faire d'omelettes sans casser d'oeufs, mais faisons le moins de mal dans la mesure du possible. <br><br>\nSoyons nombreux à utiliser ce site, pour contraindre les fabriquants à se soucier de notre environnement. <br>\nPartagez Massivement ! pour préserver notre environnement."
    },
    index: {
      title: "Achat Éco-Responsable — compare l'impact environnemental",
      h2Compare: "Articles à comparer",
      addBtn: "➕ Ajouter un article",
      removeBtn: "➖ Supprimer le dernier",
      runBtn: "Lancer l'analyse",
      hintMinUrls: "Ajoute au minimum 2 URLs de pages des articles.",
      introIntro: "Compare des articles et choisis le plus écologique :",
      introDesire: "Tu désires te procurer un produit, mais tu as plusieurs choix possibles et tu hésites lequel choisir.",
      introHelp: "Nous t'aidons à faire un choix Eco-Responsable.",
      introPaste: "Tu n'as juste qu'à coller les URLs des pages de tes objets, et nous t'aidons à faire les choix Eco-Responsables :",
      li1: "Emprunte Carbone", li2: "Fabrication ayant consommé le moins d'eau", li3: "Celui qui contient le plus de matériel recyclable",
      introRank: "Nous te les classons selon l'ordre le plus respectueux de l'environnement.",
      h3Shortlist: "Exemples de sites pour tester",
      catMarketPriority: "Marketplaces prioritaires", catMarketGeneral: "Marketplaces généralistes",
      catRetailers: "Grands détaillants (Canada)", catFashion: "Mode & plein air",
      catHome: "Maison & rénovation", catRefurb: "Reconditionné / seconde main", catEco: "Sélection « éco »",
      shortlistHint: "Ces liens sont des <b>exemples de formats</b> (tous pointant vers de vrais produits). click pour voir le type de pages que nous pouvons vous analyser...",
      h2Results: "Classement écologique (décroissant)", h3Details: "Détails par article",
      js: {
        urlLabel: "URL de l'article", urlPlaceholder: "https://exemple.com/produit",
        analyzing: "Analyse en cours… ⏳", importedMsg: "{n} lien(s) importé(s) depuis la recherche par images.",
        needTwoUrls: "Ajoute au moins 2 URL.", unknownError: "Erreur inconnue", errorPrefix: "Erreur : ",
        scoreLabel: "Score global :", rankLabel: "Rang :", openPage: "Ouvrir la page",
        lblMaterials: "Matériaux", lblWater: "Eau (L)", lblEnergy: "Énergie (kWh)", lblCo2e: "CO₂e (kg)",
        lblBiodeg: "Bio-dégradabilité", lblRecyc: "Recyclabilité", lblDurability: "Durabilité/Réparabilité",
        lblCertifications: "Certifications", lblPackaging: "Packaging", lblTransport: "Transport",
        lblNotes: "Notes", lblConfidence: "Confiance", viewSubscores: "Voir sous-scores", errorsTitle: "URLs en erreur :"
      }
    },
    images: {
      title: "Recherche par images — Achat Responsable",
      h1: "Recherche par images",
      introLine1: "Uploadez une image → puis nous retrouvons les points de vente en ligne :",
      introLine2: "Uploade l'image d'un objet que tu désires acheter ( accessoire - vêtement - véhicule - ... ) et nous t'aidons à le retrouver en ligne. Avec si possible le prix qui est affiché.",
      labelChooseImage: "Choisir une image", hintFormats: "Formats usuels (jpg, png, jpeg, gif, webp). Taille <5Mo svp",
      btnSearch: "Lancer la recherche", h3Selected: "Image sélectionnée", searchForLabel: "🔍 Recherche lancée pour :",
      h3Matches: "Correspondances trouvées", noMatches: "Aucune correspondance trouvée.",
      noDescErrorHtml: "Le LLM n'a pas pu décrire l'image — vérifie que <code>OPENAI_API_KEY</code> est bien configuré.",
      noKeyErrorHtml: "Vérifie que <code>GOOGLE_CSE_KEY</code> et <code>GOOGLE_CSE_ID</code> sont bien configurés."
    },
    exemples: {
      title: "Quels articles comparer ? Guide et exemples — Achat Éco-Responsable",
      h1: "Quels articles puis-je comparer ?",
      introText: "Pas d'idée par où commencer ? Voici un guide avec des exemples variés d'articles pour lesquels notre analyse écologique fait du sens.",
      h2Principle: "Le principe en bref",
      principleHtml: "Dès qu'un produit a une page web décrivant ses caractéristiques (matériaux, certifications, emballage, fabrication...), on peut généralement l'analyser. Plus la page contient d'informations concrètes (fiche produit détaillée, JSON-LD, labels environnementaux), meilleure sera l'analyse. L'intérêt est surtout de <b>comparer 2 articles similaires entre eux</b> (même besoin, options différentes) pour voir lequel est le plus respectueux de l'environnement.",
      h3Categories: "Catégories et exemples concrets",
      catClothingTitle: "Vêtements & textile",
      catClothingLi: ["T-shirt en coton biologique vs t-shirt en polyester classique","Manteau d'hiver en duvet vs manteau en fibres synthétiques recyclées","Jeans en denim régulier vs denim fabriqué avec moins d'eau","Chaussures de course en matériaux recyclés vs modèle classique"],
      catHomeTitle: "Maison, literie & meubles",
      catHomeLi: ["Housse de couette en coton égyptien vs en matières synthétiques","Cadre de lit en bois certifié FSC vs en composite/plastique","Matelas en mousse classique vs matelas en matériaux naturels"],
      catElecTitle: "Électronique & accessoires",
      catElecLi: ["Écouteurs sans fil de deux marques différentes","Ordinateur portable reconditionné vs neuf","Téléphone reconditionné (Back Market, Poshmark) vs neuf","Montre connectée : matériaux du boîtier, réparabilité"],
      catApplTitle: "Électroménagers",
      catApplLi: ["Grille-pain ou aspirateur : deux modèles avec certification Energy Star ou non","Petit électroménager reconditionné vs neuf"],
      catSportTitle: "Sport & plein air",
      catSportLi: ["Gourde réutilisable en inox vs en plastique","Sac à dos ou tente : matériaux imperméables recyclés vs classiques","Vélo ou équipement d'extérieur de deux marques"],
      catHouseholdTitle: "Produits ménagers & soins personnels",
      catHouseholdLi: ["Savon à vaisselle concentré vs format régulier (moins d'emballage/transport)","Shampoing en bouteille rechargeable vs bouteille jetable","Brosse à dents en bambou vs en plastique"],
      catToysTitle: "Jouets & articles pour enfants",
      catToysLi: ["Jouets en bois certifié vs jouets en plastique","Poussette ou siège d'auto : matériaux et durabilité"],
      catBagsTitle: "Bagagerie & accessoires du quotidien",
      catBagsLi: ["Sac à dos ou valise en matériaux recyclés vs classique","Portefeuille en cuir vs en matériaux véganes/recyclés"],
      catDiyTitle: "Bricolage & rénovation",
      catDiyLi: ["Peinture faible en COV vs peinture classique","Isolant écologique vs isolant standard"],
      tipHtml: "Astuce : pour un résultat pertinent, compare des articles qui répondent au <b>même besoin</b> (deux écouteurs, deux manteaux d'hiver, deux gourdes...) plutôt que des objets complètement différents.",
      h3Warning: "⚠️ Ce qui fonctionne moins bien",
      warnLiHtml: ["Les produits <b>alimentaires périssables</b> (fruits, légumes, viande...) — pas de fiche produit à analyser.","Les <b>services</b> (abonnements, voyages, assurances...) — ce ne sont pas des objets physiques.","Les pages produit très pauvres en détails (peu ou pas de description, pas de fiche technique).","Certains sites bloquent la lecture automatisée de leurs pages (ex. Best Buy actuellement)."],
      ctaText: "Prêt à essayer avec tes propres articles ?", ctaBtn: "Comparer mes articles →"
    },
    don: {
      h1: "Soutenir ce projet", eyebrow: "🌱 Merci de ton soutien.",
      interacLead: "💸 Faire un virement Interac à {email}",
      copyBtn: "📋 Copier l'adresse",
      copyToast: "Copié ! Colle cette adresse dans ton application bancaire pour envoyer le virement.",
      missionText: "Soyez généreux, mais sans aucune obligation. Tes dons servent à couvrir les frais d'hébergement du serveur et les coûts des API (analyse par intelligence artificielle, recherche d'images, recherche web) qui font fonctionner le comparateur écologique, et à m'encourager à continuer d'améliorer cet outil au service d'une consommation plus responsable.",
      canadaTitle: "🇨🇦 Tu es au Canada ?",
      canadaHint: "L'Interac fonctionne uniquement entre comptes bancaires canadiens. Clique sur ta banque pour accéder directement à ton compte en ligne et envoyer ton virement.",
      worldTitle: "🌍 Ailleurs dans le monde ?",
      worldHint: "L'Interac ne fonctionne pas hors Canada. Par carte de crédit ou de débit, où que tu sois, PayPal fonctionne sans compte requis.",
      paypalBtn: "❤️ Faire un don via PayPal",
      footerHtml: "© Tous droits réservés • Automne 2025 • Projet à intérêt Écologique, incitant les lobis industriels à faire de meilleurs choix pour la préservation de notre environnement à tous. <br>\nRéalisé par JOEL SANDÉ, du Saguenay Lac Saint-Jean. <br><br>\nJe vous le dis tout de suite : si ce projet génère suffisamment de fonds, une bonne partie ira en aide financière aux personnes en situation de handicap physique, aux jeunes filles et femmes pour les sortir de la prostitution involontaire due à leurs situations précaires, et surtout aux séniors à travers le monde, qui sont ma cause principale, car ce sont eux qui ont bâti la société dans laquelle nous sommes aujourd'hui. Maintenant qu'ils n'ont plus la force, c'est à notre tour de leur faire de petits cadeaux. Ce sont ces trois causes qui m'ont poussé à entreprendre une maîtrise en génie biomédical. Il n'y a rien de tel que de vivre et de vieillir en toute sérénité, en santé."
    }
  },

  en: {
    common: {
      notice: "💻 This site works much better on a computer or tablet (easier to copy-paste URLs) than on a phone. Mobile use isn't optimized yet, sorry for the inconvenience.",
      donateLink: "💚 Support this project (donate)",
      linkHome: "← Back to the Achat Eco-Responsable homepage",
      linkImages: "← I have an image and I'm looking for where to buy it online",
      linkExemples: "← I don't know what kind of items I can compare here (see examples)",
      footerMissionHtml: "© All rights reserved • Fall 2025 • A project for the environment, encouraging industry lobbies to make better choices to preserve our environment for everyone. <br>\nCreated by JOEL SANDÉ, from Saguenay Lac Saint-Jean. <br><br>\nMoney is fine, but our environment comes first. You can't make an omelette without breaking eggs, but let's do as little harm as possible. <br><br>\nLet's be many to use this site, to push manufacturers to care about our environment. <br>\nShare widely, to preserve our environment."
    },
    index: {
      title: "Achat Éco-Responsable — compare the environmental impact",
      h2Compare: "Items to compare", addBtn: "➕ Add an item", removeBtn: "➖ Remove the last one", runBtn: "Run the analysis",
      hintMinUrls: "Add at least 2 product page URLs.",
      introIntro: "Compare items and choose the most eco-friendly one:",
      introDesire: "You want to buy a product, but you have several options and can't decide which one to pick.",
      introHelp: "We help you make an eco-responsible choice.",
      introPaste: "Just paste the URLs of your item pages, and we'll help you make the eco-responsible choice:",
      li1: "Carbon footprint", li2: "Manufacturing that used the least water", li3: "The one containing the most recyclable material",
      introRank: "We rank them from most to least environmentally friendly.",
      h3Shortlist: "Example sites to try",
      catMarketPriority: "Priority marketplaces", catMarketGeneral: "General marketplaces",
      catRetailers: "Major retailers (Canada)", catFashion: "Fashion & outdoor gear",
      catHome: "Home & renovation", catRefurb: "Refurbished / second-hand", catEco: "\"Eco\" selection",
      shortlistHint: "These links are <b>format examples</b> (all pointing to real products). Click to see the type of pages we can analyze for you...",
      h2Results: "Ecological ranking (highest to lowest)", h3Details: "Details per item",
      js: {
        urlLabel: "Item URL", urlPlaceholder: "https://example.com/product",
        analyzing: "Analyzing… ⏳", importedMsg: "{n} link(s) imported from the image search.",
        needTwoUrls: "Add at least 2 URLs.", unknownError: "Unknown error", errorPrefix: "Error: ",
        scoreLabel: "Overall score:", rankLabel: "Rank:", openPage: "Open the page",
        lblMaterials: "Materials", lblWater: "Water (L)", lblEnergy: "Energy (kWh)", lblCo2e: "CO₂e (kg)",
        lblBiodeg: "Biodegradability", lblRecyc: "Recyclability", lblDurability: "Durability/Repairability",
        lblCertifications: "Certifications", lblPackaging: "Packaging", lblTransport: "Transport",
        lblNotes: "Notes", lblConfidence: "Confidence", viewSubscores: "View sub-scores", errorsTitle: "URLs with errors:"
      }
    },
    images: {
      title: "Image search — Achat Responsable",
      h1: "Image search",
      introLine1: "Upload an image → we'll find where to buy it online:",
      introLine2: "Upload the image of an item you want to buy (accessory - clothing - vehicle - ...) and we'll help you find it online, with the price if possible.",
      labelChooseImage: "Choose an image", hintFormats: "Common formats (jpg, png, jpeg, gif, webp). Size <5MB please",
      btnSearch: "Run the search", h3Selected: "Selected image", searchForLabel: "🔍 Search launched for:",
      h3Matches: "Matches found", noMatches: "No matches found.",
      noDescErrorHtml: "The AI couldn't describe the image — check that <code>OPENAI_API_KEY</code> is properly configured.",
      noKeyErrorHtml: "Check that <code>GOOGLE_CSE_KEY</code> and <code>GOOGLE_CSE_ID</code> are properly configured."
    },
    exemples: {
      title: "What items can I compare? Guide & examples — Achat Éco-Responsable",
      h1: "What items can I compare?",
      introText: "Not sure where to start? Here's a guide with varied examples of items where our ecological analysis makes sense.",
      h2Principle: "The idea in short",
      principleHtml: "As soon as a product has a web page describing its features (materials, certifications, packaging, manufacturing...), we can generally analyze it. The more concrete information the page has (detailed product sheet, JSON-LD, environmental labels), the better the analysis. The real value is in <b>comparing 2 similar items</b> (same need, different options) to see which one is more environmentally friendly.",
      h3Categories: "Categories and concrete examples",
      catClothingTitle: "Clothing & textiles",
      catClothingLi: ["Organic cotton t-shirt vs classic polyester t-shirt","Down winter coat vs coat made of recycled synthetic fibers","Regular denim jeans vs jeans made with less water","Running shoes made of recycled materials vs a classic model"],
      catHomeTitle: "Home, bedding & furniture",
      catHomeLi: ["Egyptian cotton duvet cover vs a synthetic-fiber one","FSC-certified wood bed frame vs composite/plastic","Classic foam mattress vs natural-material mattress"],
      catElecTitle: "Electronics & accessories",
      catElecLi: ["Wireless earbuds from two different brands","Refurbished laptop vs new","Refurbished phone (Back Market, Poshmark) vs new","Smartwatch: case materials, repairability"],
      catApplTitle: "Home appliances",
      catApplLi: ["Toaster or vacuum: two models with or without Energy Star certification","Refurbished small appliance vs new"],
      catSportTitle: "Sports & outdoor",
      catSportLi: ["Stainless steel reusable water bottle vs plastic","Backpack or tent: recycled waterproof materials vs classic","Bike or outdoor gear from two brands"],
      catHouseholdTitle: "Household & personal care products",
      catHouseholdLi: ["Concentrated dish soap vs regular size (less packaging/transport)","Refillable shampoo bottle vs disposable bottle","Bamboo toothbrush vs plastic"],
      catToysTitle: "Toys & children's items",
      catToysLi: ["Certified wooden toys vs plastic toys","Stroller or car seat: materials and durability"],
      catBagsTitle: "Bags & everyday accessories",
      catBagsLi: ["Backpack or suitcase made of recycled materials vs classic","Leather wallet vs vegan/recycled materials"],
      catDiyTitle: "DIY & renovation",
      catDiyLi: ["Low-VOC paint vs classic paint","Eco-friendly insulation vs standard insulation"],
      tipHtml: "Tip: for a relevant result, compare items that meet the <b>same need</b> (two earbuds, two winter coats, two water bottles...) rather than completely different objects.",
      h3Warning: "⚠️ What works less well",
      warnLiHtml: ["Perishable food products (fruits, vegetables, meat...) — no product sheet to analyze.","Services (subscriptions, travel, insurance...) — these aren't physical objects.","Product pages with very little detail (little or no description, no technical sheet).","Some sites block automated reading of their pages (e.g. Best Buy currently)."],
      ctaText: "Ready to try it with your own items?", ctaBtn: "Compare my items →"
    },
    don: {
      h1: "Support this project", eyebrow: "🌱 Thank you for your support.",
      interacLead: "💸 Send an Interac e-Transfer to {email}",
      copyBtn: "📋 Copy the address",
      copyToast: "Copied! Paste this address in your banking app to send the transfer.",
      missionText: "Be generous, but there's no obligation at all. Your donations help cover server hosting costs and API costs (AI analysis, image search, web search) that power the ecological comparator, and encourage me to keep improving this tool in service of more responsible consumption.",
      canadaTitle: "🇨🇦 Are you in Canada?",
      canadaHint: "Interac e-Transfer only works between Canadian bank accounts. Click your bank to go straight to your online banking and send the transfer.",
      worldTitle: "🌍 Somewhere else in the world?",
      worldHint: "Interac doesn't work outside Canada. By credit or debit card, wherever you are, PayPal works with no account required.",
      paypalBtn: "❤️ Donate via PayPal",
      footerHtml: "© All rights reserved • Fall 2025 • A project for the environment, encouraging industry lobbies to make better choices to preserve our environment for everyone. <br>\nCreated by JOEL SANDÉ, from Saguenay Lac Saint-Jean. <br><br>\nI'll say it upfront: if this project generates enough funds, a good share of it will go toward financial support for people with physical disabilities; for young women and girls, to help them escape involuntary prostitution driven by precarious circumstances; and above all for seniors around the world — my main cause, because they are the ones who built the society we live in today. Now that they no longer have the strength, it's our turn to give a little something back. These are the three causes that pushed me to pursue a Master's degree in Biomedical Engineering. There's nothing quite like living — and growing old — in peace and in good health."
    }
  },

  es: {
    common: {
      notice: "💻 Este sitio funciona mucho mejor en una computadora o tableta (más fácil para copiar y pegar las URLs) que en un teléfono. El uso móvil todavía no está optimizado, disculpa las molestias.",
      donateLink: "💚 Apoyar este proyecto (donar)",
      linkHome: "← Volver a la página de Achat Eco-Responsable",
      linkImages: "← Tengo una imagen y busco dónde comprarlo en línea",
      linkExemples: "← No sé qué tipo de artículos puedo comparar aquí (ver ejemplos)",
      footerMissionHtml: "© Todos los derechos reservados • Otoño 2025 • Un proyecto de interés ecológico, para incitar a los grupos industriales a tomar mejores decisiones para preservar nuestro medio ambiente. <br>\nRealizado por JOEL SANDÉ, de Saguenay Lac Saint-Jean. <br><br>\nEl dinero está bien, pero primero nuestro medio ambiente. No se puede hacer una tortilla sin romper huevos, pero hagamos el menor daño posible. <br><br>\nSeamos muchos usando este sitio, para obligar a los fabricantes a preocuparse por nuestro medio ambiente. <br>\n¡Comparte masivamente! Para preservar nuestro medio ambiente."
    },
    index: {
      title: "Achat Éco-Responsable — compara el impacto ambiental",
      h2Compare: "Artículos a comparar", addBtn: "➕ Añadir un artículo", removeBtn: "➖ Eliminar el último", runBtn: "Iniciar el análisis",
      hintMinUrls: "Añade al menos 2 URLs de páginas de artículos.",
      introIntro: "Compara artículos y elige el más ecológico:",
      introDesire: "Quieres comprar un producto, pero tienes varias opciones posibles y dudas cuál elegir.",
      introHelp: "Te ayudamos a hacer una elección eco-responsable.",
      introPaste: "Solo tienes que pegar las URLs de las páginas de tus artículos, y te ayudamos a hacer las elecciones eco-responsables:",
      li1: "Huella de carbono", li2: "Fabricación que consumió menos agua", li3: "El que contiene más material reciclable",
      introRank: "Los clasificamos según el orden más respetuoso con el medio ambiente.",
      h3Shortlist: "Ejemplos de sitios para probar",
      catMarketPriority: "Marketplaces prioritarios", catMarketGeneral: "Marketplaces generales",
      catRetailers: "Grandes minoristas (Canadá)", catFashion: "Moda y aire libre",
      catHome: "Hogar y renovación", catRefurb: "Reacondicionado / segunda mano", catEco: "Selección «eco»",
      shortlistHint: "Estos enlaces son <b>ejemplos de formato</b> (todos apuntan a productos reales). Haz clic para ver el tipo de páginas que podemos analizar...",
      h2Results: "Clasificación ecológica (de mayor a menor)", h3Details: "Detalles por artículo",
      js: {
        urlLabel: "URL del artículo", urlPlaceholder: "https://ejemplo.com/producto",
        analyzing: "Analizando… ⏳", importedMsg: "{n} enlace(s) importado(s) desde la búsqueda por imágenes.",
        needTwoUrls: "Añade al menos 2 URLs.", unknownError: "Error desconocido", errorPrefix: "Error: ",
        scoreLabel: "Puntuación global:", rankLabel: "Posición:", openPage: "Abrir la página",
        lblMaterials: "Materiales", lblWater: "Agua (L)", lblEnergy: "Energía (kWh)", lblCo2e: "CO₂e (kg)",
        lblBiodeg: "Biodegradabilidad", lblRecyc: "Reciclabilidad", lblDurability: "Durabilidad/Reparabilidad",
        lblCertifications: "Certificaciones", lblPackaging: "Empaque", lblTransport: "Transporte",
        lblNotes: "Notas", lblConfidence: "Confianza", viewSubscores: "Ver subpuntuaciones", errorsTitle: "URLs con errores:"
      }
    },
    images: {
      title: "Búsqueda por imágenes — Achat Responsable",
      h1: "Búsqueda por imágenes",
      introLine1: "Sube una imagen → y te ayudamos a encontrar dónde comprarlo en línea:",
      introLine2: "Sube la imagen de un artículo que quieres comprar (accesorio - ropa - vehículo - ...) y te ayudamos a encontrarlo en línea, con el precio si es posible.",
      labelChooseImage: "Elegir una imagen", hintFormats: "Formatos habituales (jpg, png, jpeg, gif, webp). Tamaño <5Mo por favor",
      btnSearch: "Iniciar la búsqueda", h3Selected: "Imagen seleccionada", searchForLabel: "🔍 Búsqueda lanzada para:",
      h3Matches: "Coincidencias encontradas", noMatches: "No se encontraron coincidencias.",
      noDescErrorHtml: "La IA no pudo describir la imagen — verifica que <code>OPENAI_API_KEY</code> esté bien configurada.",
      noKeyErrorHtml: "Verifica que <code>GOOGLE_CSE_KEY</code> y <code>GOOGLE_CSE_ID</code> estén bien configuradas."
    },
    exemples: {
      title: "¿Qué artículos puedo comparar? Guía y ejemplos — Achat Éco-Responsable",
      h1: "¿Qué artículos puedo comparar?",
      introText: "¿No sabes por dónde empezar? Aquí tienes una guía con ejemplos variados de artículos para los que nuestro análisis ecológico tiene sentido.",
      h2Principle: "El principio en resumen",
      principleHtml: "En cuanto un producto tiene una página web que describe sus características (materiales, certificaciones, empaque, fabricación...), generalmente podemos analizarlo. Cuanta más información concreta tenga la página (ficha de producto detallada, JSON-LD, etiquetas ambientales), mejor será el análisis. El interés está sobre todo en <b>comparar 2 artículos similares entre sí</b> (misma necesidad, opciones diferentes) para ver cuál es más respetuoso con el medio ambiente.",
      h3Categories: "Categorías y ejemplos concretos",
      catClothingTitle: "Ropa y textiles",
      catClothingLi: ["Camiseta de algodón orgánico vs camiseta de poliéster clásica","Abrigo de invierno de plumón vs abrigo de fibras sintéticas recicladas","Jeans de mezclilla regular vs mezclilla fabricada con menos agua","Zapatillas de running de materiales reciclados vs modelo clásico"],
      catHomeTitle: "Hogar, ropa de cama y muebles",
      catHomeLi: ["Funda nórdica de algodón egipcio vs de materiales sintéticos","Estructura de cama de madera certificada FSC vs de composite/plástico","Colchón de espuma clásico vs colchón de materiales naturales"],
      catElecTitle: "Electrónica y accesorios",
      catElecLi: ["Auriculares inalámbricos de dos marcas diferentes","Portátil reacondicionado vs nuevo","Teléfono reacondicionado (Back Market, Poshmark) vs nuevo","Reloj conectado: materiales de la caja, reparabilidad"],
      catApplTitle: "Electrodomésticos",
      catApplLi: ["Tostadora o aspiradora: dos modelos con o sin certificación Energy Star","Pequeño electrodoméstico reacondicionado vs nuevo"],
      catSportTitle: "Deporte y aire libre",
      catSportLi: ["Botella reutilizable de acero inoxidable vs de plástico","Mochila o tienda: materiales impermeables reciclados vs clásicos","Bicicleta o equipo al aire libre de dos marcas"],
      catHouseholdTitle: "Productos del hogar y cuidado personal",
      catHouseholdLi: ["Jabón lavavajillas concentrado vs formato regular (menos empaque/transporte)","Champú en botella recargable vs botella desechable","Cepillo de dientes de bambú vs de plástico"],
      catToysTitle: "Juguetes y artículos para niños",
      catToysLi: ["Juguetes de madera certificada vs juguetes de plástico","Cochecito o silla de auto: materiales y durabilidad"],
      catBagsTitle: "Bolsos y accesorios cotidianos",
      catBagsLi: ["Mochila o maleta de materiales reciclados vs clásica","Billetera de cuero vs de materiales veganos/reciclados"],
      catDiyTitle: "Bricolaje y renovación",
      catDiyLi: ["Pintura baja en COV vs pintura clásica","Aislante ecológico vs aislante estándar"],
      tipHtml: "Consejo: para un resultado relevante, compara artículos que respondan a la <b>misma necesidad</b> (dos auriculares, dos abrigos de invierno, dos botellas...) en lugar de objetos completamente diferentes.",
      h3Warning: "⚠️ Lo que funciona menos bien",
      warnLiHtml: ["Los productos alimenticios perecederos (frutas, verduras, carne...) — no hay ficha de producto que analizar.","Los servicios (suscripciones, viajes, seguros...) — no son objetos físicos.","Páginas de producto muy pobres en detalles (poca o ninguna descripción, sin ficha técnica).","Algunos sitios bloquean la lectura automatizada de sus páginas (ej. Best Buy actualmente)."],
      ctaText: "¿Listo para probar con tus propios artículos?", ctaBtn: "Comparar mis artículos →"
    },
    don: {
      h1: "Apoyar este proyecto", eyebrow: "🌱 Gracias por tu apoyo.",
      interacLead: "💸 Enviar una transferencia Interac a {email}",
      copyBtn: "📋 Copiar la dirección",
      copyToast: "¡Copiado! Pega esta dirección en tu aplicación bancaria para enviar la transferencia.",
      missionText: "Sé generoso, pero sin ninguna obligación. Tus donaciones ayudan a cubrir los costos de alojamiento del servidor y los costos de las API (análisis por inteligencia artificial, búsqueda de imágenes, búsqueda web) que hacen funcionar el comparador ecológico, y me animan a seguir mejorando esta herramienta al servicio de un consumo más responsable.",
      canadaTitle: "🇨🇦 ¿Estás en Canadá?",
      canadaHint: "Interac solo funciona entre cuentas bancarias canadienses. Haz clic en tu banco para acceder directamente a tu cuenta en línea y enviar tu transferencia.",
      worldTitle: "🌍 ¿En otro lugar del mundo?",
      worldHint: "Interac no funciona fuera de Canadá. Con tarjeta de crédito o débito, estés donde estés, PayPal funciona sin necesidad de cuenta.",
      paypalBtn: "❤️ Donar por PayPal",
      footerHtml: "© Todos los derechos reservados • Otoño 2025 • Un proyecto de interés ecológico, para incitar a los grupos industriales a tomar mejores decisiones para preservar nuestro medio ambiente. <br>\nRealizado por JOEL SANDÉ, de Saguenay Lac Saint-Jean."
    }
  },

  de: {
    common: {
      notice: "💻 Diese Website funktioniert auf einem Computer oder Tablet viel besser (einfacher, URLs zu kopieren und einzufügen) als auf einem Smartphone. Die mobile Nutzung ist noch nicht optimiert, entschuldige die Unannehmlichkeiten.",
      donateLink: "💚 Dieses Projekt unterstützen (spenden)",
      linkHome: "← Zurück zur Achat Eco-Responsable Startseite",
      linkImages: "← Ich habe ein Bild und suche, wo ich es online kaufen kann",
      linkExemples: "← Ich weiß nicht, welche Art von Artikeln ich hier vergleichen kann (Beispiele ansehen)",
      footerMissionHtml: "© Alle Rechte vorbehalten • Herbst 2025 • Ein Projekt im ökologischen Interesse, um Industrielobbys zu besseren Entscheidungen für den Schutz unserer Umwelt zu bewegen. <br>\nErstellt von JOEL SANDÉ, aus Saguenay Lac Saint-Jean. <br><br>\nGeld ist gut, aber unsere Umwelt kommt zuerst. Man kann kein Omelett machen, ohne Eier zu zerbrechen, aber lass uns so wenig Schaden wie möglich anrichten. <br><br>\nLasst uns zahlreich diese Website nutzen, um Hersteller dazu zu bringen, sich um unsere Umwelt zu kümmern. <br>\nTeile massenhaft, um unsere Umwelt zu schützen."
    },
    index: {
      title: "Achat Éco-Responsable — vergleiche die Umweltauswirkungen",
      h2Compare: "Zu vergleichende Artikel", addBtn: "➕ Artikel hinzufügen", removeBtn: "➖ Letzten entfernen", runBtn: "Analyse starten",
      hintMinUrls: "Füge mindestens 2 URLs von Artikelseiten hinzu.",
      introIntro: "Vergleiche Artikel und wähle den ökologischsten:",
      introDesire: "Du möchtest ein Produkt kaufen, hast aber mehrere Optionen und zögerst, welche du wählen sollst.",
      introHelp: "Wir helfen dir, eine umweltbewusste Wahl zu treffen.",
      introPaste: "Füge einfach die URLs der Seiten deiner Artikel ein, und wir helfen dir bei der umweltbewussten Wahl:",
      li1: "CO2-Fußabdruck", li2: "Herstellung mit dem geringsten Wasserverbrauch", li3: "Das Produkt mit dem meisten recycelbaren Material",
      introRank: "Wir ordnen sie nach der umweltfreundlichsten Reihenfolge.",
      h3Shortlist: "Beispiel-Websites zum Testen",
      catMarketPriority: "Priorisierte Marktplätze", catMarketGeneral: "Allgemeine Marktplätze",
      catRetailers: "Große Einzelhändler (Kanada)", catFashion: "Mode & Outdoor",
      catHome: "Haus & Renovierung", catRefurb: "Generalüberholt / gebraucht", catEco: "„Öko\"-Auswahl",
      shortlistHint: "Diese Links sind <b>Formatbeispiele</b> (alle verweisen auf echte Produkte). Klicke, um die Art von Seiten zu sehen, die wir für dich analysieren können...",
      h2Results: "Ökologisches Ranking (absteigend)", h3Details: "Details pro Artikel",
      js: {
        urlLabel: "URL des Artikels", urlPlaceholder: "https://beispiel.com/produkt",
        analyzing: "Analyse läuft… ⏳", importedMsg: "{n} Link(s) aus der Bildersuche importiert.",
        needTwoUrls: "Füge mindestens 2 URLs hinzu.", unknownError: "Unbekannter Fehler", errorPrefix: "Fehler: ",
        scoreLabel: "Gesamtpunktzahl:", rankLabel: "Rang:", openPage: "Seite öffnen",
        lblMaterials: "Materialien", lblWater: "Wasser (L)", lblEnergy: "Energie (kWh)", lblCo2e: "CO₂e (kg)",
        lblBiodeg: "Bioabbaubarkeit", lblRecyc: "Recyclingfähigkeit", lblDurability: "Haltbarkeit/Reparierbarkeit",
        lblCertifications: "Zertifizierungen", lblPackaging: "Verpackung", lblTransport: "Transport",
        lblNotes: "Notizen", lblConfidence: "Vertrauen", viewSubscores: "Unterbewertungen ansehen", errorsTitle: "URLs mit Fehlern:"
      }
    },
    images: {
      title: "Bildersuche — Achat Responsable",
      h1: "Bildersuche",
      introLine1: "Lade ein Bild hoch → wir finden, wo man es online kaufen kann:",
      introLine2: "Lade das Bild eines Artikels hoch, den du kaufen möchtest (Zubehör - Kleidung - Fahrzeug - ...), und wir helfen dir, ihn online zu finden, mit Preis, wenn möglich.",
      labelChooseImage: "Bild auswählen", hintFormats: "Übliche Formate (jpg, png, jpeg, gif, webp). Größe <5MB bitte",
      btnSearch: "Suche starten", h3Selected: "Ausgewähltes Bild", searchForLabel: "🔍 Suche gestartet für:",
      h3Matches: "Gefundene Übereinstimmungen", noMatches: "Keine Übereinstimmungen gefunden.",
      noDescErrorHtml: "Die KI konnte das Bild nicht beschreiben — prüfe, ob <code>OPENAI_API_KEY</code> richtig konfiguriert ist.",
      noKeyErrorHtml: "Prüfe, ob <code>GOOGLE_CSE_KEY</code> und <code>GOOGLE_CSE_ID</code> richtig konfiguriert sind."
    },
    exemples: {
      title: "Welche Artikel kann ich vergleichen? Leitfaden — Achat Éco-Responsable",
      h1: "Welche Artikel kann ich vergleichen?",
      introText: "Keine Ahnung, wo du anfangen sollst? Hier ist ein Leitfaden mit vielfältigen Beispielen von Artikeln, für die unsere ökologische Analyse Sinn macht.",
      h2Principle: "Das Prinzip kurz erklärt",
      principleHtml: "Sobald ein Produkt eine Webseite hat, die seine Eigenschaften beschreibt (Materialien, Zertifizierungen, Verpackung, Herstellung...), können wir es in der Regel analysieren. Je mehr konkrete Informationen die Seite enthält (detailliertes Produktdatenblatt, JSON-LD, Umweltlabels), desto besser die Analyse. Der eigentliche Nutzen liegt darin, <b>2 ähnliche Artikel miteinander zu vergleichen</b> (gleicher Bedarf, unterschiedliche Optionen), um zu sehen, welcher umweltfreundlicher ist.",
      h3Categories: "Kategorien und konkrete Beispiele",
      catClothingTitle: "Kleidung & Textilien",
      catClothingLi: ["Bio-Baumwoll-T-Shirt vs. klassisches Polyester-T-Shirt","Daunen-Wintermantel vs. Mantel aus recycelten Synthetikfasern","Reguläre Denim-Jeans vs. Jeans mit geringerem Wasserverbrauch hergestellt","Laufschuhe aus recycelten Materialien vs. klassisches Modell"],
      catHomeTitle: "Haus, Bettwäsche & Möbel",
      catHomeLi: ["Bettbezug aus ägyptischer Baumwolle vs. aus Synthetikmaterial","FSC-zertifiziertes Holzbettgestell vs. Verbundwerkstoff/Kunststoff","Klassische Schaumstoffmatratze vs. Matratze aus Naturmaterialien"],
      catElecTitle: "Elektronik & Zubehör",
      catElecLi: ["Kabellose Kopfhörer von zwei verschiedenen Marken","Generalüberholter Laptop vs. neu","Generalüberholtes Telefon (Back Market, Poshmark) vs. neu","Smartwatch: Gehäusematerialien, Reparierbarkeit"],
      catApplTitle: "Haushaltsgeräte",
      catApplLi: ["Toaster oder Staubsauger: zwei Modelle mit oder ohne Energy-Star-Zertifizierung","Generalüberholtes Kleingerät vs. neu"],
      catSportTitle: "Sport & Outdoor",
      catSportLi: ["Wiederverwendbare Edelstahlflasche vs. Kunststoff","Rucksack oder Zelt: recycelte wasserdichte Materialien vs. klassisch","Fahrrad oder Outdoor-Ausrüstung von zwei Marken"],
      catHouseholdTitle: "Haushalts- und Körperpflegeprodukte",
      catHouseholdLi: ["Konzentriertes Spülmittel vs. normale Größe (weniger Verpackung/Transport)","Nachfüllbare Shampooflasche vs. Einwegflasche","Bambuszahnbürste vs. Kunststoff"],
      catToysTitle: "Spielzeug & Kinderartikel",
      catToysLi: ["Zertifiziertes Holzspielzeug vs. Plastikspielzeug","Kinderwagen oder Autositz: Materialien und Haltbarkeit"],
      catBagsTitle: "Taschen & Alltagszubehör",
      catBagsLi: ["Rucksack oder Koffer aus recycelten Materialien vs. klassisch","Lederbrieftasche vs. vegane/recycelte Materialien"],
      catDiyTitle: "Heimwerken & Renovierung",
      catDiyLi: ["VOC-arme Farbe vs. klassische Farbe","Ökologische Dämmung vs. Standarddämmung"],
      tipHtml: "Tipp: Für ein relevantes Ergebnis vergleiche Artikel, die den <b>gleichen Bedarf</b> erfüllen (zwei Kopfhörer, zwei Wintermäntel, zwei Trinkflaschen...), statt völlig unterschiedliche Objekte.",
      h3Warning: "⚠️ Was weniger gut funktioniert",
      warnLiHtml: ["Verderbliche Lebensmittel (Obst, Gemüse, Fleisch...) — kein Produktdatenblatt zum Analysieren.","Dienstleistungen (Abos, Reisen, Versicherungen...) — das sind keine physischen Gegenstände.","Produktseiten mit sehr wenig Details (wenig oder keine Beschreibung, kein technisches Datenblatt).","Manche Websites blockieren das automatisierte Lesen ihrer Seiten (z. B. derzeit Best Buy)."],
      ctaText: "Bereit, es mit deinen eigenen Artikeln auszuprobieren?", ctaBtn: "Meine Artikel vergleichen →"
    },
    don: {
      h1: "Dieses Projekt unterstützen", eyebrow: "🌱 Danke für deine Unterstützung.",
      interacLead: "💸 Sende eine Interac-Überweisung an {email}",
      copyBtn: "📋 Adresse kopieren",
      copyToast: "Kopiert! Füge diese Adresse in deine Banking-App ein, um die Überweisung zu senden.",
      missionText: "Sei großzügig, aber ganz ohne Verpflichtung. Deine Spenden helfen, die Serverkosten und die API-Kosten (KI-Analyse, Bildersuche, Websuche) zu decken, die den ökologischen Vergleichsrechner betreiben, und ermutigen mich, dieses Tool für einen verantwortungsbewussteren Konsum weiter zu verbessern.",
      canadaTitle: "🇨🇦 Bist du in Kanada?",
      canadaHint: "Interac funktioniert nur zwischen kanadischen Bankkonten. Klicke auf deine Bank, um direkt zu deinem Online-Banking zu gelangen und die Überweisung zu senden.",
      worldTitle: "🌍 Woanders auf der Welt?",
      worldHint: "Interac funktioniert außerhalb Kanadas nicht. Mit Kredit- oder Debitkarte funktioniert PayPal überall, ganz ohne Konto.",
      paypalBtn: "❤️ Per PayPal spenden",
      footerHtml: "© Alle Rechte vorbehalten • Herbst 2025 • Ein Projekt im ökologischen Interesse, um Industrielobbys zu besseren Entscheidungen für den Schutz unserer Umwelt zu bewegen. <br>\nErstellt von JOEL SANDÉ, aus Saguenay Lac Saint-Jean."
    }
  },

  it: {
    common: {
      notice: "💻 Questo sito funziona molto meglio su un computer o un tablet (più facile copiare e incollare gli URL) che su un telefono. L'uso mobile non è ancora ottimizzato, scusa per l'inconveniente.",
      donateLink: "💚 Sostieni questo progetto (dona)",
      linkHome: "← Torna alla pagina di Achat Eco-Responsable",
      linkImages: "← Ho un'immagine e cerco dove comprarlo online",
      linkExemples: "← Non so che tipo di articoli posso confrontare qui (vedi esempi)",
      footerMissionHtml: "© Tutti i diritti riservati • Autunno 2025 • Un progetto di interesse ecologico, per spingere le lobby industriali a fare scelte migliori per la preservazione del nostro ambiente. <br>\nRealizzato da JOEL SANDÉ, del Saguenay Lac Saint-Jean. <br><br>\nI soldi vanno bene, ma prima il nostro ambiente. Non si può fare una frittata senza rompere le uova, ma facciamo il minor danno possibile. <br><br>\nSiamo in tanti a usare questo sito, per costringere i produttori a preoccuparsi del nostro ambiente. <br>\nCondividi in massa! Per preservare il nostro ambiente."
    },
    index: {
      title: "Achat Éco-Responsable — confronta l'impatto ambientale",
      h2Compare: "Articoli da confrontare", addBtn: "➕ Aggiungi un articolo", removeBtn: "➖ Rimuovi l'ultimo", runBtn: "Avvia l'analisi",
      hintMinUrls: "Aggiungi almeno 2 URL di pagine di articoli.",
      introIntro: "Confronta articoli e scegli il più ecologico:",
      introDesire: "Vuoi acquistare un prodotto, ma hai diverse opzioni possibili ed esiti su quale scegliere.",
      introHelp: "Ti aiutiamo a fare una scelta eco-responsabile.",
      introPaste: "Devi solo incollare gli URL delle pagine dei tuoi articoli, e ti aiutiamo a fare le scelte eco-responsabili:",
      li1: "Impronta di carbonio", li2: "Fabbricazione che ha consumato meno acqua", li3: "Quello che contiene più materiale riciclabile",
      introRank: "Li classifichiamo secondo l'ordine più rispettoso dell'ambiente.",
      h3Shortlist: "Esempi di siti da provare",
      catMarketPriority: "Marketplace prioritari", catMarketGeneral: "Marketplace generalisti",
      catRetailers: "Grandi rivenditori (Canada)", catFashion: "Moda & outdoor",
      catHome: "Casa & ristrutturazione", catRefurb: "Ricondizionato / seconda mano", catEco: "Selezione «eco»",
      shortlistHint: "Questi link sono <b>esempi di formato</b> (tutti puntano a prodotti reali). Clicca per vedere il tipo di pagine che possiamo analizzare per te...",
      h2Results: "Classifica ecologica (decrescente)", h3Details: "Dettagli per articolo",
      js: {
        urlLabel: "URL dell'articolo", urlPlaceholder: "https://esempio.com/prodotto",
        analyzing: "Analisi in corso… ⏳", importedMsg: "{n} link importati dalla ricerca per immagini.",
        needTwoUrls: "Aggiungi almeno 2 URL.", unknownError: "Errore sconosciuto", errorPrefix: "Errore: ",
        scoreLabel: "Punteggio globale:", rankLabel: "Posizione:", openPage: "Apri la pagina",
        lblMaterials: "Materiali", lblWater: "Acqua (L)", lblEnergy: "Energia (kWh)", lblCo2e: "CO₂e (kg)",
        lblBiodeg: "Biodegradabilità", lblRecyc: "Riciclabilità", lblDurability: "Durabilità/Riparabilità",
        lblCertifications: "Certificazioni", lblPackaging: "Imballaggio", lblTransport: "Trasporto",
        lblNotes: "Note", lblConfidence: "Fiducia", viewSubscores: "Vedi i sottopunteggi", errorsTitle: "URL con errori:"
      }
    },
    images: {
      title: "Ricerca per immagini — Achat Responsable",
      h1: "Ricerca per immagini",
      introLine1: "Carica un'immagine → e troviamo dove acquistarlo online:",
      introLine2: "Carica l'immagine di un articolo che vuoi acquistare (accessorio - abbigliamento - veicolo - ...) e ti aiutiamo a trovarlo online, con il prezzo se possibile.",
      labelChooseImage: "Scegli un'immagine", hintFormats: "Formati comuni (jpg, png, jpeg, gif, webp). Dimensione <5Mb per favore",
      btnSearch: "Avvia la ricerca", h3Selected: "Immagine selezionata", searchForLabel: "🔍 Ricerca avviata per:",
      h3Matches: "Corrispondenze trovate", noMatches: "Nessuna corrispondenza trovata.",
      noDescErrorHtml: "L'IA non è riuscita a descrivere l'immagine — verifica che <code>OPENAI_API_KEY</code> sia configurata correttamente.",
      noKeyErrorHtml: "Verifica che <code>GOOGLE_CSE_KEY</code> e <code>GOOGLE_CSE_ID</code> siano configurate correttamente."
    },
    exemples: {
      title: "Quali articoli posso confrontare? Guida — Achat Éco-Responsable",
      h1: "Quali articoli posso confrontare?",
      introText: "Non sai da dove cominciare? Ecco una guida con esempi variati di articoli per cui la nostra analisi ecologica ha senso.",
      h2Principle: "Il principio in breve",
      principleHtml: "Non appena un prodotto ha una pagina web che descrive le sue caratteristiche (materiali, certificazioni, imballaggio, fabbricazione...), possiamo generalmente analizzarlo. Più la pagina contiene informazioni concrete (scheda prodotto dettagliata, JSON-LD, etichette ambientali), migliore sarà l'analisi. L'interesse sta soprattutto nel <b>confrontare 2 articoli simili tra loro</b> (stesso bisogno, opzioni diverse) per vedere quale sia più rispettoso dell'ambiente.",
      h3Categories: "Categorie ed esempi concreti",
      catClothingTitle: "Abbigliamento & tessuti",
      catClothingLi: ["T-shirt in cotone biologico vs t-shirt in poliestere classico","Piumino invernale vs cappotto in fibre sintetiche riciclate","Jeans in denim regolare vs denim prodotto con meno acqua","Scarpe da corsa in materiali riciclati vs modello classico"],
      catHomeTitle: "Casa, biancheria da letto & mobili",
      catHomeLi: ["Copripiumino in cotone egiziano vs materiali sintetici","Struttura letto in legno certificato FSC vs composito/plastica","Materasso in schiuma classico vs materasso in materiali naturali"],
      catElecTitle: "Elettronica & accessori",
      catElecLi: ["Auricolari wireless di due marche diverse","Laptop ricondizionato vs nuovo","Telefono ricondizionato (Back Market, Poshmark) vs nuovo","Smartwatch: materiali della cassa, riparabilità"],
      catApplTitle: "Elettrodomestici",
      catApplLi: ["Tostapane o aspirapolvere: due modelli con o senza certificazione Energy Star","Piccolo elettrodomestico ricondizionato vs nuovo"],
      catSportTitle: "Sport & outdoor",
      catSportLi: ["Borraccia riutilizzabile in acciaio inox vs plastica","Zaino o tenda: materiali impermeabili riciclati vs classici","Bicicletta o attrezzatura outdoor di due marche"],
      catHouseholdTitle: "Prodotti per la casa & cura personale",
      catHouseholdLi: ["Detersivo piatti concentrato vs formato regolare (meno imballaggio/trasporto)","Shampoo in flacone ricaricabile vs flacone usa e getta","Spazzolino in bambù vs plastica"],
      catToysTitle: "Giocattoli & articoli per bambini",
      catToysLi: ["Giocattoli in legno certificato vs giocattoli in plastica","Passeggino o seggiolino auto: materiali e durabilità"],
      catBagsTitle: "Borse & accessori quotidiani",
      catBagsLi: ["Zaino o valigia in materiali riciclati vs classico","Portafoglio in pelle vs materiali vegani/riciclati"],
      catDiyTitle: "Fai da te & ristrutturazione",
      catDiyLi: ["Vernice a basso contenuto di COV vs vernice classica","Isolante ecologico vs isolante standard"],
      tipHtml: "Consiglio: per un risultato pertinente, confronta articoli che rispondono allo <b>stesso bisogno</b> (due auricolari, due cappotti invernali, due borracce...) piuttosto che oggetti completamente diversi.",
      h3Warning: "⚠️ Cosa funziona meno bene",
      warnLiHtml: ["I prodotti alimentari deperibili (frutta, verdura, carne...) — nessuna scheda prodotto da analizzare.","I servizi (abbonamenti, viaggi, assicurazioni...) — non sono oggetti fisici.","Pagine prodotto molto povere di dettagli (poca o nessuna descrizione, nessuna scheda tecnica).","Alcuni siti bloccano la lettura automatizzata delle loro pagine (es. Best Buy attualmente)."],
      ctaText: "Pronto a provare con i tuoi articoli?", ctaBtn: "Confronta i miei articoli →"
    },
    don: {
      h1: "Sostieni questo progetto", eyebrow: "🌱 Grazie per il tuo sostegno.",
      interacLead: "💸 Invia un bonifico Interac a {email}",
      copyBtn: "📋 Copia l'indirizzo",
      copyToast: "Copiato! Incolla questo indirizzo nella tua app bancaria per inviare il bonifico.",
      missionText: "Sii generoso, ma senza alcun obbligo. Le tue donazioni aiutano a coprire i costi di hosting del server e i costi delle API (analisi tramite intelligenza artificiale, ricerca immagini, ricerca web) che fanno funzionare il comparatore ecologico, e mi incoraggiano a continuare a migliorare questo strumento al servizio di un consumo più responsabile.",
      canadaTitle: "🇨🇦 Sei in Canada?",
      canadaHint: "Interac funziona solo tra conti bancari canadesi. Clicca sulla tua banca per accedere direttamente al tuo conto online e inviare il bonifico.",
      worldTitle: "🌍 Altrove nel mondo?",
      worldHint: "Interac non funziona fuori dal Canada. Con carta di credito o debito, ovunque tu sia, PayPal funziona senza bisogno di un account.",
      paypalBtn: "❤️ Dona con PayPal",
      footerHtml: "© Tutti i diritti riservati • Autunno 2025 • Un progetto di interesse ecologico, per spingere le lobby industriali a fare scelte migliori per la preservazione del nostro ambiente. <br>\nRealizzato da JOEL SANDÉ, del Saguenay Lac Saint-Jean."
    }
  },

  ar: {
    common: {
      notice: "💻 يعمل هذا الموقع بشكل أفضل بكثير على حاسوب أو جهاز لوحي (أسهل لنسخ ولصق الروابط) مقارنة بالهاتف. لم يتم بعد تحسين الاستخدام على الهاتف المحمول، نعتذر عن الإزعاج.",
      donateLink: "💚 دعم هذا المشروع (تبرّع)",
      linkHome: "← العودة إلى صفحة Achat Eco-Responsable",
      linkImages: "← لدي صورة وأبحث عن أماكن شرائه عبر الإنترنت",
      linkExemples: "← لا أعرف أي نوع من المنتجات يمكنني مقارنته هنا (شاهد أمثلة)",
      footerMissionHtml: "© جميع الحقوق محفوظة • خريف 2025 • مشروع ذو اهتمام بيئي، يهدف إلى دفع جماعات الضغط الصناعية لاتخاذ خيارات أفضل للحفاظ على بيئتنا. <br>\nأنجزه JOEL SANDÉ، من منطقة Saguenay Lac Saint-Jean. <br><br>\nالمال أمر جيد، لكن بيئتنا أولاً. لا يمكن صنع عجة دون كسر البيض، لكن لنقلل الضرر قدر الإمكان. <br><br>\nلنكن كثيرين في استخدام هذا الموقع، لإجبار المصنّعين على الاهتمام ببيئتنا. <br>\nشاركوا على نطاق واسع، للحفاظ على بيئتنا."
    },
    index: {
      title: "Achat Éco-Responsable — قارن الأثر البيئي",
      h2Compare: "المنتجات المراد مقارنتها", addBtn: "➕ إضافة منتج", removeBtn: "➖ حذف الأخير", runBtn: "بدء التحليل",
      hintMinUrls: "أضف رابطين على الأقل لصفحات المنتجات.",
      introIntro: "قارن بين المنتجات واختر الأكثر صداقة للبيئة:",
      introDesire: "تريد شراء منتج، لكن لديك عدة خيارات ممكنة وتتردد في أيها تختار.",
      introHelp: "نساعدك على اتخاذ خيار مسؤول بيئياً.",
      introPaste: "كل ما عليك فعله هو لصق روابط صفحات منتجاتك، ونساعدك على اتخاذ الخيارات المسؤولة بيئياً:",
      li1: "البصمة الكربونية", li2: "التصنيع الذي استهلك أقل كمية من الماء", li3: "المنتج الذي يحتوي على أكبر كمية من المواد القابلة لإعادة التدوير",
      introRank: "نرتبها حسب الأكثر احتراماً للبيئة.",
      h3Shortlist: "أمثلة على مواقع للتجربة",
      catMarketPriority: "أسواق إلكترونية ذات أولوية", catMarketGeneral: "أسواق إلكترونية عامة",
      catRetailers: "كبار تجار التجزئة (كندا)", catFashion: "الموضة والأنشطة الخارجية",
      catHome: "المنزل والتجديد", catRefurb: "مجدّد / مستعمل", catEco: "اختيارات «بيئية»",
      shortlistHint: "هذه الروابط هي <b>أمثلة على الشكل</b> (تشير جميعها إلى منتجات حقيقية). اضغط لمعرفة نوع الصفحات التي يمكننا تحليلها لك...",
      h2Results: "الترتيب البيئي (تنازلي)", h3Details: "التفاصيل حسب المنتج",
      js: {
        urlLabel: "رابط المنتج", urlPlaceholder: "https://exemple.com/produit",
        analyzing: "جارٍ التحليل… ⏳", importedMsg: "تم استيراد {n} رابط (روابط) من البحث بالصور.",
        needTwoUrls: "أضف رابطين على الأقل.", unknownError: "خطأ غير معروف", errorPrefix: "خطأ: ",
        scoreLabel: "النتيجة الإجمالية:", rankLabel: "الترتيب:", openPage: "فتح الصفحة",
        lblMaterials: "المواد", lblWater: "الماء (لتر)", lblEnergy: "الطاقة (كيلوواط ساعة)", lblCo2e: "ثاني أكسيد الكربون المكافئ (كغ)",
        lblBiodeg: "قابلية التحلل الحيوي", lblRecyc: "قابلية إعادة التدوير", lblDurability: "المتانة/قابلية الإصلاح",
        lblCertifications: "الشهادات", lblPackaging: "التغليف", lblTransport: "النقل",
        lblNotes: "ملاحظات", lblConfidence: "درجة الثقة", viewSubscores: "عرض النتائج الفرعية", errorsTitle: "روابط بها أخطاء:"
      }
    },
    images: {
      title: "البحث بالصور — Achat Responsable",
      h1: "البحث بالصور",
      introLine1: "ارفع صورة ← وسنجد لك أماكن الشراء عبر الإنترنت:",
      introLine2: "ارفع صورة منتج تريد شراءه (إكسسوار - ملابس - مركبة - ...) ونساعدك على إيجاده عبر الإنترنت، مع السعر إن أمكن.",
      labelChooseImage: "اختر صورة", hintFormats: "الصيغ الشائعة (jpg, png, jpeg, gif, webp). الحجم أقل من 5 ميغابايت من فضلك",
      btnSearch: "بدء البحث", h3Selected: "الصورة المختارة", searchForLabel: "🔍 تم إطلاق البحث عن:",
      h3Matches: "التطابقات الموجودة", noMatches: "لم يتم العثور على أي تطابق.",
      noDescErrorHtml: "لم يتمكن الذكاء الاصطناعي من وصف الصورة — تحقق من ضبط <code>OPENAI_API_KEY</code> بشكل صحيح.",
      noKeyErrorHtml: "تحقق من ضبط <code>GOOGLE_CSE_KEY</code> و<code>GOOGLE_CSE_ID</code> بشكل صحيح."
    },
    exemples: {
      title: "ما نوع المنتجات التي يمكنني مقارنتها؟ — Achat Éco-Responsable",
      h1: "ما نوع المنتجات التي يمكنني مقارنتها؟",
      introText: "لا تعرف من أين تبدأ؟ إليك دليل يضم أمثلة متنوعة من المنتجات التي يكون فيها تحليلنا البيئي منطقياً.",
      h2Principle: "الفكرة باختصار",
      principleHtml: "بمجرد أن يكون للمنتج صفحة ويب تصف خصائصه (المواد، الشهادات، التغليف، التصنيع...)، يمكننا عادة تحليله. كلما احتوت الصفحة على معلومات ملموسة أكثر (بطاقة منتج مفصّلة، JSON-LD، علامات بيئية)، كان التحليل أفضل. والفائدة الحقيقية تكمن في <b>مقارنة منتجين متشابهين</b> (نفس الحاجة، خيارات مختلفة) لمعرفة أيهما أكثر احتراماً للبيئة.",
      h3Categories: "الفئات والأمثلة الملموسة",
      catClothingTitle: "الملابس والمنسوجات",
      catClothingLi: ["قميص من القطن العضوي مقابل قميص من البوليستر التقليدي","معطف شتوي من الريش الزغبي مقابل معطف من ألياف اصطناعية معاد تدويرها","بنطال جينز عادي مقابل جينز مصنوع باستهلاك أقل للماء","حذاء رياضي من مواد معاد تدويرها مقابل نموذج تقليدي"],
      catHomeTitle: "المنزل، المفروشات والأثاث",
      catHomeLi: ["غطاء لحاف من القطن المصري مقابل مواد اصطناعية","إطار سرير خشبي معتمد من FSC مقابل مواد مركبة/بلاستيك","مرتبة إسفنجية تقليدية مقابل مرتبة من مواد طبيعية"],
      catElecTitle: "الإلكترونيات والإكسسوارات",
      catElecLi: ["سماعات لاسلكية من ماركتين مختلفتين","حاسوب محمول مجدّد مقابل جديد","هاتف مجدّد (Back Market، Poshmark) مقابل جديد","ساعة ذكية: مواد الهيكل، قابلية الإصلاح"],
      catApplTitle: "الأجهزة المنزلية",
      catApplLi: ["محمصة أو مكنسة كهربائية: طرازان بشهادة Energy Star أو بدونها","جهاز منزلي صغير مجدّد مقابل جديد"],
      catSportTitle: "الرياضة والأنشطة الخارجية",
      catSportLi: ["زجاجة ماء قابلة لإعادة الاستخدام من الفولاذ المقاوم للصدأ مقابل البلاستيك","حقيبة ظهر أو خيمة: مواد مقاومة للماء معاد تدويرها مقابل تقليدية","دراجة أو معدات خارجية من ماركتين"],
      catHouseholdTitle: "منتجات منزلية والعناية الشخصية",
      catHouseholdLi: ["سائل غسيل الصحون المركّز مقابل الحجم العادي (تغليف/نقل أقل)","زجاجة شامبو قابلة لإعادة التعبئة مقابل زجاجة يُستغنى عنها","فرشاة أسنان من الخيزران مقابل البلاستيك"],
      catToysTitle: "الألعاب ومستلزمات الأطفال",
      catToysLi: ["ألعاب خشبية معتمدة مقابل ألعاب بلاستيكية","عربة أطفال أو مقعد سيارة: المواد والمتانة"],
      catBagsTitle: "الحقائب والإكسسوارات اليومية",
      catBagsLi: ["حقيبة ظهر أو حقيبة سفر من مواد معاد تدويرها مقابل تقليدية","محفظة جلدية مقابل مواد نباتية/معاد تدويرها"],
      catDiyTitle: "الإصلاح المنزلي والتجديد",
      catDiyLi: ["طلاء منخفض المركبات العضوية المتطايرة مقابل طلاء تقليدي","عازل بيئي مقابل عازل قياسي"],
      tipHtml: "نصيحة: للحصول على نتيجة ذات صلة، قارن منتجات تلبي <b>نفس الحاجة</b> (سماعتان، معطفان شتويان، زجاجتا ماء...) بدلاً من أشياء مختلفة تماماً.",
      h3Warning: "⚠️ ما لا يعمل بشكل جيد",
      warnLiHtml: ["المنتجات الغذائية القابلة للتلف (فواكه، خضروات، لحوم...) — لا توجد بطاقة منتج لتحليلها.","الخدمات (اشتراكات، سفر، تأمين...) — ليست أشياء مادية.","صفحات المنتجات الفقيرة جداً بالتفاصيل (وصف قليل أو معدوم، لا بطاقة تقنية).","بعض المواقع تحظر القراءة الآلية لصفحاتها (مثل Best Buy حالياً)."],
      ctaText: "مستعد لتجربة ذلك مع منتجاتك الخاصة؟", ctaBtn: "قارن منتجاتي ←"
    },
    don: {
      h1: "دعم هذا المشروع", eyebrow: "🌱 شكراً لدعمك.",
      interacLead: "💸 إرسال حوالة Interac إلى {email}",
      copyBtn: "📋 نسخ العنوان",
      copyToast: "تم النسخ! الصق هذا العنوان في تطبيقك المصرفي لإرسال الحوالة.",
      missionText: "كن كريماً، ولكن دون أي التزام. تساعد تبرعاتك في تغطية تكاليف استضافة الخادم وتكاليف واجهات البرمجة (تحليل بالذكاء الاصطناعي، بحث بالصور، بحث على الويب) التي تشغّل أداة المقارنة البيئية، وتشجعني على الاستمرار في تحسين هذه الأداة في خدمة استهلاك أكثر مسؤولية.",
      canadaTitle: "🇨🇦 هل أنت في كندا؟",
      canadaHint: "تعمل خدمة Interac فقط بين الحسابات المصرفية الكندية. اضغط على بنكك للوصول مباشرة إلى حسابك عبر الإنترنت وإرسال الحوالة.",
      worldTitle: "🌍 في مكان آخر من العالم؟",
      worldHint: "لا تعمل خدمة Interac خارج كندا. عبر بطاقة ائتمان أو خصم، أينما كنت، تعمل PayPal دون الحاجة لحساب.",
      paypalBtn: "❤️ تبرّع عبر PayPal",
      footerHtml: "© جميع الحقوق محفوظة • خريف 2025 • مشروع ذو اهتمام بيئي، يهدف إلى دفع جماعات الضغط الصناعية لاتخاذ خيارات أفضل للحفاظ على بيئتنا. <br>\nأنجزه JOEL SANDÉ، من منطقة Saguenay Lac Saint-Jean."
    }
  },

  zh: {
    common: {
      notice: "💻 本网站在电脑或平板电脑上使用效果远优于手机（更方便复制粘贴网址）。移动端体验尚未优化，敬请谅解。",
      donateLink: "💚 支持本项目（捐赠）",
      linkHome: "← 返回 Achat Eco-Responsable 首页",
      linkImages: "← 我有一张图片，想找在线购买的地方",
      linkExemples: "← 我不知道这里可以比较哪些类型的商品（查看示例）",
      footerMissionHtml: "© 版权所有 • 2025年秋季 • 一个关注生态环境的项目，旨在推动产业游说团体做出更有利于保护我们共同环境的选择。<br>\n由 JOEL SANDÉ 制作，来自 Saguenay Lac Saint-Jean 地区。<br><br>\n钱固然重要，但环境优先。虽然不打破鸡蛋就做不成煎蛋卷，但我们应尽量减少伤害。<br><br>\n希望更多人使用本网站，促使制造商更加关注我们的环境。<br>\n请大力分享，共同保护我们的环境。"
    },
    index: {
      title: "Achat Éco-Responsable — 比较环境影响",
      h2Compare: "待比较的商品", addBtn: "➕ 添加一件商品", removeBtn: "➖ 删除最后一项", runBtn: "开始分析",
      hintMinUrls: "请至少添加2个商品页面的网址。",
      introIntro: "比较商品，选择最环保的一款：",
      introDesire: "你想购买一件产品，但有多个可选项，不知道该选哪个。",
      introHelp: "我们帮助你做出对环境负责任的选择。",
      introPaste: "你只需粘贴商品页面的网址，我们就会帮你做出对环境负责任的选择：",
      li1: "碳足迹", li2: "生产过程耗水量最少", li3: "含可回收材料最多的产品",
      introRank: "我们会按照最环保的顺序为它们排名。",
      h3Shortlist: "可试用的示例网站",
      catMarketPriority: "优先电商平台", catMarketGeneral: "综合电商平台",
      catRetailers: "大型零售商（加拿大）", catFashion: "时尚与户外用品",
      catHome: "家居与装修", catRefurb: "翻新/二手商品", catEco: "「环保」精选",
      shortlistHint: "这些链接是<b>格式示例</b>（均指向真实商品）。点击查看我们可以为你分析的页面类型……",
      h2Results: "生态排名（由高到低）", h3Details: "各商品详情",
      js: {
        urlLabel: "商品网址", urlPlaceholder: "https://example.com/product",
        analyzing: "分析中… ⏳", importedMsg: "已从图片搜索导入 {n} 个链接。",
        needTwoUrls: "请至少添加2个网址。", unknownError: "未知错误", errorPrefix: "错误：",
        scoreLabel: "总分：", rankLabel: "排名：", openPage: "打开页面",
        lblMaterials: "材料", lblWater: "用水量（升）", lblEnergy: "能耗（千瓦时）", lblCo2e: "二氧化碳当量（千克）",
        lblBiodeg: "可生物降解性", lblRecyc: "可回收性", lblDurability: "耐用性/可维修性",
        lblCertifications: "认证", lblPackaging: "包装", lblTransport: "运输",
        lblNotes: "备注", lblConfidence: "置信度", viewSubscores: "查看细项评分", errorsTitle: "出错的网址："
      }
    },
    images: {
      title: "以图搜索 — Achat Responsable",
      h1: "以图搜索",
      introLine1: "上传一张图片 → 我们帮你找到在线购买的地方：",
      introLine2: "上传你想购买的商品图片（配件 - 服装 - 车辆 - ……），我们会帮你在网上找到它，并尽可能显示价格。",
      labelChooseImage: "选择一张图片", hintFormats: "常见格式（jpg, png, jpeg, gif, webp）。请控制在5MB以内",
      btnSearch: "开始搜索", h3Selected: "已选图片", searchForLabel: "🔍 搜索内容：",
      h3Matches: "找到的匹配结果", noMatches: "未找到匹配结果。",
      noDescErrorHtml: "AI 无法描述该图片 — 请检查 <code>OPENAI_API_KEY</code> 是否配置正确。",
      noKeyErrorHtml: "请检查 <code>GOOGLE_CSE_KEY</code> 和 <code>GOOGLE_CSE_ID</code> 是否配置正确。"
    },
    exemples: {
      title: "我可以比较哪些商品？指南与示例 — Achat Éco-Responsable",
      h1: "我可以比较哪些商品？",
      introText: "不知道从何开始？这里有一份指南，列举了各种适合用我们的生态分析工具进行比较的商品示例。",
      h2Principle: "原理简述",
      principleHtml: "只要一款产品有描述其特性（材料、认证、包装、生产工艺等）的网页，我们通常就能对其进行分析。页面包含的具体信息越多（详细的产品资料、JSON-LD 结构化数据、环保标签），分析效果就越好。这个工具的真正价值在于<b>比较两件类似的商品</b>（满足相同需求、不同选项），从而看出哪一款更加环保。",
      h3Categories: "分类与具体示例",
      catClothingTitle: "服装与纺织品",
      catClothingLi: ["有机棉T恤 对比 传统聚酯纤维T恤","羽绒冬季外套 对比 再生合成纤维外套","普通牛仔裤 对比 耗水量更低工艺制成的牛仔裤","再生材料跑鞋 对比 传统款式"],
      catHomeTitle: "家居、床品与家具",
      catHomeLi: ["埃及棉被套 对比 合成纤维材质","FSC认证木质床架 对比 复合板材/塑料","传统海绵床垫 对比 天然材料床垫"],
      catElecTitle: "电子产品与配件",
      catElecLi: ["两个不同品牌的无线耳机","翻新笔记本电脑 对比 全新款","翻新手机（Back Market、Poshmark）对比 全新款","智能手表：表壳材料、可维修性"],
      catApplTitle: "家用电器",
      catApplLi: ["烤面包机或吸尘器：有无Energy Star认证的两款型号","翻新小家电 对比 全新款"],
      catSportTitle: "运动与户外",
      catSportLi: ["不锈钢可重复使用水壶 对比 塑料水壶","背包或帐篷：再生防水材料 对比 传统材料","两个品牌的自行车或户外装备"],
      catHouseholdTitle: "家居清洁与个人护理用品",
      catHouseholdLi: ["浓缩洗洁精 对比 普通规格（更少包装/运输）","可补充装洗发水瓶 对比 一次性瓶装","竹制牙刷 对比 塑料牙刷"],
      catToysTitle: "玩具与儿童用品",
      catToysLi: ["认证木制玩具 对比 塑料玩具","婴儿车或安全座椅：材料与耐用性"],
      catBagsTitle: "包袋与日常配件",
      catBagsLi: ["再生材料背包或行李箱 对比 传统款式","真皮钱包 对比 素食/再生材料钱包"],
      catDiyTitle: "DIY 与装修",
      catDiyLi: ["低VOC涂料 对比 传统涂料","环保隔热材料 对比 标准隔热材料"],
      tipHtml: "小贴士：为了得到有意义的结果，请比较满足<b>相同需求</b>的商品（两副耳机、两件冬季外套、两个水壶……），而不是完全不同的物品。",
      h3Warning: "⚠️ 效果不佳的情况",
      warnLiHtml: ["易腐食品（水果、蔬菜、肉类等）— 没有可供分析的产品资料。","服务类项目（订阅、旅行、保险等）— 不是实体商品。","内容非常有限的产品页面（描述很少或没有，没有技术规格表）。","部分网站会屏蔽自动化读取（例如目前的 Best Buy）。"],
      ctaText: "准备好用你自己的商品试一试了吗？", ctaBtn: "比较我的商品 →"
    },
    don: {
      h1: "支持本项目", eyebrow: "🌱 感谢你的支持。",
      interacLead: "💸 通过 Interac 转账至 {email}",
      copyBtn: "📋 复制地址",
      copyToast: "已复制！请在你的银行应用中粘贴此地址以发送转账。",
      missionText: "慷慨解囊，但绝无强制。你的捐赠将用于支付服务器托管费用以及支撑本生态比较工具运行的各类API费用（人工智能分析、图片搜索、网络搜索），并激励我持续改进这个致力于更负责任消费的工具。",
      canadaTitle: "🇨🇦 你在加拿大吗？",
      canadaHint: "Interac 转账仅支持加拿大银行账户之间使用。点击你的银行，直接进入你的网上银行发送转账。",
      worldTitle: "🌍 在世界其他地方？",
      worldHint: "Interac 在加拿大境外无法使用。无论你身在何处，使用信用卡或借记卡，PayPal 均可使用，无需注册账户。",
      paypalBtn: "❤️ 通过 PayPal 捐赠",
      footerHtml: "© 版权所有 • 2025年秋季 • 一个关注生态环境的项目，旨在推动产业游说团体做出更有利于保护我们共同环境的选择。<br>\n由 JOEL SANDÉ 制作，来自 Saguenay Lac Saint-Jean 地区。"
    }
  }
};

function getEcoLang() {
  return localStorage.getItem('ecoLang') || 'fr';
}

function setEcoLang(lang) {
  localStorage.setItem('ecoLang', lang);
}

function applyEcoDir(lang) {
  document.documentElement.lang = lang;
  document.documentElement.dir = ECO_RTL_LANGS.includes(lang) ? 'rtl' : 'ltr';
}

function ecoT(lang) {
  return ECO_TRANSLATIONS[lang] || ECO_TRANSLATIONS.fr;
}

/* Construit (ou reconstruit) le sélecteur de langue fixe en haut à droite,
   et appelle applyFn(lang) au chargement + à chaque changement de langue. */
function buildEcoLangSwitcher(applyFn) {
  let currentLang = getEcoLang();
  applyEcoDir(currentLang);
  applyFn(currentLang);

  let switcher = document.getElementById('langSwitcher');
  if (!switcher) {
    switcher = document.createElement('div');
    switcher.id = 'langSwitcher';
    document.body.prepend(switcher);
  }

  function render() {
    switcher.innerHTML = '';
    ECO_LANGS.forEach(lang => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'lang-btn' + (lang === currentLang ? ' active' : '');
      btn.textContent = lang.toUpperCase();
      btn.addEventListener('click', () => {
        currentLang = lang;
        setEcoLang(lang);
        applyEcoDir(lang);
        render();
        applyFn(lang);
      });
      switcher.appendChild(btn);
    });
  }
  render();
}
