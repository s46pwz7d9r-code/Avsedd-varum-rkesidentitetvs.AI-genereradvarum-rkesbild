"""
GEO Brand Identity Study
========================
Syfte: Undersöka skillnaden mellan företags avsedda varumärkesidentitet
och den varumärkesbild som genereras av AI-baserade sökmotorer (GEO).

Kör varje fråga N_ITERATIONS gånger per motor och sparar rådata + analys.
Källanalys sker direkt på löptexten i varje svar (inga URL-citat krävs).
"""

import re
import json
import time
import random
import statistics
from datetime import datetime
from pathlib import Path

import openai
from google import genai
import anthropic
import pandas as pd
from textblob import TextBlob

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

OPENAI_API_KEY = "DIN_NYCKEL"
GEMINI_API_KEY = "DIN_NYCKEL"
CLAUDE_API_KEY = "DIN_NYCKEL"
N_ITERATIONS  = 100
DELAY_SECONDS = 1.2
OUTPUT_DIR    = Path("geo_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# VARUMÄRKEN
# ─────────────────────────────────────────────

BRANDS = [
    {
        "name": "Blocket",
        "intended_identity": {
            "attributes": [
                "enkelhet", "trygghet", "schysst", "hållbarhet", "cirkulär",
                "BankID", "köpskydd", "återbruk", "hela Sveriges marknadsplats",
                "demokratisering", "gemenskap", "lokal", "säkerhet", "förtroende",
                "andrahand", "cirkulär ekonomi", "positiv kraft"
            ],
            "sentiment": "positiv",
            "positioning": "Sveriges ledande marknadsplats för begagnat",
        },
    },
]

# Prompt – svenska
QUERY_TEMPLATE_SV = (
    "Ge en beskrivning och analys av varumärket {brand}. "
    "Redogör för {brand}s kärnvärden, egenskaper och positionering, "
    "samt de kännetecken som förknippas med varumärket. "
    "Ange vilka källor, webbplatser eller referenser som ligger till grund "
    "för informationen."
)

ACTIVE_TEMPLATE = QUERY_TEMPLATE_SV


# ─────────────────────────────────────────────
# KÄLLMÖNSTER – söks direkt i löptexten
# Lägg till egna mönster per varumärke vid behov
# ─────────────────────────────────────────────

SOURCE_PATTERNS = {
    # OWNED
    "blocket.se":               ("owned",             [r'blocket\.se', r'www\.blocket', r'blockets? officiell', 
                                                       r'blocketpaketet'r'blockets? egen kommunikation', r'blockets? (egna )?webbplats', r'årsredovisning']),
    "schibsted.com":            ("owned",             [r'schibsted']),
    # EARNED
    "Dagens Industri":          ("earned",            [r'dagens industri', r'\bdi\.se\b']),
    "Svenska Dagbladet":        ("earned",            [r'svenska dagbladet', r'\bsvd\.se\b']),
    "Breakit":                  ("earned",            [r'breakit']),
    "Market.se":                ("earned",            [r'market\.se']),
    "Nyhetsmedier (generellt)": ("earned",            [r'nyhetsartikel', r'affärspress', r'nyhetsmedier',
                                                       r'medierapportering', r'nyheter', r'pressmeddelande'r'externa analyser',
                                                       r'externa källor', r'branschanalys', r'oberoende analys']),
    # SHARED
    "Sociala medier": ("shared", [r'sociala medier.*källa', r'källa.*sociala medier',
                               r'instagram\.com', r'facebook\.com', r'linkedin\.com',
                               r'blocket.*blogg.*källa', r'källa.*instagram']),
    # EJ KATEGORISERBAR
    "Allmän kunskap":           ("ej_kategoriserbar", [r'allmän.*kunskap', r'allmänt.*känd',
                                                       r'allmänt tillgänglig', r'kollektiv.*uppfattning',
                                                       r'personlig.*erfarenhet', r'allmän bedömning',
                                                       r'branschföljare', r'bred.*förståelse',
                                                       r'långvarigt.*känd', r'observation']),
    "Inga specifika källor":    ("ej_kategoriserbar", [r'inga specifika', r'inte refererats',
                                                       r'specifika.*citerade']),
}


# ─────────────────────────────────────────────
# API-WRAPPERS
# ─────────────────────────────────────────────

def query_chatgpt(prompt: str) -> dict:
    client   = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return {
        "text":   response.choices[0].message.content,
        "model":  response.model,
        "tokens": response.usage.total_tokens,
    }


def query_gemini(prompt: str) -> dict:
    client   = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )
    return {
        "text":   response.text,
        "model":  "gemini-3.1-flash-lite",
        "tokens": None,
    }

def query_claude(prompt: str) -> dict:
    client   = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text":   response.content[0].text,
        "model":  "claude-sonnet-4-6",
        "tokens": response.usage.input_tokens + response.usage.output_tokens,
    }

ENGINES = {
    "chatgpt": query_chatgpt,
    "gemini":  query_gemini,
    "claude":  query_claude,
}


# ─────────────────────────────────────────────
# ANALYS-FUNKTIONER
# ─────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    blob         = TextBlob(text)
    polarity     = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.1:
        label = "positiv"
    elif polarity < -0.1:
        label = "negativ"
    else:
        label = "neutral"
    return {
        "polarity":     round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "label":        label,
    }


def score_attribute_overlap(text: str, intended_attributes: list) -> dict:
    text_lower = text.lower()
    hits   = [a for a in intended_attributes if a.lower() in text_lower]
    misses = [a for a in intended_attributes if a.lower() not in text_lower]
    ratio  = len(hits) / len(intended_attributes) if intended_attributes else 0
    return {
        "hits":    hits,
        "misses":  misses,
        "ratio":   round(ratio, 4),
        "n_hits":  len(hits),
        "n_total": len(intended_attributes),
    }


def extract_sources_section(text: str) -> str:
    """
    Försöker plocka ut källavsnittet ur svaret.
    Söker igenom hela texten om ingen källrubrik hittas.
    """
    m = re.search(
        r'(?:###?\s*Käll|###?\s*Source|Denna information|Information.*hämtad|'
        r'Baserat på|baseras på|enligt uppgifter från|information.*hämtad)(.*?)(?:\n---|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    # Om källrubrik hittas, returnera den sektionen
    if m:
        return m.group(0)
    # Annars – sök igenom hela svaret
    return text


def classify_sources_from_text(text: str) -> dict:
    """
    Genomsöker löptexten efter källreferenser och klassificerar dem
    som owned / earned / shared / ej_kategoriserbar enligt SOURCE_PATTERNS.
    """
    section = extract_sources_section(text)

    found_sources = {}
    for source_name, (category, patterns) in SOURCE_PATTERNS.items():
        match = any(re.search(p, section, re.IGNORECASE) for p in patterns)
        found_sources[source_name] = {
            "found":    1 if match else 0,
            "category": category,
        }

    peso_counts = {"owned": 0, "earned": 0, "shared": 0, "ej_kategoriserbar": 0}
    for info in found_sources.values():
        if info["found"]:
            peso_counts[info["category"]] += 1

    total    = sum(peso_counts.values()) or 1
    peso_pct = {k: round(v / total, 4) for k, v in peso_counts.items()}

    identified = [name for name, info in found_sources.items() if info["found"]]

    return {
        "binary":      found_sources,
        "peso_counts": peso_counts,
        "peso_pct":    peso_pct,
        "identified":  identified,
        "n_sources":   len(identified),
    }


# ─────────────────────────────────────────────
# DATAINSAMLING
# ─────────────────────────────────────────────

def run_study():
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records = []

    for brand in BRANDS:
        brand_name = brand["name"]
        intended   = brand["intended_identity"]
        prompt     = ACTIVE_TEMPLATE.format(brand=brand_name)

        print(f"\n{'='*60}")
        print(f"Varumärke: {brand_name}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"{'='*60}")

        for engine_name, engine_fn in ENGINES.items():
            print(f"\n  → Motor: {engine_name.upper()} ({N_ITERATIONS} iterationer)")
            engine_records = []

            for i in range(1, N_ITERATIONS + 1):
                try:
                    result     = engine_fn(prompt)
                    text       = result.get("text", "")
                    sentiment  = analyze_sentiment(text)
                    attributes = score_attribute_overlap(text, intended["attributes"])
                    sources    = classify_sources_from_text(text)

                    # Binär kolumn per källtyp
                    src_binary = {
                        f"src_{re.sub(r'[^a-z0-9]', '_', name.lower())}": info["found"]
                        for name, info in sources["binary"].items()
                    }

                    record = {
                        "timestamp":              datetime.now().isoformat(),
                        "brand":                  brand_name,
                        "engine":                 engine_name,
                        "iteration":              i,
                        "prompt":                 prompt,
                        "response":               text,
                        "response_len":           len(text),
                        "tokens":                 result.get("tokens"),
                        # Sentiment (UF1)
                        "sentiment_polarity":     sentiment["polarity"],
                        "sentiment_subjectivity": sentiment["subjectivity"],
                        "sentiment_label":        sentiment["label"],
                        # Attributöverlapp (UF1)
                        "attr_hits":              attributes["n_hits"],
                        "attr_total":             attributes["n_total"],
                        "attr_ratio":             attributes["ratio"],
                        "attr_hits_list":         json.dumps(attributes["hits"],   ensure_ascii=False),
                        "attr_misses_list":       json.dumps(attributes["misses"], ensure_ascii=False),
                        # PESO-aggregat (UF3)
                        "peso_owned":             sources["peso_counts"]["owned"],
                        "peso_earned":            sources["peso_counts"]["earned"],
                        "peso_shared":            sources["peso_counts"]["shared"],
                        "peso_ej_kat":            sources["peso_counts"]["ej_kategoriserbar"],
                        "peso_pct_owned":         sources["peso_pct"]["owned"],
                        "peso_pct_earned":        sources["peso_pct"]["earned"],
                        "peso_pct_shared":        sources["peso_pct"]["shared"],
                        "n_sources_identified":   sources["n_sources"],
                        "sources_identified":     " | ".join(sources["identified"]),
                        # Binära källflaggor
                        **src_binary,
                    }

                    engine_records.append(record)
                    all_records.append(record)

                    if i % 10 == 0:
                        avg_ratio  = statistics.mean(r["attr_ratio"]      for r in engine_records)
                        avg_owned  = statistics.mean(r["peso_pct_owned"]  for r in engine_records)
                        avg_earned = statistics.mean(r["peso_pct_earned"] for r in engine_records)
                        print(f"    [{i:>3}/{N_ITERATIONS}]  R_attr={avg_ratio:.2f}  "
                              f"owned={avg_owned:.0%}  earned={avg_earned:.0%}")
                    else:
                        print(f"    [{i:>3}/{N_ITERATIONS}]  ✓")

                except Exception as e:
                    print(f"    [FEL vid iteration {i}]: {e}")

                time.sleep(DELAY_SECONDS + random.uniform(0, 0.5))

            brand_safe   = brand_name.lower().replace(" ", "_")
            partial_path = OUTPUT_DIR / f"{timestamp}_{brand_safe}_{engine_name}_raw.csv"
            pd.DataFrame(engine_records).to_csv(partial_path, index=False, encoding="utf-8-sig")
            print(f"    ✓ Delresultat sparat: {partial_path.name}")

    df_all   = pd.DataFrame(all_records)
    raw_path = OUTPUT_DIR / f"{timestamp}_all_raw.csv"
    df_all.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Rådata sparad: {raw_path}")

    if df_all.empty:
        print("Ingen data samlades in – kontrollera API-nycklarna.")
        return df_all, None

    summary_path = OUTPUT_DIR / f"{timestamp}_summary.csv"
    summary = (
        df_all.groupby(["brand", "engine"])
        .agg(
            n_obs              = ("iteration",           "count"),
            mean_attr_ratio    = ("attr_ratio",          "mean"),
            std_attr_ratio     = ("attr_ratio",          "std"),
            mean_sentiment_pol = ("sentiment_polarity",  "mean"),
            std_sentiment_pol  = ("sentiment_polarity",  "std"),
            pct_positive_sent  = ("sentiment_label",     lambda x: (x == "positiv").mean()),
            mean_peso_owned    = ("peso_pct_owned",      "mean"),
            mean_peso_earned   = ("peso_pct_earned",     "mean"),
            mean_peso_shared   = ("peso_pct_shared",     "mean"),
            mean_n_sources     = ("n_sources_identified","mean"),
        )
        .round(4)
        .reset_index()
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"✓ Sammanfattning sparad: {summary_path}")
    print("\n" + summary.to_string(index=False))

    return df_all, summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("GEO Brand Identity Study")
    print(f"Startar {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Motorer: {list(ENGINES.keys())}")
    print(f"Varumärken: {[b['name'] for b in BRANDS]}")
    print(f"Iterationer per kombination: {N_ITERATIONS}")
    print()

    df, summary = run_study()
    print("\n✓ Studie klar.")
