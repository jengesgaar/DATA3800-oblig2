"""
Korrelasjon mellom nedbør og avling.

Kjøres med:  python correlation.py

Hvis datasettet fra cleaning_merge.py ikke finnes, bruker den testdata i stedet.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- oppsett ---

DATAFIL = Path("datasets/analysedata.csv")
UTMAPPE = Path("datasets/korrelasjon")

# Disse må stemme overens med det cleaning_merge.py lager.
# TODO: dobbeltsjekk navnene
MAAL = "yield_deviation"

FORKLARENDE = [
    "rainfall_sum",          # sum nedbør i vekstsesongen
    "rainfall_std",          # hvor ujevnt regnet kom
    "rainfall_planting",     # nedbør rundt såing
    "rainfall_harvesting",   # nedbør rundt høsting
    "percentage_wet",        # andel våte dager
    "percentage_dry",        # andel tørre dager
    "longest_dryspell",      # lengste tørkeperiode i dager
    "days_with_frost",
]

KOL_FYLKE, KOL_VEKST, KOL_AAR = "county", "crop", "year"
GRUPPER = [KOL_FYLKE, KOL_VEKST, KOL_AAR]

# Fylker med færre år enn dette hopper vi over. Med 8 år må korrelasjonen
# opp i 0.71 for å bli signifikant.
MIN_AAR_PER_FYLKE = 8


def sjekk_kolonner(df, kreves=None):
    """Sjekker at kolonnene vi trenger faktisk finnes."""
    kreves = kreves or (FORKLARENDE + [MAAL] + GRUPPER)
    mangler = [k for k in kreves if k not in df.columns]
    if mangler:
        raise KeyError(f"Datasettet mangler {mangler}. Fant: {sorted(df.columns)}")


def _korr(x, y, metode):
    f = stats.spearmanr if metode == "spearman" else stats.pearsonr
    r, p = f(x, y)
    return float(r), float(p)

# --- analyse ---

def korrelasjonsmatrise(df, metode="spearman"):
    """Alle tallkolonnene korrelert mot hverandre.

    Bruker Spearman som standard. Den bryr seg bare om rekkefølgen på
    verdiene, ikke avstanden mellom dem, så et ekstremt år som 2018
    trekker ikke like mye.
    """
    sjekk_kolonner(df)
    return df[FORKLARENDE + [MAAL]].corr(method=metode).round(3)


def korrelasjon_med_pverdi(df, metode="spearman"):
    """Samme tall, men med p-verdi og n så vi ser hvor mye vi kan stole på dem."""
    sjekk_kolonner(df)
    rader = []
    for variabel in FORKLARENDE:
        par = df[[variabel, MAAL]].dropna()
        if len(par) < 4:
            rader.append({"variabel": variabel, "n": len(par),
                          "korrelasjon": np.nan, "p_verdi": np.nan})
            continue
        r, p = _korr(par[variabel], par[MAAL], metode)
        rader.append({"variabel": variabel, "n": len(par),
                      "korrelasjon": round(r, 3), "p_verdi": round(p, 4)})

    ut = pd.DataFrame(rader)
    # Vi tester 8 variabler samtidig, så vi ganger p-verdien med antall
    # tester for å ikke bli lurt av tilfeldige utslag.
    ut["p_justert"] = (ut.p_verdi * len(FORKLARENDE)).clip(upper=1.0).round(4)
    ut["signifikant_5pct"] = ut.p_justert < 0.05
    return ut


def per_vekst(df, metode="spearman"):
    """En tabell per vekst.

    Hele poenget med oppgaven er jo å se om vekstene reagerer ulikt, og
    det forsvinner hvis vi slår dem sammen.
    """
    sjekk_kolonner(df)
    deler = []
    for vekst, gruppe in df.groupby(KOL_VEKST):
        d = korrelasjon_med_pverdi(gruppe, metode)
        d.insert(0, KOL_VEKST, vekst)
        deler.append(d)
    return pd.concat(deler, ignore_index=True)


def per_fylke(df, variabel="rainfall_sum", metode="spearman"):
    """En korrelasjon per fylke og vekst, for å se om det varierer regionalt.

    Fylker med for få år droppes. En korrelasjon regnet på fem punkter
    ser like overbevisende ut i tabellen som en regnet på tjue.
    """
    sjekk_kolonner(df, [variabel, MAAL, KOL_FYLKE, KOL_VEKST])
    rader = []
    for (fylke, vekst), gruppe in df.groupby([KOL_FYLKE, KOL_VEKST]):
        par = gruppe[[variabel, MAAL]].dropna()
        if len(par) < MIN_AAR_PER_FYLKE:
            continue
        r, p = _korr(par[variabel], par[MAAL], metode)
        rader.append({KOL_FYLKE: fylke, KOL_VEKST: vekst, "n": len(par),
                      "korrelasjon": round(r, 3), "p_verdi": round(p, 4)})
    return (pd.DataFrame(rader)
              .sort_values([KOL_VEKST, "korrelasjon"], ascending=[True, False]))

# --- figurer ---

def varmekart(matrise, tittel, fil):
    """Varmekart med matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, aks = plt.subplots(figsize=(1.1 * len(matrise.columns) + 3,
                                     0.9 * len(matrise.index) + 2.5))
    bilde = aks.imshow(matrise.values, cmap="RdBu_r", vmin=-1, vmax=1)
    aks.set_xticks(range(len(matrise.columns)))
    aks.set_xticklabels(matrise.columns, rotation=40, ha="right")
    aks.set_yticks(range(len(matrise.index)))
    aks.set_yticklabels(matrise.index)

    # Skriv tallet inni hver rute, hvit tekst der bakgrunnen er for mørk
    for i in range(matrise.shape[0]):
        for j in range(matrise.shape[1]):
            v = matrise.values[i, j]
            if pd.notna(v):
                aks.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                         color="white" if abs(v) > 0.55 else "black")

    aks.set_title(tittel)
    fig.colorbar(bilde, ax=aks, shrink=.8, label="Korrelasjon")
    fig.tight_layout()
    fil.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fil, dpi=150)
    plt.close(fig)


def kjor_alt(df, utmappe=UTMAPPE):
    """Kjører alle analysene og lagrer resultatene."""
    utmappe.mkdir(parents=True, exist_ok=True)
    resultat = {
        "matrise_spearman": korrelasjonsmatrise(df, "spearman"),
        "matrise_pearson":  korrelasjonsmatrise(df, "pearson"),
        "samlet":           korrelasjon_med_pverdi(df),
        "per_vekst":        per_vekst(df),
        "per_fylke":        per_fylke(df),
    }
    for navn, tabell in resultat.items():
        tabell.to_csv(utmappe / f"{navn}.csv", index=navn.startswith("matrise"))

    varmekart(resultat["matrise_spearman"], "Korrelasjoner (Spearman)",
              utmappe / "varmekart_samlet.png")
    varmekart(resultat["per_vekst"].pivot(index=KOL_VEKST, columns="variabel",
                                          values="korrelasjon"),
              "Korrelasjon mot avlingsavvik, per vekst",
              utmappe / "varmekart_per_vekst.png")
    return resultat

# --- testdata ---

def testdata(fro=42):
    """Tilfeldige data med samme kolonner som det ekte datasettet.

    Hvete og bygg er satt opp til å tape på mye nedbør, havre ikke. Da kan
    jeg se om koden fanger det opp.
    """
    rng = np.random.default_rng(fro)
    effekt = {"Hvete": -0.45, "Bygg": -0.20, "Havre": 0.05}
    rader = []
    for fylke in ["Østlandet", "Innlandet", "Vestfold", "Agder",
                  "Rogaland", "Trøndelag", "Nordland"]:
        normal = rng.uniform(250, 550)
        for aar in range(2001, 2025):
            sum_mm = normal * rng.normal(1.0, 0.22)
            avvik = sum_mm / normal - 1
            våt = float(np.clip(rng.normal(0.45 + 0.30 * avvik, 0.06), 0, 1))
            for vekst, e in effekt.items():
                rader.append({
                    KOL_FYLKE: fylke, KOL_VEKST: vekst, KOL_AAR: aar,
                    MAAL: e * avvik + rng.normal(0, 0.10),
                    "rainfall_sum": round(sum_mm, 1),
                    "rainfall_std": round(abs(rng.normal(4, 1)), 2),
                    "rainfall_planting": round(sum_mm * rng.uniform(.15, .3), 1),
                    "rainfall_harvesting": round(sum_mm * rng.uniform(.2, .35), 1),
                    "percentage_wet": round(våt, 3),
                    "percentage_dry": round(1 - våt, 3),
                    "longest_dryspell": int(np.clip(rng.normal(10 - 8 * avvik, 3), 0, 40)),
                    "days_with_frost": int(np.clip(rng.normal(4 - 2 * avvik, 2), 0, 30)),
                })
    return pd.DataFrame(rader)


def main():
    if DATAFIL.exists():
        df = pd.read_csv(DATAFIL)
        print(f"Leste {len(df)} rader fra {DATAFIL}")
    else:
        df = testdata()
        print(f"Fant ikke {DATAFIL}, bruker testdata ({len(df)} rader).")
        print("NB: tallene under betyr ingenting, de er tilfeldige.\n")

    resultat = kjor_alt(df)
    print(resultat["per_vekst"].to_string(index=False))
    print(f"\nLagret i {UTMAPPE}")


if __name__ == "__main__":
    main()
