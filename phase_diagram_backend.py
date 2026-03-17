#!/usr/bin/env python3
# phase_diagram_backend.py
# Modified code from Carlos
import os
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fetchtemps(Mat, DF, n_steps=101):
    """
    - Mat: list/tuple of 3 strings e.g. ["La", "Sr", "MnO3"]
    - DF: pandas DataFrame already loaded from CSV
    - n_steps: number of x steps (default 101)
    Returns: (NDF, CDF) pandas DataFrames with columns ["x","T","Name"]
    """
    # Lists to store the retrieved data
    XNT = []; NT = []
    XCT = []; CT = []
    CName = []; NName = []
    # --- new DOI lists ---
    CDOI = []; NDOI = []

    # Adding log
    log_rows = []

    N = len(Mat)
    step = n_steps
    Qa = Mat[N - 1]
    X = np.linspace(0, 1, step)

    for x in X:
        # Adding cases for x=0,  x=1 and in between
        
        # print(x)
        
        if x == 0:
            Mu = Mat[0] + Qa
        elif x == 1:
            Mu = Mat[1] + Qa
        else:
            a = str(1 - x).zfill(2)[0:4]
            b = str(x).zfill(2)[0:4]
            Mu = Mat[0] + a + Mat[1] + b + Qa

        # protect against NaN in Names column with na=False
        # Main composition
        Da = DF[DF['Names'].str.contains(Mu, na=False)]
        
        Nx = len(Da['Names']) if 'Names' in Da else 0
        if Nx > 0:
            
            # Adding log
            for _, row in Da.iterrows():
                log_rows.append({
                    "Original_Name": row["Names"],
                    "Mu": Mu,
                    "Type": row["Type"],
                    "Normalised_Value": row["Normalised Value"],
                    "ID": row["_id"],
                    "DOI": row["DOI"],
                })
            
            Dneel = Da.loc[Da['Type'] == "Néel"]
            DCurie = Da.loc[Da['Type'] == 'Curie']
            Nneel = len(Dneel['Names'])
            Ncurie = len(DCurie['Names'])
            if Nneel > 0:
                xnn = x * np.ones(Nneel)
                XNT += list(xnn)
                NT += list(Dneel['Normalised Value'])
                NName += [Mu for _ in range(len(list(xnn)))]
                # append DOIs for Néel entries
                NDOI += list(Dneel['DOI'])
            if Ncurie > 0:
                xcn = x * np.ones(Ncurie)
                XCT += list(xcn)
                CT += list(DCurie['Normalised Value'])
                CName += [Mu for _ in range(len(list(xcn)))]
                # append DOIs for Curie entries
                CDOI += list(DCurie['DOI'])


    # --- Fractional matches (p/q with q=2..9). x is fraction of Mat[1] => x = (q-p)/q ---
    for q in range(2, 10):
        for p in range(1, q):
            Mu_frac = f"{Mat[0]}{p}/{q}{Mat[1]}{q-p}/{q}{Qa}"
            Df = DF[DF['Names'].str.contains(Mu_frac, na=False)]
            if Df.empty:
                continue
            x_frac = (q - p) / q
            for _, row in Df.iterrows():
                if row.get("Type") == "Néel":
                    XNT.append(x_frac); NT.append(row["Normalised Value"]); NName.append(Mu_frac)
                    NDOI.append(row.get("DOI", None))
                elif row.get("Type") == "Curie":
                    XCT.append(x_frac); CT.append(row["Normalised Value"]); CName.append(Mu_frac)
                    CDOI.append(row.get("DOI", None))
            
                # Logging
                log_rows.append({
                    "Original_Name": row["Names"],
                    "Mu": Mu_frac,
                    "Type": row["Type"],
                    "Normalised_Value": row["Normalised Value"],
                    "ID": row.get("_id", None),
                    "DOI": row["DOI"],
                })
    
    # --- Two-decimal explicit checks for 0.10,0.20,...,0.90 ---
    for i in range(1, 10):
        x_val = i / 10.0
        a2 = f"{1-x_val:.2f}"
        b2 = f"{x_val:.2f}"
        Mu2 = Mat[0] + a2 + Mat[1] + b2 + Qa
    
        Dm = DF[DF['Names'].str.contains(Mu2, na=False)]
        if Dm.empty:
            continue
    
        for _, row in Dm.iterrows():
            if row.get("Type") == "Néel":
                XNT.append(x_val); NT.append(row["Normalised Value"]); NName.append(Mu2)
                NDOI.append(row.get("DOI", None))
            elif row.get("Type") == "Curie":
                XCT.append(x_val); CT.append(row["Normalised Value"]); CName.append(Mu2)
                CDOI.append(row.get("DOI", None))
    
            # Logging (direct)
            log_rows.append({
                "Original_Name": row["Names"],
                "Mu": Mu2,
                "Type": row["Type"],
                "Normalised_Value": row["Normalised Value"],
                "ID": row.get("_id", None),
                "DOI": row["DOI"],
            })
    
    # Parse / clean numeric strings (best-effort; keep behaviour similar to original)
    def _parse_list_to_floats(lst):
        try:
            out = [str(t).strip() for t in lst]
            
            out = [s.replace("∼", "").replace("~", "").replace(">", "") for s in out]
            
            out = [s.split("±")[0] for s in out]
            out = [s.split("–")[0] for s in out]
            out = [s.split("-")[0] for s in out]
            
            out = [s.strip() for s in out]
            
            return [float(s) if s else np.nan for s in out]
        except Exception:
            return []

    NT = _parse_list_to_floats(NT)
    CT = _parse_list_to_floats(CT)

    # Build DataFrames
    NDF = pd.DataFrame(columns=["x", "T", "Name", "DOI"])
    if XNT and NT and len(XNT) == len(NT):
        NDF["x"] = XNT
        NDF["T"] = NT
        NDF["Name"] = NName
        NDF["DOI"] = NDOI

    CDF = pd.DataFrame(columns=["x", "T", "Name", "DOI"])
    if XCT and CT and len(XCT) == len(CT):
        CDF["x"] = XCT
        CDF["T"] = CT
        CDF["Name"] = CName
        CDF["DOI"] = CDOI

    
    # print(len(CDF["Name"]))
    
    # ADding log
    LOG_DIR = os.path.join("/app/storage", "materials_nollm_log",)    
    os.makedirs(LOG_DIR, exist_ok=True)
    filename = f"{Mat[0]}_(1-x)_{Mat[1]}_(x)_{Mat[2]}.csv"
    LOG_PATH = os.path.join(LOG_DIR, filename)
    
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(LOG_PATH, mode="w", index=False,
                      header=not os.path.exists(LOG_PATH))

    return NDF, CDF


def generate_phase_diagram(materials, csv_path=None, n_steps=101, save_path=None):
    """
    Compute Neel/Curie averages from CSV and return PNG bytes.
    - materials: list/tuple of three strings, e.g. ["La","Sr","MnO3"]
    - csv_path: optional path to phase_transitions.csv (default: ./phase_transitions.csv)
    - n_steps: number of composition steps (default 101)
    - save_path: optional filepath to save the PNG (if provided)
    Returns: tuple (png_bytes, saved_path_or_None)
    Raises ValueError on missing/no-data or FileNotFoundError on missing CSV.
    """
    
    #------------------------------- Errors -----------------------------------
    if not (isinstance(materials, (list, tuple)) and len(materials) == 3):
        raise ValueError("materials must be a list/tuple of three strings")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    #------------------------------- Errors -----------------------------------
    
    df = pd.read_csv(csv_path)

    NDF, CDF = fetchtemps(materials, df, n_steps=n_steps)

    return NDF, CDF

    # # If no data for both, signal to caller
    # if (NDF.empty if isinstance(NDF, pd.DataFrame) else True) and (CDF.empty if isinstance(CDF, pd.DataFrame) else True):
    #     raise ValueError("No Neel or Curie data found for the requested materials/composition range")

    # # Compute averages and std (ddof=0 to follow original script)
    # Xneel = sorted(list(set(list(NDF["x"])))) if not NDF.empty else []
    # Xcurie = sorted(list(set(list(CDF["x"])))) if not CDF.empty else []

    # AVNT = []; SDNT = []
    # AVCT = []; SDCT = []

    # for x in Xneel:
    #     dn = NDF.loc[NDF["x"] == x]
    #     U = dn["T"].mean()
    #     SU = dn["T"].std(ddof=0)
    #     AVNT.append(U)
    #     SDNT.append(SU)

    # for x in Xcurie:
    #     dm = CDF.loc[CDF["x"] == x]
    #     U = dm["T"].mean()
    #     SU = dm["T"].std(ddof=0)
    #     AVCT.append(U)
    #     SDCT.append(SU)

    # # Plot
    # fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    # # if Xneel:
    # #     ax.errorbar(Xneel, AVNT, SDNT, color='b', marker='o', mfc='b', ms=5, label="NEEL")
    # # if Xcurie:
    # #     ax.errorbar(Xcurie, AVCT, SDCT, color='r', marker='o', mfc='r', ms=5, label="CURIE")

    # # Scatter raw points used in averages
    # if not CDF.empty:
    #     ax.scatter(CDF["x"], CDF["T"], marker=">", s=20, alpha=0.75, label="CURIE")

    # if not NDF.empty:
    #     ax.scatter(NDF["x"], NDF["T"], s=20, alpha=0.75, label="NEEL")
    
    # ax.set_xlim([0, 1])
    # # keep the same y-limits as in the original top-level script
    # ax.set_ylim([100, 500])
    # ax.set_ylabel("Temperature(K)")
    # ax.set_xlabel("Composition x")
    # ax.legend()
    # ax.grid(True)

    # buf = io.BytesIO()
    # fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
    # plt.close(fig)
    # buf.seek(0)
    # png_bytes = buf.read()

    # saved = None
    # if save_path:
    #     try:
    #         with open(save_path, "wb") as f:
    #             f.write(png_bytes)
    #         saved = os.path.abspath(save_path)
    #     except Exception:
    #         # don't fail on save errors; still return bytes
    #         saved = None

    # return png_bytes, saved