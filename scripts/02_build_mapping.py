"""ARKit -> MetaHuman face board (GUI) esleme tablosunu uret.

Zincir:

    ARKit pozu  --(Epic posemap)-->  raw CTRL_expressions.*
                --(DNA GUI->raw tersi)-->  GUI CTRL_*.t[xy]  = face board kemigi

Neden GUI hedefliyoruz: addon'un evaluate() akisi her karede
`mapGUIToRawControls()` cagirip raw kontrolleri GUI'den yeniden ureti-
yor, yani dogrudan yazilan raw degerler eziliyor. Ayrintili gerekce:
docs/api-notes.md.

Girdi:
  --posemap   ARKitRemap'in PA_MetaHuman_ARKit_Mapping.posemap.json'u
  --controls  01_dump_dna_controls.py ciktisi (DNA'dan gercek isimler)

Cikti:
  mapping/arkit_to_mh.json       uygulanacak tablo
  mapping/_generated/build_report.json  kapsam + atlananlar + uyarilar

Calistirma (Blender gerekmez, saf python):
  python scripts/02_build_mapping.py --posemap <...> --controls <...>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# Epic'in pose asset'i orneklenerek cikarildigi icin (poseIndexToTimeRule
# i/(N-1) + baseline cikarma) tabloya cok kucuk yanlis pozitifler sizmis.
# Ornegin CheekPuff ve TongueOut kayitlarinda browLateralL 0.031 agirlikla
# gorunuyor; kas anatomisi acisindan anlamsiz, klasik ornekleme artefakti.
DEFAULT_EPSILON = 0.05

# Ayni GUI eksenine dusen katkilar toplanir; toplam bu araliga kirpilir.
# Aralik DNA'daki from/to degerlerinin birlesiminden turetilir.
SUM_THEN_CLAMP = True

# Tablonun anahtarlari MediaPipe'in KENDI kategori isimleridir. Boylece
# detector ciktisi ile mapping arasinda isim cevirme katmani hic olmaz --
# plandaki "Left/Right vs L/R" tuzagi bastan kalkar. Epic'in pose asset'i
# PascalCase kullaniyor; donusum tek kurallik: ilk harfi kucult.
#
# MediaPipe FaceLandmarker 52 kategori dondurur, ilki '_neutral'dir ve
# atlanir -> 51 gercek skor. ARKit'in 52'lik setinden farki: MediaPipe
# tongueOut URETMEZ. iPhone Live Link uretir, o yuzden tabloda tutuluyor.
MEDIAPIPE_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
ARKIT_ONLY = ["tongueOut"]  # sadece iPhone Live Link

# Epic'in pose asset'inden TURETILEMEYEN, elle yazilan esleme.
#
# mouthClose'un posemap'te sifirdan farkli tek kaydi yok, ama kanalin
# anlami net: "cene aciligindan BAGIMSIZ olarak dudaklarin kapanma
# miktari". MetaHuman'da bunun karsiligi lipsTogether kontrolleri --
# ust ve alt dudagi birbirine getiren cift.
#
# Bu olmadan "m", "b", "p" seslerinde cene hafif acikken dudaklarin
# kapanmasi gerekirken agiz acik kaliyor; konusmayi yapay gosteren
# sey buydu.
#
# TURETILMEDI, ANLAMDAN YAZILDI. Epic'in gercek agirliklari farkli
# olabilir; provenance'ta ayrica isaretleniyor.
AUTHORED = {
    "mouthClose": [
        {"gui": "CTRL_L_mouth_lipsTogetherU", "axis": "y", "weight": 1.0},
        {"gui": "CTRL_R_mouth_lipsTogetherU", "axis": "y", "weight": 1.0},
        {"gui": "CTRL_L_mouth_lipsTogetherD", "axis": "y", "weight": 1.0},
        {"gui": "CTRL_R_mouth_lipsTogetherD", "axis": "y", "weight": 1.0},
    ],
}
CANONICAL = set(MEDIAPIPE_BLENDSHAPES) | set(ARKIT_ONLY)


def to_mediapipe_name(epic_pose_name: str) -> str:
    """'EyeBlinkLeft' -> 'eyeBlinkLeft'. Kural tek satir, el tablosu yok."""
    return epic_pose_name[0].lower() + epic_pose_name[1:] if epic_pose_name else epic_pose_name


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_raw_lookup(controls: dict) -> dict[str, str]:
    """posemap'in 'ctrl_expressions_browraiseinl' formatini DNA'daki
    'CTRL_expressions.browRaiseInL' ismine baglar. Buyuk/kucuk harf ve
    ayrac farkini normalize eder; tahmin yok, birebir join."""
    return {name.replace(".", "_").lower(): name for name in controls["rawControlNames"]}


def invert_gui_to_raw(controls: dict) -> dict[str, list[dict]]:
    """Her raw kontrol icin onu suren GUI eksenlerini ve ters cevirme
    katsayilarini cikar.

    DNA parcali-dogrusal tutuyor:  raw = slope * gui + cut,  gui in [from, to]
    Tersi:                          gui = (raw - cut) / slope
    """
    gui_names = controls["guiControlNames"]
    raw_names = controls["rawControlNames"]
    idx_in = controls["guiToRawInputIndices"]
    idx_out = controls["guiToRawOutputIndices"]
    v_from = controls["guiToRawFromValues"]
    v_to = controls["guiToRawToValues"]
    v_slope = controls["guiToRawSlopeValues"]
    v_cut = controls["guiToRawCutValues"]

    per_raw: dict[str, list[dict]] = collections.defaultdict(list)
    for row in range(len(idx_in)):
        full = gui_names[idx_in[row]]
        bone, axis = full.split(".")
        per_raw[raw_names[idx_out[row]]].append(
            {
                "gui": bone,
                # DNA 'tx'/'ty' der, Blender pose_bone.location.x/.y bekler
                "axis": axis.lstrip("t").lower(),
                "from": v_from[row],
                "to": v_to[row],
                "slope": v_slope[row],
                "cut": v_cut[row],
            }
        )
    return per_raw


def solve_gui_value(rows: list[dict], raw_value: float) -> tuple[dict | None, str | None]:
    """raw_value'yu uretecek GUI katkisini sec.

    Tek satirli kontrollerde (243/263) dogrudan cozulur. Cok satirlilar
    lipSticky/neckSwallow gibi 'faz' kontrolleri: ayni slider uzerinde
    ucgen rampalar, tek bir raw degerini iki farkli gui degeri uretebilir.
    Bunlari belirsiz isaretleyip disarida birakiyoruz.
    """
    candidates = []
    for row in rows:
        if row["slope"] == 0:
            continue
        gui_value = (raw_value - row["cut"]) / row["slope"]
        lo, hi = min(row["from"], row["to"]), max(row["from"], row["to"])
        # kucuk kayan nokta toleransi
        if lo - 1e-6 <= gui_value <= hi + 1e-6:
            candidates.append((row, gui_value))

    if not candidates:
        return None, "hicbir GUI araligi bu degeri uretmiyor"
    if len(candidates) > 1:
        return None, f"belirsiz: {len(candidates)} GUI araligi ayni degeri uretiyor"

    row, gui_value = candidates[0]
    return {"gui": row["gui"], "axis": row["axis"], "weight": gui_value}, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posemap", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("mapping/arkit_to_mh.json"))
    parser.add_argument("--report", type=Path, default=Path("mapping/_generated/build_report.json"))
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    args = parser.parse_args()

    posemap = load_json(args.posemap)
    controls = load_json(args.controls)

    raw_lookup = build_raw_lookup(controls)
    per_raw = invert_gui_to_raw(controls)

    # ARKit pozu -> {raw control: agirlik}
    by_pose: dict[str, dict[str, float]] = collections.OrderedDict()
    for record in posemap["records"]:
        by_pose.setdefault(record["arkitPoseName"], {})[record["sourceMhaCurveName"]] = record["weight"]

    mapping: dict[str, list[dict]] = collections.OrderedDict()
    report = {
        "sourcePosemap": posemap.get("assetPath"),
        "sourceDna": controls.get("sourceDna"),
        "epsilon": args.epsilon,
        "droppedBelowEpsilon": [],
        "unresolvedCurve": [],
        "ambiguous": [],
        "excludedPoses": [],
        "missingFromPosemap": [],
        "poseCount": 0,
        "channelCount": 0,
    }

    for pose_name, curves in by_pose.items():
        name = to_mediapipe_name(pose_name)
        # Kanonik olmayanlari disarida birak. Pose asset'inde 'Pose_4'..'Pose_17'
        # gibi isimsiz girdiler var; hepsi ayni degerlere ornekleniyor (TongueOut
        # ile ozdes), yani ayirt edilememis ekstra dil pozlari. Cop veri.
        if name not in CANONICAL:
            report["excludedPoses"].append({"pose": pose_name, "reason": "kanonik ARKit ismi degil"})
            continue

        # ayni GUI eksenine birden fazla raw dusebilir -> topla
        accumulated: dict[tuple[str, str], float] = collections.OrderedDict()

        for curve_name, weight in sorted(curves.items()):
            if abs(weight) < args.epsilon:
                report["droppedBelowEpsilon"].append(
                    {"pose": pose_name, "curve": curve_name, "weight": weight}
                )
                continue

            raw_name = raw_lookup.get(curve_name)
            if raw_name is None:
                report["unresolvedCurve"].append({"pose": pose_name, "curve": curve_name})
                continue

            rows = per_raw.get(raw_name)
            if not rows:
                report["unresolvedCurve"].append(
                    {"pose": pose_name, "curve": curve_name, "reason": "DNA'da GUI surucusu yok"}
                )
                continue

            contribution, problem = solve_gui_value(rows, weight)
            if contribution is None:
                report["ambiguous"].append(
                    {"pose": pose_name, "curve": curve_name, "weight": weight, "reason": problem}
                )
                continue

            key = (contribution["gui"], contribution["axis"])
            accumulated[key] = accumulated.get(key, 0.0) + contribution["weight"]

        if accumulated:
            mapping[name] = [
                {"gui": gui, "axis": axis, "weight": round(value, 6)}
                for (gui, axis), value in accumulated.items()
            ]
            report["channelCount"] += len(accumulated)

    # elle yazilan eslemeleri ekle, ama once hedeflerinin DNA'da gercekten
    # var oldugunu dogrula -- isim uydurmus olmayalim
    known_gui = {n.split(".")[0] for n in controls["guiControlNames"]}
    report["authored"] = []
    for name, entries in AUTHORED.items():
        if name in mapping:
            report["authored"].append({"pose": name, "reason": "posemap'ten geldi, elle yazilan atlandi"})
            continue
        missing_gui = [e["gui"] for e in entries if e["gui"] not in known_gui]
        if missing_gui:
            report["authored"].append({"pose": name, "reason": f"DNA'da yok: {missing_gui}"})
            continue
        mapping[name] = entries
        report["channelCount"] += len(entries)
        report["authored"].append({"pose": name, "reason": "eklendi", "channels": len(entries)})

    report["poseCount"] = len(mapping)
    report["missingFromPosemap"] = [n for n in MEDIAPIPE_BLENDSHAPES if n not in mapping]

    # anahtarlari MediaPipe'in kendi sirasinda tut, okumasi kolay olsun
    mapping = collections.OrderedDict(
        (n, mapping[n]) for n in MEDIAPIPE_BLENDSHAPES + ARKIT_ONLY if n in mapping
    )

    # GUI ekseni basina gecerli aralik: kirpma icin. DNA'daki from/to birlesimi.
    limits: dict[str, dict[str, float]] = {}
    for rows in per_raw.values():
        for row in rows:
            key = f"{row['gui']}.{row['axis']}"
            lo, hi = min(row["from"], row["to"]), max(row["from"], row["to"])
            current = limits.setdefault(key, {"min": lo, "max": hi})
            current["min"] = min(current["min"], lo)
            current["max"] = max(current["max"], hi)

    document = {
        "schemaVersion": 1,
        "description": (
            "ARKit blendshape -> MetaHuman face board GUI kontrol ekseni. "
            "Uygulama: her ARKit skoru icin katkilari topla, sonra guiLimits ile kirp, "
            "sonucu face board pose_bone.location.<axis> olarak yaz."
        ),
        "provenance": {
            "posemap": posemap.get("assetPath"),
            "posemapExtraction": posemap.get("extractionMethod"),
            "posemapCaveat": (
                "Epic'in asset'inden ORNEKLENEREK cikarilmis, birebir asset degeri degil "
                "(poseIndexToTimeRule i/(N-1), baseline cikarma). Kucuk agirliklar epsilon ile elendi."
            ),
            "dna": controls.get("sourceDna"),
            "combineRule": "sum_then_clamp" if SUM_THEN_CLAMP else "max",
            "authoredChannels": sorted(AUTHORED),
            "authoredCaveat": (
                "Bu kanallar Epic'in asset'inden TURETILMEDI, anlamdan elle yazildi. "
                "Bkz. scripts/02_build_mapping.py AUTHORED."
            ),
        },
        # MediaPipe'in urettigi ama tabloda karsiligi olmayan kanallar. Bunlar
        # BILINEN bosluk; unknown_shapes() bunlari uyari olarak saymaz, yoksa
        # her karede calan bir alarm gercek isim kaymasini gizler.
        "knownUnmapped": report["missingFromPosemap"],
        "guiLimits": limits,
        "arkit": mapping,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    covered = len(MEDIAPIPE_BLENDSHAPES) - len(report["missingFromPosemap"])
    print(f"MediaPipe kapsama={covered}/{len(MEDIAPIPE_BLENDSHAPES)}")
    print(f"toplam poz={report['poseCount']} (tongueOut dahil) kanal={report['channelCount']}")
    print(f"  epsilon alti elenen : {len(report['droppedBelowEpsilon'])}")
    print(f"  cozulemeyen curve   : {len(report['unresolvedCurve'])}")
    print(f"  belirsiz            : {len(report['ambiguous'])}")
    print(f"  disarida birakilan  : {len(report['excludedPoses'])}")
    if report["missingFromPosemap"]:
        print(f"  MediaPipe'de olup tabloda OLMAYAN: {report['missingFromPosemap']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
