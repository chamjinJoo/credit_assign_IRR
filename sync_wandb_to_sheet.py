"""
sync_wandb_to_sheet.py
======================
wandb에서 eval/success_rate (마지막 값)를 가져와
로컬 SMAX_results.xlsx 파일을 업데이트하는 스크립트.

사전 준비
---------
1. wandb 로그인:
       wandb login

2. 패키지 설치:
       pip install wandb openpyxl numpy

3. Google Sheet를 xlsx로 다운로드:
       Google Sheet → 파일 → 다운로드 → Microsoft Excel(.xlsx)
       → 파일명을 SMAX_results.xlsx 로 저장 후 스크립트와 같은 폴더에 위치

사용법
------
    # 테스트 (파일 변경 없이 결과만 출력)
    python sync_wandb_to_sheet.py --dry-run

    # 특정 조합만 업데이트
    python sync_wandb_to_sheet.py --map 3m --condition "Tar, w=3" --model pci+ippo ippo

    # 전체 업데이트
    python sync_wandb_to_sheet.py
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import wandb

# ============================================================
# ★ 설정부 — 필요에 따라 수정하세요
# ============================================================
WANDB_ENTITY  = "chamjin-korea-advanced-institute-of-science-and-technology"
WANDB_PROJECT = "pci-marl-smax-jax"
METRIC_KEY    = "eval/success_rate"

XLSX_PATH     = Path("SMAX_results.xlsx")   # 스크립트와 같은 폴더에 위치

# ============================================================
# 이름 매핑 테이블
# ============================================================

MODEL_MAP = {
    "pci_ippo":  "pci+ippo",
    "pci_mappo": "pci+mappo",
    "ippo":      "ippo",
    "mappo":     "mappo",
}
MODEL_KEYS = sorted(MODEL_MAP.keys(), key=len, reverse=True)

MAP_DISPLAY = {
    "smacv2_10_units": "smacv2_10units",
    "smacv2_5_units":  "smacv2_5units",
    "3m":              "3m",
    "2s3z":            "2s3z",
    "3s5z":            "3s5z",
}
MAP_KEYS = sorted(MAP_DISPLAY.keys(), key=len, reverse=True)

CONDITION_MAP = {
    "tar_w=8": "Tar, w=8",
    "tar_w=6": "Tar, w=6",
    "tar_w=5": "Tar, w=5",
    "tar_w=4": "Tar, w=4",
    "tar_w=3": "Tar, w=3",
    "tar":     "Tar",
    "cb":      "C.B.",
    "":        "vanila",
}
KNOWN_CONDITIONS = sorted(CONDITION_MAP.keys(), key=len, reverse=True)

SHEET_MODELS = ["pci+ippo", "ippo", "pci+mappo", "mappo"]

# ============================================================
# run name 파싱
# ============================================================

def parse_run_name(run_name: str):
    """
    예시:
      jax_mappo_smax_smacv2_5_units_tar_w=3_seed42          <- clean
      jax_pci_ippo_smax_smacv2_5_units_seed42               <- clean (vanilla)
      jax_pci_ippo_smax_smacv2_5_units_tar_w=3_seed2_fixloss-h5   <- memo (seed 뒤)
      jax_pci_ippo_smax_smacv2_5_units_tar_w=3_fixloss-h5_seed1   <- memo (seed 앞)

    반환: {"model", "map", "condition", "seed", "has_memo"} 또는 None
    """
    name = run_name.strip()

    if name.startswith("jax_"):
        name = name[4:]
    else:
        return None

    model_key = None
    for m in MODEL_KEYS:
        if name.startswith(m + "_"):
            model_key = m
            name = name[len(m) + 1:]
            break
    if model_key is None:
        return None

    if name.startswith("smax_"):
        name = name[5:]
    else:
        return None

    map_key = None
    for m in MAP_KEYS:
        if name.startswith(m):
            map_key = m
            name = name[len(m):].lstrip("_")
            break
    if map_key is None:
        return None

    seed_match = re.search(r'seed(\d+)', name)
    if seed_match is None:
        return None
    seed = int(seed_match.group(1))

    before_seed = name[:seed_match.start()].strip("_")
    after_seed  = name[seed_match.end():].strip("_")

    condition_key    = ""
    remaining_before = before_seed
    for cond in KNOWN_CONDITIONS:
        if cond and cond in before_seed:
            condition_key    = cond
            remaining_before = before_seed.replace(cond, "").strip("_")
            break

    has_memo = bool(after_seed) or bool(remaining_before)

    return {
        "model":     MODEL_MAP[model_key],
        "map":       MAP_DISPLAY[map_key],
        "condition": CONDITION_MAP[condition_key],
        "seed":      seed,
        "has_memo":  has_memo,
    }

# ============================================================
# wandb 수집 (clean run 우선)
# ============================================================

def fetch_wandb_results() -> dict:
    """
    중복 처리 우선순위:
      1순위 - clean run (메모 없음)
      2순위 - memo run  (clean이 없을 때만)
      동종 중복 -> 먼저 수집된 것 유지
    """
    api  = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

    results   = {}
    memo_flag = {}
    skipped   = []

    for run in runs:
        parsed = parse_run_name(run.name)
        if parsed is None:
            skipped.append(f"  [파싱 실패] {run.name}")
            continue

        try:
            history = run.history(keys=[METRIC_KEY], pandas=False)
            values  = [r[METRIC_KEY] for r in history
                       if METRIC_KEY in r and r[METRIC_KEY] is not None]
        except Exception as e:
            skipped.append(f"  [히스토리 오류] {run.name}: {e}")
            continue

        if not values:
            skipped.append(f"  [메트릭 없음] {run.name}")
            continue

        value    = values[-1]
        has_memo = parsed["has_memo"]
        key      = (parsed["map"], parsed["condition"], parsed["seed"], parsed["model"])

        if key in results:
            prev_memo = memo_flag[key]
            if prev_memo and not has_memo:
                print(f"  [중복-교체] clean이 memo를 대체: {run.name}")
                results[key]   = value
                memo_flag[key] = False
            elif not prev_memo and has_memo:
                print(f"  [중복-무시] clean 유지, memo 무시: {run.name}")
            else:
                kind = "memo" if has_memo else "clean"
                print(f"  [중복-무시] 동종({kind}) 중복, 먼저 수집된 값 유지: {run.name}")
            continue

        results[key]   = value
        memo_flag[key] = has_memo

        tag = " [memo]" if has_memo else ""
        print(f"  [수집{tag}] {run.name}")
        print(f"         -> {parsed['map']} | {parsed['condition']} | "
              f"seed={parsed['seed']} | {parsed['model']} = {value:.5f}")

    if skipped:
        print("\n[스킵된 runs]")
        print("\n".join(skipped))

    return results

# ============================================================
# 필터링
# ============================================================

def apply_filter(results: dict, maps, conditions, models) -> dict:
    """지정된 map/condition/model 조합만 남긴다. 빈 리스트 = 전체 허용."""
    def norm(s):
        return s.strip().lower()

    f_maps  = [norm(m) for m in maps]       if maps       else []
    f_conds = [norm(c) for c in conditions] if conditions else []
    f_mods  = [norm(m) for m in models]     if models     else []

    return {
        k: v for k, v in results.items()
        if (not f_maps  or norm(k[0]) in f_maps)
        and (not f_conds or norm(k[1]) in f_conds)
        and (not f_mods  or norm(k[3]) in f_mods)
    }

# ============================================================
# xlsx 업데이트
# ============================================================

def find_header_row(ws, required_cols):
    """required_cols를 모두 포함하는 헤더 행 번호(1-based)와 열 매핑 반환."""
    for row in ws.iter_rows():
        vals = [str(c.value).strip() if c.value is not None else "" for c in row]
        if all(rc in vals for rc in required_cols):
            return row[0].row, {v: i + 1 for i, v in enumerate(vals)}
    return None, {}


def update_detail_table(ws, results: dict, dry_run: bool) -> int:
    """Map | Variant | Seed | model... 테이블 업데이트."""
    header_row, col_map = find_header_row(ws, ["Map", "Variant", "Seed"])
    if header_row is None:
        print("  [경고] Detail 테이블 헤더(Map/Variant/Seed)를 찾지 못했습니다.")
        return 0

    map_c      = col_map.get("Map")
    variant_c  = col_map.get("Variant")
    seed_c     = col_map.get("Seed")
    model_cols = {m: col_map.get(m) for m in SHEET_MODELS}

    n = 0
    for row in ws.iter_rows(min_row=header_row + 1):
        def get(c):
            return str(row[c - 1].value).strip() if c and row[c - 1].value is not None else ""

        row_map     = get(map_c)
        row_variant = get(variant_c)
        row_seed_s  = get(seed_c)

        if not row_map or not row_seed_s:
            continue
        try:
            row_seed = int(row_seed_s)
        except ValueError:
            continue

        for model, mcol in model_cols.items():
            if mcol is None:
                continue
            key = (row_map, row_variant, row_seed, model)
            if key not in results:
                continue

            value = results[key]
            cell  = row[mcol - 1]
            print(f"  {'[DRY]' if dry_run else '[업데이트]'} "
                  f"{cell.coordinate} <- {value:.5f}  "
                  f"({row_map} | {row_variant} | seed={row_seed} | {model})")
            if not dry_run:
                cell.value = round(value, 5)
            n += 1

    return n


def update_summary_table(ws, results: dict, dry_run: bool) -> int:
    """Models 헤더가 있는 요약 테이블 업데이트 (mean (std) 형식)."""
    summary_header_row = None
    for row in ws.iter_rows():
        if any(str(c.value).strip() == "Models" for c in row if c.value):
            summary_header_row = row[0].row
            break
    if summary_header_row is None:
        print("  [경고] Summary 테이블 헤더(Models)를 찾지 못했습니다.")
        return 0

    model_header_row = summary_header_row + 1
    model_row_cells  = list(ws.iter_rows(
        min_row=model_header_row, max_row=model_header_row))[0]
    model_col_map = {
        str(c.value).strip(): c.column
        for c in model_row_cells if c.value
    }

    grouped = defaultdict(list)
    for (mp, cond, seed, model), v in results.items():
        grouped[(mp, cond, model)].append(v)

    current_map = ""
    n = 0
    for row in ws.iter_rows(min_row=model_header_row + 1):
        first_cell = row[0]
        first_val  = str(first_cell.value).strip() if first_cell.value else ""

        if first_val in MAP_DISPLAY.values():
            current_map = first_val
            continue

        variant = first_val
        if not current_map or not variant:
            continue

        for model, mcol in model_col_map.items():
            if model not in SHEET_MODELS:
                continue
            vals = grouped.get((current_map, variant, model), [])
            if not vals:
                continue

            mean = np.mean(vals)
            std  = np.std(vals)
            text = f"{mean:.3f} ({std:.3f})"
            cell = ws.cell(row=first_cell.row, column=mcol)

            print(f"  {'[DRY]' if dry_run else '[요약 업데이트]'} "
                  f"{cell.coordinate} <- {text}  "
                  f"({current_map} | {variant} | {model})")
            if not dry_run:
                cell.value = text
            n += 1

    return n

# ============================================================
# 메인
# ============================================================

def main():
    valid_maps   = list(MAP_DISPLAY.values())
    valid_conds  = list(CONDITION_MAP.values())
    valid_models = list(MODEL_MAP.values())

    parser = argparse.ArgumentParser(
        description="wandb -> SMAX_results.xlsx 동기화",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--map", dest="maps", nargs="+", default=[],
        metavar="MAP",
        help=(
            f"업데이트할 맵 (공백 구분).\n"
            f"선택지: {valid_maps}\n"
            f"예) --map 3m smacv2_5units"
        ),
    )
    parser.add_argument(
        "--condition", dest="conditions", nargs="+", default=[],
        metavar="COND",
        help=(
            f"업데이트할 condition (공백 구분, 쉼표 포함 시 따옴표 필요).\n"
            f"선택지: {valid_conds}\n"
            f"예) --condition 'Tar, w=3' vanila"
        ),
    )
    parser.add_argument(
        "--model", dest="models", nargs="+", default=[],
        metavar="MODEL",
        help=(
            f"업데이트할 모델 (공백 구분).\n"
            f"선택지: {valid_models}\n"
            f"예) --model pci+ippo ippo"
        ),
    )
    parser.add_argument(
        "--xlsx", default=str(XLSX_PATH), metavar="FILE",
        help=f"대상 xlsx 파일 경로 (기본값: {XLSX_PATH})",
    )
    parser.add_argument("--dry-run",    action="store_true",
                        help="파일 변경 없이 결과만 출력")
    parser.add_argument("--no-summary", action="store_true",
                        help="Summary 테이블 업데이트 건너뜀")
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)

    print("=" * 60)
    print(f"wandb 프로젝트: {WANDB_ENTITY}/{WANDB_PROJECT}")
    print(f"대상 파일     : {xlsx_path}")
    print(f"Dry-run       : {args.dry_run}")
    if args.maps:       print(f"map 필터      : {args.maps}")
    if args.conditions: print(f"condition 필터: {args.conditions}")
    if args.models:     print(f"model 필터    : {args.models}")
    print("=" * 60)

    # 1. xlsx 존재 확인
    if not xlsx_path.exists():
        print(f"\n[오류] 파일을 찾을 수 없습니다: {xlsx_path}")
        print("Google Sheet -> 파일 -> 다운로드 -> Microsoft Excel(.xlsx) 후 재시도하세요.")
        sys.exit(1)

    # 2. wandb 수집
    print("\n[1/3] wandb 데이터 수집 중...")
    results = fetch_wandb_results()
    print(f"\n총 {len(results)}개 결과 수집 완료.")

    if not results:
        print("[종료] 수집된 결과가 없습니다.")
        sys.exit(0)

    # 3. 필터 적용
    results = apply_filter(results, args.maps, args.conditions, args.models)
    print(f"필터 후 {len(results)}개 결과 대상.")
    if not results:
        print("[종료] 필터 조건에 맞는 결과가 없습니다.")
        sys.exit(0)

    # 4. xlsx 로드 및 업데이트
    print(f"\n[2/3] {xlsx_path} 로드 중...")
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    print(f"  시트: {ws.title}  (row={ws.max_row}, col={ws.max_column})")

    print("\n[3/3] 셀 업데이트 중...")

    print("\n  --- Detail 테이블 ---")
    n_detail = update_detail_table(ws, results, dry_run=args.dry_run)

    n_summary = 0
    if not args.no_summary:
        print("\n  --- Summary 테이블 ---")
        n_summary = update_summary_table(ws, results, dry_run=args.dry_run)

    # 5. 저장
    if not args.dry_run:
        wb.save(xlsx_path)
        print(f"\n  저장 완료: {xlsx_path}")

    print("\n" + "=" * 60)
    print(f"완료! detail={n_detail}셀, summary={n_summary}셀 업데이트")
    if args.dry_run:
        print("(dry-run 모드: 파일 변경 없음)")
    print("=" * 60)


if __name__ == "__main__":
    main()
