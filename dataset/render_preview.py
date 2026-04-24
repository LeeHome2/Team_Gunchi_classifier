"""DXF → PNG 렌더링 + extents 메타 JSON."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, Optional

import ezdxf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ezdxf import bbox as ezbbox
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PREVIEW_DIR

_RENDER_CONFIG = None
try:
    from ezdxf.addons.drawing.config import ColorPolicy, Configuration
    # lineweight 관련 설정 제거 — 기본값 사용 (벽체 두께 과장 방지).
    # 흰 배경에 검은 선으로만 강제, 선 두께는 도면 원본 크기 따름.
    _RENDER_CONFIG = Configuration(
        color_policy=ColorPolicy.BLACK,
    )
except Exception:
    pass


def render_dxf_to_png(dxf_path, output_dir, *, figsize=(12, 12), dpi=120, max_size_mb=4.5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # HATCH/SOLID 제거: 채움 패턴이 도면 전체를 덮어버리는 문제 방지
    # (원본 DXF 파일은 안 건드림. 메모리 상 doc만 수정)
    removed = 0
    for e in list(msp):
        if e.dxftype() in ("HATCH", "SOLID"):
            msp.delete_entity(e)
            removed += 1

    ext = ezbbox.extents(msp, fast=True)
    extents_data = None
    if ext is not None and ext.has_data:
        extents_data = {
            "min_x": float(ext.extmin.x), "min_y": float(ext.extmin.y),
            "max_x": float(ext.extmax.x), "max_y": float(ext.extmax.y),
        }

    file_id = dxf_path.stem
    png_path = output_dir / f"{file_id}.png"
    meta_path = output_dir / f"{file_id}.preview.json"

    current_dpi = dpi
    attempts = 0
    size_mb = 0.0
    while attempts < 4:
        fig, ax = plt.subplots(figsize=figsize, dpi=current_dpi)
        ax.set_aspect("equal"); ax.axis("off")
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        try:
            if _RENDER_CONFIG is not None:
                Frontend(ctx, backend, config=_RENDER_CONFIG).draw_layout(msp, finalize=True)
            else:
                Frontend(ctx, backend).draw_layout(msp, finalize=True)
        except Exception as e:
            print(f"  [warn] 부분 렌더 실패: {e}", file=sys.stderr)
        fig.savefig(str(png_path), dpi=current_dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        size_mb = png_path.stat().st_size / (1024 * 1024)
        if size_mb <= max_size_mb:
            break
        current_dpi = int(current_dpi * 0.75)
        attempts += 1

    img_w = img_h = 0
    try:
        from PIL import Image
        with Image.open(png_path) as img:
            img_w, img_h = img.size
    except Exception:
        pass

    meta = {
        "file_id": file_id, "png_path": str(png_path),
        "image_size_mb": round(size_mb, 2), "dpi_final": current_dpi,
        "dpi_attempts": attempts + 1, "figsize_inches": list(figsize),
        "image_width_px": img_w, "image_height_px": img_h,
        "extents_cad": extents_data,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-i", "--input")
    g.add_argument("--input-dir")
    ap.add_argument("-o", "--output-dir", default=str(PREVIEW_DIR))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--figsize", type=int, default=12)
    ap.add_argument("--max-size-mb", type=float, default=4.5)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    paths = [Path(args.input)] if args.input else sorted(Path(args.input_dir).glob("*.dxf"))
    if args.limit and not args.input:
        paths = paths[: args.limit]

    ok = fail = 0
    for p in paths:
        try:
            meta = render_dxf_to_png(p, output_dir,
                figsize=(args.figsize, args.figsize),
                dpi=args.dpi, max_size_mb=args.max_size_mb)
            print(f"OK  {p.name}: {meta['image_size_mb']}MB ({meta['image_width_px']}x{meta['image_height_px']}, dpi={meta['dpi_final']})")
            ok += 1
        except Exception as e:
            print(f"ERR {p.name}: {e}")
            fail += 1
    print(f"\n완료: 성공 {ok} / 실패 {fail}")


if __name__ == "__main__":
    main()
