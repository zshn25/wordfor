"""Minify JS, CSS, and HTML files for production deployment.

Usage:
    python minify.py              # minify in-place
    python minify.py --dry-run    # show savings without modifying files
"""

import argparse
import sys
from pathlib import Path

WORDFOR_DIR = Path(__file__).resolve().parent.parent

FILES = {
    "js": [WORDFOR_DIR / "app.js", WORDFOR_DIR / "sw.js"],
    "css": [WORDFOR_DIR / "style.css"],
    "html": [
        WORDFOR_DIR / "index.html",
        WORDFOR_DIR / "about.html",
        WORDFOR_DIR / "bench.html",
    ],
}


def minify_js(text):
    import rjsmin

    return rjsmin.jsmin(text)


def minify_css(text):
    import rcssmin

    return rcssmin.cssmin(text)


def minify_html(text):
    import htmlmin

    return htmlmin.minify(
        text,
        remove_comments=True,
        remove_empty_space=True,
        reduce_boolean_attributes=True,
        remove_optional_attribute_quotes=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Minify static assets")
    parser.add_argument("--dry-run", action="store_true", help="Show savings only")
    args = parser.parse_args()

    minifiers = {"js": minify_js, "css": minify_css, "html": minify_html}
    total_before = 0
    total_after = 0

    for ftype, paths in FILES.items():
        for path in paths:
            if not path.exists():
                print(f"  SKIP: {path.name} not found")
                continue
            original = path.read_text(encoding="utf-8")
            before = len(original.encode("utf-8"))
            minified = minifiers[ftype](original)
            after = len(minified.encode("utf-8"))
            pct = (1 - after / before) * 100 if before else 0

            total_before += before
            total_after += after

            status = "DRY RUN" if args.dry_run else "OK"
            print(
                f"  {path.name:20s}  {before:>8,} -> {after:>8,} bytes  ({pct:5.1f}% smaller)  [{status}]"
            )

            if not args.dry_run:
                path.write_text(minified, encoding="utf-8")

    pct_total = (1 - total_after / total_before) * 100 if total_before else 0
    print(
        f"\n  Total: {total_before:,} -> {total_after:,} bytes ({pct_total:.1f}% smaller)"
    )


if __name__ == "__main__":
    main()
