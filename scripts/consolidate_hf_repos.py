"""Consolidate 112 HuggingFace repos into 16 (one per model).

Uploads local checkpoint files and results into a single consolidated repo
per model with subdirectories for each experiment/mode combination. Generates
a comprehensive README.md model card and optionally deletes old repos.

Why this exists: Managing 112 separate repos (96 model + 16 results) is
unwieldy. One repo per model with all 6 variants inside is cleaner for
users, easier to maintain, and provides a single landing page per model.

Usage:
    uv run python scripts/consolidate_hf_repos.py --dry-run
    uv run python scripts/consolidate_hf_repos.py --slug pythia-70m --no-delete
    uv run python scripts/consolidate_hf_repos.py --no-delete
    uv run python scripts/consolidate_hf_repos.py --delete-only
    uv run python scripts/consolidate_hf_repos.py --delay 2.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from generate_model_cards import (
    ALL_EXPERIMENTS,
    ALL_MODES,
    HF_NAMESPACE,
    VARIANT_DIRS,
    generate_consolidated_card,
    load_aggregated_results,
)

SCRIPTS_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPTS_DIR / "checkpoints"


# ---------------------------------------------------------------------------
# Consolidation logic
# ---------------------------------------------------------------------------


def consolidate_model(
    api: object | None,
    slug: str,
    agg: dict,
    dry_run: bool,
    delay: float,
) -> bool:
    """Consolidate one model's repos into a single consolidated repo.

    Uploads local checkpoint files (from scripts/checkpoints/) and results
    (from scripts/fulltrain_*.json) into tzervas/bwsk-{slug} with subdirs.

    Args:
        api: HfApi instance (None for dry_run).
        slug: Model slug (e.g. "pythia-70m").
        agg: Aggregated results dict.
        dry_run: If True, skip actual uploads.
        delay: Seconds between API calls.

    Returns:
        True if all operations succeeded.
    """
    repo_id = f"{HF_NAMESPACE}/bwsk-{slug}"
    success = True

    # Create repo
    if dry_run:
        print(f"  [DRY RUN] Would create repo: {repo_id}")
    else:
        try:
            api.create_repo(repo_id, exist_ok=True)  # type: ignore[union-attr]
            print(f"  Created repo: {repo_id}")
            time.sleep(delay)
        except Exception as e:
            print(f"  ERROR creating repo {repo_id}: {e}")
            return False

    # Upload checkpoint files for each variant
    for exp in ALL_EXPERIMENTS:
        for mode in ALL_MODES:
            variant_dir = VARIANT_DIRS[(exp, mode)]
            best_path = CHECKPOINT_DIR / slug / exp / mode / "best_model"

            if not best_path.exists():
                print(f"  SKIP {slug}/{variant_dir}: no checkpoint at {best_path}")
                continue

            if dry_run:
                files = list(best_path.iterdir())
                print(f"  [DRY RUN] Would upload {len(files)} files to {repo_id}/{variant_dir}/")
            else:
                try:
                    api.upload_folder(  # type: ignore[union-attr]
                        folder_path=str(best_path),
                        path_in_repo=variant_dir,
                        repo_id=repo_id,
                        commit_message=(f"Add {variant_dir} model weights"),
                    )
                    print(f"  Uploaded: {repo_id}/{variant_dir}/")
                    time.sleep(delay)
                except Exception as e:
                    print(f"  ERROR uploading {variant_dir}: {e}")
                    success = False

            # Upload per-run training_results.json into variant dir
            run_json = SCRIPTS_DIR / f"fulltrain_{slug}_{exp}_{mode}.json"
            if run_json.exists():
                if dry_run:
                    print(f"  [DRY RUN] Would upload {variant_dir}/training_results.json")
                else:
                    try:
                        api.upload_file(  # type: ignore[union-attr]
                            path_or_fileobj=str(run_json),
                            path_in_repo=(f"{variant_dir}/training_results.json"),
                            repo_id=repo_id,
                            commit_message=(f"Add {variant_dir} training results"),
                        )
                        time.sleep(delay)
                    except Exception as e:
                        print(f"  ERROR uploading {variant_dir}/training_results.json: {e}")
                        success = False

    # Upload aggregated results.json at root
    agg_json = SCRIPTS_DIR / f"fulltrain_{slug}_results.json"
    if agg_json.exists():
        if dry_run:
            print("  [DRY RUN] Would upload results.json")
        else:
            try:
                api.upload_file(  # type: ignore[union-attr]
                    path_or_fileobj=str(agg_json),
                    path_in_repo="results.json",
                    repo_id=repo_id,
                    commit_message="Add aggregated training results",
                )
                time.sleep(delay)
            except Exception as e:
                print(f"  ERROR uploading results.json: {e}")
                success = False

    # Generate and upload consolidated README.md
    card = generate_consolidated_card(agg)
    if dry_run:
        if not card.startswith("---\n"):
            print(f"  WARNING: {repo_id} card missing YAML frontmatter")
            success = False
        print(f"  [DRY RUN] Would upload README.md ({len(card)} chars)")
    else:
        try:
            api.upload_file(  # type: ignore[union-attr]
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add consolidated BWSK model card",
            )
            print("  Uploaded README.md")
            time.sleep(delay)
        except Exception as e:
            print(f"  ERROR uploading README.md: {e}")
            success = False

    return success


def get_old_repo_ids(slug: str) -> list[str]:
    """Get all old repo IDs for a model slug.

    Returns the 6 model repos + 1 results repo that will be replaced
    by the consolidated repo.

    Args:
        slug: Model slug.

    Returns:
        List of old repo ID strings.
    """
    repos = []
    for exp in ALL_EXPERIMENTS:
        for mode in ALL_MODES:
            repos.append(f"{HF_NAMESPACE}/bwsk-{slug}-{exp}-{mode}")
    repos.append(f"{HF_NAMESPACE}/bwsk-{slug}-full-training-results")
    return repos


def delete_old_repos(
    api: object | None,
    slug: str,
    dry_run: bool,
    delay: float,
) -> bool:
    """Delete the old per-variant repos for a model.

    Args:
        api: HfApi instance (None for dry_run).
        slug: Model slug.
        dry_run: If True, skip actual deletion.
        delay: Seconds between API calls.

    Returns:
        True if all deletions succeeded.
    """
    repos = get_old_repo_ids(slug)
    success = True

    for repo_id in repos:
        if dry_run:
            print(f"  [DRY RUN] Would delete: {repo_id}")
        else:
            try:
                api.delete_repo(repo_id)  # type: ignore[union-attr]
                print(f"  Deleted: {repo_id}")
                time.sleep(delay)
            except Exception as e:
                # 404 is fine — repo may already be gone
                if "404" in str(e):
                    print(f"  Already gone: {repo_id}")
                else:
                    print(f"  ERROR deleting {repo_id}: {e}")
                    success = False

    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Consolidate HuggingFace repos from 112 to 16."""
    parser = argparse.ArgumentParser(
        description=("Consolidate 112 HuggingFace repos into 16 (one per model)")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without touching HF",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Process only one model slug (e.g. pythia-70m)",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Consolidate without deleting old repos",
    )
    parser.add_argument(
        "--delete-only",
        action="store_true",
        help="Only delete old repos (after verifying consolidation)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between API calls (default: 2.0)",
    )
    args = parser.parse_args()

    # Load data
    all_results = load_aggregated_results()
    if not all_results:
        print("ERROR: No fulltrain_*_results.json files found")
        sys.exit(1)

    slugs = sorted(all_results.keys())
    if args.slug:
        if args.slug not in all_results:
            print(f"ERROR: Unknown slug '{args.slug}'")
            print(f"Available: {', '.join(slugs)}")
            sys.exit(1)
        slugs = [args.slug]

    print(f"Found {len(all_results)} models, processing {len(slugs)}")

    # Initialize HF API if not dry-run
    api = None
    if not args.dry_run:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
        except ImportError:
            print("ERROR: huggingface_hub not installed. Use --dry-run or install it.")
            sys.exit(1)

    consolidated = 0
    deleted = 0
    errors = 0

    if not args.delete_only:
        print("\n=== Consolidating repos ===\n")
        for slug in slugs:
            agg = all_results[slug]
            print(f"\n--- {agg['model_name']} ({slug}) ---")
            ok = consolidate_model(api, slug, agg, args.dry_run, args.delay)
            if ok:
                consolidated += 1
            else:
                errors += 1

    if not args.no_delete and not args.dry_run:
        # Confirm before deletion
        if not args.delete_only:
            print(f"\n=== Consolidation complete: {consolidated} models ===")
        total_to_delete = sum(len(get_old_repo_ids(s)) for s in slugs)
        print(f"\nAbout to delete {total_to_delete} old repos for {len(slugs)} models.")
        confirm = input("Type 'yes' to confirm deletion: ")
        if confirm.strip().lower() != "yes":
            print("Deletion cancelled.")
        else:
            print("\n=== Deleting old repos ===\n")
            for slug in slugs:
                print(f"\n--- Deleting old repos for {slug} ---")
                ok = delete_old_repos(api, slug, False, args.delay)
                if ok:
                    deleted += 1
                else:
                    errors += 1
    elif not args.no_delete and args.dry_run:
        print("\n=== Old repos that would be deleted ===\n")
        for slug in slugs:
            delete_old_repos(None, slug, True, 0)
            deleted += 1

    # Summary
    action = "validated" if args.dry_run else "completed"
    print(f"\nDone ({action}):")
    if not args.delete_only:
        print(f"  Consolidated: {consolidated} models")
    if not args.no_delete:
        print(f"  Deleted: {deleted} models' old repos")
    if errors:
        print(f"  Errors: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
