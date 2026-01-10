#!/usr/bin/env python3
"""
Main knowledge base analysis script
Scans repository and creates structured knowledge base
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from categorize_content import Categorizer
from pipeline_detector import PipelineDetector
from unified_systems import UnifiedSystemDetector
from ui_ux_extractor import UIUXExtractor
from knowledge_base_builder import KnowledgeBaseBuilder
from overlap_analyzer import OverlapAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze repository and create knowledge base"
    )
    
    # Use environment variables for container-friendly defaults
    default_repo = os.getenv("KB_REPO_DIR", "/app/repo")
    default_output = os.getenv("KB_OUTPUT_DIR", "/app/output")
    default_mode = os.getenv("KB_ANALYSIS_MODE", "full")
    default_format = os.getenv("KB_OUTPUT_FORMAT", "both")
    default_pipeline = os.getenv("KB_PIPELINE_FILTER", "all")
    default_ui_ux = os.getenv("KB_INCLUDE_UI_UX", "true").lower() == "true"
    
    parser.add_argument("--repo-dir", type=str, default=default_repo,
                        help="Repository directory to analyze")
    parser.add_argument("--output-dir", type=str, default=default_output,
                        help="Output directory for knowledge base")
    parser.add_argument("--mode", type=str, default=default_mode,
                        choices=["full", "incremental", "specific-topic", "specific-pipeline"],
                        help="Analysis mode")
    parser.add_argument("--format", type=str, default=default_format,
                        choices=["json", "markdown", "both"],
                        help="Output format")
    parser.add_argument("--pipeline", type=str, default=default_pipeline,
                        choices=["plugins", "daw", "standalone", "all"],
                        help="Filter by pipeline type")
    parser.add_argument("--include-ui-ux", action="store_true", default=default_ui_ux,
                        help="Include UI/UX analysis")
    parser.add_argument("--topic", type=str, default=None,
                        help="Specific topic to analyze (for specific-topic mode)")
    
    args = parser.parse_args()
    
    repo_dir = Path(args.repo_dir)
    output_dir = Path(args.output_dir)
    
    if not repo_dir.exists():
        logger.error(f"Repository directory not found: {repo_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DAiW Knowledge Base Analyzer")
    logger.info("=" * 60)
    logger.info(f"Repository: {repo_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Pipeline filter: {args.pipeline}")
    logger.info(f"Include UI/UX: {args.include_ui_ux}")
    
    # Initialize analyzers
    logger.info("\nInitializing analyzers...")
    categorizer = Categorizer()
    pipeline_detector = PipelineDetector()
    system_detector = UnifiedSystemDetector()
    kb_builder = KnowledgeBaseBuilder(output_dir)
    
    ui_ux_extractor = None
    if args.include_ui_ux:
        ui_ux_extractor = UIUXExtractor()
    
    overlap_analyzer = OverlapAnalyzer()
    
    # Scan repository
    logger.info("\nScanning repository...")
    all_files = list(repo_dir.rglob("*"))
    # Filter out common ignore patterns
    ignore_patterns = {".git", "__pycache__", ".DS_Store", "node_modules", 
                      "build", "dist", ".venv", "venv", ".pytest_cache"}
    files = [f for f in all_files if f.is_file() and 
             not any(part in ignore_patterns for part in f.parts)]
    
    logger.info(f"Found {len(files)} files to analyze")
    
    # Categorize content
    logger.info("\nCategorizing content...")
    categorized = categorizer.categorize_files(files, repo_dir)
    
    # Detect pipelines
    logger.info("\nDetecting pipeline types...")
    pipeline_components = pipeline_detector.detect_pipelines(files, repo_dir)
    
    # Detect unified systems
    logger.info("\nDetecting unified systems...")
    unified_systems = system_detector.detect_systems(files, repo_dir, categorized)
    
    # Extract UI/UX if requested
    ui_ux_data = {}
    if args.include_ui_ux and ui_ux_extractor:
        logger.info("\nExtracting UI/UX components...")
        ui_ux_data = ui_ux_extractor.extract_ui_ux(files, repo_dir, pipeline_components)
    
    # Analyze overlap
    logger.info("\nAnalyzing shared components...")
    shared_components = overlap_analyzer.analyze_overlap(
        pipeline_components, unified_systems, ui_ux_data
    )
    
    # Build knowledge base
    logger.info("\nBuilding knowledge base structure...")
    kb_builder.build(
        categorized=categorized,
        pipeline_components=pipeline_components,
        unified_systems=unified_systems,
        ui_ux_data=ui_ux_data,
        shared_components=shared_components,
        pipeline_filter=args.pipeline,
        output_format=args.format
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Knowledge base analysis complete!")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
