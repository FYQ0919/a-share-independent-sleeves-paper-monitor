import argparse
import json

from app.config import settings
from app.pipeline import ResearchPipeline
from app.storage import Storage


def main():
    parser = argparse.ArgumentParser(description="A股量化研究工作台")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行一次选股和研报任务")
    run_parser.add_argument("--mode", choices=["demo", "live"], default=settings.data_mode)
    run_parser.add_argument("--notify", action="store_true", help="推送到已配置渠道")
    args = parser.parse_args()

    pipeline = ResearchPipeline(settings, Storage(settings.database_path))
    if args.command == "run":
        result = pipeline.run(mode=args.mode, notify=args.notify)
        print(json.dumps({
            "run_id": result.run_id,
            "mode": result.mode,
            "candidates": len(result.candidates),
            "report": str(settings.report_dir),
            "notifications": result.notifications,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

