from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import Settings
from app.pipeline import ResearchPipeline


def build_scheduler(settings: Settings, pipeline: ResearchPipeline):
    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
    scheduler.add_job(
        lambda: pipeline.run(mode=settings.data_mode, notify=settings.push_on_schedule),
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=settings.schedule_hour,
            minute=settings.schedule_minute,
            timezone="Asia/Shanghai",
        ),
        id="daily_research",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    if settings.factor_auto_update_enabled:
        def update_factor_registry():
            from scripts.update_factor_registry import run_auto_update

            return run_auto_update()

        scheduler.add_job(
            update_factor_registry,
            trigger=CronTrigger(
                day=settings.factor_update_day,
                hour=settings.factor_update_hour,
                minute=settings.factor_update_minute,
                timezone="Asia/Shanghai",
            ),
            id="monthly_factor_update",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=86400,
        )
    return scheduler
