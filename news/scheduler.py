"""
ARGUS V5 — Scheduled News Scraper Daemon
Coordinates 4x daily market news scraping and FinBERT scoring using APScheduler
with cron triggers tuned to Indian market hours (Asia/Kolkata timezone).
"""

import logging
from typing import Optional, Dict, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from news.rss_scraper import RSSNewsScraper

logger = logging.getLogger("argus.news_scheduler")


class NewsScheduler:
    """
    Automated BackgroundScheduler running 4x daily at:
    - 08:00 IST (Pre-Market Catalyst Discovery)
    - 10:00 IST (Morning Session Reaction)
    - 12:00 IST (Midday Institutional Positioning)
    - 14:00 IST (Power Hour Inflow/Outflow Sentiment)
    """

    def __init__(self, scraper: Optional[RSSNewsScraper] = None, timezone: str = "Asia/Kolkata"):
        self.scraper = scraper or RSSNewsScraper()
        self.timezone = timezone
        self.scheduler = BackgroundScheduler(timezone=self.timezone)
        self._is_running = False
        self._last_run_result = None

    def _job_wrapper(self):
        try:
            res = self.scraper.scrape_and_process()
            self._last_run_result = res
        except Exception as e:
            pass

    def start(self):
        """Register the 4x daily cron triggers and start background scheduler."""
        if self._is_running:
            return

        # 08:00, 10:00, 12:00, 14:00 IST
        cron_hours = "8,10,12,14"
        trigger = CronTrigger(hour=cron_hours, minute="0", timezone=self.timezone)

        self.scheduler.add_job(
            self._job_wrapper,
            trigger=trigger,
            id="argus_news_scraper_job",
            replace_existing=True,
            misfire_grace_time=300,
        )

        self.scheduler.start()
        self._is_running = True

    @property
    def running(self) -> bool:
        return self._is_running

    def get_jobs(self):
        return self.scheduler.get_jobs()

    def shutdown(self, wait: bool = False):
        self.stop()

    def stop(self):
        """Stop the background scheduler."""
        if self._is_running:
            self.scheduler.shutdown(wait=False)
            self._is_running = False

    def run_now(self) -> Dict[str, Any]:
        """Manually trigger an immediate news scrape and FinBERT scoring cycle."""
        res = self.scraper.scrape_and_process()
        self._last_run_result = res
        return res

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self._is_running,
            "timezone": self.timezone,
            "scheduled_hours": [8, 10, 12, 14],
            "last_run_result": self._last_run_result,
        }


def start_news_scheduler(scraper: Optional[RSSNewsScraper] = None, feature_store=None) -> NewsScheduler:
    """Factory helper to start the 4x daily news scheduler."""
    if scraper is None and feature_store is not None:
        scraper = RSSNewsScraper(feature_store=feature_store)
    scheduler = NewsScheduler(scraper=scraper)
    scheduler.start()
    return scheduler
