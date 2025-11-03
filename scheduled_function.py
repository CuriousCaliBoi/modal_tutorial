import modal
from datetime import datetime

app = modal.App("example-scheduled")


# Run every hour
@app.function(schedule=modal.Cron("0 * * * *"))
def hourly_job():
    print(f"Hourly job running at {datetime.now()}")


# Run every day at midnight
@app.function(schedule=modal.Cron("0 0 * * *"))
def daily_job():
    print(f"Daily job running at {datetime.now()}")


# Run every 5 minutes
@app.function(schedule=modal.Period(minutes=5))
def frequent_job():
    print(f"Frequent job running at {datetime.now()}")


# To deploy scheduled functions, use:
# modal deploy scheduled_function.py


