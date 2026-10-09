"""Decision logic of the account cost guard Lambda (infra/cost_guard). No AWS calls."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra" / "cost_guard" / "src"))

import cost_guard as cg  # noqa: E402

NOW = dt.datetime(2026, 10, 4, 12, 0, tzinfo=dt.timezone.utc)
CFG = cg.Config(dry_run=False)


def samples(hours, value, step_min=5):
    """5-minute samples covering the last ``hours`` hours, all equal to ``value``."""
    n = int(hours * 60 / step_min)
    return [(NOW - dt.timedelta(minutes=step_min * i + 1), value) for i in range(n)]


def facts(hours_up, cpu=None, gpu=None, tags=None, **kw):
    return cg.InstanceFacts(
        instance_id="i-1", name="box", instance_type="g5.2xlarge",
        launch_time=NOW - dt.timedelta(hours=hours_up), tags=tags or {},
        cpu=samples(4, 0.1) if cpu is None else cpu, gpu=gpu or [], **kw)


def test_idle_four_hours_low_cpu_no_gpu_metrics_stops():
    d = cg.decide_instance(facts(5), NOW, CFG)
    assert d.stop_reason and d.stop_reason.startswith("idle")


def test_not_idle_before_window_elapsed():
    assert cg.decide_instance(facts(3), NOW, CFG).stop_reason is None


def test_one_busy_hour_keeps_instance():
    cpu = samples(3, 0.1) + [(NOW - dt.timedelta(hours=3, minutes=m), 15.0) for m in range(1, 60, 5)]
    assert cg.decide_instance(facts(5, cpu=cpu), NOW, CFG).stop_reason is None


def test_gpu_activity_keeps_instance_even_with_low_cpu():
    d = cg.decide_instance(facts(5, gpu=samples(1, 40.0)), NOW, CFG)
    assert d.stop_reason is None and "GPU busy" in d.info[0]


def test_gpu_idle_metrics_still_allow_stop():
    assert cg.decide_instance(facts(5, gpu=samples(4, 0.0)), NOW, CFG).stop_reason


def test_metrics_gap_never_reads_as_idle():
    assert cg.decide_instance(facts(5, cpu=samples(2, 0.1)), NOW, CFG).stop_reason is None


def test_runtime_notice_once_per_start():
    busy = samples(4, 30.0)
    d = cg.decide_instance(facts(37, cpu=busy), NOW, CFG)
    assert [n[0] for n in d.notices] == [cg.RUNTIME_NOTICE_TAG]
    start_id = str(int((NOW - dt.timedelta(hours=37)).timestamp()))
    again = cg.decide_instance(facts(37, cpu=busy, tags={cg.RUNTIME_NOTICE_TAG: start_id}), NOW, CFG)
    assert again.notices == []


def test_warning_before_hard_stop_then_stop():
    busy = samples(4, 30.0)
    warn = cg.decide_instance(facts(67, cpu=busy), NOW, CFG)
    assert cg.STOP_WARNING_TAG in [n[0] for n in warn.notices] and warn.stop_reason is None
    stop = cg.decide_instance(facts(72.5, cpu=busy), NOW, CFG)
    assert stop.stop_reason and "72" in stop.stop_reason


def test_exemption_tag_blocks_stops_but_notifies():
    tags = {cg.EXEMPT_TAG: (NOW + dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ")}
    d = cg.decide_instance(facts(80, tags=tags), NOW, CFG)
    assert d.stop_reason is None
    assert any("NOT stopped" in n[1] for n in d.notices)


def test_expired_or_bad_exemption_is_ignored():
    past = {cg.EXEMPT_TAG: "2026-10-01"}
    assert cg.decide_instance(facts(80, tags=past), NOW, CFG).stop_reason
    assert cg.exempt_until({cg.EXEMPT_TAG: "soon"}, NOW) is None


def test_service_managed_instances_are_never_stopped():
    asg = {"aws:autoscaling:groupName": "g"}
    assert cg.decide_instance(facts(80, tags=asg), NOW, CFG).stop_reason is None
    assert cg.decide_instance(facts(80, lifecycle="spot"), NOW, CFG).stop_reason is None


def test_gpu_type_detection():
    assert all(map(cg.is_gpu_instance, ["ml.g5.2xlarge", "ml.p4d.24xlarge", "g6e.xlarge", "p5.48xlarge"]))
    assert not any(map(cg.is_gpu_instance, ["ml.m5.xlarge", "ml.c5.2xlarge", "ml.trn1.2xlarge", ""]))


def test_jobs_over_cap_keeps_oldest():
    jobs = [cg.Job("batch", f"j{i}", NOW - dt.timedelta(minutes=i)) for i in range(5)]
    assert [j.name for j in cg.jobs_over_cap(jobs, 3)] == ["j1", "j0"]
    assert cg.jobs_over_cap(jobs, 128) == []


def test_gpu_alert_dedupe():
    assert not cg.gpu_alert_due(16, {}, NOW, CFG)
    assert cg.gpu_alert_due(17, {}, NOW, CFG)
    recent = {"gpu_alert_at": (NOW - dt.timedelta(hours=1)).isoformat(), "gpu_alert_count": 17}
    assert not cg.gpu_alert_due(18, recent, NOW, CFG)
    assert cg.gpu_alert_due(30, recent, NOW, CFG)  # grew a lot -> re-alert
    stale = {"gpu_alert_at": (NOW - dt.timedelta(hours=7)).isoformat(), "gpu_alert_count": 17}
    assert cg.gpu_alert_due(17, stale, NOW, CFG)


def test_parse_stop_time_and_config_env(monkeypatch):
    assert cg.parse_stop_time("User initiated (2026-09-01 12:00:00 GMT)") == dt.datetime(
        2026, 9, 1, 12, tzinfo=dt.timezone.utc)
    assert cg.parse_stop_time("") is None
    monkeypatch.setenv("IDLE_CPU_PCT", "2.5")
    monkeypatch.setenv("DRY_RUN", "false")
    cfg = cg.Config.from_env({"max_active_jobs": "10"})
    assert (cfg.idle_cpu_pct, cfg.dry_run, cfg.max_active_jobs) == (2.5, False, 10)
