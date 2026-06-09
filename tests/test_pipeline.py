"""
syslog-DFP pipeline test suite.

Tier 1 — CPU-only unit tests (no GPU, no Morpheus). Run anywhere with:
    pytest tests/ -m "not gpu and not morpheus"

Tier 2 — GPU operation tests (cuDF/CuPy only, no Morpheus stages). Run on GPU host:
    pytest tests/ -m "gpu and not morpheus"

Tier 3 — Full Morpheus stage tests (requires the container). Run inside Docker:
    pytest tests/ -m morpheus

To run everything the current environment supports (auto-skips unavailable tiers):
    pytest tests/
"""

import json
import math
import os
import re
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# ── GPU availability ──────────────────────────────────────────────────────────
try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="RAPIDS cuDF/CuPy not available")

# ── Morpheus stack availability ───────────────────────────────────────────────
# morpheus_app.py has module-level side effects (RMM init, FEEDBACK_DIR creation)
# so we import it once here inside a try/except.
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

MORPHEUS_AVAILABLE = False
app = None
try:
    import morpheus_app as app  # type: ignore
    MORPHEUS_AVAILABLE = True
except Exception:
    pass

requires_morpheus = pytest.mark.skipif(
    not MORPHEUS_AVAILABLE, reason="Morpheus container stack not available"
)

# ── Constants — mirrored exactly from morpheus_app.py ─────────────────────────
NUM_LOG_FEATURES = 8
FEATURE_COLS = [
    "feature_log_count", "feature_priority", "feature_event_type",
    "feature_is_security_event",
    "feature_hour_sin", "feature_hour_cos", "feature_day_sin", "feature_day_cos",
    "feature_user_hash", "feature_msg_len", "feature_proc_hash",
]
for _i in range(NUM_LOG_FEATURES):
    FEATURE_COLS.append(f"feature_log_v{_i}")
NUM_FEATURES = len(FEATURE_COLS)

# ── Regexes — mirrored from GPUPreprocessingStage.__init__ ────────────────────
IP_REGEX        = r"from\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
HOST_REGEX      = r"(?:<\d+>)?([A-Z][a-z]{2}\s+\d+\s\d+:\d+:\d+)\s+([a-zA-Z0-9\.\-_]+)"
USER_REGEX      = r"(?:user|for user|User|User=)\s*([a-zA-Z0-9\.\-_]+)"
PROCESS_REGEX   = r"(?:<\d+>)?(?:[A-Z][a-z]{2}\s+\d+\s\d+:\d+:\d+)\s+[a-zA-Z0-9\.\-_]+\s+([a-zA-Z0-9\.\-_]+)(?:\[\d+\])?: "
RFC5424_TS_REGEX = r"(?:<\d+>)\d+\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"

# ── CPU helpers — exact replicas of key pipeline functions ───────────────────

def parse_timestamps(messages: pd.Series, now: datetime) -> pd.Series:
    """Replica of timestamp parsing in GPUPreprocessingStage._process."""
    year_str = str(now.year) + ' '
    ts_raw   = messages.str.extract(HOST_REGEX, expand=True)[0]
    ts_norm  = ts_raw.str.replace(r'\s+', ' ', regex=True).str.strip()
    parsed_3164 = pd.to_datetime(year_str + ts_norm, format="%Y %b %d %H:%M:%S", errors='coerce')
    ts_5424     = messages.str.extract(RFC5424_TS_REGEX, expand=True)[0]
    parsed_5424 = pd.to_datetime(ts_5424, format="%Y-%m-%dT%H:%M:%S", errors='coerce')
    return parsed_3164.fillna(parsed_5424).fillna(pd.Timestamp(now))


def scale_score(raw: float) -> float:
    """Replica of the anomaly scoring formula in ManualInferenceStage._process."""
    if raw <= 1.5:
        return raw * 2.0
    return 3.0 + math.log1p(max(0.0, raw - 1.5)) * 2.0


# ── Sample syslog messages ─────────────────────────────────────────────────────
RFC3164 = "Jan  5 14:32:01 web-server-01 sshd[1234]: Failed password for user admin from 192.168.1.100 port 22 ssh2"
RFC5424 = "<13>1 2026-01-05T14:32:01 web-server-01 sshd 1234 - - Failed password for user admin"
CRON    = "Jun  9 03:00:01 db-server-02 CRON[5678]: (root) CMD (/usr/sbin/logrotate /etc/logrotate.conf)"
BROKEN  = "this is not a valid syslog message at all"


# =============================================================================
# TIER 1: CPU-ONLY UNIT TESTS
# =============================================================================

class TestTimestampParsing:
    NOW = datetime(2026, 6, 9, 12, 0, 0)

    def test_rfc3164_month_day_time(self):
        result = parse_timestamps(pd.Series([RFC3164]), self.NOW)
        ts = result.iloc[0]
        assert ts.month == 1 and ts.day == 5
        assert ts.hour == 14 and ts.minute == 32 and ts.second == 1

    def test_rfc3164_year_from_now(self):
        # RFC 3164 has no year; it should be inferred from `now`
        result = parse_timestamps(pd.Series([RFC3164]), self.NOW)
        assert result.iloc[0].year == self.NOW.year

    def test_rfc5424_full_date(self):
        result = parse_timestamps(pd.Series([RFC5424]), self.NOW)
        ts = result.iloc[0]
        assert ts.year == 2026 and ts.month == 1 and ts.day == 5
        assert ts.hour == 14 and ts.minute == 32

    def test_malformed_falls_back_to_now(self):
        result = parse_timestamps(pd.Series([BROKEN]), self.NOW)
        ts = result.iloc[0]
        assert ts.year == self.NOW.year
        assert ts.month == self.NOW.month
        assert ts.day == self.NOW.day

    def test_mixed_batch_no_nats(self):
        msgs = pd.Series([RFC3164, RFC5424, BROKEN, CRON])
        result = parse_timestamps(msgs, self.NOW)
        assert len(result) == 4
        assert result.notna().all(), "Fallback must fill every row; no NaT allowed"

    def test_output_length_preserved(self):
        msgs = pd.Series([RFC3164] * 500)
        assert len(parse_timestamps(msgs, self.NOW)) == 500

    def test_rfc5424_not_confused_with_3164(self):
        # RFC 5424 timestamp is ISO format; 3164 parser should produce NaT for it,
        # then 5424 parser picks it up.
        result = parse_timestamps(pd.Series([RFC5424]), self.NOW)
        assert result.iloc[0].year == 2026  # 5424 year, not inferred from now


class TestRegexExtraction:
    def test_host_regex_timestamp_group(self):
        m = re.search(HOST_REGEX, RFC3164)
        assert m is not None and "14:32:01" in m.group(1)

    def test_host_regex_hostname_group(self):
        m = re.search(HOST_REGEX, RFC3164)
        assert m is not None and m.group(2) == "web-server-01"

    def test_host_regex_no_match_on_malformed(self):
        assert re.search(HOST_REGEX, BROKEN) is None

    def test_ip_regex_extracts_address(self):
        m = re.search(IP_REGEX, RFC3164)
        assert m is not None and m.group(1) == "192.168.1.100"

    def test_ip_regex_absent(self):
        assert re.search(IP_REGEX, CRON) is None

    def test_user_regex_keyword_for(self):
        m = re.search(USER_REGEX, RFC3164)
        assert m is not None and m.group(1) == "admin"

    def test_process_regex_sshd(self):
        m = re.search(PROCESS_REGEX, RFC3164)
        assert m is not None and m.group(1) == "sshd"

    def test_process_regex_cron(self):
        m = re.search(PROCESS_REGEX, CRON)
        assert m is not None and m.group(1) == "CRON"


class TestScoringFormula:
    def test_zero_maps_to_zero(self):
        assert scale_score(0.0) == pytest.approx(0.0)

    def test_linear_boundary_1_5(self):
        assert scale_score(1.5) == pytest.approx(3.0)

    def test_linear_midpoint(self):
        assert scale_score(1.0) == pytest.approx(2.0)

    def test_log_region_above_boundary(self):
        # Just above 1.5 the log region begins; score > 3.0
        assert scale_score(1.501) > 3.0

    def test_log_region_does_not_explode(self):
        # Very high raw loss should stay well below 10 before clipping
        assert scale_score(20.0) < 10.0
        assert scale_score(100.0) < 15.0  # clipping happens outside this fn

    def test_monotonically_increasing(self):
        raws = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0]
        scores = [scale_score(r) for r in raws]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"Not monotone at index {i}"

    def test_learning_state_forces_zero(self):
        # Emulate `cp.where(is_learning, 0.0, scaled)` for a high raw loss
        raw = 9.0
        is_learning = True
        score = 0.0 if is_learning else min(scale_score(raw), 10.0)
        assert score == 0.0

    def test_score_clipped_to_10(self):
        # The production code clips to [0, 10]; confirm values above 10 exist before clip
        assert scale_score(200.0) > 10.0  # unclipped — proves clip is necessary


class TestEPSCalculation:
    def test_30_second_window(self):
        ts = pd.to_datetime(["2026-06-09 10:00:00", "2026-06-09 10:00:30"])
        diff = pd.Timedelta(ts.max() - ts.min())
        assert diff.total_seconds() == pytest.approx(30.0)

    def test_same_second_floors_to_one(self):
        ts = pd.to_datetime(["2026-06-09 10:00:00", "2026-06-09 10:00:00"])
        diff = pd.Timedelta(ts.max() - ts.min())
        duration_sec = max(diff.total_seconds(), 1.0)
        assert duration_sec == pytest.approx(1.0)

    def test_feature_log_count_realistic_range(self):
        # 1000 events over 60 s → EPS=1000/60*60=1000 → log1p(1000)/10 ≈ 0.69
        eps = 1000.0
        feature = math.log1p(eps) / 10.0
        assert 0.0 < feature < 2.0

    def test_pd_timedelta_wraps_numpy_timedelta64(self):
        # cuDF returns numpy.timedelta64 from max()-min(); pd.Timedelta must accept it
        import numpy as np
        delta = np.timedelta64(90, 's')
        wrapped = pd.Timedelta(delta)
        assert wrapped.total_seconds() == pytest.approx(90.0)


class TestRollingWindow:
    def test_accumulation_appends_rows(self):
        cache = {"h": pd.DataFrame({"f": [1.0, 2.0]})}
        new_rows = pd.DataFrame({"f": [3.0]})
        cache["h"] = pd.concat([cache["h"], new_rows], ignore_index=True).tail(50000)
        assert len(cache["h"]) == 3

    def test_cap_enforced(self):
        max_rows = 100
        cache = {"h": pd.DataFrame({"f": range(90)})}
        new_batch = pd.DataFrame({"f": range(20)})
        cache["h"] = pd.concat([cache["h"], new_batch], ignore_index=True).tail(max_rows)
        assert len(cache["h"]) == max_rows

    def test_index_clean_after_concat(self):
        # Simulate joining batches with non-zero-starting indices
        cache = {"h": pd.DataFrame({"f": range(50)}, index=range(50, 100))}
        new = pd.DataFrame({"f": range(10)}, index=range(200, 210))
        cache["h"] = pd.concat([cache["h"], new], ignore_index=True).tail(50000)
        assert list(cache["h"].index) == list(range(len(cache["h"])))

    def test_eligibility_threshold_at_100(self):
        cache = {
            "host-a": pd.DataFrame({"f": range(99)}),
            "host-b": pd.DataFrame({"f": range(100)}),
            "host-c": pd.DataFrame({"f": range(200)}),
        }
        eligible = [uid for uid, df in cache.items() if len(df) >= 100]
        assert "host-a" not in eligible
        assert "host-b" in eligible and "host-c" in eligible

    def test_snapshot_is_a_copy(self):
        cache = {"h": pd.DataFrame({"f": [1.0, 2.0]})}
        snapshot = cache["h"].copy()
        cache["h"].loc[0, "f"] = 99.0
        assert snapshot.loc[0, "f"] == pytest.approx(1.0)


class TestFeedbackProcessing:
    def test_missing_features_default_to_zero(self):
        partial = {"feature_priority": 0.5, "feature_msg_len": 0.3}
        filled = {k: float(partial.get(k, 0.0)) for k in FEATURE_COLS}
        assert len(filled) == NUM_FEATURES
        assert filled["feature_log_count"] == pytest.approx(0.0)
        assert filled["feature_priority"] == pytest.approx(0.5)
        assert filled["feature_log_v7"] == pytest.approx(0.0)

    def test_all_features_present_preserved(self):
        full = {k: 0.42 for k in FEATURE_COLS}
        filled = {k: float(full.get(k, 0.0)) for k in FEATURE_COLS}
        assert all(v == pytest.approx(0.42) for v in filled.values())

    def test_extra_keys_stripped(self):
        features = {k: 1.0 for k in FEATURE_COLS}
        features["not_a_real_feature"] = 99.0
        filled = {k: float(features.get(k, 0.0)) for k in FEATURE_COLS}
        assert "not_a_real_feature" not in filled

    def test_feedback_cap_at_100(self, tmp_path):
        feedback_file = tmp_path / "host1.json"
        existing = [{"feature_priority": float(i)} for i in range(100)]
        feedback_file.write_text(json.dumps(existing))
        loaded = json.loads(feedback_file.read_text())
        loaded.append({k: 0.0 for k in FEATURE_COLS})
        capped = loaded[-100:]
        assert len(capped) == 100
        # Oldest entry (i=0) was displaced
        assert capped[0].get("feature_priority") == pytest.approx(1.0)

    def test_corrupted_feedback_file_resets(self, tmp_path):
        feedback_file = tmp_path / "host2.json"
        feedback_file.write_text("not valid json {{{")
        try:
            result = json.loads(feedback_file.read_text())
        except (json.JSONDecodeError, ValueError):
            result = []
        assert result == []

    def test_feedback_df_filled_for_missing_cols(self):
        rows = [{"feature_priority": 0.5}]
        df = pd.DataFrame(rows)
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df = df[FEATURE_COLS]
        assert list(df.columns) == FEATURE_COLS
        assert df["feature_log_count"].iloc[0] == pytest.approx(0.0)


# =============================================================================
# TIER 2: GPU OPERATION TESTS  (cuDF/CuPy; no Morpheus stages)
# =============================================================================

@pytest.mark.gpu
@requires_gpu
class TestGPUHashFeatures:
    """Verify the CuPy hash operations produce correctly-shaped, in-range outputs."""

    def test_user_hash_shape(self):
        users = cudf.Series(["root", "admin", "nobody", "root", "admin"])
        result = (cp.abs(users.hash_values().values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
        assert result.shape == (5,)

    def test_user_hash_in_range(self):
        users = cudf.Series(["root", "admin", "nobody", "daemon"] * 500)
        result = (cp.abs(users.hash_values().values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
        assert float(cp.min(result)) >= 0.0
        assert float(cp.max(result)) < 1.0

    def test_proc_hash_in_range(self):
        procs = cudf.Series(["sshd", "cron", "nginx", "systemd"] * 500)
        result = (cp.abs(procs.hash_values().values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
        assert float(cp.min(result)) >= 0.0
        assert float(cp.max(result)) < 1.0

    def test_log_hash_seeds_produce_different_distributions(self):
        msgs = cudf.Series(["sshd: Failed password for root from 10.0.0.1"] * 300)
        means = []
        for i in range(NUM_LOG_FEATURES):
            r = (cp.abs(msgs.hash_values(seed=i).values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
            means.append(round(float(cp.mean(r)), 4))
        # At least half the seeds should produce distinct means
        assert len(set(means)) > NUM_LOG_FEATURES // 2

    def test_hash_shape_large_batch(self):
        n = 16384
        series = cudf.Series([f"user_{i}" for i in range(n)])
        result = (cp.abs(series.hash_values().values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
        assert result.shape[0] == n

    def test_abs_eliminates_negatives(self):
        # hash_values() can return negative int32; abs + modulo must always yield >= 0
        series = cudf.Series([str(i) for i in range(1000)])
        raw_hashes = series.hash_values().values  # CuPy int32 array
        result = (cp.abs(raw_hashes) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
        assert float(cp.min(result)) >= 0.0


@pytest.mark.gpu
@requires_gpu
class TestGPUTemporalFeatures:
    """sin/cos features must stay in [-1, 1] for any valid timestamp."""

    def _timestamps(self, n=500):
        ts_np = np.array(
            [f"2026-06-09T{h:02d}:{m:02d}:00" for h, m in ((i % 24, (i * 3) % 60) for i in range(n))],
            dtype="datetime64[ns]",
        )
        return cudf.Series(ts_np)

    def test_hour_sin_range(self):
        hours_cp = (self._timestamps().dt.hour + self._timestamps().dt.minute / 60.0).values
        vals = cp.sin(2 * cp.float32(cp.pi) * hours_cp / cp.float32(24.0)).astype(cp.float32)
        assert float(cp.min(vals)) >= -1.0 - 1e-5
        assert float(cp.max(vals)) <= 1.0 + 1e-5

    def test_hour_cos_range(self):
        hours_cp = (self._timestamps().dt.hour + self._timestamps().dt.minute / 60.0).values
        vals = cp.cos(2 * cp.float32(cp.pi) * hours_cp / cp.float32(24.0)).astype(cp.float32)
        assert float(cp.min(vals)) >= -1.0 - 1e-5
        assert float(cp.max(vals)) <= 1.0 + 1e-5

    def test_day_sin_range(self):
        days_cp = self._timestamps().dt.weekday.astype("float32").values
        vals = cp.sin(2 * cp.float32(cp.pi) * days_cp / cp.float32(7.0)).astype(cp.float32)
        assert float(cp.min(vals)) >= -1.0 - 1e-5
        assert float(cp.max(vals)) <= 1.0 + 1e-5

    def test_shape_preserved(self):
        n = 1024
        ts = self._timestamps(n)
        hours_cp = (ts.dt.hour + ts.dt.minute / 60.0).values
        assert hours_cp.shape[0] == n

    def test_midnight_and_noon_are_antipodal(self):
        ts = cudf.Series(np.array(["2026-06-09T00:00:00", "2026-06-09T12:00:00"], dtype="datetime64[ns]"))
        hours_cp = (ts.dt.hour + ts.dt.minute / 60.0).values
        sin_vals = cp.sin(2 * cp.float32(cp.pi) * hours_cp / cp.float32(24.0)).astype(cp.float32)
        # sin(0) = 0.0, sin(pi) = 0.0 — both should be near zero
        assert abs(float(sin_vals[0])) < 1e-5
        assert abs(float(sin_vals[1])) < 1e-5


@pytest.mark.gpu
@requires_gpu
class TestAccumulatorIndexSafety:
    """Guard against the duplicate-index regression that caused column size mismatches."""

    def _gdf(self, n):
        """Make a cuDF DataFrame whose index starts at 0 — exactly like a Kafka message."""
        return cudf.DataFrame({"message": [RFC3164] * n}, index=range(n))

    def test_concat_without_ignore_creates_duplicates(self):
        bad = cudf.concat([self._gdf(10), self._gdf(10)])
        assert not bad.index.is_unique

    def test_concat_with_ignore_index_is_clean(self):
        good = cudf.concat([self._gdf(10), self._gdf(10), self._gdf(5)], ignore_index=True)
        assert good.index.is_unique
        assert list(good.index) == list(range(25))

    def test_hash_assignment_after_clean_concat(self):
        n = 200
        df = cudf.concat([self._gdf(n)], ignore_index=True)
        hash_vals = (cp.abs(df["message"].hash_values().values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
        df["feature_user_hash"] = hash_vals  # must not raise column-size mismatch
        assert len(df["feature_user_hash"]) == n

    def test_timestamp_assignment_after_clean_concat(self):
        n = 100
        df = cudf.concat([self._gdf(n)], ignore_index=True)
        ts_np = np.array(["2026-06-09T10:00:00"] * n, dtype="datetime64[ns]")
        df["timestamp"] = cudf.Series(ts_np)  # positional assignment must succeed
        assert len(df["timestamp"]) == n
        assert str(df["timestamp"].dtype) == "datetime64[ns]"

    def test_multiple_feature_assignments_no_mismatch(self):
        """Full sweep of all hash + temporal feature assignments on a clean-indexed batch."""
        n = 500
        df = cudf.concat([self._gdf(n)], ignore_index=True)
        msgs = df["message"]

        # Hash features
        users = msgs.str.extract(r"user\s+(\w+)", expand=True)[0].fillna("none")
        procs = msgs.str.extract(r"\s+(\w+)\[", expand=True)[0].fillna("unknown")
        df["feature_user_hash"] = (cp.abs(users.hash_values().values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
        df["feature_proc_hash"] = (cp.abs(procs.hash_values().values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
        for i in range(NUM_LOG_FEATURES):
            df[f"feature_log_v{i}"] = (cp.abs(msgs.hash_values(seed=i).values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)

        # Temporal features
        ts_np = np.array(["2026-06-09T10:00:00"] * n, dtype="datetime64[ns]")
        df["timestamp"] = cudf.Series(ts_np)
        hours_cp = (df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0).values
        days_cp  = df["timestamp"].dt.weekday.astype("float32").values
        df["feature_hour_sin"] = cp.sin(2 * cp.float32(cp.pi) * hours_cp / cp.float32(24.0)).astype(cp.float32)
        df["feature_hour_cos"] = cp.cos(2 * cp.float32(cp.pi) * hours_cp / cp.float32(24.0)).astype(cp.float32)
        df["feature_day_sin"]  = cp.sin(2 * cp.float32(cp.pi) * days_cp / cp.float32(7.0)).astype(cp.float32)
        df["feature_day_cos"]  = cp.cos(2 * cp.float32(cp.pi) * days_cp / cp.float32(7.0)).astype(cp.float32)

        # Every assigned column must have exactly n rows
        assigned = (
            ["feature_user_hash", "feature_proc_hash"]
            + [f"feature_log_v{i}" for i in range(NUM_LOG_FEATURES)]
            + ["feature_hour_sin", "feature_hour_cos", "feature_day_sin", "feature_day_cos"]
        )
        for col in assigned:
            assert len(df[col]) == n, f"{col} has wrong length"


# =============================================================================
# TIER 3: FULL MORPHEUS STAGE TESTS  (requires container)
# =============================================================================

def _make_msg(messages):
    """Build a minimal ControlMessage wrapping a cuDF DataFrame of syslog strings."""
    from morpheus.messages import ControlMessage, MessageMeta
    gdf = cudf.DataFrame({"message": messages})
    msg = ControlMessage()
    msg.payload(MessageMeta(gdf))
    return msg


@pytest.mark.morpheus
@requires_morpheus
class TestGPUPreprocessingStage:
    def setup_method(self):
        from morpheus.config import Config
        self.stage = app.GPUPreprocessingStage(Config())

    def test_all_feature_cols_present(self):
        out = self.stage._process(_make_msg([RFC3164, RFC5424, CRON] * 50))
        with out.payload().mutable_dataframe() as df:
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            assert not missing, f"Missing columns: {missing}"

    def test_row_count_unchanged(self):
        n = 150
        out = self.stage._process(_make_msg([RFC3164] * n))
        with out.payload().mutable_dataframe() as df:
            assert len(df) == n

    def test_hash_features_in_unit_range(self):
        out = self.stage._process(_make_msg([RFC3164, RFC5424, CRON] * 100))
        with out.payload().mutable_dataframe() as df:
            hash_cols = (
                ["feature_user_hash", "feature_proc_hash"]
                + [f"feature_log_v{i}" for i in range(NUM_LOG_FEATURES)]
            )
            for col in hash_cols:
                mn, mx = float(df[col].min()), float(df[col].max())
                assert mn >= 0.0,  f"{col} min={mn} is negative"
                assert mx < 1.0,   f"{col} max={mx} >= 1.0"

    def test_trig_features_in_unit_range(self):
        out = self.stage._process(_make_msg([RFC3164] * 200))
        with out.payload().mutable_dataframe() as df:
            for col in ["feature_hour_sin", "feature_hour_cos", "feature_day_sin", "feature_day_cos"]:
                mn, mx = float(df[col].min()), float(df[col].max())
                assert mn >= -1.0 - 1e-5, f"{col} min={mn}"
                assert mx <=  1.0 + 1e-5, f"{col} max={mx}"

    def test_source_column_populated(self):
        out = self.stage._process(_make_msg([RFC3164] * 50))
        with out.payload().mutable_dataframe() as df:
            assert "source" in df.columns
            # RFC3164 has a hostname; most rows should not be "unknown_source"
            assert not (df["source"] == "unknown_source").all()

    def test_timestamp_is_datetime64(self):
        out = self.stage._process(_make_msg([RFC3164] * 30))
        with out.payload().mutable_dataframe() as df:
            assert "timestamp" in df.columns
            assert "datetime64" in str(df["timestamp"].dtype)

    def test_feature_col_dtype_float32(self):
        out = self.stage._process(_make_msg([RFC3164] * 20))
        with out.payload().mutable_dataframe() as df:
            float_cols = [c for c in FEATURE_COLS if c != "feature_log_count"]
            for col in float_cols:
                assert str(df[col].dtype) == "float32", f"{col} dtype={df[col].dtype}"


@pytest.mark.morpheus
@requires_morpheus
class TestTritonFallback:
    """When Triton is unreachable every row must have triton_classified=False."""

    def setup_method(self):
        from morpheus.config import Config
        from morpheus.utils.column_info import ColumnInfo
        self.stage = app.StringTritonInferenceStage(
            Config(),
            model_name="cyber_log_pipeline",
            server_url="localhost:19999",   # nothing listening; guaranteed failure
            output_mapping=[ColumnInfo(name="label", dtype="int64")],
        )

    def test_all_rows_classified_false(self):
        out = self.stage._process(_make_msg([RFC3164] * 30))
        with out.payload().mutable_dataframe() as df:
            assert "triton_classified" in df.columns
            assert bool((df["triton_classified"] == False).all())

    def test_no_exception_propagated(self):
        result = self.stage._process(_make_msg([RFC3164] * 10))
        assert result is not None

    def test_row_count_preserved_on_failure(self):
        n = 50
        out = self.stage._process(_make_msg([RFC3164] * n))
        with out.payload().mutable_dataframe() as df:
            assert len(df) == n
