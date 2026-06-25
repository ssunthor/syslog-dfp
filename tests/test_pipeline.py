"""
syslog-DFP pipeline test suite.

Tier 1 — CPU-only unit tests (no GPU, no Morpheus). Run anywhere with:
    pytest tests/ -m "not gpu and not morpheus and not torch"

Tier 1.5 — PyTorch CPU tests (reason attribution). Run with torch installed:
    pytest tests/ -m torch

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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
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


# ── Reason attribution replica (CPU) ─────────────────────────────────────────

REASON_MAP = {
    0: "Volume Anomaly", 1: "Severity Spike", 2: "Unusual Event", 3: "Security Flag",
    4: "Time Anomaly (Hour)", 5: "Time Anomaly (Hour)", 6: "Time Anomaly (Day)", 7: "Time Anomaly (Day)",
    8: "User Anomaly", 9: "Message Length Anomaly", 10: "Process Anomaly",
}
for _ri in range(11, NUM_FEATURES):
    REASON_MAP[_ri] = "Log Content Anomaly"


def attribute_reason(abs_errors: torch.Tensor) -> np.ndarray:
    """Replica of ManualInferenceStage relative-error reason attribution."""
    feat_median = torch.median(abs_errors, dim=0).values.unsqueeze(0)
    feat_iqr = (
        torch.quantile(abs_errors, 0.75, dim=0)
        - torch.quantile(abs_errors, 0.25, dim=0)
    ).unsqueeze(0).clamp(min=0.01)
    relative_errors = (abs_errors - feat_median) / feat_iqr
    _, reason_indices = torch.max(relative_errors, dim=1)
    return reason_indices.numpy()


def _baseline_errors(batch_size: int = 200) -> torch.Tensor:
    """Generate realistic baseline reconstruction errors: time features have
    higher raw error (the exact problem the relative attribution fixes)."""
    errors = torch.rand(batch_size, NUM_FEATURES) * 0.3
    # Simulate the autoencoder's inherently higher error on cyclical features
    errors[:, 4] += 0.8  # hour_sin
    errors[:, 5] += 0.7  # hour_cos
    errors[:, 6] += 0.5  # day_sin
    errors[:, 7] += 0.5  # day_cos
    return errors


@pytest.mark.torch
@requires_torch
class TestReasonAttribution:
    """Verify that relative-error reason attribution correctly identifies every
    feature as the anomaly cause when that feature has an outlier error."""

    @pytest.mark.parametrize("feature_idx", range(NUM_FEATURES))
    def test_spiked_feature_is_selected(self, feature_idx):
        """When one feature has a large outlier error in a single row, that row's
        reason code should point to the spiked feature."""
        errors = _baseline_errors(200)
        target_row = 42
        # Inject a spike much larger than anything else in that feature's column
        errors[target_row, feature_idx] += 5.0
        reasons = attribute_reason(errors)
        assert reasons[target_row] == feature_idx, (
            f"Expected feature {feature_idx} ({FEATURE_COLS[feature_idx]}) "
            f"but got {reasons[target_row]} ({FEATURE_COLS[reasons[target_row]]})"
        )

    @pytest.mark.parametrize("feature_idx", range(NUM_FEATURES))
    def test_reason_maps_to_known_string(self, feature_idx):
        assert feature_idx in REASON_MAP, f"Feature {feature_idx} has no reason label"
        assert isinstance(REASON_MAP[feature_idx], str)

    def test_time_features_do_not_dominate_when_uniform(self):
        """The old argmax approach always picked time features. With relative
        attribution, a batch where all rows have similar errors should NOT
        overwhelmingly pick any single feature category."""
        errors = _baseline_errors(500)
        reasons = attribute_reason(errors)
        unique_reasons = set(reasons.tolist())
        # Should see at least 4 distinct feature indices selected
        assert len(unique_reasons) >= 4, (
            f"Only {len(unique_reasons)} distinct reasons in a uniform batch: {unique_reasons}"
        )

    def test_old_argmax_would_always_pick_time(self):
        """Confirm the baseline errors recreate the original bug — raw argmax
        selects time features (index 4 or 5) for the vast majority of rows."""
        errors = _baseline_errors(500)
        raw_max_indices = torch.argmax(errors, dim=1).numpy()
        time_count = np.isin(raw_max_indices, [4, 5, 6, 7]).sum()
        assert time_count > 400, (
            f"Baseline should cause time dominance, but only {time_count}/500 rows were time"
        )

    def test_each_reason_string_reachable(self):
        """Every reason string in the map must be reachable by spiking the
        corresponding feature."""
        seen_reasons = set()
        for feature_idx in range(NUM_FEATURES):
            errors = _baseline_errors(100)
            errors[0, feature_idx] += 5.0
            reasons = attribute_reason(errors)
            seen_reasons.add(REASON_MAP[int(reasons[0])])
        expected = set(REASON_MAP.values())
        assert seen_reasons == expected, f"Missing: {expected - seen_reasons}"

    def test_multiple_spikes_picks_largest(self):
        """When two features spike, the larger spike should win."""
        errors = _baseline_errors(100)
        row = 10
        errors[row, 0] += 3.0   # Volume Anomaly
        errors[row, 9] += 6.0   # Message Length — bigger spike
        reasons = attribute_reason(errors)
        assert reasons[row] == 9, (
            f"Expected feature 9 (larger spike) but got {reasons[row]}"
        )

    def test_batch_size_one(self):
        """Edge case: single-row batch. Median and IQR are degenerate but the
        function should not crash."""
        errors = torch.rand(1, NUM_FEATURES) * 0.5
        errors[0, 8] = 3.0
        reasons = attribute_reason(errors)
        assert reasons[0] == 8

    def test_all_features_identical_error_no_crash(self):
        """When all features have the same error, any selection is valid —
        just verify it doesn't crash or return out-of-range indices."""
        errors = torch.ones(50, NUM_FEATURES) * 0.5
        reasons = attribute_reason(errors)
        assert all(0 <= r < NUM_FEATURES for r in reasons)

    def test_reason_codes_shape_matches_batch(self):
        batch_size = 317
        errors = _baseline_errors(batch_size)
        reasons = attribute_reason(errors)
        assert reasons.shape == (batch_size,)

    @pytest.mark.parametrize("feature_idx,expected_label", [
        (0, "Volume Anomaly"),
        (1, "Severity Spike"),
        (2, "Unusual Event"),
        (3, "Security Flag"),
        (4, "Time Anomaly (Hour)"),
        (5, "Time Anomaly (Hour)"),
        (6, "Time Anomaly (Day)"),
        (7, "Time Anomaly (Day)"),
        (8, "User Anomaly"),
        (9, "Message Length Anomaly"),
        (10, "Process Anomaly"),
        (11, "Log Content Anomaly"),
        (18, "Log Content Anomaly"),
    ])
    def test_feature_index_to_label_mapping(self, feature_idx, expected_label):
        assert REASON_MAP[feature_idx] == expected_label

    def test_log_content_features_distinguishable(self):
        """Features 11–18 all map to 'Log Content Anomaly' but the attribution
        should still pick the specific index with the highest relative error."""
        errors = _baseline_errors(100)
        row = 0
        # Spike feature_log_v5 (index 16) harder than the others
        errors[row, 16] += 5.0
        reasons = attribute_reason(errors)
        assert reasons[row] == 16


# ── Minimal CPU AutoEncoder (mirrors Morpheus AutoEncoder interface) ─────────

def _build_cpu_autoencoder(input_dim: int = NUM_FEATURES):
    """Build a tiny AutoEncoder matching the production architecture."""
    encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 12), torch.nn.ReLU(),
        torch.nn.Linear(12, 8), torch.nn.ReLU(),
    )
    decoder = torch.nn.Sequential(
        torch.nn.Linear(8, 12), torch.nn.ReLU(),
        torch.nn.Linear(12, input_dim),
    )
    model = torch.nn.Sequential(encoder, decoder)
    return model


def _train_autoencoder(model, data: np.ndarray, epochs: int = 30):
    """Train on normalized data; register saved_means/saved_scales buffers."""
    means = np.mean(data, axis=0)
    stds = np.maximum(np.std(data, axis=0), 0.1)
    normalized = (data - means) / stds

    model.register_buffer("saved_means", torch.tensor(means, dtype=torch.float32))
    model.register_buffer("saved_scales", torch.tensor(stds, dtype=torch.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    tensor_data = torch.tensor(normalized, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        out = model(tensor_data)
        loss = loss_fn(out, tensor_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def _run_inference(model, data: np.ndarray):
    """Run the full ManualInferenceStage scoring + reason attribution on CPU.
    Returns (scores, reason_labels) matching production logic."""
    means = model.saved_means.numpy()
    stds = model.saved_scales.numpy()
    normalized = (data - means) / stds
    input_t = torch.tensor(normalized, dtype=torch.float32)

    with torch.no_grad():
        reconstruction = model(input_t)
        abs_errors = torch.abs(input_t - reconstruction)
        max_err = torch.max(abs_errors, dim=1).values
        mean_err = torch.mean(abs_errors, dim=1)
        loss = (0.7 * mean_err + 0.3 * max_err).to(torch.float32)

        # Relative reason attribution (matches production code)
        feat_median = torch.median(abs_errors, dim=0).values.unsqueeze(0)
        feat_iqr = (
            torch.quantile(abs_errors, 0.75, dim=0)
            - torch.quantile(abs_errors, 0.25, dim=0)
        ).unsqueeze(0).clamp(min=0.01)
        relative_errors = (abs_errors - feat_median) / feat_iqr
        _, reason_indices = torch.max(relative_errors, dim=1)

    scores_np = loss.numpy()
    scaled = np.where(scores_np <= 1.5, scores_np * 2.0,
                      3.0 + np.log1p(np.maximum(0, scores_np - 1.5)) * 2.0)
    scaled = np.clip(scaled, 0.0, 10.0)

    reason_codes = reason_indices.numpy()
    labels = [REASON_MAP[int(r)] for r in reason_codes]
    return scaled, labels, abs_errors.numpy()


# ── Scenario-based feature profiles ─────────────────────────────────────────

def _ssh_server_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n rows of 'normal SSH server' behavior.
    Daytime activity, sshd process, medium priority, stable content."""
    data = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    data[:, 0] = rng.normal(0.30, 0.05, n).clip(0, 1)     # log_count: moderate
    data[:, 1] = rng.normal(0.86, 0.02, n).clip(0, 1)      # priority: info(6)/7
    data[:, 2] = rng.normal(0.35, 0.03, n).clip(0, 1)      # event_type: stable
    data[:, 3] = 0.0                                         # is_security_event: no
    # Daytime hours (9-17) — cyclical encoding
    hours = rng.uniform(9, 17, n)
    data[:, 4] = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    data[:, 5] = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    # Weekdays only (Mon-Fri = 0-4)
    days = rng.uniform(0, 4, n)
    data[:, 6] = np.sin(2 * np.pi * days / 7).astype(np.float32)
    data[:, 7] = np.cos(2 * np.pi * days / 7).astype(np.float32)
    data[:, 8] = rng.normal(0.42, 0.02, n).clip(0, 1)      # user_hash: root/admin
    data[:, 9] = rng.normal(0.25, 0.04, n).clip(0, 1)      # msg_len: short
    data[:, 10] = rng.normal(0.55, 0.02, n).clip(0, 1)     # proc_hash: sshd
    # Content hashes: stable, tightly clustered
    for i in range(11, NUM_FEATURES):
        data[:, i] = rng.normal(0.30 + 0.03 * (i - 11), 0.02, n).clip(0, 1)
    return data


def _new_webserver_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n rows of 'nginx just installed' behavior.
    More logs, different priority, new process, 24/7 traffic, longer messages,
    totally different content hashes. Some security events (TLS errors)."""
    data = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    data[:, 0] = rng.normal(0.70, 0.08, n).clip(0, 1)      # log_count: doubled
    data[:, 1] = rng.normal(0.14, 0.03, n).clip(0, 1)      # priority: notice(1)/7
    data[:, 2] = rng.normal(0.72, 0.05, n).clip(0, 1)      # event_type: shifted
    data[:, 3] = rng.choice([0.0, 1.0], n, p=[0.85, 0.15]) # is_security: 15% yes
    # 24/7 traffic — uniform hours, includes weekends
    hours = rng.uniform(0, 24, n)
    data[:, 4] = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    data[:, 5] = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    days = rng.uniform(0, 7, n)
    data[:, 6] = np.sin(2 * np.pi * days / 7).astype(np.float32)
    data[:, 7] = np.cos(2 * np.pi * days / 7).astype(np.float32)
    data[:, 8] = rng.normal(0.78, 0.03, n).clip(0, 1)      # user_hash: www-data
    data[:, 9] = rng.normal(0.60, 0.06, n).clip(0, 1)      # msg_len: longer (access logs)
    data[:, 10] = rng.normal(0.22, 0.02, n).clip(0, 1)     # proc_hash: nginx (new)
    # Content hashes: completely different message templates
    for i in range(11, NUM_FEATURES):
        data[:, i] = rng.normal(0.70 + 0.02 * (i - 11), 0.03, n).clip(0, 1)
    return data


def _lateral_movement_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n rows of lateral movement: same processes but new users,
    security events, unusual hours."""
    data = _ssh_server_profile(n, rng)
    data[:, 3] = rng.choice([0.0, 1.0], n, p=[0.4, 0.6])  # is_security: 60% yes
    data[:, 8] = rng.normal(0.91, 0.03, n).clip(0, 1)      # user_hash: unknown users
    # Late night activity (1-4 AM)
    hours = rng.uniform(1, 4, n)
    data[:, 4] = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    data[:, 5] = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    return data


def _crypto_miner_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n rows of crypto-mining activity: massive log volume spike,
    new process, but similar message patterns otherwise."""
    data = _ssh_server_profile(n, rng)
    data[:, 0] = rng.normal(0.92, 0.03, n).clip(0, 1)      # log_count: 3x normal
    data[:, 10] = rng.normal(0.88, 0.02, n).clip(0, 1)     # proc_hash: xmrig/minerd
    data[:, 9] = rng.normal(0.15, 0.03, n).clip(0, 1)      # msg_len: short status lines
    return data


@pytest.mark.torch
@requires_torch
class TestScenarioWebServerInstall:
    """End-to-end scenario: train a model on SSH-server behavior, then infer
    against 'someone just installed nginx' logs. Validates that the full
    pipeline — normalization, autoencoder, scoring, reason attribution —
    produces the expected multi-feature anomaly response."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        self.rng = np.random.default_rng(seed=42)
        self.model = _build_cpu_autoencoder()
        baseline = _ssh_server_profile(2000, self.rng)
        _train_autoencoder(self.model, baseline, epochs=50)
        self.baseline_data = _ssh_server_profile(200, self.rng)
        self.anomaly_data = _new_webserver_profile(200, self.rng)

    def test_baseline_scores_low(self):
        """Normal SSH traffic should score well below the anomaly threshold."""
        scores, _, _ = _run_inference(self.model, self.baseline_data)
        assert np.mean(scores) < 4.0, f"Baseline mean score {np.mean(scores):.2f} is too high"
        assert np.median(scores) < 3.0

    def test_webserver_scores_elevated(self):
        """New web server traffic should produce clearly elevated scores."""
        scores, _, _ = _run_inference(self.model, self.anomaly_data)
        assert np.mean(scores) > 2.0, f"Anomaly mean score {np.mean(scores):.2f} is too low"

    def test_webserver_scores_higher_than_baseline(self):
        baseline_scores, _, _ = _run_inference(self.model, self.baseline_data)
        anomaly_scores, _, _ = _run_inference(self.model, self.anomaly_data)
        assert np.mean(anomaly_scores) > np.mean(baseline_scores) * 1.5

    def test_multiple_reason_categories_triggered(self):
        """A web server install should trigger several DIFFERENT anomaly
        categories, not just one."""
        _, labels, _ = _run_inference(self.model, self.anomaly_data)
        unique_reasons = set(labels)
        assert len(unique_reasons) >= 3, (
            f"Expected at least 3 distinct anomaly reasons, got {unique_reasons}"
        )

    def test_process_anomaly_detected(self):
        """nginx is a new process — Process Anomaly should appear."""
        _, labels, _ = _run_inference(self.model, self.anomaly_data)
        assert "Process Anomaly" in labels, (
            f"Process Anomaly not found in: {set(labels)}"
        )

    def test_volume_or_content_anomaly_detected(self):
        """Doubled log volume and new message templates should trigger at
        least one of Volume Anomaly or Log Content Anomaly."""
        _, labels, _ = _run_inference(self.model, self.anomaly_data)
        assert "Volume Anomaly" in labels or "Log Content Anomaly" in labels, (
            f"Neither Volume nor Content anomaly found in: {set(labels)}"
        )

    def test_no_single_reason_dominates(self):
        """No single reason should account for more than 70% of the rows."""
        _, labels, _ = _run_inference(self.model, self.anomaly_data)
        from collections import Counter
        counts = Counter(labels)
        total = len(labels)
        for reason, count in counts.items():
            assert count / total < 0.70, (
                f"'{reason}' dominates with {count}/{total} = {count/total:.0%}"
            )

    def test_per_feature_errors_show_expected_spikes(self):
        """The raw per-feature errors should be elevated for the features we
        know changed (process, volume, content) vs baseline."""
        _, _, anomaly_errors = _run_inference(self.model, self.anomaly_data)
        _, _, baseline_errors = _run_inference(self.model, self.baseline_data)
        anomaly_mean = np.mean(anomaly_errors, axis=0)
        baseline_mean = np.mean(baseline_errors, axis=0)
        # Process hash (idx 10) should have notably higher error
        assert anomaly_mean[10] > baseline_mean[10], "Process hash error should be elevated"
        # At least half of the content hash features (11-18) should be elevated
        content_elevated = sum(
            1 for i in range(11, NUM_FEATURES) if anomaly_mean[i] > baseline_mean[i]
        )
        assert content_elevated >= 4, (
            f"Only {content_elevated}/8 content features elevated"
        )

    def test_baseline_reasons_are_diverse(self):
        """Even baseline traffic should show varied reasons (not all one type),
        confirming the relative attribution works for 'normal' data too."""
        _, labels, _ = _run_inference(self.model, self.baseline_data)
        unique = set(labels)
        assert len(unique) >= 2, f"Baseline reasons not diverse: {unique}"


@pytest.mark.torch
@requires_torch
class TestScenarioLateralMovement:
    """Scenario: attacker performing lateral movement — same host, but new
    users, security events, and unusual hours."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        self.rng = np.random.default_rng(seed=99)
        self.model = _build_cpu_autoencoder()
        baseline = _ssh_server_profile(2000, self.rng)
        _train_autoencoder(self.model, baseline, epochs=50)
        self.anomaly_data = _lateral_movement_profile(200, self.rng)

    def test_scores_elevated(self):
        scores, _, _ = _run_inference(self.model, self.anomaly_data)
        assert np.mean(scores) > 2.0

    def test_user_anomaly_detected(self):
        _, labels, _ = _run_inference(self.model, self.anomaly_data)
        assert "User Anomaly" in labels, f"User Anomaly not in {set(labels)}"

    def test_security_flag_or_time_anomaly_detected(self):
        _, labels, _ = _run_inference(self.model, self.anomaly_data)
        has_security = "Security Flag" in labels
        has_time = "Time Anomaly (Hour)" in labels
        assert has_security or has_time, (
            f"Expected Security Flag or Time Anomaly (Hour), got: {set(labels)}"
        )

    def test_multiple_reasons(self):
        _, labels, _ = _run_inference(self.model, self.anomaly_data)
        assert len(set(labels)) >= 2


@pytest.mark.torch
@requires_torch
class TestScenarioCryptoMiner:
    """Scenario: crypto miner installed — massive volume spike and new process,
    but same user and similar message structure."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        self.rng = np.random.default_rng(seed=77)
        self.model = _build_cpu_autoencoder()
        baseline = _ssh_server_profile(2000, self.rng)
        _train_autoencoder(self.model, baseline, epochs=50)
        self.anomaly_data = _crypto_miner_profile(200, self.rng)

    def test_scores_elevated(self):
        scores, _, _ = _run_inference(self.model, self.anomaly_data)
        assert np.mean(scores) > 2.0

    def test_volume_anomaly_detected(self):
        _, labels, _ = _run_inference(self.model, self.anomaly_data)
        assert "Volume Anomaly" in labels, f"Volume Anomaly not in {set(labels)}"

    def test_process_anomaly_detected(self):
        _, labels, _ = _run_inference(self.model, self.anomaly_data)
        assert "Process Anomaly" in labels, f"Process Anomaly not in {set(labels)}"


# ── Threat pattern matching (CPU-only, no torch) ────────────────────────────

from collections import Counter

# Replica of ThreatPatternStage._define_patterns() for CPU testing.
# We import the actual pattern functions if morpheus_app is available,
# otherwise replicate the logic so tests run anywhere.
if MORPHEUS_AVAILABLE and hasattr(app, 'ThreatPatternStage'):
    _PATTERNS = app.ThreatPatternStage.PATTERNS
else:
    # Standalone replicas — keep in sync with morpheus_app.py
    def _p_brute_force(reasons, _scores, n):
        sec = reasons.get("Security Flag", 0)
        ratio = sec / max(n, 1)
        if sec >= 5 and ratio > 0.3:
            conf = min(1.0, 0.4 + (sec - 5) * 0.06 + (ratio - 0.3) * 1.0)
            return ("Brute Force Attack", f"{sec} security events", round(conf, 2))
        return None

    def _p_lateral(reasons, _scores, _n):
        user_count = reasons.get("User Anomaly", 0)
        time_count = reasons.get("Time Anomaly (Hour)", 0)
        sec_count = reasons.get("Security Flag", 0)
        if user_count >= 2 and (time_count >= 1 or sec_count >= 1):
            signals = min(user_count, 5) + min(time_count, 3) + min(sec_count, 3)
            conf = min(1.0, 0.3 + signals * 0.07)
            return ("Lateral Movement", "", round(conf, 2))
        return None

    def _p_service(reasons, _scores, _n):
        proc_count = reasons.get("Process Anomaly", 0)
        content_count = reasons.get("Log Content Anomaly", 0)
        volume_count = reasons.get("Volume Anomaly", 0)
        if proc_count >= 2 and (content_count >= 3 or volume_count >= 1):
            signals = min(proc_count, 6) + min(content_count, 8) + min(volume_count, 4)
            conf = min(1.0, 0.3 + signals * 0.05)
            return ("New Service Detected", "", round(conf, 2))
        return None

    def _p_exfil(reasons, scores, _n):
        vol_count = reasons.get("Volume Anomaly", 0)
        high_scores = sum(1 for s in scores if s > 6.0)
        if vol_count >= 3 and high_scores >= 3:
            conf = min(1.0, 0.4 + (vol_count - 3) * 0.08 + (high_scores - 3) * 0.08)
            return ("Possible Data Exfiltration", "", round(conf, 2))
        return None

    def _p_privesc(reasons, scores, _n):
        user_count = reasons.get("User Anomaly", 0)
        sec_count = reasons.get("Security Flag", 0)
        high_scores = sum(1 for s in scores if s > 7.0)
        if user_count >= 1 and sec_count >= 2 and high_scores >= 2:
            conf = min(1.0, 0.5 + (sec_count - 2) * 0.1 + (high_scores - 2) * 0.1)
            return ("Privilege Escalation Attempt", "", round(conf, 2))
        return None

    def _p_crypto(reasons, _scores, _n):
        proc_count = reasons.get("Process Anomaly", 0)
        vol_count = reasons.get("Volume Anomaly", 0)
        if proc_count >= 2 and vol_count >= 2 and not reasons.get("Security Flag", 0):
            conf = min(1.0, 0.3 + (proc_count - 2) * 0.1 + (vol_count - 2) * 0.1)
            return ("Suspicious Process Activity", "", round(conf, 2))
        return None

    def _p_drift(reasons, _scores, n):
        distinct = sum(1 for v in reasons.values() if v >= 2)
        if distinct >= 4 and n >= 10:
            conf = min(1.0, 0.4 + (distinct - 4) * 0.15 + (n - 10) * 0.01)
            return ("Major Behavioral Shift", "", round(conf, 2))
        return None

    def _p_offhours(reasons, _scores, n):
        tc = reasons.get("Time Anomaly (Hour)", 0) + reasons.get("Time Anomaly (Day)", 0)
        ratio = tc / max(n, 1)
        if tc >= 5 and ratio > 0.5:
            conf = min(1.0, 0.3 + (tc - 5) * 0.05 + (ratio - 0.5) * 1.0)
            return ("Off-Hours Activity", "", round(conf, 2))
        return None

    _PATTERNS = [_p_brute_force, _p_lateral, _p_service, _p_exfil, _p_privesc, _p_crypto, _p_drift, _p_offhours]


def _evaluate_patterns(reasons_dict, scores_list):
    """Run all patterns against a reason/score set. Returns list of (label, confidence) tuples."""
    reasons = Counter(reasons_dict)
    n = sum(reasons.values())
    matches = []
    for fn in _PATTERNS:
        result = fn(reasons, scores_list, n)
        if result:
            matches.append((result[0], result[2]))
    return matches


def _matched_labels(reasons_dict, scores_list):
    """Convenience: return just the set of matched pattern labels."""
    return {label for label, _ in _evaluate_patterns(reasons_dict, scores_list)}


def _get_confidence(reasons_dict, scores_list, pattern_name):
    """Return the confidence for a specific pattern, or None if not matched."""
    for label, conf in _evaluate_patterns(reasons_dict, scores_list):
        if label == pattern_name:
            return conf
    return None


class TestThreatPatterns:
    """Verify each threat pattern fires on its expected input and does NOT
    fire on unrelated inputs."""

    def test_brute_force_triggers(self):
        reasons = {"Security Flag": 8, "User Anomaly": 2}
        assert "Brute Force Attack" in _matched_labels(reasons, [5.0] * 10)

    def test_brute_force_needs_threshold(self):
        reasons = {"Security Flag": 3}
        assert "Brute Force Attack" not in _matched_labels(reasons, [5.0] * 3)

    def test_lateral_movement_triggers(self):
        reasons = {"User Anomaly": 3, "Time Anomaly (Hour)": 2, "Security Flag": 1}
        assert "Lateral Movement" in _matched_labels(reasons, [4.0] * 6)

    def test_lateral_movement_needs_user_plus_time_or_security(self):
        reasons = {"User Anomaly": 3}
        assert "Lateral Movement" not in _matched_labels(reasons, [4.0] * 3)

    def test_service_install_triggers(self):
        reasons = {"Process Anomaly": 3, "Log Content Anomaly": 5, "Volume Anomaly": 2}
        assert "New Service Detected" in _matched_labels(reasons, [4.0] * 10)

    def test_service_install_proc_plus_volume(self):
        reasons = {"Process Anomaly": 3, "Volume Anomaly": 2}
        assert "New Service Detected" in _matched_labels(reasons, [4.0] * 5)

    def test_service_install_needs_process(self):
        reasons = {"Log Content Anomaly": 10, "Volume Anomaly": 5}
        assert "New Service Detected" not in _matched_labels(reasons, [4.0] * 15)

    def test_data_exfiltration_triggers(self):
        reasons = {"Volume Anomaly": 5}
        assert "Possible Data Exfiltration" in _matched_labels(reasons, [7.0] * 5)

    def test_data_exfiltration_needs_high_scores(self):
        reasons = {"Volume Anomaly": 5}
        assert "Possible Data Exfiltration" not in _matched_labels(reasons, [3.0] * 5)

    def test_privilege_escalation_triggers(self):
        reasons = {"User Anomaly": 2, "Security Flag": 3}
        assert "Privilege Escalation Attempt" in _matched_labels(reasons, [8.0] * 5)

    def test_crypto_mining_triggers(self):
        reasons = {"Process Anomaly": 3, "Volume Anomaly": 4}
        assert "Suspicious Process Activity" in _matched_labels(reasons, [5.0] * 7)

    def test_crypto_mining_suppressed_by_security_flag(self):
        reasons = {"Process Anomaly": 3, "Volume Anomaly": 4, "Security Flag": 1}
        assert "Suspicious Process Activity" not in _matched_labels(reasons, [5.0] * 8)

    def test_behavioral_drift_triggers(self):
        reasons = {
            "Volume Anomaly": 3, "Process Anomaly": 2,
            "User Anomaly": 2, "Log Content Anomaly": 4,
        }
        assert "Major Behavioral Shift" in _matched_labels(reasons, [4.0] * 11)

    def test_behavioral_drift_needs_enough_events(self):
        reasons = {
            "Volume Anomaly": 2, "Process Anomaly": 2,
            "User Anomaly": 2, "Log Content Anomaly": 2,
        }
        assert "Major Behavioral Shift" not in _matched_labels(reasons, [4.0] * 8)

    def test_off_hours_triggers(self):
        reasons = {"Time Anomaly (Hour)": 8, "Volume Anomaly": 1}
        assert "Off-Hours Activity" in _matched_labels(reasons, [4.0] * 9)

    def test_off_hours_needs_majority(self):
        reasons = {"Time Anomaly (Hour)": 3, "Volume Anomaly": 10}
        assert "Off-Hours Activity" not in _matched_labels(reasons, [4.0] * 13)

    def test_normal_traffic_no_patterns(self):
        reasons = {"Volume Anomaly": 1, "Log Content Anomaly": 1}
        assert _evaluate_patterns(reasons, [3.0] * 2) == []

    def test_multiple_patterns_can_fire(self):
        reasons = {
            "Security Flag": 6, "User Anomaly": 3,
            "Time Anomaly (Hour)": 2,
        }
        labels = _matched_labels(reasons, [8.0] * 11)
        assert "Brute Force Attack" in labels
        assert "Lateral Movement" in labels

    def test_webserver_install_scenario(self):
        """Simulates the reason distribution from a new web server install."""
        reasons = {
            "Process Anomaly": 5, "Log Content Anomaly": 8,
            "Volume Anomaly": 4, "Severity Spike": 2,
            "Message Length Anomaly": 3,
        }
        labels = _matched_labels(reasons, [5.0] * 22)
        assert "New Service Detected" in labels
        assert "Brute Force Attack" not in labels

    def test_crypto_miner_scenario(self):
        """Simulates the reason distribution from a crypto miner."""
        reasons = {"Process Anomaly": 6, "Volume Anomaly": 8, "Message Length Anomaly": 2}
        assert "Suspicious Process Activity" in _matched_labels(reasons, [6.0] * 16)

    def test_all_patterns_reachable(self):
        """Every defined pattern label must be triggerable."""
        expected = {
            "Brute Force Attack", "Lateral Movement", "New Service Detected",
            "Possible Data Exfiltration", "Privilege Escalation Attempt",
            "Suspicious Process Activity", "Major Behavioral Shift", "Off-Hours Activity",
        }
        scenarios = [
            ({"Security Flag": 8}, [5.0] * 8),
            ({"User Anomaly": 3, "Time Anomaly (Hour)": 2}, [4.0] * 5),
            ({"Process Anomaly": 3, "Log Content Anomaly": 5}, [4.0] * 8),
            ({"Volume Anomaly": 5}, [7.0] * 5),
            ({"User Anomaly": 2, "Security Flag": 3}, [8.0] * 5),
            ({"Process Anomaly": 4, "Volume Anomaly": 4}, [5.0] * 8),
            ({"Volume Anomaly": 3, "Process Anomaly": 2, "User Anomaly": 2, "Log Content Anomaly": 4}, [4.0] * 11),
            ({"Time Anomaly (Hour)": 8}, [4.0] * 8),
        ]
        seen = set()
        for reasons, scores in scenarios:
            for label, _ in _evaluate_patterns(reasons, scores):
                seen.add(label)
        assert seen == expected, f"Missing: {expected - seen}"

    # ── Confidence scoring tests ─────────────────────────────────────────

    def test_confidence_is_between_0_and_1(self):
        reasons = {"Security Flag": 8, "User Anomaly": 2}
        for _, conf in _evaluate_patterns(reasons, [5.0] * 10):
            assert 0.0 <= conf <= 1.0

    def test_confidence_increases_with_stronger_signal(self):
        low = _get_confidence({"Security Flag": 5, "Volume Anomaly": 10}, [5.0] * 15, "Brute Force Attack")
        high = _get_confidence({"Security Flag": 10, "Volume Anomaly": 5}, [5.0] * 15, "Brute Force Attack")
        assert low is not None and high is not None
        assert high > low, f"Expected higher confidence with more signals: {high} vs {low}"

    def test_barely_triggered_has_low_confidence(self):
        conf = _get_confidence({"Security Flag": 5, "Volume Anomaly": 10}, [5.0] * 15, "Brute Force Attack")
        assert conf is not None and conf < 0.7

    def test_overwhelming_signal_has_high_confidence(self):
        conf = _get_confidence({"Security Flag": 20}, [8.0] * 20, "Brute Force Attack")
        assert conf is not None and conf >= 0.9

    def test_each_pattern_returns_confidence(self):
        scenarios = [
            ({"Security Flag": 8}, [5.0] * 8),
            ({"User Anomaly": 3, "Time Anomaly (Hour)": 2}, [4.0] * 5),
            ({"Process Anomaly": 3, "Log Content Anomaly": 5}, [4.0] * 8),
            ({"Volume Anomaly": 5}, [7.0] * 5),
            ({"User Anomaly": 2, "Security Flag": 3}, [8.0] * 5),
            ({"Process Anomaly": 4, "Volume Anomaly": 4}, [5.0] * 8),
            ({"Volume Anomaly": 3, "Process Anomaly": 2, "User Anomaly": 2, "Log Content Anomaly": 4}, [4.0] * 11),
            ({"Time Anomaly (Hour)": 8}, [4.0] * 8),
        ]
        for reasons, scores in scenarios:
            matches = _evaluate_patterns(reasons, scores)
            assert len(matches) > 0
            for label, conf in matches:
                assert 0.0 < conf <= 1.0, f"{label} has invalid confidence {conf}"


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
