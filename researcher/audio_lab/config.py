"""Frequency bands, window sizes, and decision thresholds for Audio Audit Lab.

Numeric windows and cutoffs that the spec does not fix live here.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "output" / "audio"
DEFAULT_DB_PATH = ROOT / "data" / "audio_lab.db"
MODEL_DIR = ROOT / "data" / "audio_models"

FEATURE_SR = 22050

# Spectral frames used for bands, centroid, rolloff, and flatness.
N_FFT = 4096
HOP_LENGTH = 1024

# Pause and noise-floor frames. Short enough that a 5 s gap is within ±0.3 s.
FRAME_S = 0.05
PAUSE_DBFS = -50.0
NOISE_FLOOR_QUIET_FRACTION = 0.05
# Kadrlar shu darajadan past bo'lsa raqamli sukunat; shovqin pog'onasi hisobiga kirmaydi.
NOISE_FLOOR_DIGITAL_SILENCE_DBFS = -100.0
# Separate from the 5% noise-floor estimate. Quietest fraction for artifact measures.
ARTIFACT_QUIET_FRACTION = 0.10
ARTIFACT_FLATNESS_LOW_HZ = 2000.0
ARTIFACT_FLATNESS_HIGH_HZ = 8000.0

RMS_STEP_S = 1.0
BAND_WINDOW_S = 10.0
SHORT_TERM_WINDOW_S = 3.0
SHORT_TERM_STEP_S = 1.0

# Self-similarity. Hop divides a 4 s phrase so the lag can land on 4.0 s.
REP_WINDOW_S = 2.0
REP_HOP_S = 0.5
# Lags closer than one window are overlap, not a repeat.
REP_MIN_LAG_S = 2.0
# Longer files widen the hop so the matrix stays bounded.
REP_MAX_WINDOWS = 1600
N_MFCC = 13

ONSET_HOP_LENGTH = 512

TEMPO_BPM_MIN = 40.0
TEMPO_BPM_MAX = 240.0
# Peak dominance (best − second) / best. Below this, tempo is not reliable.
TEMPO_RELIABLE_MIN_CONFIDENCE = 0.12
# A steady tone can still show a tempogram bump. Require real onsets.
TEMPO_RELIABLE_MIN_ONSETS_PER_SEC = 0.4

# Best-minus-second Krumhansl correlation. Below this, the key is not reliable.
KEY_RELIABLE_MIN_MARGIN = 0.04

# Low-pass estimate: power below the reference spectrum by this many dB.
LOWPASS_DROP_DB = 60.0

BANDS = (
    ("sub_bass", 20.0, 60.0),
    ("bass", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("high_mid", 2000.0, 6000.0),
    ("high", 6000.0, 16000.0),
)

# Krumhansl & Kessler 1982 pitch-class profiles, C at index 0.
MAJOR_PROFILE = (
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
)
MINOR_PROFILE = (
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
)
PITCH_CLASS_NAMES = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif", ".webm")

# Segmentation. Feature rate stays 22050 even when per-track measures use --sr.
SEG_SR = 22050
SEG_CHUNK_S = 600.0
SEG_CHUNK_OVERLAP_S = 2.0
SEG_HOP_S = 1.0
SEG_WINDOW_S = 2.0
SEG_KERNEL_HALF_S = (20, 30, 60)
SEG_SILENCE_MIN_S = 0.5
SEG_DIP_DB = 2.0
SEG_DIP_MIN_S = 2.0
SEG_DIP_MAX_S = 8.0
SEG_DIP_MEDIAN_S = 21
SEG_LEVEL_WINDOW_S = 3.0
# Combined boundary score. Silence is strong so a gap wins over its two edges.
SEG_W_NOVELTY = 1.0
SEG_W_SILENCE = 1.4
SEG_W_DIP = 0.9
# Adaptive gate: median + k * MAD. One expect-tracks retry scales k by this.
SEG_K = 2.6
SEG_K_RETRY_SCALE = 0.30
SEG_MIN_TRACK_S = 90.0
SEG_CONF_HIGH_MAD = 8.0
SEG_CONF_MED_MAD = 5.0
SEG_DUP_MEAN_COSINE = 0.98
SEG_DUP_CHROMA_COSINE = 0.95
SEG_FIRST_CUE_MAX_S = 5.0
SEG_TRACKLIST_TOL_S = 5.0
SEG_LONG_FILE_S = 15 * 60
SEG_PROFILE_MIN_TRACKS = 8
# Suggest this rate for per-track band/spectral measures. Segmentation stays at SEG_SR.
SEG_CLARITY_SR = 44100
# Sample-clip groups. Average-linkage cosine distance above this starts another group.
CLIP_GROUP_DISTANCE = 1.0
