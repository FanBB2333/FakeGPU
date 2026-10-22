"""The modeled CUDA caching allocator against PyTorch's behavior.

Expected numbers come from allocation traces run once on an Ampere GPU
(sm_80) with torch 2.9.1 and 2.10.0, and from
c10/cuda/CUDACachingAllocator.cpp, c10/cuda/CUDAAllocatorConfig.cpp,
c10/core/AllocatorConfig.cpp and torch/csrc/cuda/Module.cpp of
v2.9.1-v2.14.1.  Where the sources agree, the trace is checked under the
oldest and newest version; elsewhere each version gets its own expectation.
"""

from __future__ import annotations

import itertools
import pytest

from fakegpu._allocator import _DeviceMemoryTracker


MIB = 1024**2
GIB = 1024**3
SHARED_VERSIONS = [(2, 9), (2, 14)]


def _tracker(conf=None, *, version=(2, 14), total=80 * GIB, **kwargs):
    kwargs.setdefault("context_bytes", 0)
    return _DeviceMemoryTracker(
        [total],
        caching_allocator=True,
        torch_version=version,
        alloc_conf=conf,
        **kwargs,
    )


class _Trace:
    """Replays torch.empty(n, dtype=torch.uint8) allocations and frees."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.live = {}
        self.keys = itertools.count(1)
        self.steps = []

    def record(self):
        self.steps.append(
            (self.tracker.memory_allocated(0), self.tracker.memory_reserved(0))
        )

    def alloc(self, name, nbytes, *, stream=0, record=True):
        key = next(self.keys)
        self.tracker.allocate(key, int(nbytes), 0, stream=stream)
        self.live[name] = key
        if record:
            self.record()
        return key

    def free(self, name, *, record=True):
        self.tracker.release(self.live.pop(name))
        if record:
            self.record()

    def empty_cache(self):
        self.tracker.empty_cache()
        self.record()

    def counters(self):
        stats = self.tracker.memory_stats(0)
        return {
            key: stats[key]
            for key in (
                "num_alloc_retries",
                "num_ooms",
                "num_sync_all_streams",
                "num_device_alloc",
                "num_device_free",
            )
        }


def _basic(trace):
    for name, nbytes in (
        ("a", 4),
        ("b", 4000),
        ("c", 600 * 1024),
        ("d", MIB),
        ("e", MIB + 1),
        ("f", 5 * MIB),
        ("g", 9 * MIB + MIB // 2),
        ("h", 10 * MIB),
        ("i", 25 * MIB + 3),
        ("j", 300 * MIB),
    ):
        trace.alloc(name, nbytes)
    trace.free("j")
    trace.alloc("k", 100 * MIB)
    trace.alloc("l", 150 * MIB)
    trace.free("k")
    trace.alloc("m", 120 * MIB)
    trace.free("l")
    trace.alloc("n", 333 * MIB + 7)
    trace.free("i")
    trace.empty_cache()
    trace.alloc("p", 250 * MIB)
    trace.free("p")
    trace.alloc("q", 240 * MIB)
    trace.free("q")
    trace.alloc("r", 220 * MIB)
    trace.free("r")
    for name in list(trace.live):
        trace.free(name, record=False)
    trace.empty_cache()


BASIC_ALLOCATED = [
    512, 4608, 619008, 1667584, 2716672, 7959552, 17921024, 28406784,
    55669760, 370242560, 55669760, 160527360, 317813760, 212956160,
    338785280, 181498880, 531723264, 504460288, 504460288, 766604288,
    504460288, 756118528, 504460288, 735147008, 504460288, 0,
]
BASIC_CASES = {
    None: (
        BASIC_ALLOCATED,
        [
            2097152, 2097152, 2097152, 2097152, 23068672, 23068672, 23068672,
            33554432, 60817408, 375390208, 375390208, 375390208, 375390208,
            375390208, 501219328, 501219328, 851443712, 851443712, 509607936,
            771751936, 771751936, 771751936, 771751936, 771751936, 771751936, 0,
        ],
        {"num_sync_all_streams": 2, "num_device_alloc": 8, "num_device_free": 8},
    ),
    "max_split_size_mb:200": (
        BASIC_ALLOCATED[:21] + [766604288] + BASIC_ALLOCATED[22:],
        [
            2097152, 2097152, 2097152, 2097152, 23068672, 23068672, 23068672,
            33554432, 60817408, 375390208, 375390208, 480247808, 637534208,
            637534208, 763363328, 763363328, 1113587712, 1113587712, 509607936,
            771751936, 771751936, 771751936, 771751936, 1002438656, 1002438656, 0,
        ],
        {"num_sync_all_streams": 2, "num_device_alloc": 11, "num_device_free": 11},
    ),
    "max_split_size_mb:200,max_non_split_rounding_mb:50": (
        BASIC_ALLOCATED[:21] + [766604288, 504460288, 766604288]
        + BASIC_ALLOCATED[24:],
        [
            2097152, 2097152, 2097152, 2097152, 23068672, 23068672, 23068672,
            33554432, 60817408, 375390208, 375390208, 480247808, 637534208,
            637534208, 763363328, 763363328, 1113587712, 1113587712, 509607936,
            771751936, 771751936, 771751936, 771751936, 771751936, 771751936, 0,
        ],
        {"num_sync_all_streams": 2, "num_device_alloc": 10, "num_device_free": 10},
    ),
    "roundup_power2_divisions:4": (
        [
            512, 4608, 659968, 1708544, 3019264, 8262144, 18747904, 29233664,
            58593792, 394138112, 58593792, 176034304, 343806464, 226365952,
            360583680, 192811520, 595464704, 566104576, 566104576, 834540032,
            566104576, 834540032, 566104576, 800985600, 566104576, 0,
        ],
        [
            2097152, 2097152, 2097152, 2097152, 23068672, 23068672, 23068672,
            33554432, 62914560, 398458880, 398458880, 398458880, 398458880,
            398458880, 532676608, 532676608, 935329792, 935329792, 570425344,
            838860800, 838860800, 838860800, 838860800, 838860800, 838860800, 0,
        ],
        {"num_sync_all_streams": 2, "num_device_alloc": 8, "num_device_free": 8},
    ),
    "expandable_segments:True": (
        [
            512, 4608, 619008, 1667584, 2716672, 7959552, 17921024, 28406784,
            54621696, 369194496, 54621696, 159479296, 316765696, 211908096,
            337737216, 180450816, 529627136, 503412224, 503412224, 765556224,
            503412224, 755070464, 503412224, 734098944, 503412224, 0,
        ],
        [
            2097152, 2097152, 2097152, 2097152, 23068672, 23068672, 23068672,
            44040192, 65011712, 379584512, 379584512, 379584512, 379584512,
            379584512, 463470592, 463470592, 799014912, 799014912, 526385152,
            778043392, 778043392, 778043392, 778043392, 778043392, 778043392, 0,
        ],
        # Expandable segments map and unmap pages: device frees count
        # unmaps, and the segment stats are not touched.
        {"num_sync_all_streams": 2, "num_device_alloc": 8, "num_device_free": 4},
    ),
}


@pytest.mark.parametrize("version", SHARED_VERSIONS)
@pytest.mark.parametrize("conf", list(BASIC_CASES), ids=str)
def test_allocation_trace_matches_real_allocator(conf, version) -> None:
    allocated, reserved, counters = BASIC_CASES[conf]
    trace = _Trace(_tracker(conf, version=version))
    _basic(trace)
    assert [step[0] for step in trace.steps] == allocated
    assert [step[1] for step in trace.steps] == reserved
    assert trace.counters() == {
        "num_alloc_retries": 0,
        "num_ooms": 0,
        **counters,
    }


def test_expandable_segments_keep_segment_and_split_stats_flat() -> None:
    trace = _Trace(_tracker("expandable_segments:True"))
    trace.alloc("a", 4)
    trace.alloc("b", 25 * MIB + 3)
    stats = trace.tracker.memory_stats(0)
    assert stats["segment.all.current"] == 0
    assert stats["inactive_split.all.current"] == 0
    assert stats["reserved_bytes.all.current"] == 2 * MIB + 40 * MIB
    segments = trace.tracker.allocator_snapshot()
    assert [segment["is_expandable"] for segment in segments] == [True, True]


SIZES = [
    1, 511, 513, 4000, 600 * 1024, 1363149, 3 * MIB, 5 * MIB + 1,
    9 * MIB + MIB // 2, 25 * MIB + 3, 70 * MIB + 1, 300 * MIB, 333 * MIB + 7,
    600 * MIB + 5, 1100 * MIB + 3,
]
SIZE_CASES = {
    None: [
        (512, 2097152), (512, 2097152), (1024, 2097152), (4096, 2097152),
        (614400, 2097152), (1363456, 20971520), (3145728, 20971520),
        (5243392, 20971520), (9961472, 20971520), (27262976, 27262976),
        (73400832, 75497472), (314572800, 314572800), (350224384, 350224384),
        (629146112, 631242752), (1153434112, 1155530752),
    ],
    "roundup_power2_divisions:4": [
        (512, 2097152), (512, 2097152), (1024, 2097152), (4096, 2097152),
        (655360, 2097152), (1572864, 20971520), (3145728, 20971520),
        (6291456, 20971520), (10485760, 10485760), (29360128, 29360128),
        (83886080, 83886080), (335544320, 335544320), (402653184, 402653184),
        (671088640, 671088640), (1342177280, 1342177280),
    ],
    "expandable_segments:True": [
        (512, 2097152), (512, 2097152), (1024, 2097152), (4096, 2097152),
        (614400, 2097152), (1363456, 20971520), (3145728, 20971520),
        (5243392, 20971520), (9961472, 20971520), (26214912, 41943040),
        (73400832, 83886080), (314572800, 314572800), (349176320, 356515840),
        (629146112, 650117120), (1153434112, 1174405120),
    ],
}
ROUNDUP_LIST = "roundup_power2_divisions:[1:4,64:8,>:2]"
ROUNDUP_LIST_SIZES = [
    (512, 2097152), (512, 2097152), (1024, 2097152), (4096, 2097152),
    (655360, 2097152), (1572864, 20971520), (3145728, 20971520),
    (5243392, 20971520), (9961472, 20971520), (27262976, 27262976),
    None, (402653184, 402653184), (402653184, 402653184),
    (805306368, 805306368), (1610612736, 1610612736),
]


def _sizes(tracker):
    rows = []
    for size in SIZES:
        before = (tracker.memory_allocated(0), tracker.memory_reserved(0))
        tracker.allocate(1, size, 0)
        rows.append(
            (
                tracker.memory_allocated(0) - before[0],
                tracker.memory_reserved(0) - before[1],
            )
        )
        tracker.release(1)
        tracker.empty_cache()
    return rows


@pytest.mark.parametrize("version", SHARED_VERSIONS)
@pytest.mark.parametrize("conf", list(SIZE_CASES), ids=str)
def test_request_rounding_and_segment_sizes(conf, version) -> None:
    assert _sizes(_tracker(conf, version=version)) == SIZE_CASES[conf]


@pytest.mark.parametrize(
    ("version", "seventy_mib"),
    [
        # parseRoundUpPower2Divisions: up to 2.9 ">" refills from the last
        # listed interval itself (here 64 MiB -> 2 divisions); from 2.10 it
        # starts after it, so 64 MiB keeps 8 divisions.
        ((2, 9), (100663296, 100663296)),
        ((2, 10), (75497472, 75497472)),
        ((2, 14), (75497472, 75497472)),
    ],
)
def test_per_interval_roundup_list(version, seventy_mib) -> None:
    expected = [seventy_mib if row is None else row for row in ROUNDUP_LIST_SIZES]
    assert _sizes(_tracker(ROUNDUP_LIST, version=version)) == expected


def _gc_trace(conf, version):
    tracker = _tracker(conf, version=version, total=4 * GIB)
    tracker.set_memory_fraction(400 * MIB / (4 * GIB), 0)
    trace = _Trace(tracker)
    trace.alloc("x1", 100 * MIB)
    trace.alloc("x2", 60 * MIB)
    trace.alloc("x3", 80 * MIB)
    trace.free("x1")
    trace.free("x2")
    trace.alloc("y", 30 * MIB)
    trace.free("x3")
    trace.alloc("z", 120 * MIB)
    return trace


@pytest.mark.parametrize("version", SHARED_VERSIONS)
def test_garbage_collection_threshold_frees_old_blocks_under_a_fraction(
    version,
) -> None:
    plain = _gc_trace(None, version)
    assert plain.steps[-1] == (150 * MIB, 360 * MIB)
    assert plain.counters()["num_device_free"] == 0

    collected = _gc_trace("garbage_collection_threshold:0.5", version)
    assert collected.steps[-1] == (150 * MIB, 260 * MIB)
    assert collected.counters()["num_device_alloc"] == 4
    assert collected.counters()["num_device_free"] == 1


@pytest.mark.parametrize("version", SHARED_VERSIONS)
def test_failed_cudamalloc_releases_oversize_blocks_first(version) -> None:
    tracker = _tracker("max_split_size_mb:200", version=version, total=4 * GIB)
    tracker.set_memory_fraction(350 * MIB / (4 * GIB), 0)
    trace = _Trace(tracker)
    trace.alloc("j", 300 * MIB)
    trace.free("j")
    trace.alloc("k", 100 * MIB)
    assert trace.steps[-1] == (100 * MIB, 100 * MIB)
    assert trace.counters() == {
        "num_alloc_retries": 0,
        "num_ooms": 0,
        "num_sync_all_streams": 0,
        "num_device_alloc": 2,
        "num_device_free": 1,
    }

    # Without max_split_size_mb the cached 300 MiB block is split instead.
    tracker = _tracker(version=version, total=4 * GIB)
    tracker.set_memory_fraction(350 * MIB / (4 * GIB), 0)
    trace = _Trace(tracker)
    trace.alloc("j", 300 * MIB)
    trace.free("j")
    trace.alloc("k", 100 * MIB)
    assert trace.steps[-1] == (100 * MIB, 300 * MIB)
    assert trace.counters()["num_device_alloc"] == 1


def _oom_message(version, conf=None):
    import torch

    tracker = _tracker(conf, version=version, total=4 * GIB)
    tracker.set_memory_fraction(256 * MIB / (4 * GIB), 0)
    trace = _Trace(tracker)
    trace.alloc("a", 100 * MIB)
    trace.alloc("b", 100 * MIB)
    with pytest.raises(torch.OutOfMemoryError) as excinfo:
        trace.alloc("c", 100 * MIB)
    assert trace.counters() == {
        "num_alloc_retries": 1,
        "num_ooms": 1,
        "num_sync_all_streams": 1,
        "num_device_alloc": 2,
        "num_device_free": 0,
    }
    return str(excinfo.value)


HEAD = (
    "CUDA out of memory. Tried to allocate 100.00 MiB. GPU 0 has a total "
    "capacity of 4.00 GiB of which 3.80 GiB is free. 256.00 MiB allowed; Of "
    "the allocated memory 200.00 MiB is allocated by PyTorch, and 0 bytes is "
    "reserved by PyTorch but unallocated."
)
HINT = (
    " If reserved but unallocated memory is large try setting "
    "{var}=expandable_segments:True to avoid fragmentation.  See "
    "documentation for Memory Management  ({url})"
)
OLD_URL = "https://pytorch.org/docs/stable/notes/cuda.html#environment-variables"
NEW_URL = (
    "https://docs.pytorch.org/docs/stable/notes/cuda.html"
    "#optimizing-memory-usage-with-pytorch-cuda-alloc-conf"
)


@pytest.mark.parametrize(
    ("version", "conf", "tail"),
    [
        ((2, 9), None, HINT.format(var="PYTORCH_CUDA_ALLOC_CONF", url=OLD_URL)),
        (
            (2, 9),
            "expandable_segments:True",
            HINT.format(var="PYTORCH_CUDA_ALLOC_CONF", url=OLD_URL),
        ),
        ((2, 10), None, HINT.format(var="PYTORCH_ALLOC_CONF", url=OLD_URL)),
        ((2, 13), None, HINT.format(var="PYTORCH_CUDA_ALLOC_CONF", url=NEW_URL)),
        ((2, 14), None, HINT.format(var="PYTORCH_CUDA_ALLOC_CONF", url=NEW_URL)),
        ((2, 14), "expandable_segments:True", ""),
    ],
)
def test_out_of_memory_message_per_version(version, conf, tail) -> None:
    # The real message also says how much memory the process holds when
    # NVML is available; FakeCUDA leaves that sentence out.
    assert _oom_message(version, conf) == HEAD + tail


@pytest.mark.parametrize("version", SHARED_VERSIONS)
def test_streams_keep_separate_free_lists_and_record_stream_delays_reuse(
    version,
) -> None:
    tracker = _tracker(version=version)
    trace = _Trace(tracker)
    trace.alloc("a", 50 * MIB, stream=7)
    trace.free("a")
    # A block cached for stream 7 is not reused on the default stream.
    trace.alloc("b", 50 * MIB)
    assert trace.steps[-1] == (50 * MIB, 100 * MIB)
    segments = tracker.allocator_snapshot()
    assert sorted(segment["stream"] for segment in segments) == [0, 7]
    trace.free("b")

    key = trace.alloc("c", 30 * MIB)
    tracker.record_stream(key, 7)
    trace.free("c")
    stats = tracker.memory_stats(0)
    assert stats["allocated_bytes.all.current"] == 0
    assert stats["active_bytes.all.current"] == 30 * MIB
    assert stats["active.all.current"] == 1
    assert stats["requested_bytes.all.current"] == 30 * MIB
    default = next(s for s in tracker.allocator_snapshot() if s["stream"] == 0)
    assert [(b["size"], b["state"]) for b in default["blocks"]] == [
        (30 * MIB, "active_pending_free"),
        (20 * MIB, "inactive"),
    ]

    # The next allocation processes the completed event and reuses it.
    trace.alloc("d", 30 * MIB)
    assert trace.steps[-1] == (30 * MIB, 100 * MIB)
    trace.free("d")
    key = trace.alloc("e", 4000)
    tracker.record_stream(key, 7)
    trace.free("e")
    trace.empty_cache()
    assert trace.steps[-1] == (0, 0)
    assert trace.counters() == {
        "num_alloc_retries": 0,
        "num_ooms": 0,
        "num_sync_all_streams": 1,
        "num_device_alloc": 3,
        "num_device_free": 3,
    }


def test_expandable_segments_are_per_stream() -> None:
    tracker = _tracker("expandable_segments:True")
    trace = _Trace(tracker)
    trace.alloc("a", 50 * MIB, stream=7)
    trace.free("a")
    trace.alloc("b", 50 * MIB)
    assert trace.steps[-1] == (50 * MIB, 120 * MIB)
    assert sorted(
        (segment["stream"], segment["total_size"])
        for segment in tracker.allocator_snapshot()
    ) == [(0, 60 * MIB), (7, 60 * MIB)]


# (PYTORCH_CUDA_ALLOC_CONF, 2.9 error, 2.10+ error); None means accepted.
CONFIG_ERRORS = [
    (
        "max_split_size_mb:1",
        "CachingAllocator option max_split_size_mb too small, must be > 20",
        "CachingAllocator option max_split_size_mb too small, must be >= 20",
    ),
    (
        "max_split_size_mb:20",
        "CachingAllocator option max_split_size_mb too small, must be > 20",
        None,
    ),
    (
        "bogus_key:3",
        "Unrecognized CachingAllocator option: bogus_key",
        "Unrecognized key 'bogus_key' in CUDA allocator config.",
    ),
    (
        "expandable_segments:true",
        "Expected a single True/False argument for expandable_segments",
        "Expected 'True' or 'False' at index 2 in ConfigTokenizer but got 'true'",
    ),
    (
        "garbage_collection_threshold:1.5",
        "garbage_collect_threshold too big, set it 0.0~1.0",
        "garbage_collect_threshold is invalid, set it in (0.0, 1.0)",
    ),
    (
        "garbage_collection_threshold:0",
        "garbage_collect_threshold too small, set it 0.0~1.0",
        "garbage_collect_threshold is invalid, set it in (0.0, 1.0)",
    ),
    (
        "roundup_power2_divisions:3",
        "For roundups, the divisions has to be power of 2 ",
        "For roundups, the divisions has to be power of 2 ",
    ),
    (
        "roundup_power2_divisions:[100:4]",
        "For roundups, the intervals have to be power of 2 ",
        "For roundups, the intervals have to be power of 2 ",
    ),
    (
        "backend:foo",
        "Unknown allocator backend, options are native and cudaMallocAsync",
        "Unknown allocator backend, options are native and cudaMallocAsync",
    ),
    (
        "max_non_split_rounding_mb:5",
        "CachingAllocator option max_non_split_rounding_mb too small, must be > 20",
        "CachingAllocator option max_non_split_rounding_mb too small, must be >= 20",
    ),
    (
        "pinned_num_register_threads:3",
        "Number of register threads has to be power of 2 ",
        "Number of register threads has to be power of 2, got 3",
    ),
    ("max_split_size_mb: 100", None, None),
    ("release_lock_on_cudamalloc:True", None, None),
]


@pytest.mark.parametrize(("conf", "legacy_error", "current_error"), CONFIG_ERRORS)
def test_allocator_config_errors_follow_the_installed_torch(
    conf, legacy_error, current_error
) -> None:
    for version, error, error_type in (
        ((2, 9), legacy_error, RuntimeError),
        ((2, 10), current_error, ValueError),
        ((2, 14), current_error, ValueError),
    ):
        tracker = _tracker(conf, version=version)
        if error is None:
            tracker.check_initialized()
            assert tracker.allocate(1, 4, 0)
            continue
        # Raised at CUDA initialization and again on every allocation.
        with pytest.raises(error_type) as excinfo:
            tracker.check_initialized()
        assert type(excinfo.value) is error_type
        assert str(excinfo.value) == error
        with pytest.raises(error_type, match="^" + _escape(error) + "$"):
            tracker.allocate(1, 4, 0)


def _escape(text):
    import re

    return re.escape(text)


def test_allocator_config_number_parsing_errors() -> None:
    with pytest.raises(RuntimeError, match="^stoi$"):
        _tracker("max_split_size_mb:abc", version=(2, 9)).check_initialized()
    with pytest.raises(RuntimeError, match="^stoull$"):
        _tracker("max_split_size_mb:abc", version=(2, 14)).check_initialized()
    with pytest.raises(
        RuntimeError, match="^Error parsing CachingAllocator settings, expected :$"
    ):
        _tracker("max_split_size_mb", version=(2, 9)).check_initialized()


def test_large_segment_size_is_a_key_from_torch_2_11() -> None:
    for version in ((2, 9), (2, 10)):
        with pytest.raises((RuntimeError, ValueError), match="large_segment_size_mb"):
            _tracker("large_segment_size_mb:40", version=version).check_initialized()
    with pytest.raises(
        ValueError,
        match=r"^CachingAllocator option large_segment_size_mb must be > 10 MB$",
    ):
        _tracker("large_segment_size_mb:10", version=(2, 14)).check_initialized()
    # The split limit follows the large segment size.
    with pytest.raises(ValueError, match="must be >= 40$"):
        _tracker(
            "large_segment_size_mb:40,max_split_size_mb:30", version=(2, 14)
        ).check_initialized()
    tracker = _tracker("large_segment_size_mb:40", version=(2, 14))
    tracker.allocate(1, 5 * MIB, 0)
    assert tracker.memory_reserved(0) == 40 * MIB


def test_unified_alloc_conf_variable_from_torch_2_10() -> None:
    # Up to 2.9 only PYTORCH_CUDA_ALLOC_CONF configures the CUDA allocator.
    environ = {"PYTORCH_ALLOC_CONF": "max_split_size_mb:100"}
    legacy = _tracker(version=(2, 9), environ=environ)
    assert legacy.memory_stats(0)["max_split_size"] == -1
    current = _tracker(version=(2, 10), environ=environ)
    assert current.memory_stats(0)["max_split_size"] == 100 * MIB
    bogus = {"PYTORCH_ALLOC_CONF": "bogus_key:3"}
    _tracker(version=(2, 9), environ=bogus).check_initialized()
    with pytest.raises(ValueError, match="bogus_key"):
        _tracker(version=(2, 10), environ=bogus).check_initialized()
    # PYTORCH_CUDA_ALLOC_CONF wins over the unified name.
    both = {
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:200",
        "PYTORCH_ALLOC_CONF": "max_split_size_mb:100",
    }
    assert (
        _tracker(version=(2, 14), environ=both).memory_stats(0)["max_split_size"]
        == 200 * MIB
    )


def test_per_process_memory_fraction_key_from_torch_2_10() -> None:
    tracker = _tracker("per_process_memory_fraction:0.25", version=(2, 14), total=GIB)
    assert tracker.get_memory_fraction(0) == 0.25
    with pytest.raises(RuntimeError, match="Unrecognized CachingAllocator option"):
        _tracker("per_process_memory_fraction:0.25", version=(2, 9)).check_initialized()


def test_memory_fraction_semantics_per_version() -> None:
    # Up to 2.9 any call caps cudaMalloc, also 1.0; later 1.0 lifts the cap.
    legacy = _tracker(version=(2, 9), total=10)
    legacy.set_memory_fraction(1.0, 0)
    assert legacy._devices[0].allowed_memory_maximum == 10
    current = _tracker(version=(2, 14), total=10)
    current.set_memory_fraction(1.0, 0)
    assert current._devices[0].allowed_memory_maximum is None
    # 2.13 switched the cap from truncation to std::round.
    for version, expected in (((2, 9), 2), ((2, 12), 2), ((2, 13), 3)):
        tracker = _tracker(version=version, total=10)
        tracker.set_memory_fraction(0.25, 0)
        assert tracker._devices[0].allowed_memory_maximum == expected


def test_memory_stats_keys_follow_the_installed_torch() -> None:
    legacy = list(_tracker(version=(2, 9)).memory_stats_as_nested_dict(0))
    assert legacy == [
        "num_alloc_retries", "num_ooms", "max_split_size", "num_sync_all_streams",
        "num_device_alloc", "num_device_free", "allocation", "segment", "active",
        "inactive_split", "allocated_bytes", "reserved_bytes", "active_bytes",
        "inactive_split_bytes", "requested_bytes", "oversize_allocations",
        "oversize_segments",
    ]
    current = list(_tracker(version=(2, 14)).memory_stats_as_nested_dict(0))
    assert current == (
        legacy[:6]
        + ["num_oom_rejections"]
        + legacy[6:12]
        + ["reserved_bytes_by_private_pools"]
        + legacy[12:]
    )
    assert "num_oom_rejections" in _tracker(version=(2, 12)).memory_stats(0)
    assert "num_oom_rejections" not in _tracker(version=(2, 11)).memory_stats(0)


def test_host_memory_stats_keys_follow_the_installed_torch() -> None:
    assert list(_tracker(version=(2, 9)).host_memory_stats()) == [
        "num_host_alloc", "num_host_free", "allocation", "segment",
        "allocated_bytes", "reserved_bytes", "host_alloc_time", "host_free_time",
    ]
    assert list(_tracker(version=(2, 10)).host_memory_stats()) == [
        "num_host_alloc", "num_host_free", "allocations", "active_requests",
        "allocated_bytes", "active_bytes", "host_alloc_time", "host_free_time",
    ]


def test_mem_get_info_subtracts_the_modeled_cuda_context(monkeypatch) -> None:
    monkeypatch.delenv("FAKEGPU_CUDA_CONTEXT_BYTES", raising=False)
    tracker = _DeviceMemoryTracker([4 * GIB], torch_version=(2, 14))
    assert tracker.context_bytes == 630_718_464
    tracker.allocate(1, 4000, 0)
    assert tracker.mem_get_info(0) == (4 * GIB - 2 * MIB - 630_718_464, 4 * GIB)

    monkeypatch.setenv("FAKEGPU_CUDA_CONTEXT_BYTES", str(100 * MIB))
    tracker = _DeviceMemoryTracker([4 * GIB], torch_version=(2, 14))
    assert tracker.mem_get_info(0) == (4 * GIB - 100 * MIB, 4 * GIB)
    # Free memory never goes below zero on small profiles.
    small = _DeviceMemoryTracker([64 * MIB], torch_version=(2, 14))
    assert small.mem_get_info(0) == (0, 64 * MIB)


def _history_run(version, context="all"):
    tracker = _tracker(version=version)
    tracker.record_memory_history("all", context, "all", 1000, False)
    tracker.allocate(1, 4000, 0)
    tracker.allocate(2, 25 * MIB, 0)
    tracker.release(1)
    tracker.empty_cache()
    return tracker, tracker.snapshot_dict(None)


@pytest.mark.parametrize(
    ("version", "trace_keys", "top_keys"),
    [
        (
            (2, 9),
            ["action", "addr", "size", "stream", "time_us", "compile_context"],
            ["segments", "device_traces", "allocator_settings", "external_annotations"],
        ),
        (
            (2, 14),
            [
                "action", "addr", "size", "stream", "time_us", "compile_context",
                "user_metadata", "pool_id",
            ],
            [
                "segments", "device_traces", "allocator_settings",
                "external_annotations", "host_segments", "host_traces",
            ],
        ),
    ],
)
def test_memory_history_records_allocator_events(version, trace_keys, top_keys) -> None:
    tracker, snapshot = _history_run(version)
    assert list(snapshot) == top_keys
    entries = snapshot["device_traces"][0]
    assert [(entry["action"], entry["size"], entry["stream"]) for entry in entries] == [
        ("segment_alloc", 2 * MIB, 0),
        ("alloc", 4000, 0),
        ("segment_alloc", 26 * MIB, 0),
        ("alloc", 25 * MIB, 0),
        ("free_requested", 4000, 0),
        ("free_completed", 4000, 0),
        ("segment_free", 2 * MIB, 0),
    ]
    # Frames are added last, and only when a context was recorded.
    assert list(entries[0]) == trace_keys + ["frames"]
    assert entries[1]["frames"][0]["name"] == "_history_run"
    assert set(entries[1]["frames"][0]) == {"name", "filename", "line"}
    segment = snapshot["segments"][0]
    assert list(segment)[-2:] == ["blocks", "frames"]
    assert list(segment["blocks"][0]) == [
        "address", "size", "requested_size", "state", "frames",
    ]
    settings = snapshot["allocator_settings"]
    assert settings["PYTORCH_CUDA_ALLOC_CONF"] == ""
    assert settings["max_split_size"] == -1
    assert settings["roundup_power2_divisions"] == {
        str(1 << index): 0 for index in range(16)
    }
    assert ("max_round_threshold" in settings) == (version >= (2, 13))

    # The traces are read before the snapshot records its own entry.
    again = tracker.snapshot_dict(None)["device_traces"][0]
    assert again[-1]["action"] == "snapshot"
    assert again[-1]["addr"] == 0

    tracker.record_memory_history(None, None, "all", 0, False)
    assert tracker.snapshot_dict(None)["device_traces"] == [[]]
    assert not tracker.is_history_enabled()


def test_memory_history_without_context_has_no_frames() -> None:
    tracker = _tracker(version=(2, 9))
    tracker.record_memory_history("all", None, "all", 10, False)
    tracker.allocate(1, 4000, 0)
    entries = tracker.snapshot_dict(None)["device_traces"][0]
    assert [list(entry) for entry in entries] == [
        ["action", "addr", "size", "stream", "time_us", "compile_context"]
    ] * 2
    assert [entry["action"] for entry in entries] == ["segment_alloc", "alloc"]
    # "state" keeps a single entry; skip_actions drops actions.
    tracker.record_memory_history("state", "all", "all", 10, True)
    tracker.allocate(2, 4000, 0)
    assert len(tracker.snapshot_dict(None)["device_traces"][0]) == 1
    tracker.record_memory_history(
        "all", "all", "all", 10, True, False, False, ["alloc"]
    )
    tracker.allocate(3, 4000, 0)
    assert tracker.snapshot_dict(None)["device_traces"][0] == []
    with pytest.raises(RuntimeError, match="^Unknown TraceEntry action: nope$"):
        tracker.record_memory_history("all", "all", "all", 10, True, False, False, ["nope"])
    with pytest.raises(RuntimeError, match="expected stacks to be 'python', or 'all'"):
        tracker.record_memory_history("all", "all", "cpp", 10, True)


def test_memory_history_ring_buffer_keeps_the_newest_entries() -> None:
    tracker = _tracker(version=(2, 14))
    tracker.record_memory_history("all", None, "all", 3, False)
    for key in range(1, 4):
        tracker.allocate(key, 4000, 0)
    entries = tracker.snapshot_dict((0, 0, True))["device_traces"][0]
    assert [entry["action"] for entry in entries] == ["alloc", "alloc", "alloc"]
    assert [entry["addr"] for entry in entries] == sorted(
        entry["addr"] for entry in entries
    )


def test_snapshot_argument_checks_per_version() -> None:
    legacy = _tracker(version=(2, 9))
    assert legacy.snapshot_dict((0, 0))["segments"] == []
    with pytest.raises(RuntimeError, match="^mempool_id must be a tuple of 2 integers$"):
        legacy.snapshot_dict((1, 2, 3, 4))
    current = _tracker(version=(2, 14))
    current.allocate(1, 4000, 0)
    assert len(current.snapshot_dict((0, 0, False))["segments"]) == 1
    assert current.snapshot_dict((0, 0, False))["device_traces"] == []
    # Only the default pool exists, so a private pool has no segments.
    assert current.snapshot_dict((1, 2))["segments"] == []
    with pytest.raises(RuntimeError, match="^Expected tuple of size 2 or 3$"):
        current.snapshot_dict((1, 2, 3, 4))
    with pytest.raises(RuntimeError, match="^Expected tuple or None$"):
        current.snapshot_dict([0, 0])


def test_raw_alloc_and_delete() -> None:
    tracker = _tracker()
    pointer = tracker.raw_alloc(1000, 0, 0)
    assert isinstance(pointer, int) and pointer > 0
    assert tracker.memory_allocated(0) == 1024
    tracker.raw_delete(pointer)
    assert tracker.memory_allocated(0) == 0
    assert tracker.raw_alloc(0, 0, 0) == 0
    tracker.raw_delete(0)
    with pytest.raises(RuntimeError, match="^invalid device pointer: 0x"):
        tracker.raw_delete(pointer)
    pointer = tracker.raw_alloc(2 * MIB, 5, 0)
    (segment,) = [s for s in tracker.allocator_snapshot() if s["stream"] == 5]
    assert (segment["segment_type"], segment["total_size"]) == ("large", 20 * MIB)


def test_set_allocator_settings_at_runtime() -> None:
    tracker = _tracker(version=(2, 14))
    tracker.set_allocator_settings("expandable_segments:True")
    tracker.allocate(1, 25 * MIB + 3, 0)
    assert tracker.memory_reserved(0) == 40 * MIB
    assert tracker.allocator_settings() == "expandable_segments:True"
    # The runtime parser checks the accelerator keys first.
    with pytest.raises(
        ValueError, match="^Unrecognized key 'bogus' in Accelerator allocator config.$"
    ):
        tracker.set_allocator_settings("bogus:1")
    with pytest.raises(RuntimeError, match="Allocator backend parsed at runtime"):
        tracker.set_allocator_settings("backend:cudaMallocAsync")
