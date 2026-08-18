import sys
import types

import pytest

from medsel.utils.device import pick_device, resolve_dtype, supports_bf16


def fake_torch(capability=(8, 0), *, cuda=True, bf16_flag=True):
    """A stand-in for torch reporting a chosen GPU generation.

    The hardware has to be simulated because the flag under test is precisely the one that
    misreports on Pascal, and no dev machine here has a card old enough to reproduce it.
    """
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda,
            is_bf16_supported=lambda: bf16_flag,
            get_device_capability=lambda index=0: capability,
        ),
        backends=types.SimpleNamespace(mps=None),
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
    )


@pytest.fixture
def gpu(monkeypatch):
    def install(capability, *, bf16_flag=True, cuda=True):
        monkeypatch.setitem(
            sys.modules, "torch", fake_torch(capability, cuda=cuda, bf16_flag=bf16_flag)
        )

    return install


class TestSupportsBf16:
    def test_rejects_pascal_even_though_torch_claims_support(self, gpu):
        gpu((6, 1), bf16_flag=True)
        assert supports_bf16("cuda") is False

    def test_rejects_turing(self, gpu):
        gpu((7, 5))
        assert supports_bf16("cuda") is False

    def test_accepts_ampere(self, gpu):
        gpu((8, 0))
        assert supports_bf16("cuda") is True

    def test_false_on_cpu(self, gpu):
        gpu((8, 0), cuda=False)
        assert supports_bf16("cpu") is False


class TestResolveDtype:
    @pytest.mark.parametrize(
        ("capability", "expected"),
        [((6, 1), "float32"), ((7, 0), "float16"), ((7, 5), "float16"), ((8, 0), "bfloat16")],
    )
    def test_auto_follows_the_generation(self, gpu, capability, expected):
        gpu(capability)
        assert resolve_dtype("auto", "cuda") == expected

    def test_auto_on_cpu_is_fp32(self, gpu):
        gpu((8, 0), cuda=False)
        assert resolve_dtype("auto", "cpu") == "float32"

    def test_explicit_request_overrides_the_hardware(self, gpu):
        gpu((6, 1))
        assert resolve_dtype("bf16", "cuda") == "bfloat16"

    def test_unknown_dtype_raises(self, gpu):
        gpu((8, 0))
        with pytest.raises(ValueError, match="unknown dtype"):
            resolve_dtype("float8", "cuda")


class TestPickDevice:
    def test_prefers_cuda(self, gpu):
        gpu((8, 0))
        assert pick_device() == "cuda"

    def test_falls_back_to_cpu(self, gpu):
        gpu((8, 0), cuda=False)
        assert pick_device() == "cpu"
