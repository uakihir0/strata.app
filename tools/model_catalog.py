"""Model files and family metadata shared by the macOS setup and MCP surfaces."""

HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "b22d729eae29b5796f76fb70f91aef549b9fc52c",
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",
}


def hf(repo: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{HF_REVISIONS[repo]}/"


MODELS = {
    "Q2_0": {"about": "2-bit quantization, fastest", "download_gb": 66.4, "families": ("qwen",)},
    "IQ2_XS": {"about": "2-bit importance quantization", "download_gb": 68.0},
    "IQ3_XXS": {"about": "3-bit importance quantization", "download_gb": 75.8},
    "IQ3_S": {"about": "3.5-bit importance quantization, highest quality", "download_gb": 83.6,
              "families": ("qwen",)},
    "IQ1_M": {"about": "Coder quantization with half the original experts", "download_gb": 58.4,
              "families": ("coder",)},
    "UD-Q4_K_XL": {"about": "Experimental Unsloth 4-bit quantization; unavailable in the macOS port",
                   "download_gb": 111.3, "families": ("unsloth",), "experimental": True},
}

CONTEXTS = [8192, 32768, 65536, 131072, 262144, 393216, 524288]

FAMILIES = {
    "qwen": {
        "title": "Qwen3.8-Flash-Next",
        "by": "Qwen; GSQ-RCO quants by ISTA-DASLab",
        "about": "the original model",
        "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF") + "{q}/",
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
        "tag": "",
        "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
        "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "name": "qwen3.8-flash-next",
    },
    "swift": {
        "title": "Swift 1.5",
        "by": "UkisAI fine-tune of Qwen3.8-Flash-Next",
        "about": "a fine-tune that uses shorter reasoning by its authors' report",
        "hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
        "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
        "tag": "swift-",
        "mmproj_hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
        "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf",
        "name": "swift-1.5",
        "license": "Swift Open License 1.0",
    },
    "coder": {
        "title": "Qwen3.8-Flash-Next Coder",
        "by": "ISTA-DASLab",
        "about": "a version with half the experts, selected for coding and tool use",
        "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF") + "{q}/",
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
        "tag": "coder-",
        "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"),
        "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "name": "qwen3.8-flash-next-coder",
        "profile": "expert-profile-coder.bin",
    },
    "unsloth": {
        "title": "Qwen3.8-Flash-Next (Unsloth)",
        "by": "Unsloth",
        "about": "experimental 4-bit quantization; not supported by the macOS port",
        "hf": hf("unsloth/Qwen3.8-Flash-Next-GGUF") + "{q}/",
        "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf",
        "shards": 4,
        "tag": "unsloth-",
        "experimental": True,
        "vision": False,
    },
}
