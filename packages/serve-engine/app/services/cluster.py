"""Cluster health for L.A.I.L Status (source of truth).

Default topology is this machine as probed (hostname, LAN, Tailscale, RoCE)
plus any RoCE peers that answer ping (ARP table, or a /24-or-tighter QSFP
scan when the table is cold after reboot). LAIL_CLUSTER_JSON or
gitignored data/cluster.json still override.

Probes:
  - local node via metadata + live NICs
  - remote nodes via passwordless SSH + a small JSON collector
  - QSFP fabric reachability between discovered interconnect IPs
  - whether a model is loaded on each node and whether multi-node looks aligned
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import platform
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import metadata

log = logging.getLogger(__name__)

# Used only if live NIC/hostname probes fail.
_FALLBACK_CLUSTER = {
    "name": "local",
    "nodes": [
        {
            "id": "local",
            "label": "this host",
            "role": "head",
            "local": True,
            "vllm_url": "http://127.0.0.1:8000",
        },
    ],
}

_SKIP_IFACES = {"lo", "docker0", "tailscale0"}
_SKIP_IFACE_PREFIXES = ("br-", "veth", "virbr", "cni", "flannel", "wg")
_DEFAULT_VLLM_URL = "http://127.0.0.1:8000"


def _node_vllm_url(node: dict[str, Any] | None = None) -> str:
    """Live serve URL for a node. LAB_BASE_URL, then :8000, when unset."""
    if node:
        url = str(node.get("vllm_url") or "").strip()
        if url:
            return url
    env = (os.environ.get("LAB_BASE_URL") or "").strip()
    return env or _DEFAULT_VLLM_URL
