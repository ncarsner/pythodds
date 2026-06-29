#!/usr/bin/env python3
"""Command-line utility for subnet mask calculations.

Given an IPv4 address with optional CIDR prefix length, computes the network
address, broadcast address, first and last usable IP, subnet mask, usable
host count, and number of classful networks.

Usage examples:
  subnet 192.168.1.50/24
  subnet 10.0.0.1 --cidr 8
  subnet 172.16.5.100/20
  subnet 192.168.1.50/24 --format json
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


class SubnetResult(NamedTuple):
    input_address: str
    cidr: int
    network_address: str
    broadcast_address: str
    first_ip: str
    last_ip: str
    subnet_mask: str
    hosts: int
    networks: int


def _classful_prefix(ip: ipaddress.IPv4Address) -> int:
    """Return classful prefix length: 8 (Class A), 16 (Class B), 24 (Class C/D/E)."""
    first_octet = int(ip) >> 24
    if first_octet < 128:
        return 8
    if first_octet < 192:
        return 16
    return 24


def compute_subnet(ip_str: str, cidr: int | None = None) -> SubnetResult:
    """Compute subnet information for a given IPv4 address and prefix length.

    Args:
        ip_str: IPv4 address, optionally with CIDR notation (e.g. "192.168.1.50/24").
        cidr: Prefix length override (0–32); supersedes any CIDR embedded in ip_str.

    Returns:
        SubnetResult with all computed fields.

    Raises:
        ValueError: If the address or prefix length is invalid.
    """
    if "/" in ip_str:
        raw_ip, suffix = ip_str.split("/", 1)
        try:
            embedded_cidr: int | None = int(suffix)
        except ValueError:
            raise ValueError(f"invalid CIDR prefix: {suffix!r}")
    else:
        raw_ip = ip_str
        embedded_cidr = None

    try:
        ip = ipaddress.IPv4Address(raw_ip)
    except ipaddress.AddressValueError as exc:
        raise ValueError(f"invalid IPv4 address: {raw_ip!r}") from exc

    prefix = (
        cidr
        if cidr is not None
        else (embedded_cidr if embedded_cidr is not None else 32)
    )

    if not 0 <= prefix <= 32:
        raise ValueError(f"CIDR prefix must be 0–32, got {prefix}")

    network = ipaddress.IPv4Network(f"{raw_ip}/{prefix}", strict=False)

    subnet_mask = str(network.netmask)
    network_addr = str(network.network_address)
    broadcast_addr = str(network.broadcast_address)

    if prefix == 32:
        first_ip = last_ip = str(ip)
        hosts = 1
    elif prefix == 31:
        # RFC 3021: both addresses usable on point-to-point links
        first_ip = network_addr
        last_ip = broadcast_addr
        hosts = 2
    else:
        first_ip = str(network.network_address + 1)
        last_ip = str(network.broadcast_address - 1)
        hosts = network.num_addresses - 2

    classful = _classful_prefix(network.network_address)
    networks = 2 ** (prefix - classful) if prefix >= classful else 1

    return SubnetResult(
        input_address=str(ip),
        cidr=prefix,
        network_address=network_addr,
        broadcast_address=broadcast_addr,
        first_ip=first_ip,
        last_ip=last_ip,
        subnet_mask=subnet_mask,
        hosts=hosts,
        networks=networks,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subnet mask calculator: network, broadcast, host range, mask, and counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  subnet 192.168.1.50/24
  subnet 10.0.0.1 --cidr 8
  subnet 172.16.5.100/20 --format json
""",
    )
    parser.add_argument(
        "address",
        metavar="ADDRESS",
        help="IPv4 address with optional CIDR notation (e.g. 192.168.1.50/24)",
    )
    parser.add_argument(
        "--cidr",
        "-c",
        type=int,
        metavar="PREFIX",
        help="prefix length 0–32; overrides any CIDR embedded in ADDRESS (default: 32)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="output format: table (default) or json",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_table(result: SubnetResult) -> str:
    w = 18
    lines = [
        "Subnet Information",
        "=" * 40,
        f"{'Input:':<{w}} {result.input_address}/{result.cidr}",
        f"{'Network:':<{w}} {result.network_address}",
        f"{'Broadcast:':<{w}} {result.broadcast_address}",
        f"{'First IP:':<{w}} {result.first_ip}",
        f"{'Last IP:':<{w}} {result.last_ip}",
        f"{'Subnet mask:':<{w}} {result.subnet_mask}",
        f"{'Hosts:':<{w}} {result.hosts:,}",
        f"{'Networks:':<{w}} {result.networks:,}",
    ]
    return "\n".join(lines)


def format_json(result: SubnetResult) -> str:
    return json.dumps(
        {
            "input_address": result.input_address,
            "cidr": result.cidr,
            "network_address": result.network_address,
            "broadcast_address": result.broadcast_address,
            "first_ip": result.first_ip,
            "last_ip": result.last_ip,
            "subnet_mask": result.subnet_mask,
            "hosts": result.hosts,
            "networks": result.networks,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        result = compute_subnet(args.address, cidr=args.cidr)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(format_json(result))
    else:
        print(format_table(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
