"""Tests for subnet mask calculator utility."""

import ipaddress
import json

import pytest

from src.utils.subnet import (
    _classful_prefix,
    compute_subnet,
    format_json,
    format_table,
    main,
)

# ---------------------------------------------------------------------------
# _classful_prefix
# ---------------------------------------------------------------------------


def test_classful_prefix_class_a():
    assert _classful_prefix(ipaddress.IPv4Address("10.0.0.1")) == 8


def test_classful_prefix_class_a_boundary():
    assert _classful_prefix(ipaddress.IPv4Address("0.0.0.1")) == 8
    assert _classful_prefix(ipaddress.IPv4Address("127.255.255.255")) == 8


def test_classful_prefix_class_b():
    assert _classful_prefix(ipaddress.IPv4Address("172.16.0.1")) == 16


def test_classful_prefix_class_b_boundary():
    assert _classful_prefix(ipaddress.IPv4Address("128.0.0.1")) == 16
    assert _classful_prefix(ipaddress.IPv4Address("191.255.255.255")) == 16


def test_classful_prefix_class_c():
    assert _classful_prefix(ipaddress.IPv4Address("192.168.1.1")) == 24
    assert _classful_prefix(ipaddress.IPv4Address("192.0.0.1")) == 24


# ---------------------------------------------------------------------------
# compute_subnet — input parsing
# ---------------------------------------------------------------------------


def test_compute_subnet_cidr_embedded():
    r = compute_subnet("192.168.1.50/24")
    assert r.cidr == 24
    assert r.input_address == "192.168.1.50"


def test_compute_subnet_cidr_override_takes_precedence():
    r = compute_subnet("192.168.1.50/24", cidr=25)
    assert r.cidr == 25


def test_compute_subnet_bare_ip_defaults_to_32():
    r = compute_subnet("10.1.2.3")
    assert r.cidr == 32


def test_compute_subnet_bare_ip_with_cidr_arg():
    r = compute_subnet("10.0.0.1", cidr=8)
    assert r.cidr == 8


# ---------------------------------------------------------------------------
# compute_subnet — address fields
# ---------------------------------------------------------------------------


def test_compute_subnet_class_c_24():
    r = compute_subnet("192.168.1.50/24")
    assert r.network_address == "192.168.1.0"
    assert r.broadcast_address == "192.168.1.255"
    assert r.first_ip == "192.168.1.1"
    assert r.last_ip == "192.168.1.254"
    assert r.subnet_mask == "255.255.255.0"
    assert r.hosts == 254
    assert r.networks == 1


def test_compute_subnet_class_a_8():
    r = compute_subnet("10.0.0.0/8")
    assert r.subnet_mask == "255.0.0.0"
    assert r.hosts == 2**24 - 2
    assert r.networks == 1


def test_compute_subnet_class_b_16():
    r = compute_subnet("172.16.0.0/16")
    assert r.subnet_mask == "255.255.0.0"
    assert r.hosts == 2**16 - 2
    assert r.networks == 1


def test_compute_subnet_prefix_32():
    r = compute_subnet("192.168.1.1/32")
    assert r.first_ip == "192.168.1.1"
    assert r.last_ip == "192.168.1.1"
    assert r.hosts == 1
    assert r.network_address == "192.168.1.1"
    assert r.broadcast_address == "192.168.1.1"


def test_compute_subnet_prefix_31():
    r = compute_subnet("10.0.0.0/31")
    assert r.first_ip == "10.0.0.0"
    assert r.last_ip == "10.0.0.1"
    assert r.hosts == 2


# ---------------------------------------------------------------------------
# compute_subnet — network count
# ---------------------------------------------------------------------------


def test_networks_class_c_subnets():
    # /26 from class C (/24): 2^(26-24) = 4 networks
    r = compute_subnet("192.168.1.0/26")
    assert r.networks == 4
    assert r.hosts == 62


def test_networks_class_b_subnets():
    # /24 from class B (/16): 2^(24-16) = 256 networks
    r = compute_subnet("172.16.0.0/24")
    assert r.networks == 256


def test_networks_supernet():
    # /6 is smaller than class A /8 → 1 (supernet)
    r = compute_subnet("10.0.0.0/6")
    assert r.networks == 1


def test_networks_class_b_supernet():
    # /12 is smaller than class B /16 → 1
    r = compute_subnet("172.16.0.0/12")
    assert r.networks == 1


# ---------------------------------------------------------------------------
# compute_subnet — error handling
# ---------------------------------------------------------------------------


def test_invalid_ip_address():
    with pytest.raises(ValueError, match="invalid IPv4 address"):
        compute_subnet("999.999.999.999/24")


def test_invalid_embedded_cidr_non_numeric():
    with pytest.raises(ValueError, match="invalid CIDR prefix"):
        compute_subnet("192.168.1.1/abc")


def test_cidr_too_large():
    with pytest.raises(ValueError, match="CIDR prefix must be 0"):
        compute_subnet("192.168.1.1", cidr=33)


def test_cidr_negative():
    with pytest.raises(ValueError, match="CIDR prefix must be 0"):
        compute_subnet("192.168.1.1", cidr=-1)


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


def test_format_table_contains_all_fields():
    r = compute_subnet("192.168.1.50/24")
    out = format_table(r)
    assert "192.168.1.0" in out
    assert "192.168.1.255" in out
    assert "192.168.1.1" in out
    assert "192.168.1.254" in out
    assert "255.255.255.0" in out
    assert "254" in out
    assert "Subnet Information" in out


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------


def test_format_json_structure():
    r = compute_subnet("192.168.1.50/24")
    data = json.loads(format_json(r))
    assert data["network_address"] == "192.168.1.0"
    assert data["broadcast_address"] == "192.168.1.255"
    assert data["first_ip"] == "192.168.1.1"
    assert data["last_ip"] == "192.168.1.254"
    assert data["subnet_mask"] == "255.255.255.0"
    assert data["hosts"] == 254
    assert data["cidr"] == 24
    assert data["input_address"] == "192.168.1.50"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_table_output(capsys):
    rc = main(["192.168.1.50/24"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "192.168.1.0" in out


def test_main_json_output(capsys):
    rc = main(["192.168.1.50/24", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["network_address"] == "192.168.1.0"


def test_main_cidr_flag(capsys):
    rc = main(["10.0.0.1", "--cidr", "16"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "10.0.0.0" in out


def test_main_invalid_address_returns_2(capsys):
    rc = main(["not.valid/24"])
    assert rc == 2
    assert "Error" in capsys.readouterr().err


def test_main_invalid_cidr_returns_2(capsys):
    rc = main(["192.168.1.1", "--cidr", "99"])
    assert rc == 2
    assert "Error" in capsys.readouterr().err
