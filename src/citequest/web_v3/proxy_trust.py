"""Explicit network trust for a private API behind its HTTPS load balancer."""
from ipaddress import ip_network


def proxy_networks(values: list[str]) -> list[str]:
    if not isinstance(values, list) or len(values) > 8:
        raise ValueError("configure at most eight explicit proxy networks")
    result = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("proxy trust requires IP addresses or canonical CIDRs")
        try:
            network = str(ip_network(value, strict=True))
        except ValueError:
            raise ValueError("proxy trust requires IP addresses or canonical CIDRs, never wildcards") from None
        if network not in result:
            result.append(network)
    return result
