"""
Upstream computation for the security gate (src/security_gate.py): a real
dependency-vulnerability scan, queried from the public OSV database
(https://osv.dev) for three pinned dependency sets, and frozen to JSON so the
gate's tests run against a fixed, reproducible snapshot rather than a live
feed that changes daily.

  - requirements_legacy_web.txt      old jinja2 / requests / urllib3 pins
  - requirements_legacy_imaging.txt  old pyyaml / pillow pins
  - requirements-ci.txt (repo root)  this project's own CI dependencies

Only GitHub Security Advisory (GHSA) records are kept, because those carry a
severity label (CRITICAL / HIGH / MODERATE / LOW) in database_specific.severity;
OSV's PYSEC records for the same vulnerabilities are aliases of the GHSA ones
and carry no label, so keeping both would double-count.

This is the security-gate analogue of computing fairness metrics from a trained
model: the gate consumes the scan, it does not perform it.
"""
import datetime
import json
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "security_scans"
SETS = {
    "legacy_web": OUT_DIR / "requirements_legacy_web.txt",
    "legacy_imaging": OUT_DIR / "requirements_legacy_imaging.txt",
    "project_ci": PROJECT_ROOT / "requirements-ci.txt",
}


def read_pins(path):
    pins = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            pins.append((name.strip(), version.strip()))
    return pins


def query_osv(pins):
    body = {"queries": [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in pins]}
    req = urllib.request.Request("https://api.osv.dev/v1/querybatch", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    batch = json.load(urllib.request.urlopen(req, timeout=60))["results"]
    findings = []
    for (name, version), res in zip(pins, batch):
        vulns = list(res.get("vulns", []))
        token = res.get("next_page_token")
        while token:  # querybatch paginates per query; fetch the remaining pages
            more_req = urllib.request.Request(
                "https://api.osv.dev/v1/query",
                data=json.dumps({"package": {"name": name, "ecosystem": "PyPI"}, "version": version, "page_token": token}).encode(),
                headers={"Content-Type": "application/json"})
            more = json.load(urllib.request.urlopen(more_req, timeout=60))
            vulns += more.get("vulns", [])
            token = more.get("next_page_token")
        seen = set()
        for v in vulns:
            if v["id"] in seen:
                continue
            seen.add(v["id"])
            if not v["id"].startswith("GHSA-"):
                continue
            full = json.load(urllib.request.urlopen(f"https://api.osv.dev/v1/vulns/{v['id']}", timeout=60))
            findings.append({
                "id": v["id"],
                "package": name,
                "version": version,
                "severity": (full.get("database_specific") or {}).get("severity", "UNKNOWN"),
                "aliases": full.get("aliases", []),
            })
    return findings


def main():
    today = datetime.date.today().isoformat()
    for key, path in SETS.items():
        pins = read_pins(path)
        findings = query_osv(pins)
        report = {"scanner": "OSV.dev querybatch (GHSA records only)", "snapshot_date": today,
                  "dependency_file": path.relative_to(PROJECT_ROOT).as_posix(),
                  "packages": [f"{n}=={v}" for n, v in pins], "findings": findings}
        out = OUT_DIR / f"scan_{key}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        counts = {}
        for f in findings:
            counts[f["severity"]] = counts.get(f["severity"], 0) + 1
        print(f"{key}: {len(findings)} findings {counts} -> {out.name}")


if __name__ == "__main__":
    main()
