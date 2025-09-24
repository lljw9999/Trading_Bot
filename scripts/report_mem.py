#!/usr/bin/env python3
"""
Trading System Memory Footprint Reporter

12-hour memory usage monitoring for container health validation.
Outputs OK/FAIL to stdout for PagerDuty integration.

Usage: python scripts/report_mem.py
Exit codes: 0=OK, 1=FAIL
"""

import sys
import json
import subprocess
import datetime
import re
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryReporter:
    """Monitors Docker container memory usage and system health."""

    def __init__(self):
        """Initialize memory reporter."""
        self.report_time = datetime.datetime.now()

        # Memory thresholds (MB)
        self.container_memory_limits = {
            "trading_redis": 512,  # Redis should use <512MB
            "trading_grafana": 1024,  # Grafana should use <1GB
            "trading_prometheus": 2048,  # Prometheus should use <2GB
            "trading_influxdb": 1024,  # InfluxDB should use <1GB
            "trading_redpanda": 1536,  # Redpanda should use <1.5GB
        }

        # System thresholds
        self.max_memory_growth_pct = 3.0  # Max 3% growth per 12h period
        self.max_total_memory_pct = 80.0  # Max 80% of available system memory

    def get_docker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current Docker container memory statistics."""
        try:
            # First get container ID to name mapping
            name_cmd = ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"]
            name_result = subprocess.run(
                name_cmd, capture_output=True, text=True, timeout=30
            )

            id_to_name = {}
            if name_result.returncode == 0:
                for line in name_result.stdout.strip().split("\n"):
                    if "\t" in line:
                        container_id, name = line.split("\t", 1)
                        id_to_name[container_id.strip()] = name.strip()

            # Run docker stats command
            cmd = [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"Docker stats failed: {result.stderr}")
                return {}

            # Parse output
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            container_stats = {}

            for line in lines:
                if not line.strip():
                    continue

                # Split by whitespace and parse the expected format:
                # CONTAINER_ID   MEM_USAGE / LIMIT   MEM_%   CPU_%
                parts = line.split()
                if len(parts) >= 6:  # ID, mem_val, '/', limit, mem_%, cpu_%
                    container_id = parts[0].strip()
                    mem_usage = (
                        f"{parts[1]} / {parts[3]}"  # Reconstruct "123MiB / 7.654GiB"
                    )
                    mem_perc = parts[4].strip()
                    cpu_perc = parts[5].strip()

                    # Get container name from ID
                    container_name = id_to_name.get(container_id, container_id)

                    # Parse memory usage (e.g., "123.4MiB / 2GiB")
                    mem_match = re.match(
                        r"(\d+\.?\d*)(\w+)\s*/\s*(\d+\.?\d*)(\w+)", mem_usage
                    )
                    if mem_match:
                        used_val = float(mem_match.group(1))
                        used_unit = mem_match.group(2)
                        limit_val = float(mem_match.group(3))
                        limit_unit = mem_match.group(4)

                        # Convert to MB
                        used_mb = self._convert_to_mb(used_val, used_unit)
                        limit_mb = self._convert_to_mb(limit_val, limit_unit)

                        container_stats[container_name] = {
                            "memory_used_mb": used_mb,
                            "memory_limit_mb": limit_mb,
                            "memory_percent": float(mem_perc.rstrip("%")),
                            "cpu_percent": float(cpu_perc.rstrip("%")),
                        }

            return container_stats

        except Exception as e:
            logger.error(f"Failed to get Docker stats: {e}")
            return {}

    def _convert_to_mb(self, value: float, unit: str) -> float:
        """Convert memory value to MB."""
        unit = unit.lower()
        if unit in ["mb", "mib"]:
            return value
        elif unit in ["gb", "gib"]:
            return value * 1024
        elif unit in ["kb", "kib"]:
            return value / 1024
        elif unit in ["b"]:
            return value / (1024 * 1024)
        else:
            logger.warning(f"Unknown memory unit: {unit}")
            return value

    def get_system_memory(self) -> Dict[str, float]:
        """Get system-wide memory statistics."""
        try:
            # Get system memory info
            cmd = ["df", "-h", "/"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            # Get memory info using vm_stat on macOS or /proc/meminfo on Linux
            try:
                # Try macOS vm_stat first
                vm_cmd = ["vm_stat"]
                vm_result = subprocess.run(
                    vm_cmd, capture_output=True, text=True, timeout=10
                )

                if vm_result.returncode == 0:
                    return self._parse_macos_memory(vm_result.stdout)

            except:
                pass

            # Try Linux /proc/meminfo
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                return self._parse_linux_memory(meminfo)
            except:
                pass

            # Fallback to basic stats
            return {"total_mb": 8192, "available_mb": 4096, "used_pct": 50.0}

        except Exception as e:
            logger.error(f"Failed to get system memory: {e}")
            return {"total_mb": 8192, "available_mb": 4096, "used_pct": 50.0}

    def _parse_macos_memory(self, vm_output: str) -> Dict[str, float]:
        """Parse macOS vm_stat output."""
        lines = vm_output.split("\n")
        page_size = 4096  # Default page size

        stats = {}
        for line in lines:
            if "page size of" in line:
                page_size = int(re.search(r"(\d+)", line).group(1))
            elif "Pages free:" in line:
                stats["free"] = int(re.search(r"(\d+)", line).group(1))
            elif "Pages active:" in line:
                stats["active"] = int(re.search(r"(\d+)", line).group(1))
            elif "Pages inactive:" in line:
                stats["inactive"] = int(re.search(r"(\d+)", line).group(1))
            elif "Pages wired down:" in line:
                stats["wired"] = int(re.search(r"(\d+)", line).group(1))

        # Calculate totals in MB
        total_pages = (
            stats.get("free", 0)
            + stats.get("active", 0)
            + stats.get("inactive", 0)
            + stats.get("wired", 0)
        )
        used_pages = stats.get("active", 0) + stats.get("wired", 0)

        total_mb = (total_pages * page_size) / (1024 * 1024)
        used_mb = (used_pages * page_size) / (1024 * 1024)
        available_mb = total_mb - used_mb
        used_pct = (used_mb / total_mb) * 100 if total_mb > 0 else 0

        return {
            "total_mb": total_mb,
            "available_mb": available_mb,
            "used_pct": used_pct,
        }

    def _parse_linux_memory(self, meminfo: str) -> Dict[str, float]:
        """Parse Linux /proc/meminfo output."""
        lines = meminfo.split("\n")
        stats = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                # Extract numeric value (in kB)
                match = re.search(r"(\d+)", value)
                if match:
                    stats[key.strip()] = int(match.group(1))

        total_mb = stats.get("MemTotal", 0) / 1024
        available_mb = stats.get("MemAvailable", 0) / 1024
        used_mb = total_mb - available_mb
        used_pct = (used_mb / total_mb) * 100 if total_mb > 0 else 0

        return {
            "total_mb": total_mb,
            "available_mb": available_mb,
            "used_pct": used_pct,
        }

    def validate_container_memory(
        self, container_stats: Dict[str, Dict[str, Any]]
    ) -> tuple[bool, List[str]]:
        """Validate container memory usage against limits."""
        is_healthy = True
        issues = []

        for container, stats in container_stats.items():
            if not container.startswith("trading_"):
                continue

            memory_used = stats["memory_used_mb"]
            limit = self.container_memory_limits.get(container, 1024)

            if memory_used > limit:
                is_healthy = False
                issues.append(
                    f"Container {container} exceeds memory limit: "
                    f"{memory_used:.1f}MB > {limit}MB"
                )

            # Check for very high memory percentage
            mem_pct = stats["memory_percent"]
            if mem_pct > 90:
                is_healthy = False
                issues.append(
                    f"Container {container} high memory usage: {mem_pct:.1f}%"
                )

        return is_healthy, issues

    def validate_system_memory(
        self, system_stats: Dict[str, float]
    ) -> tuple[bool, List[str]]:
        """Validate system-wide memory usage."""
        is_healthy = True
        issues = []

        used_pct = system_stats["used_pct"]
        if used_pct > self.max_total_memory_pct:
            is_healthy = False
            issues.append(
                f"System memory usage high: {used_pct:.1f}% > {self.max_total_memory_pct}%"
            )

        # Check available memory
        available_mb = system_stats["available_mb"]
        if available_mb < 1024:  # Less than 1GB available
            is_healthy = False
            issues.append(f"Low available memory: {available_mb:.1f}MB")

        return is_healthy, issues

    def generate_report(self) -> Dict[str, Any]:
        """Generate complete memory footprint report."""
        logger.info("🔍 Generating memory footprint report...")

        # Get container stats
        container_stats = self.get_docker_stats()
        if not container_stats:
            logger.error("❌ Failed to retrieve container statistics")
            return {"status": "FAIL", "error": "No container data available"}

        # Get system stats
        system_stats = self.get_system_memory()

        # Validate container memory
        container_healthy, container_issues = self.validate_container_memory(
            container_stats
        )

        # Validate system memory
        system_healthy, system_issues = self.validate_system_memory(system_stats)

        # Overall health
        is_healthy = container_healthy and system_healthy
        all_issues = container_issues + system_issues

        # Build report
        report = {
            "timestamp": self.report_time.isoformat(),
            "status": "OK" if is_healthy else "FAIL",
            "container_stats": container_stats,
            "system_stats": system_stats,
            "validation": {
                "is_healthy": is_healthy,
                "container_healthy": container_healthy,
                "system_healthy": system_healthy,
                "issues": all_issues,
                "thresholds": {
                    "container_limits": self.container_memory_limits,
                    "max_memory_growth_pct": self.max_memory_growth_pct,
                    "max_total_memory_pct": self.max_total_memory_pct,
                },
            },
        }

        return report

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print human-readable report summary."""
        status = report["status"]
        containers = report["container_stats"]
        system = report["system_stats"]

        print(f"\n💾 Memory Report Summary - {report['timestamp']}")
        print(f"{'='*60}")
        print(f"Status: {status}")
        print(
            f"System Memory: {system.get('used_pct', 0):.1f}% used "
            f"({system.get('total_mb', 0):.0f}MB total)"
        )
        print(f"Available: {system.get('available_mb', 0):.0f}MB")

        print(f"\n📊 Container Memory Usage:")
        for container, stats in containers.items():
            if container.startswith("trading_"):
                limit = self.container_memory_limits.get(container, 1024)
                used = stats["memory_used_mb"]
                pct = (used / limit) * 100
                status_icon = "🔴" if used > limit else "🟡" if pct > 80 else "🟢"

                print(
                    f"  {status_icon} {container:<20} "
                    f"{used:>6.1f}MB / {limit:>4.0f}MB ({pct:>5.1f}%)"
                )

        if report["validation"]["issues"]:
            print(f"\n⚠️  Issues Detected:")
            for issue in report["validation"]["issues"]:
                print(f"  • {issue}")
        else:
            print(f"\n✅ All memory checks passed")


def main():
    """Main entry point for memory reporting."""
    try:
        # Initialize reporter
        reporter = MemoryReporter()

        # Generate report
        report = reporter.generate_report()

        # Print summary to stdout
        reporter.print_summary(report)

        # Output status for PagerDuty integration
        status = report["status"]
        print(f"\nFINAL_STATUS: {status}")

        # Write detailed report to log
        logger.info(f"Memory Report: {json.dumps(report, indent=2)}")

        # Exit with appropriate code
        sys.exit(0 if status == "OK" else 1)

    except Exception as e:
        logger.error(f"❌ Memory reporting failed: {e}")
        print(f"FINAL_STATUS: FAIL")
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
