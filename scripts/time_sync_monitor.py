#!/usr/bin/env python3
"""
Time Sync Integrity Monitor

Monitors system time synchronization and detects clock skew:
- Checks NTP/PTP synchronization status
- Measures clock drift vs authoritative time sources
- Alerts when drift exceeds 150ms threshold
- Exports metrics to Prometheus for alerting
- Critical for compliance and replay accuracy
"""

import argparse
import json
import logging
import subprocess
import time
import socket
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis
    import requests

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("time_sync_monitor")


class TimeSyncMonitor:
    """
    Monitors time synchronization and detects dangerous clock skew.
    Essential for compliance, audit trails, and replay accuracy.
    """

    def __init__(self):
        """Initialize time sync monitor."""
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # Time sync configuration
        self.config = {
            "clock_skew_threshold_ms": 150,  # Alert if >150ms drift
            "warning_threshold_ms": 75,  # Warning if >75ms drift
            "ntp_servers": [
                "time.nist.gov",
                "pool.ntp.org",
                "time.google.com",
                "time.cloudflare.com",
            ],
            "ptp_check_command": ["ptp4l", "-s"],  # PTP daemon status
            "chrony_check_command": ["chronyc", "tracking"],  # Chrony NTP client
            "ntpq_check_command": ["ntpq", "-p"],  # Classic NTP client
            "check_interval_seconds": 30,  # Check every 30 seconds
            "prometheus_port": 9090,  # Prometheus metrics port
            "alert_retention_hours": 24,  # Keep alerts for 24 hours
        }

        logger.info("Initialized time synchronization monitor")

    def get_system_time_status(self) -> Dict[str, any]:
        """
        Get comprehensive system time synchronization status.

        Returns:
            System time status and synchronization health
        """
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "system_time": {
                    "utc_time": datetime.now(timezone.utc).isoformat(),
                    "local_time": datetime.now().isoformat(),
                    "timezone": str(datetime.now().astimezone().tzinfo),
                },
                "ntp_status": self._check_ntp_status(),
                "ptp_status": self._check_ptp_status(),
                "clock_skew": self._measure_clock_skew(),
                "sync_health": {},
            }

            # Determine overall sync health
            skew_ms = status["clock_skew"].get("max_skew_ms", float("inf"))
            ntp_synced = status["ntp_status"].get("synchronized", False)
            ptp_synced = status["ptp_status"].get("synchronized", False)

            status["sync_health"] = {
                "healthy": skew_ms <= self.config["clock_skew_threshold_ms"]
                and (ntp_synced or ptp_synced),
                "synchronized": ntp_synced or ptp_synced,
                "skew_within_threshold": skew_ms
                <= self.config["clock_skew_threshold_ms"],
                "max_skew_ms": skew_ms,
                "sync_method": (
                    "PTP" if ptp_synced else ("NTP" if ntp_synced else "NONE")
                ),
            }

            return status

        except Exception as e:
            logger.error(f"Error getting time status: {e}")
            return {"error": str(e)}

    def _check_ntp_status(self) -> Dict[str, any]:
        """Check NTP synchronization status."""
        try:
            ntp_status = {
                "synchronized": False,
                "stratum": None,
                "offset_ms": None,
                "jitter_ms": None,
                "peer_count": 0,
                "method": None,
            }

            # Try chrony first (modern NTP client)
            chrony_result = self._check_chrony_status()
            if chrony_result.get("success", False):
                ntp_status.update(chrony_result)
                ntp_status["method"] = "chrony"
                return ntp_status

            # Fall back to classic ntpq
            ntpq_result = self._check_ntpq_status()
            if ntpq_result.get("success", False):
                ntp_status.update(ntpq_result)
                ntp_status["method"] = "ntpq"
                return ntp_status

            # Manual NTP check as fallback
            manual_result = self._manual_ntp_check()
            if manual_result.get("success", False):
                ntp_status.update(manual_result)
                ntp_status["method"] = "manual"
                return ntp_status

            ntp_status["error"] = "No NTP client available"
            return ntp_status

        except Exception as e:
            return {"error": str(e), "synchronized": False}

    def _check_chrony_status(self) -> Dict[str, any]:
        """Check chrony NTP daemon status."""
        try:
            result = subprocess.run(
                self.config["chrony_check_command"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return {"success": False, "reason": "chrony not available"}

            # Parse chrony output
            lines = result.stdout.strip().split("\n")
            chrony_status = {"success": True}

            for line in lines:
                if "Stratum" in line:
                    chrony_status["stratum"] = int(line.split()[-1])
                elif "System time" in line:
                    # Extract offset in seconds, convert to ms
                    offset_str = line.split()[-2]
                    offset_sec = float(offset_str)
                    chrony_status["offset_ms"] = offset_sec * 1000
                elif "RMS offset" in line:
                    jitter_str = line.split()[-2]
                    jitter_sec = float(jitter_str)
                    chrony_status["jitter_ms"] = jitter_sec * 1000

            # Consider synchronized if stratum <= 15 and offset reasonable
            chrony_status["synchronized"] = (
                chrony_status.get("stratum", 16) <= 15
                and abs(chrony_status.get("offset_ms", float("inf"))) < 1000
            )

            return chrony_status

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def _check_ntpq_status(self) -> Dict[str, any]:
        """Check classic NTP daemon status."""
        try:
            result = subprocess.run(
                self.config["ntpq_check_command"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return {"success": False, "reason": "ntpq not available"}

            # Parse ntpq output
            lines = result.stdout.strip().split("\n")
            ntpq_status = {"success": True, "peer_count": 0}

            synchronized = False
            best_offset = float("inf")

            for line in lines:
                if line.startswith("*"):  # Synchronized peer
                    synchronized = True
                    parts = line.split()
                    if len(parts) >= 9:
                        try:
                            offset_ms = float(parts[8])
                            if abs(offset_ms) < abs(best_offset):
                                best_offset = offset_ms
                        except:
                            pass

                if line.startswith((" ", "*", "+", "-", "x")):
                    ntpq_status["peer_count"] += 1

            ntpq_status["synchronized"] = synchronized
            if best_offset != float("inf"):
                ntpq_status["offset_ms"] = best_offset

            return ntpq_status

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def _manual_ntp_check(self) -> Dict[str, any]:
        """Manual NTP time check against NTP servers."""
        try:
            # Simple SNTP client implementation
            best_offset = float("inf")
            successful_queries = 0

            for ntp_server in self.config["ntp_servers"][:2]:  # Check first 2 servers
                try:
                    offset = self._query_ntp_server(ntp_server)
                    if offset is not None:
                        successful_queries += 1
                        if abs(offset) < abs(best_offset):
                            best_offset = offset
                except Exception as e:
                    logger.debug(f"NTP query to {ntp_server} failed: {e}")
                    continue

            if successful_queries > 0:
                return {
                    "success": True,
                    "synchronized": abs(best_offset) < 1.0,  # Within 1 second
                    "offset_ms": best_offset * 1000,
                    "servers_queried": successful_queries,
                }
            else:
                return {"success": False, "reason": "No NTP servers reachable"}

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def _query_ntp_server(self, server: str, timeout: float = 5.0) -> Optional[float]:
        """Query NTP server and return time offset in seconds."""
        try:
            # Create NTP packet (simplified SNTP client)
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(timeout)

            # NTP packet format: 48 bytes, first byte = 0x1B (version 3, mode 3)
            ntp_packet = b"\x1b" + 47 * b"\0"

            # Record local time before sending
            t1 = time.time()
            client.sendto(ntp_packet, (server, 123))

            # Receive response
            response, address = client.recvfrom(48)
            t4 = time.time()

            client.close()

            # Extract server timestamp (bytes 40-43, network byte order)
            # NTP epoch is 1900-01-01, Unix epoch is 1970-01-01 (difference = 2208988800 seconds)
            server_time_ntp = struct.unpack("!I", response[40:44])[0]
            server_time_unix = server_time_ntp - 2208988800

            # Calculate offset: server_time - local_time
            offset = server_time_unix - ((t1 + t4) / 2)

            return offset

        except Exception as e:
            logger.debug(f"NTP query failed for {server}: {e}")
            return None

    def _check_ptp_status(self) -> Dict[str, any]:
        """Check PTP (Precision Time Protocol) status."""
        try:
            # Try ptp4l status
            result = subprocess.run(
                ["pmc", "-u", "-b", "0", "GET", "CURRENT_DATA_SET"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # Parse PTP output (simplified)
                ptp_status = {
                    "synchronized": "SLAVE" in result.stdout,
                    "method": "ptp4l",
                    "success": True,
                }
                return ptp_status

            # Fallback: check if PTP daemon is running
            ps_result = subprocess.run(
                ["pgrep", "-f", "ptp4l"], capture_output=True, timeout=5
            )

            return {
                "synchronized": False,
                "running": ps_result.returncode == 0,
                "success": ps_result.returncode == 0,
                "method": "process_check",
            }

        except Exception as e:
            return {"synchronized": False, "error": str(e)}

    def _measure_clock_skew(self) -> Dict[str, any]:
        """Measure clock skew against multiple time sources."""
        try:
            skew_measurements = []

            # Query multiple NTP servers
            for server in self.config["ntp_servers"][:3]:
                try:
                    offset = self._query_ntp_server(server, timeout=3.0)
                    if offset is not None:
                        skew_ms = offset * 1000
                        skew_measurements.append(
                            {
                                "server": server,
                                "skew_ms": skew_ms,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Skew measurement failed for {server}: {e}")

            if not skew_measurements:
                return {"error": "No servers available for skew measurement"}

            # Calculate statistics
            skews = [m["skew_ms"] for m in skew_measurements]
            max_skew = max(abs(s) for s in skews)
            avg_skew = sum(skews) / len(skews)

            return {
                "measurements": skew_measurements,
                "max_skew_ms": max_skew,
                "avg_skew_ms": avg_skew,
                "server_count": len(skew_measurements),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": str(e)}

    def check_time_sync_health(self) -> Dict[str, any]:
        """Check overall time synchronization health."""
        try:
            logger.debug("⏰ Checking time synchronization health")

            # Get full time status
            time_status = self.get_system_time_status()

            # Determine health and alerting
            sync_health = time_status.get("sync_health", {})
            healthy = sync_health.get("healthy", False)
            max_skew_ms = sync_health.get("max_skew_ms", float("inf"))

            health_check = {
                "timestamp": datetime.now().isoformat(),
                "healthy": healthy,
                "synchronized": sync_health.get("synchronized", False),
                "max_skew_ms": max_skew_ms,
                "sync_method": sync_health.get("sync_method", "NONE"),
                "alert_level": self._determine_alert_level(max_skew_ms),
                "full_status": time_status,
            }

            # Store metrics in Redis
            if self.redis_client:
                self._store_time_metrics(health_check)

            return health_check

        except Exception as e:
            logger.error(f"Error checking time sync health: {e}")
            return {"error": str(e), "healthy": False}

    def _determine_alert_level(self, skew_ms: float) -> str:
        """Determine alert level based on skew magnitude."""
        if skew_ms >= self.config["clock_skew_threshold_ms"]:
            return "CRITICAL"
        elif skew_ms >= self.config["warning_threshold_ms"]:
            return "WARNING"
        else:
            return "OK"

    def _store_time_metrics(self, health_check: Dict[str, any]):
        """Store time synchronization metrics in Redis."""
        try:
            # Store current metrics
            metrics_key = "metrics:time_sync"
            metrics_data = {
                "healthy": int(health_check.get("healthy", False)),
                "synchronized": int(health_check.get("synchronized", False)),
                "max_skew_ms": health_check.get("max_skew_ms", -1),
                "timestamp": health_check.get("timestamp"),
                "sync_method": health_check.get("sync_method", "NONE"),
                "alert_level": health_check.get("alert_level", "UNKNOWN"),
            }

            self.redis_client.hmset(metrics_key, metrics_data)
            self.redis_client.expire(metrics_key, 300)  # 5 minute TTL

            # Store time series data
            ts_key = f"timeseries:clock_skew:{datetime.now().strftime('%Y%m%d%H%M')}"
            self.redis_client.lpush(ts_key, health_check["max_skew_ms"])
            self.redis_client.ltrim(ts_key, 0, 59)  # Keep last 60 measurements
            self.redis_client.expire(ts_key, 7200)  # 2 hour TTL

            # Generate alerts if needed
            if health_check.get("alert_level") in ["WARNING", "CRITICAL"]:
                self._generate_time_sync_alert(health_check)

        except Exception as e:
            logger.error(f"Error storing time metrics: {e}")

    def _generate_time_sync_alert(self, health_check: Dict[str, any]):
        """Generate time synchronization alert."""
        try:
            alert_level = health_check.get("alert_level", "UNKNOWN")
            max_skew_ms = health_check.get("max_skew_ms", 0)
            sync_method = health_check.get("sync_method", "NONE")

            alert = {
                "timestamp": datetime.now().isoformat(),
                "alert_type": "time_sync_drift",
                "level": alert_level,
                "message": f"{alert_level}: Clock skew {max_skew_ms:.1f}ms (threshold: {self.config['clock_skew_threshold_ms']}ms)",
                "max_skew_ms": max_skew_ms,
                "sync_method": sync_method,
                "synchronized": health_check.get("synchronized", False),
            }

            # Store in Redis alerts
            self.redis_client.lpush("alerts:time_sync", json.dumps(alert))
            self.redis_client.ltrim("alerts:time_sync", 0, 99)  # Keep last 100 alerts

            logger.warning(f"⚠️ TIME SYNC ALERT: {alert['message']}")

        except Exception as e:
            logger.error(f"Error generating time sync alert: {e}")

    def run_monitoring_cycle(self) -> Dict[str, any]:
        """Run single time sync monitoring cycle."""
        try:
            logger.debug("⏰ Running time sync monitoring cycle")

            cycle_results = {
                "timestamp": datetime.now().isoformat(),
                "cycle_type": "time_sync_monitoring",
            }

            # Check time sync health
            health_check = self.check_time_sync_health()
            cycle_results["health_check"] = health_check

            # Log significant events
            if not health_check.get("healthy", True):
                logger.warning(
                    f"Time sync unhealthy: {health_check.get('max_skew_ms', 0):.1f}ms skew"
                )

            return cycle_results

        except Exception as e:
            logger.error(f"Error in time sync monitoring cycle: {e}")
            return {"error": str(e)}

    def run_monitoring_daemon(self):
        """Run time sync monitor as continuous daemon."""
        logger.info("⏰ Starting time synchronization monitoring daemon")

        try:
            while True:
                cycle_results = self.run_monitoring_cycle()

                # Log health status changes
                health_check = cycle_results.get("health_check", {})
                if health_check.get("alert_level") in ["WARNING", "CRITICAL"]:
                    logger.warning(
                        f"Time sync issue: {health_check.get('max_skew_ms', 0):.1f}ms skew"
                    )

                # Wait before next cycle
                time.sleep(self.config["check_interval_seconds"])

        except KeyboardInterrupt:
            logger.info("Time sync monitoring daemon stopped by user")
        except Exception as e:
            logger.error(f"Time sync monitoring daemon error: {e}")

    def export_prometheus_metrics(self) -> str:
        """Export time sync metrics in Prometheus format."""
        try:
            health_check = self.check_time_sync_health()

            metrics = []

            # Clock skew metric
            skew_ms = health_check.get("max_skew_ms", -1)
            if skew_ms >= 0:
                metrics.append(f"time_sync_clock_skew_milliseconds {skew_ms}")

            # Sync health boolean
            healthy = int(health_check.get("healthy", False))
            metrics.append(f"time_sync_healthy {healthy}")

            # Synchronized boolean
            synchronized = int(health_check.get("synchronized", False))
            metrics.append(f"time_sync_synchronized {synchronized}")

            # Sync method
            sync_method = health_check.get("sync_method", "NONE")
            for method in ["NTP", "PTP", "NONE"]:
                value = 1 if sync_method == method else 0
                metrics.append(f'time_sync_method{{method="{method.lower()}"}} {value}')

            return "\n".join(metrics) + "\n"

        except Exception as e:
            logger.error(f"Error exporting Prometheus metrics: {e}")
            return f"# Error: {e}\n"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Time Synchronization Monitor")

    parser.add_argument(
        "--mode",
        choices=["check", "monitor", "daemon", "metrics"],
        default="check",
        help="Monitor mode",
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("⏰ Starting Time Synchronization Monitor")

    try:
        monitor = TimeSyncMonitor()

        if args.mode == "check":
            results = monitor.check_time_sync_health()
            print(f"\n⏰ TIME SYNC STATUS:")
            print(json.dumps(results, indent=2))

        elif args.mode == "monitor":
            results = monitor.run_monitoring_cycle()
            print(f"\n📊 MONITORING CYCLE:")
            print(json.dumps(results, indent=2))

        elif args.mode == "daemon":
            monitor.run_monitoring_daemon()
            return 0

        elif args.mode == "metrics":
            metrics = monitor.export_prometheus_metrics()
            print(metrics)
            return 0

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        # Return appropriate exit code
        if args.mode in ["check", "monitor"]:
            return 0 if results.get("healthy", False) else 1
        else:
            return 0

    except Exception as e:
        logger.error(f"Error in time sync monitor: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
