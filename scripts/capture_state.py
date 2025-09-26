#!/usr/bin/env python3
"""
One-Shot Incident Snapshot
Single command that captures full system state for audits/post-mortems
"""

import os
import sys
import json
import time
import tarfile
import tempfile
import subprocess
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("capture_state")


class StateCapture:
    """System state capture for incident analysis."""

    def __init__(self):
        """Initialize state capture."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Configuration
        self.config = {
            "snapshot_dir": "/tmp",
            "s3_bucket": os.getenv("INCIDENT_BUCKET", "trading-incident-snapshots"),
            "s3_region": os.getenv("AWS_REGION", "us-east-1"),
            "ipfs_node": os.getenv("IPFS_NODE", "http://localhost:5001"),
            "max_log_size_mb": 100,
            "retention_days": 30,
        }

        # Initialize AWS S3 client
        try:
            self.s3_client = boto3.client("s3", region_name=self.config["s3_region"])
        except (NoCredentialsError, Exception) as e:
            logger.warning(f"S3 client initialization failed: {e}")
            self.s3_client = None

        logger.info("📸 State Capture initialized")
        logger.info(f"   Snapshot dir: {self.config['snapshot_dir']}")
        logger.info(f"   S3 bucket: {self.config['s3_bucket']}")

    def collect_feature_flags(self) -> Dict[str, Any]:
        """Collect current feature flags state."""
        try:
            flags = self.redis.hgetall("features:flags") or {}

            # Convert string values to appropriate types
            processed_flags = {}
            for key, value in flags.items():
                try:
                    # Try to parse as int/float/bool
                    if value.lower() in ["true", "false"]:
                        processed_flags[key] = value.lower() == "true"
                    elif value.isdigit():
                        processed_flags[key] = int(value)
                    elif "." in value and value.replace(".", "").isdigit():
                        processed_flags[key] = float(value)
                    else:
                        processed_flags[key] = value
                except:
                    processed_flags[key] = value

            return {
                "flags": processed_flags,
                "total_flags": len(processed_flags),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error collecting feature flags: {e}")
            return {"error": str(e)}

    def collect_positions(self) -> Dict[str, Any]:
        """Collect current position state."""
        try:
            positions = {}

            # Collect positions from different strategies
            strategies = ["RL", "BASIS", "MM"]
            for strategy in strategies:
                strategy_positions = {}

                # Get strategy-specific position data
                for symbol in ["BTC", "ETH", "SOL"]:
                    pos_key = f"strategy:{strategy}:{symbol}:position"
                    position_data = self.redis.get(pos_key)
                    if position_data:
                        try:
                            strategy_positions[symbol] = json.loads(position_data)
                        except:
                            strategy_positions[symbol] = position_data

                if strategy_positions:
                    positions[strategy] = strategy_positions

            # Get portfolio-level positions
            portfolio_pos = self.redis.get("portfolio_positions")
            if portfolio_pos:
                try:
                    positions["portfolio"] = json.loads(portfolio_pos)
                except:
                    positions["portfolio"] = portfolio_pos

            # Get risk metrics
            risk_metrics = self.redis.hgetall("risk:stats") or {}

            return {
                "positions": positions,
                "risk_metrics": risk_metrics,
                "total_strategies": len(positions),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error collecting positions: {e}")
            return {"error": str(e)}

    def collect_orders(self) -> Dict[str, Any]:
        """Collect current active orders."""
        try:
            orders = {}

            # Check different order streams for recent orders
            order_streams = [
                "orders:binance",
                "orders:coinbase",
                "orders:ftx",
                "orders:dydx",
                "orders:all",
            ]

            for stream in order_streams:
                try:
                    # Get recent orders from stream
                    stream_data = self.redis.xrevrange(stream, count=50)
                    if stream_data:
                        stream_orders = []
                        for stream_id, order_data in stream_data:
                            order_info = {"stream_id": stream_id, **order_data}
                            stream_orders.append(order_info)
                        orders[stream] = stream_orders
                except:
                    continue

            # Get active order counts
            active_counts = {}
            for venue in ["binance", "coinbase", "ftx", "dydx"]:
                count_key = f"orders:active:{venue}"
                count = self.redis.get(count_key)
                if count:
                    active_counts[venue] = int(count)

            return {
                "recent_orders": orders,
                "active_counts": active_counts,
                "total_streams": len([s for s in orders.values() if s]),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error collecting orders: {e}")
            return {"error": str(e)}

    def collect_model_hashes(self) -> Dict[str, Any]:
        """Collect model and code hashes."""
        try:
            model_info = {}

            # Get model hash from Redis
            model_hash = self.redis.get("model:hash")
            if model_hash:
                model_info["model_hash"] = model_hash

            # Get model version info
            model_version = self.redis.get("model:version")
            if model_version:
                model_info["model_version"] = model_version

            # Get Git commit hash
            try:
                git_hash = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=project_root,
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
                model_info["git_hash"] = git_hash
            except:
                model_info["git_hash"] = "unknown"

            # Get Git branch
            try:
                git_branch = (
                    subprocess.check_output(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=project_root,
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
                model_info["git_branch"] = git_branch
            except:
                model_info["git_branch"] = "unknown"

            # Get deployment info
            active_color = self.redis.get("mode:active_color")
            if active_color:
                model_info["active_deployment"] = active_color

            return {"model_info": model_info, "timestamp": time.time()}

        except Exception as e:
            logger.error(f"Error collecting model hashes: {e}")
            return {"error": str(e)}

    def collect_service_status(self) -> Dict[str, Any]:
        """Collect service status information."""
        try:
            services = {}

            # Check systemd services
            service_names = ["redis-server", "ops_bot", "trading_bot", "risk_monitor"]

            for service in service_names:
                try:
                    result = subprocess.run(
                        ["systemctl", "is-active", service],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    services[service] = {
                        "status": result.stdout.strip(),
                        "active": result.returncode == 0,
                    }
                except:
                    services[service] = {"status": "unknown", "active": False}

            # Get Redis info
            try:
                redis_info = self.redis.info()
                redis_status = {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                    "uptime_in_seconds": redis_info.get("uptime_in_seconds", 0),
                    "redis_version": redis_info.get("redis_version", "unknown"),
                }
                services["redis"] = redis_status
            except:
                services["redis"] = {"error": "connection_failed"}

            # Get system load
            try:
                with open("/proc/loadavg", "r") as f:
                    loadavg = f.read().strip().split()[:3]
                services["system_load"] = {
                    "1min": float(loadavg[0]),
                    "5min": float(loadavg[1]),
                    "15min": float(loadavg[2]),
                }
            except:
                services["system_load"] = {"error": "unavailable"}

            return {
                "services": services,
                "total_services": len(services),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error collecting service status: {e}")
            return {"error": str(e)}

    def collect_prometheus_gauges(self) -> Dict[str, Any]:
        """Collect key Prometheus metrics from Redis."""
        try:
            metrics = {}

            # Key metric patterns to collect
            metric_patterns = [
                "metric:*",
                "pnl:*",
                "risk:*",
                "gpu:*",
                "strategy:*:pnl*",
                "venue:*:score*",
            ]

            for pattern in metric_patterns:
                keys = self.redis.keys(pattern)
                for key in keys[:50]:  # Limit to prevent overload
                    value = self.redis.get(key)
                    if value:
                        try:
                            # Try to parse as number
                            metrics[key] = float(value)
                        except:
                            metrics[key] = value

            # Get specific high-priority metrics
            priority_metrics = {
                "capital_effective": self.redis.get("risk:capital_effective"),
                "capital_cap": self.redis.get("risk:capital_cap"),
                "system_mode": self.redis.get("mode"),
                "total_pnl": self.redis.get("pnl:total"),
                "active_positions": self.redis.get("positions:count"),
                "gpu_memory": self.redis.get("gpu:mem_frac"),
            }

            for key, value in priority_metrics.items():
                if value is not None:
                    try:
                        metrics[f"priority:{key}"] = float(value)
                    except:
                        metrics[f"priority:{key}"] = value

            return {
                "metrics": metrics,
                "total_metrics": len(metrics),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error collecting Prometheus gauges: {e}")
            return {"error": str(e)}

    def collect_log_files(self, temp_dir: Path) -> List[str]:
        """Collect relevant log files."""
        try:
            log_files = []
            log_directories = ["/var/log/trader", "/tmp", project_root / "logs"]

            max_size_bytes = self.config["max_log_size_mb"] * 1024 * 1024

            for log_dir in log_directories:
                log_path = Path(log_dir)
                if not log_path.exists():
                    continue

                # Find log files
                for pattern in ["*.log", "*.out", "*.err"]:
                    for log_file in log_path.glob(pattern):
                        try:
                            if log_file.stat().st_size > max_size_bytes:
                                # Truncate large files
                                truncated_file = temp_dir / f"truncated_{log_file.name}"
                                with open(log_file, "rb") as src:
                                    src.seek(-max_size_bytes, 2)  # Last N bytes
                                    with open(truncated_file, "wb") as dst:
                                        dst.write(src.read())
                                log_files.append(str(truncated_file))
                            else:
                                log_files.append(str(log_file))
                        except Exception as e:
                            logger.debug(f"Error processing log file {log_file}: {e}")

            return log_files

        except Exception as e:
            logger.error(f"Error collecting log files: {e}")
            return []

    def create_state_bundle(self) -> Dict[str, Any]:
        """Create comprehensive state bundle."""
        try:
            logger.info("🔍 Collecting system state...")

            # Collect all state components
            state_bundle = {
                "snapshot_metadata": {
                    "timestamp": time.time(),
                    "datetime_utc": datetime.now(timezone.utc).isoformat() + "Z",
                    "hostname": os.uname().nodename,
                    "system": os.uname().sysname,
                    "capture_version": "1.0.0",
                },
                "feature_flags": self.collect_feature_flags(),
                "positions": self.collect_positions(),
                "orders": self.collect_orders(),
                "model_hashes": self.collect_model_hashes(),
                "service_status": self.collect_service_status(),
                "prometheus_gauges": self.collect_prometheus_gauges(),
            }

            logger.info("✅ State collection completed")
            return state_bundle

        except Exception as e:
            logger.error(f"Error creating state bundle: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def create_tarball(self, state_bundle: Dict[str, Any]) -> str:
        """Create tarball with state bundle and logs."""
        try:
            timestamp = int(time.time())
            tarball_path = (
                Path(self.config["snapshot_dir"]) / f"snapshot_{timestamp}.tgz"
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Write state bundle JSON
                state_file = temp_path / "state.json"
                with open(state_file, "w") as f:
                    json.dump(state_bundle, f, indent=2, default=str)

                # Collect log files
                log_files = self.collect_log_files(temp_path)

                # Create tarball
                with tarfile.open(tarball_path, "w:gz") as tar:
                    # Add state file
                    tar.add(state_file, arcname="state.json")

                    # Add log files
                    for log_file in log_files:
                        try:
                            log_path = Path(log_file)
                            tar.add(log_file, arcname=f"logs/{log_path.name}")
                        except Exception as e:
                            logger.debug(f"Error adding log file {log_file}: {e}")

                logger.info(f"📦 Created tarball: {tarball_path}")
                return str(tarball_path)

        except Exception as e:
            logger.error(f"Error creating tarball: {e}")
            raise

    def upload_to_s3(self, tarball_path: str) -> Optional[str]:
        """Upload tarball to S3."""
        if not self.s3_client:
            logger.warning("S3 client not available - skipping S3 upload")
            return None

        try:
            tarball_name = Path(tarball_path).name
            s3_key = f"incidents/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/{tarball_name}"

            self.s3_client.upload_file(tarball_path, self.config["s3_bucket"], s3_key)

            s3_url = f"s3://{self.config['s3_bucket']}/{s3_key}"
            logger.info(f"📤 Uploaded to S3: {s3_url}")
            return s3_url

        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            return None

    def upload_to_ipfs(self, tarball_path: str) -> Optional[str]:
        """Upload tarball to IPFS."""
        try:
            # Use IPFS CLI if available
            result = subprocess.run(
                ["ipfs", "add", tarball_path],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # Extract CID from output
                lines = result.stdout.strip().split("\n")
                if lines:
                    # Format: "added <CID> <filename>"
                    cid = lines[-1].split()[1]
                    ipfs_url = f"ipfs://{cid}"
                    logger.info(f"📤 Uploaded to IPFS: {ipfs_url}")
                    return cid

            logger.warning("IPFS upload failed - CLI not available")
            return None

        except subprocess.TimeoutExpired:
            logger.error("IPFS upload timed out")
            return None
        except FileNotFoundError:
            logger.warning("IPFS CLI not found - skipping IPFS upload")
            return None
        except Exception as e:
            logger.error(f"Error uploading to IPFS: {e}")
            return None

    def store_incident_record(self, snapshot_info: Dict[str, Any]):
        """Store incident snapshot record in Redis."""
        try:
            incident_record = {
                "timestamp": int(time.time()),
                "snapshot_file": snapshot_info.get("tarball_path"),
                "s3_url": snapshot_info.get("s3_url"),
                "ipfs_cid": snapshot_info.get("ipfs_cid"),
                "size_bytes": snapshot_info.get("size_bytes", 0),
                "components_captured": snapshot_info.get("components", []),
            }

            # Store in audit incidents stream
            self.redis.xadd("audit:incidents", incident_record)

            # Keep latest snapshot info
            self.redis.set(
                "incident:latest_snapshot", json.dumps(incident_record, default=str)
            )

            logger.info(f"📝 Stored incident record: {incident_record['timestamp']}")

        except Exception as e:
            logger.error(f"Error storing incident record: {e}")

    def capture_snapshot(self) -> Dict[str, Any]:
        """Main method to capture complete system snapshot."""
        try:
            capture_start = time.time()
            logger.info("📸 Starting incident snapshot capture...")

            # Create state bundle
            state_bundle = self.create_state_bundle()

            # Create tarball
            tarball_path = self.create_tarball(state_bundle)
            tarball_size = Path(tarball_path).stat().st_size

            # Upload to storage backends
            s3_url = self.upload_to_s3(tarball_path)
            ipfs_cid = self.upload_to_ipfs(tarball_path)

            # Compile results
            snapshot_info = {
                "status": "completed",
                "tarball_path": tarball_path,
                "size_bytes": tarball_size,
                "s3_url": s3_url,
                "ipfs_cid": ipfs_cid,
                "components": list(state_bundle.keys()),
                "capture_duration": time.time() - capture_start,
                "timestamp": time.time(),
            }

            # Store incident record
            self.store_incident_record(snapshot_info)

            logger.info(
                f"✅ Snapshot capture completed in {snapshot_info['capture_duration']:.1f}s"
            )
            logger.info(f"   Tarball: {tarball_path} ({tarball_size:,} bytes)")
            if s3_url:
                logger.info(f"   S3: {s3_url}")
            if ipfs_cid:
                logger.info(f"   IPFS: ipfs://{ipfs_cid}")

            return snapshot_info

        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            return {"status": "error", "error": str(e), "timestamp": time.time()}

    def get_status_report(self) -> Dict[str, Any]:
        """Get snapshot system status."""
        try:
            # Get latest snapshot info
            latest_snapshot = self.redis.get("incident:latest_snapshot")
            if latest_snapshot:
                try:
                    latest_info = json.loads(latest_snapshot)
                except:
                    latest_info = None
            else:
                latest_info = None

            # Count recent incidents
            try:
                recent_incidents = self.redis.xlen("audit:incidents")
            except:
                recent_incidents = 0

            status = {
                "service": "incident_snapshot",
                "timestamp": time.time(),
                "config": self.config,
                "latest_snapshot": latest_info,
                "total_incidents": recent_incidents,
                "s3_available": self.s3_client is not None,
                "ipfs_available": subprocess.run(
                    ["which", "ipfs"], capture_output=True
                ).returncode
                == 0,
            }

            return status

        except Exception as e:
            return {
                "service": "incident_snapshot",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }


def main():
    """Main entry point for state capture."""
    import argparse

    parser = argparse.ArgumentParser(description="One-Shot Incident Snapshot")
    parser.add_argument(
        "--capture", action="store_true", help="Capture system snapshot"
    )
    parser.add_argument("--status", action="store_true", help="Show status report")

    args = parser.parse_args()

    # Create state capture
    capture = StateCapture()

    if args.status:
        # Show status report
        status = capture.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.capture:
        # Capture snapshot
        result = capture.capture_snapshot()
        print(json.dumps(result, indent=2, default=str))

        if result.get("status") != "error":
            sys.exit(0)
        else:
            sys.exit(1)

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
