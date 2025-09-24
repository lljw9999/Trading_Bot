#!/usr/bin/env python3
"""
SBOM (Software Bill of Materials) Generator
Generates comprehensive SBOM for model artifacts, dependencies, and supply chain
"""
import os
import sys
import json
import yaml
import hashlib
import datetime
import pathlib
import subprocess
import pkg_resources
from pathlib import Path
from datetime import timezone


def get_python_dependencies():
    """Get all installed Python packages."""
    dependencies = []

    try:
        installed_packages = [d for d in pkg_resources.working_set]

        for package in installed_packages:
            dep_info = {
                "name": package.project_name,
                "version": package.version,
                "location": package.location,
                "type": "python_package",
            }

            # Try to get more metadata
            try:
                metadata = package.get_metadata("METADATA")
                if metadata:
                    lines = metadata.split("\n")
                    for line in lines:
                        if line.startswith("Home-page:"):
                            dep_info["homepage"] = line.split(":", 1)[1].strip()
                        elif line.startswith("Author:"):
                            dep_info["author"] = line.split(":", 1)[1].strip()
                        elif line.startswith("License:"):
                            dep_info["license"] = line.split(":", 1)[1].strip()
            except Exception:
                pass

            dependencies.append(dep_info)

    except Exception as e:
        print(f"Warning: Could not enumerate Python packages: {e}")

    return dependencies


def get_system_dependencies():
    """Get system-level dependencies."""
    dependencies = []

    # Check for common system tools
    system_tools = ["git", "docker", "curl", "wget", "make"]

    for tool in system_tools:
        try:
            result = subprocess.run(["which", tool], capture_output=True, text=True)
            if result.returncode == 0:
                path = result.stdout.strip()

                # Try to get version
                version = "unknown"
                try:
                    version_result = subprocess.run(
                        [tool, "--version"], capture_output=True, text=True, timeout=5
                    )
                    if version_result.returncode == 0:
                        version = version_result.stdout.strip().split("\n")[0][
                            :100
                        ]  # First line, truncated
                except Exception:
                    pass

                dependencies.append(
                    {
                        "name": tool,
                        "version": version,
                        "path": path,
                        "type": "system_tool",
                    }
                )
        except Exception:
            continue

    return dependencies


def get_model_artifacts():
    """Get model artifacts and their metadata."""
    artifacts = []

    # Scan common artifact directories
    artifact_dirs = [
        "checkpoints",
        "artifacts/last_good",
        "model_cards",
        "artifacts/validation",
    ]

    for base_dir in artifact_dirs:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(
                        (".pt", ".pth", ".onnx", ".pkl", ".json", ".yaml", ".md")
                    ):
                        file_path = os.path.join(root, file)

                        try:
                            stat_info = os.stat(file_path)
                            file_hash = calculate_file_hash(file_path)

                            artifacts.append(
                                {
                                    "path": file_path,
                                    "name": file,
                                    "type": "model_artifact",
                                    "size_bytes": stat_info.st_size,
                                    "modified_time": datetime.datetime.fromtimestamp(
                                        stat_info.st_mtime, tz=timezone.utc
                                    ).isoformat(),
                                    "sha256": file_hash,
                                }
                            )
                        except Exception as e:
                            print(f"Warning: Could not process {file_path}: {e}")

    return artifacts


def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception:
        return "unknown"


def get_git_metadata():
    """Get Git repository metadata."""
    try:
        # Get current commit
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
        if commit_result.returncode != 0:
            return None

        commit_hash = commit_result.stdout.strip()

        # Get commit details
        log_result = subprocess.run(
            ["git", "log", "-1", "--format=%H|%an|%ae|%at|%s"],
            capture_output=True,
            text=True,
        )

        if log_result.returncode == 0:
            parts = log_result.stdout.strip().split("|")
            if len(parts) >= 5:
                return {
                    "commit_hash": parts[0],
                    "author_name": parts[1],
                    "author_email": parts[2],
                    "commit_time": datetime.datetime.fromtimestamp(
                        int(parts[3]), tz=timezone.utc
                    ).isoformat(),
                    "commit_message": parts[4],
                    "type": "git_metadata",
                }

        return {"commit_hash": commit_hash, "type": "git_metadata"}

    except Exception:
        return None


def get_environment_info():
    """Get environment and runtime information."""
    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "architecture": os.uname().machine if hasattr(os, "uname") else "unknown",
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "user": os.getenv("USER", "unknown"),
        "pwd": os.getcwd(),
        "type": "environment_info",
    }

    # Add key environment variables
    relevant_env_vars = ["PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HOME"]
    env_info["environment_variables"] = {}

    for var in relevant_env_vars:
        value = os.getenv(var)
        if value:
            # Truncate long paths for readability
            if len(value) > 200:
                value = value[:200] + "..."
            env_info["environment_variables"][var] = value

    return env_info


def generate_sbom():
    """Generate comprehensive SBOM."""
    print("📦 Generating Software Bill of Materials (SBOM)...")

    timestamp = datetime.datetime.now(timezone.utc)

    sbom = {
        "sbom_version": "1.0",
        "format": "custom",  # Could be upgraded to SPDX/CycloneDX later
        "generated_at": timestamp.isoformat(),
        "generated_by": "automated_pipeline",
        "document_id": f"sbom_{timestamp.strftime('%Y%m%d_%H%M%SZ')}",
        "tool": {"name": "sbom_generator.py", "version": "1.0", "vendor": "Internal"},
        "components": [],
    }

    # Add Python dependencies
    print("  📚 Scanning Python dependencies...")
    python_deps = get_python_dependencies()
    sbom["components"].extend(python_deps)
    print(f"     Found {len(python_deps)} Python packages")

    # Add system dependencies
    print("  🔧 Scanning system dependencies...")
    system_deps = get_system_dependencies()
    sbom["components"].extend(system_deps)
    print(f"     Found {len(system_deps)} system tools")

    # Add model artifacts
    print("  🤖 Scanning model artifacts...")
    model_artifacts = get_model_artifacts()
    sbom["components"].extend(model_artifacts)
    print(f"     Found {len(model_artifacts)} model artifacts")

    # Add Git metadata
    print("  📝 Getting Git metadata...")
    git_metadata = get_git_metadata()
    if git_metadata:
        sbom["components"].append(git_metadata)
        print(f"     Git commit: {git_metadata.get('commit_hash', 'N/A')[:8]}")

    # Add environment info
    print("  🌍 Getting environment info...")
    env_info = get_environment_info()
    sbom["components"].append(env_info)

    # Add summary statistics
    sbom["summary"] = {
        "total_components": len(sbom["components"]),
        "python_packages": len(
            [c for c in sbom["components"] if c.get("type") == "python_package"]
        ),
        "system_tools": len(
            [c for c in sbom["components"] if c.get("type") == "system_tool"]
        ),
        "model_artifacts": len(
            [c for c in sbom["components"] if c.get("type") == "model_artifact"]
        ),
        "git_metadata": 1 if git_metadata else 0,
        "environment_info": 1,
    }

    print(f"  ✅ SBOM generated with {sbom['summary']['total_components']} components")

    return sbom


def sign_sbom(sbom_file):
    """Sign SBOM with simple hash-based signature."""
    print("✍️ Signing SBOM...")

    try:
        # Calculate SBOM hash
        sbom_hash = calculate_file_hash(sbom_file)

        # Create signature metadata
        signature = {
            "signature_version": "1.0",
            "signed_at": datetime.datetime.now(timezone.utc).isoformat(),
            "signer": "automated_pipeline",
            "algorithm": "SHA256",
            "sbom_file": os.path.basename(sbom_file),
            "sbom_hash": sbom_hash,
            "signature_type": "hash_based",  # Would be upgraded to PKI in production
        }

        signature_file = sbom_file.replace(".json", ".sig")
        with open(signature_file, "w") as f:
            json.dump(signature, f, indent=2)

        print(f"  ✅ SBOM signature: {signature_file}")
        return signature_file

    except Exception as e:
        print(f"  ❌ SBOM signing failed: {e}")
        return None


def verify_sbom(sbom_file, signature_file):
    """Verify SBOM signature."""
    print("🔍 Verifying SBOM signature...")

    try:
        # Load signature
        with open(signature_file, "r") as f:
            signature = json.load(f)

        # Calculate current hash
        current_hash = calculate_file_hash(sbom_file)

        # Verify hash
        if current_hash == signature.get("sbom_hash"):
            print("  ✅ SBOM signature verification PASSED")
            return True
        else:
            print("  ❌ SBOM signature verification FAILED")
            return False

    except Exception as e:
        print(f"  ❌ SBOM signature verification ERROR: {e}")
        return False


def main():
    """Main SBOM generator function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate SBOM for model artifacts")
    parser.add_argument(
        "--output",
        "-o",
        default="artifacts/sbom",
        help="Output directory for SBOM files",
    )
    parser.add_argument("--verify", help="Verify existing SBOM signature file")
    args = parser.parse_args()

    if args.verify:
        # Verification mode
        signature_file = args.verify
        sbom_file = signature_file.replace(".sig", ".json")

        if not os.path.exists(sbom_file) or not os.path.exists(signature_file):
            print("❌ SBOM or signature file not found")
            return 1

        if verify_sbom(sbom_file, signature_file):
            print("✅ SBOM verification successful")
            return 0
        else:
            print("❌ SBOM verification failed")
            return 1

    # Generation mode
    print("📦 SBOM Generator for Supply-Chain Hardening")
    print("=" * 50)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    # Generate SBOM
    sbom = generate_sbom()

    # Save SBOM
    sbom_file = output_dir / f"sbom_{timestamp}.json"
    with open(sbom_file, "w") as f:
        json.dump(sbom, f, indent=2, default=str)

    print(f"📄 SBOM saved: {sbom_file}")

    # Sign SBOM
    signature_file = sign_sbom(str(sbom_file))

    # Verify signature
    if signature_file:
        verify_sbom(str(sbom_file), signature_file)

    # Create latest symlinks
    latest_sbom = output_dir / "sbom_latest.json"
    latest_sig = output_dir / "sbom_latest.sig"

    if latest_sbom.exists():
        latest_sbom.unlink()
    if latest_sig.exists():
        latest_sig.unlink()

    latest_sbom.symlink_to(sbom_file.name)
    if signature_file:
        latest_sig.symlink_to(os.path.basename(signature_file))

    # Summary
    print("=" * 50)
    print(f"✅ SBOM Generation Complete")
    print(f"Components: {sbom['summary']['total_components']}")
    print(f"SBOM File: {sbom_file}")
    print(f"Signature: {signature_file if signature_file else 'N/A'}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
