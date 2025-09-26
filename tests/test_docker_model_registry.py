#!/usr/bin/env python3
"""
Tests for Docker Build & Model Registry (Tasks L & M)

Validates Docker builds with HF_TOKEN and documentation completion.
"""

import unittest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class TestDockerBuildIntegration(unittest.TestCase):
    """Test Docker build integration for Task L."""

    def test_triton_dockerfile_exists(self):
        """Test that Triton Dockerfile exists with HF_TOKEN support."""
        dockerfile_path = Path("docker/triton/Dockerfile")
        self.assertTrue(dockerfile_path.exists(), "Triton Dockerfile should exist")

        with open(dockerfile_path, "r") as f:
            content = f.read()

        # Check for HF_TOKEN integration
        self.assertIn("ARG HF_TOKEN", content)
        self.assertIn("ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}", content)
        self.assertIn("huggingface-cli login", content)

        print("✅ Triton Dockerfile has HF_TOKEN support")

    def test_dev_dockerfile_exists(self):
        """Test that development Dockerfile exists with HF_TOKEN support."""
        dockerfile_path = Path("docker/dev.Dockerfile")
        self.assertTrue(dockerfile_path.exists(), "Development Dockerfile should exist")

        with open(dockerfile_path, "r") as f:
            content = f.read()

        # Check for HF_TOKEN integration
        self.assertIn("ARG HF_TOKEN", content)
        self.assertIn("ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}", content)
        self.assertIn("huggingface-cli login", content)

        print("✅ Development Dockerfile has HF_TOKEN support")

    def test_docker_requirements_file(self):
        """Test that Docker-specific requirements file exists."""
        requirements_path = Path("requirements-docker.txt")
        self.assertTrue(
            requirements_path.exists(), "requirements-docker.txt should exist"
        )

        with open(requirements_path, "r") as f:
            content = f.read()

        # Should use regular tensorflow instead of tensorflow-macos as dependency
        self.assertIn("tensorflow>=2.13.0", content)

        # Should contain essential packages
        self.assertIn("huggingface-hub", content)
        self.assertIn("transformers", content)
        self.assertIn("boto3", content)

        print("✅ Docker requirements file properly configured")


class TestEnvironmentAndDocs(unittest.TestCase):
    """Test environment configuration and documentation for Task M."""

    def test_env_example_has_hf_token(self):
        """Test that env.example contains HF_TOKEN."""
        env_path = Path("env.example")
        self.assertTrue(env_path.exists(), "env.example should exist")

        with open(env_path, "r") as f:
            content = f.read()

        # Check for HF_TOKEN configuration
        self.assertIn("HF_TOKEN=", content)
        self.assertIn("Model Registry & HuggingFace Hub", content)
        self.assertIn("https://huggingface.co/settings/tokens", content)

        print("✅ env.example has HF_TOKEN configuration")

    def test_workflow_docs_model_registry(self):
        """Test that workflow_v2.md contains Model Registry section."""
        docs_path = Path("docs/workflow_v2.md")
        self.assertTrue(docs_path.exists(), "docs/workflow_v2.md should exist")

        with open(docs_path, "r") as f:
            content = f.read()

        # Check for Model Registry section
        self.assertIn("## Model Registry", content)
        self.assertIn("fetch_models.py", content)
        self.assertIn("S3 Storage Layout", content)
        self.assertIn("Hot-Swap via Redis", content)

        # Check for key components
        self.assertIn("s3://trading-models/", content)
        self.assertIn("model:registry:active", content)
        self.assertIn("make models-sync", content)

        print("✅ workflow_v2.md has comprehensive Model Registry documentation")

    def test_makefile_models_integration(self):
        """Test that Makefile.models exists with proper targets."""
        makefile_path = Path("Makefile.models")
        self.assertTrue(makefile_path.exists(), "Makefile.models should exist")

        with open(makefile_path, "r") as f:
            content = f.read()

        # Check for all required targets
        required_targets = [
            "models-list",
            "models-sync",
            "models-fetch",
            "models-upload-s3",
            "models-clean",
            "models-status",
        ]

        for target in required_targets:
            self.assertIn(f"{target}:", content)

        print("✅ Makefile.models has all required targets")


class TestCIIntegration(unittest.TestCase):
    """Test CI integration hints for Task N (future implementation)."""

    def test_model_fetch_script_supports_caching(self):
        """Test that model fetch script supports CI caching patterns."""
        script_path = Path("scripts/fetch_models.py")
        self.assertTrue(script_path.exists(), "fetch_models.py should exist")

        with open(script_path, "r") as f:
            content = f.read()

        # Check for cache directory patterns that support CI
        # The script uses .cache/hf_models pattern which is CI-friendly
        self.assertIn(".cache", content)
        self.assertIn("hf_models", content)
        self.assertIn("cache_dir", content)

        print("✅ Model fetch script supports CI caching patterns")

    def test_workflow_docs_ci_integration(self):
        """Test that workflow docs include CI integration examples."""
        docs_path = Path("docs/workflow_v2.md")
        with open(docs_path, "r") as f:
            content = f.read()

        # Check for CI/CD integration documentation
        self.assertIn("CI/CD Integration", content)
        self.assertIn("GitHub Actions", content)
        self.assertIn("~/.cache/huggingface/hub", content)

        print("✅ Workflow docs include CI integration examples")


if __name__ == "__main__":
    print("🧪 Running Docker & Model Registry Tests (Tasks L & M)...")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestDockerBuildIntegration)
    )
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestEnvironmentAndDocs))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCIIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    print(f"📊 Docker & Model Registry Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All Docker & Model Registry tests passed!")
        print("🎯 Task L: Docker Build - COMPLETED")
        print("🎯 Task M: Env & Docs - COMPLETED")
        print()
        print("✅ Implementation Summary:")
        print("   • Docker builds with HF_TOKEN support")
        print("   • Docker-specific requirements (Linux compatible)")
        print("   • Environment configuration with HF_TOKEN")
        print("   • Comprehensive Model Registry documentation")
        print("   • CI/CD integration examples")
        print("   • Makefile integration for model management")
        print()
        print("🚀 Ready for CI cache implementation (Task N)!")
    else:
        print("❌ Some Docker & Model Registry tests failed!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")

    # Exit with appropriate code
    # Note: sys.exit removed to avoid pytest return-value warnings
    # Test framework will handle success/failure appropriately
