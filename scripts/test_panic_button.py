#!/usr/bin/env python3
"""
Panic Button System Test Script

Tests the emergency panic button functionality to ensure it works correctly
before deployment to production.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.panic_button import PanicButton


def test_panic_button_functionality():
    """Test panic button system functionality."""
    print("🧪 Testing Panic Button System")
    print("=" * 50)

    panic_button = PanicButton()

    # Test 1: Check initial status
    print("\n📊 Test 1: Check initial panic status")
    status = panic_button.check_panic_status()
    print(f"Initial status: {json.dumps(status, indent=2)}")

    # Test 2: Execute mock panic sequence
    print("\n🚨 Test 2: Execute panic sequence (test mode)")
    results = panic_button.execute_panic_sequence(
        reason="System test - not a real emergency", initiated_by="test_script"
    )

    print(f"Panic execution results:")
    print(f"  Overall success: {results.get('overall_success', False)}")
    print(f"  Execution time: {results.get('execution_time_seconds', 0):.1f}s")
    print(f"  Steps completed: {len(results.get('sequence_steps', {}))}")

    # Show detailed results for key steps
    steps = results.get("sequence_steps", {})
    for step_name, step_result in steps.items():
        success = step_result.get("success", False)
        icon = "✅" if success else "❌"
        print(f"  {icon} {step_name}: {'SUCCESS' if success else 'FAILED'}")

    # Test 3: Check post-panic status
    print("\n📊 Test 3: Check post-panic status")
    post_status = panic_button.check_panic_status()
    print(f"Post-panic status: {json.dumps(post_status, indent=2)}")

    # Test 4: Clear panic mode
    print("\n🔓 Test 4: Clear panic mode")
    clear_result = panic_button.clear_panic_mode(cleared_by="test_script")
    print(f"Clear result: {json.dumps(clear_result, indent=2)}")

    # Test 5: Final status check
    print("\n📊 Test 5: Final status check")
    final_status = panic_button.check_panic_status()
    print(f"Final status: {json.dumps(final_status, indent=2)}")

    # Summary
    print("\n" + "=" * 50)
    overall_success = results.get("overall_success", False)
    if overall_success:
        print("✅ PANIC BUTTON TEST: PASSED")
        print("   All core functionality working correctly")
    else:
        print("❌ PANIC BUTTON TEST: FAILED")
        print("   Some functionality needs attention")

    return overall_success


if __name__ == "__main__":
    try:
        success = test_panic_button_functionality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)
