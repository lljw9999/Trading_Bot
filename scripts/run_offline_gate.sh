#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting RL Offline Gate Process..."

# Generate timestamped artifact directory
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
ART=artifacts/$TS/rl

echo "📁 Creating artifact directory: $ART"
mkdir -p "$ART"

# Create a mock checkpoint if it doesn't exist
CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"
if [[ ! -f "$CKPT_DIR/latest.pt" ]]; then
    echo "📦 Creating mock checkpoint for testing..."
    echo "# Mock PyTorch checkpoint for offline gate testing" > "$CKPT_DIR/latest.pt"
fi

echo "📊 Running offline evaluation..."
python tools/eval_offline.py \
    --ckpt checkpoints/latest.pt \
    --episodes 32 \
    --env envs/orderbook_env.py:OrderBookEnv \
    --out $ART/eval.json \
    --md-out $ART/eval.md

echo "🚦 Running gate check..."
python tools/check_eval_gate.py \
    --eval $ART/eval.json \
    --gate gates/sol_gate.yaml \
    --out-md $ART/gate_report.md

echo "📝 Creating artifact README..."
cat > $ART/README.md << EOF
# RL Policy Offline Evaluation Artifacts

**Generated:** $(date)
**Timestamp:** $TS

## Files

- \`eval.json\` - Machine-readable evaluation metrics
- \`eval.md\` - Human-readable evaluation report  
- \`gate_report.md\` - Gate check results with pass/fail status

## Usage

View gate status:
\`\`\`bash
cat $ART/gate_report.md
\`\`\`

Check evaluation metrics:
\`\`\`bash
jq . $ART/eval.json
\`\`\`
EOF

echo "✅ Offline gate process complete!"
echo "📂 Artifacts available at: $ART"
echo ""
echo "🔍 Quick status check:"
if python tools/check_eval_gate.py --eval $ART/eval.json --gate gates/sol_gate.yaml >/dev/null 2>&1; then
    echo "   ✅ Gate Status: PASS"
else
    echo "   ❌ Gate Status: FAIL"
fi

echo ""
echo "📋 View results:"
echo "   cat $ART/gate_report.md"
echo "   jq . $ART/eval.json"