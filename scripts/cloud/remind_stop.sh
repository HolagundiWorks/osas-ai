#!/usr/bin/env bash
# Print the manual stop steps. This script does not call a cloud API.
set -euo pipefail
cat <<'EOF'
Training is finished only if train_result.json exists and steps_completed is set.
This repository did not provision a cloud GPU.

Stop billing yourself:
- RunPod: open the Pods page and stop or terminate the pod.
- Lambda Cloud: open Instances and terminate the instance.
- Any other host: terminate the GPU instance from its console. Closing an SSH session does not stop billing.

Do not upload customer drawings, proprietary models or credentials with the checkpoints.
Copy only the adapter directory and manifest to storage you control, then terminate the GPU.
EOF
