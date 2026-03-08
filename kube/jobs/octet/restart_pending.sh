#!/bin/bash
# Script to periodically check a Kubernetes job and delete/recreate it when pending.
# If the job is running, no action is taken.
#
# Usage: ./combine_restart_pending.sh <job.yaml> [interval_seconds]
#   job.yaml: path to the Kubernetes Job YAML file
#   interval_seconds: optional, default 600 (10 minutes)

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <job.yaml> [interval_seconds]"
  echo "  job.yaml: path to the Kubernetes Job YAML file"
  echo "  interval_seconds: optional, default 600 (10 minutes)"
  exit 1
fi

if [ ! -f "$1" ]; then
  echo "Error: Job file not found: $1"
  exit 1
fi
JOB_FILE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
INTERVAL="${2:-600}"

JOB_NAME=$(kubectl apply -f "$JOB_FILE" --dry-run=client -o jsonpath='{.metadata.name}' 2>/dev/null)
if [ -z "$JOB_NAME" ]; then
  echo "Error: Could not parse job name from $JOB_FILE (is it a valid Kubernetes Job YAML?)"
  exit 1
fi

echo "Monitoring job: $JOB_NAME (from $JOB_FILE)"
echo "Will delete and recreate when pending, every ${INTERVAL}s interval"
echo "Press Ctrl+C to stop"
echo "---"

while true; do
  if kubectl get job "$JOB_NAME" &>/dev/null; then
    # Check if any pods are running
    RUNNING_COUNT=$(kubectl get pods -l job-name="$JOB_NAME" --field-selector=status.phase=Running -o name 2>/dev/null | wc -l | tr -d ' ')
    SUCCEEDED=$(kubectl get job "$JOB_NAME" -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")
    FAILED=$(kubectl get job "$JOB_NAME" -o jsonpath='{.status.failed}' 2>/dev/null || echo "0")

    if [ "$RUNNING_COUNT" -gt 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S'): Job $JOB_NAME is running ($RUNNING_COUNT pod(s)). No action needed."
    elif [ "${SUCCEEDED:-0}" -gt 0 ] || [ "${FAILED:-0}" -gt 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S'): Job $JOB_NAME has completed (succeeded=$SUCCEEDED, failed=$FAILED). Skipping."
    else
      # Job exists, no running pods, not completed -> pending
      echo "$(date '+%Y-%m-%d %H:%M:%S'): Job $JOB_NAME is pending. Deleting and recreating..."
      kubectl delete job "$JOB_NAME" --wait=false
      # Wait for job to be fully deleted before recreating
      while kubectl get job "$JOB_NAME" &>/dev/null; do
        sleep 2
      done
      kubectl apply -f "$JOB_FILE"
      echo "$(date '+%Y-%m-%d %H:%M:%S'): Job recreated."
    fi
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Job $JOB_NAME does not exist. Creating..."
    kubectl apply -f "$JOB_FILE"
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S'): Sleeping for $((INTERVAL / 60)) minutes..."
  sleep "$INTERVAL"
done
