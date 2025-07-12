#!/bin/bash
# deploy-hippocampal-experiment.sh
# Deployment script for hippocampal oscillations experiment on NRP cluster

set -e

echo "=========================================="
echo "Axolotl Hippocampal Oscillations Deployment"
echo "Sample: 2023-12-03-e-Hc112823_avv9hckcr1"
echo "Conditions: baseline, optogenetics, kainic acid"
echo "Oscillations: ripples, sharp wave ripples, HFOs"
echo "=========================================="

# Configuration
NAMESPACE="braingeneers"
EXPERIMENT_NAME="hippocampal-oscillations-2023-12-03"
SAMPLE_NAME="2023-12-03-e-Hc112823_avv9hckcr1"

# Check if we're in the right directory
if [ ! -d "k8s" ]; then
    echo "Error: Please run this script from the axolotl repository root directory"
    exit 1
fi

# Check kubectl connection
echo "Checking kubectl connection to NRP cluster..."
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo "Error: Cannot connect to Kubernetes cluster. Please check your kubectl configuration."
    exit 1
fi

echo "Connected to cluster: $(kubectl config current-context)"

# Check if namespace exists
if ! kubectl get namespace $NAMESPACE > /dev/null 2>&1; then
    echo "Error: Namespace '$NAMESPACE' does not exist"
    exit 1
fi

echo "Using namespace: $NAMESPACE"

# Deploy ConfigMap
echo ""
echo "Deploying configuration..."
kubectl apply -f k8s/hippocampal-oscillations-configmap.yaml

# Verify ConfigMap was created
if kubectl get configmap axolotl-hippocampal-config -n $NAMESPACE > /dev/null 2>&1; then
    echo "✓ ConfigMap deployed successfully"
else
    echo "✗ ConfigMap deployment failed"
    exit 1
fi

# Deploy Job
echo ""
echo "Deploying job..."
kubectl apply -f k8s/hippocampal-oscillations-job.yaml

# Verify Job was created
if kubectl get job axolotl-$EXPERIMENT_NAME -n $NAMESPACE > /dev/null 2>&1; then
    echo "✓ Job deployed successfully"
else
    echo "✗ Job deployment failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Job name: axolotl-$EXPERIMENT_NAME"
echo "Namespace: $NAMESPACE"
echo ""
echo "Monitor your job with:"
echo "  kubectl get job axolotl-$EXPERIMENT_NAME -n $NAMESPACE"
echo "  kubectl describe job axolotl-$EXPERIMENT_NAME -n $NAMESPACE"
echo ""
echo "View logs with:"
echo "  kubectl logs -f job/axolotl-$EXPERIMENT_NAME -n $NAMESPACE"
echo ""
echo "Check pod status with:"
echo "  kubectl get pods -l job-name=axolotl-$EXPERIMENT_NAME -n $NAMESPACE"
echo ""
echo "Results will be saved to:"
echo "  s3://braingeneers/oscillations/results/$SAMPLE_NAME/"
echo ""
echo "Expected analysis duration: 30-60 minutes"
echo "=========================================="
