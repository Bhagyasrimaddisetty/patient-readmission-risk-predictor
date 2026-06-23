#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# push_to_github.sh  —  initialise repo and push to GitHub
# Usage:  bash push_to_github.sh <github-username> <repo-name>
# Example: bash push_to_github.sh johndoe patient-readmission-predictor
# ─────────────────────────────────────────────────────────────
set -e

USERNAME=${1:?"Usage: $0 <github-username> <repo-name>"}
REPO=${2:?"Usage: $0 <github-username> <repo-name>"}

echo "▶ Configuring git …"
git init
git config user.email "${USERNAME}@users.noreply.github.com"
git config user.name  "${USERNAME}"

echo "▶ Staging files …"
git add .
git commit -m "feat: initial commit — patient readmission predictor

- End-to-end ML pipeline: Logistic Regression, Random Forest, XGBoost
- FastAPI REST API with /health, /predict, /predict/batch endpoints
- 14 pytest unit tests (preprocessing + API schema validation)
- Synthetic data generator + Makefile for one-command setup
- EDA Jupyter notebook"

echo "▶ Creating GitHub repo and pushing …"
echo ""
echo "Option A — GitHub CLI (gh):"
echo "  gh repo create ${REPO} --public --source=. --remote=origin --push"
echo ""
echo "Option B — Manual:"
echo "  1. Go to https://github.com/new and create repo: ${REPO}"
echo "  2. Run:"
echo "     git remote add origin https://github.com/${USERNAME}/${REPO}.git"
echo "     git branch -M main"
echo "     git push -u origin main"
echo ""
echo "✅ Local git repo initialised and commit created. Choose Option A or B above."
