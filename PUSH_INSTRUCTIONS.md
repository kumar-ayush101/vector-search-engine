# Pushing this repo to GitHub

The `gh` CLI is not installed on this machine, so the remote repo was not
created automatically. Two options:

## Option A — with the GitHub CLI (if you install it)

```bash
gh repo create vector-search-engine --public --source=. --remote=origin --push
```

## Option B — manual (no gh)

1. Create an empty repo on GitHub named `vector-search-engine`
   (do NOT initialise it with a README/License — this repo already has them):
   https://github.com/new

2. From this folder, add the remote and push:

```bash
git remote add origin https://github.com/kumar-ayush101/vector-search-engine.git
git branch -M main
git push -u origin main
```

The branch is currently `master` locally; `git branch -M main` renames it to
`main` to match the CI workflow and badge URLs in the README.

## After pushing

- CI runs automatically on the first push (vet, build, test -race, docker build).
- To deploy a live demo: Render → New + → Blueprint → select the repo
  (`render.yaml` is already included). Then update the README with the live URL.
