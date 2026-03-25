# Deployment to PyPi

1. **Install uv** (if you haven't):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

2. **Tag & Push to Main**

First push all changes.



Then create a tag, make sure this is the same version as the pyproject.toml (and increase if needed)
```bash 
git tag v0.2.0
```
Then push the tag: (this will update documentation with github actions)
```bash
git push origin --tags 
```

1. **Build and Publish**:
From your project root, run:

```bash
uv build && uv publish -t <token>

```

Or instead of using `-t`, when prompted for a username, enter `__token__`. For the password, paste your PyPI token (including the `pypi-` prefix).

