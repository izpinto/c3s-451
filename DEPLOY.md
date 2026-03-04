# Deployment to PyPi



1. **Install uv** (if you haven't):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```


2. **Build and Publish**:
From your project root, run:
```bash
uv build
uv publish

```


3. **Authentication**:
When prompted for a username, enter `__token__`. For the password, paste your PyPI token (including the `pypi-` prefix).

