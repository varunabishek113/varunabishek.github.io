# varunabishek.github.io

Personal blog built with [Pelican](https://getpelican.com/) and deployed to GitHub Pages via GitHub Actions.

🔗 **Live site:** [https://varunabishek113.github.io/varunabishek.github.io/](https://varunabishek113.github.io/varunabishek.github.io/)

## Setup

```bash
pip install -r requirements.txt
```

## Local Development

```bash
make devserver
```

Serves the site locally with auto-reload.

## Build for Production

```bash
make publish
```

Generates the site using `publishconf.py` into the `output/` directory.

## Deployment

Pushes to `main` automatically trigger a GitHub Actions workflow that builds and deploys the site to GitHub Pages.

## Project Structure

```
content/          # Blog posts and pages (Markdown)
content/images/   # Static images
content/pages/    # Static pages (e.g., About)
theme/            # Custom Pelican theme (Flex-based)
pelicanconf.py    # Development config
publishconf.py    # Production config
```

Metadata (`Title:`, `Date:`, etc.) must be the first thing in a post — no blank lines before it. The deploy workflow normalizes this automatically, but keeping it clean locally avoids surprises.


## Screenshots
![1775872959937](image/README/1775872959937.png)