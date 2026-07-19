# Kaifang Yao (姚凯方) — Personal Homepage

A personal academic homepage hosted on GitHub Pages at [wwwsaidthat.github.io](https://wwwsaidthat.github.io).

## Design

Modeled after a clean academic personal website style with:
- Left sidebar: photo gallery, name (English + Chinese), affiliation, social links
- Right content: Biography, News, Publications (with search/filter), Projects, Experience, Education, Services & Honors

Built with Tailwind CSS (CDN), Font Awesome icons, and Inter font. No build step needed — pure static HTML.

## How to Customize

1. **Photos**: Add your photos to `images/` named `photo1.jpg`, `photo2.jpg`, etc. (up to `photo20.jpg`). The gallery will auto-detect and display them.

2. **Profile sidebar** (lines ~130-160 in `index.html`):
   - Update your title/role, department, institution, location
   - Update social links (email, GitHub, Google Scholar, etc.)

3. **Biography** (lines ~170-180): Write about yourself, your research interests, and background.

4. **News** (around line ~185): Add news items with dates following the format.

5. **Publications** (starts around line ~300 in the `<script>` section):
   - Add entries to the `publicationsData` array following the template format
   - Mark important ones with `selected: true`

6. **Projects**, **Experience**, **Education**, **Services**, **Honors**: Fill in each section with your actual information.

## Testing Locally

Open `index.html` directly in your browser, or use any static file server:

```bash
# Using Python 3
python3 -m http.server 8000

# Then visit http://localhost:8000
```

## Deployment

Push to the `main` branch. GitHub Pages will automatically deploy from the repository root.
