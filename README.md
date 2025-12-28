# Machine Learning: Algorithms and Applications

An intelligent textbook covering machine learning algorithms and applications, built with MkDocs Material.

## Overview

This comprehensive textbook provides a rigorous yet accessible introduction to machine learning, designed for college undergraduate students. It covers:

- Supervised Learning (KNN, Decision Trees, Logistic Regression, SVMs)
- Unsupervised Learning (K-Means Clustering)
- Deep Learning (Neural Networks, CNNs, Transfer Learning)
- Practical Implementation with Python

## Features

- **Learning Graph**: 200 concepts organized into a Directed Acyclic Graph (DAG)
- **14 Taxonomies**: Concepts categorized for easy navigation
- **Interactive Navigation**: MkDocs Material theme with search and navigation features
- **Mathematical Support**: MathJax for equations and formulas
- **Code Examples**: Syntax-highlighted code blocks with copy functionality
- **Quality Validated**: All content validated with automated quality checks

## Prerequisites

- Linear algebra (matrix operations, eigenvalues/eigenvectors)
- Calculus (derivatives, chain rule, gradients)
- Programming experience (Python recommended)

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Build the Documentation

```bash
# Build the static site
mkdocs build
```

### 3. Serve Locally

```bash
# Start the development server
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

### 4. Deploy to GitHub Pages

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Project Structure

```
Textbook/
├── mkdocs.yml              # MkDocs configuration
├── requirements.txt        # Python dependencies
├── docs/                   # Documentation source
│   ├── index.md           # Homepage
│   ├── course-description.md
│   ├── stylesheets/       # Custom CSS
│   │   └── extra.css
│   └── learning-graph/    # Learning graph files
│       ├── index.md
│       ├── concept-list.md
│       ├── learning-graph.csv
│       ├── learning-graph.json
│       ├── concept-taxonomy.md
│       ├── quality-metrics.md
│       └── taxonomy-distribution.md
├── logs/                   # Generation logs
└── site/                   # Built documentation (generated)
```

## Learning Graph

The textbook uses a learning graph with:

- **200 Concepts**: Organized from foundational to advanced
- **289 Dependencies**: Showing prerequisite relationships
- **14 Categories**: Foundation, KNN, Decision Trees, Logistic Regression, Regularization, SVM, Clustering, Neural Networks, CNNs, Transfer Learning, Evaluation, Preprocessing, Optimization, Miscellaneous
- **Quality Score**: 85/100 (Good)

### Taxonomy Distribution

- Foundation Concepts (FOUND): 31 (15.5%)
- Neural Networks (NN): 37 (18.5%)
- Convolutional Networks (CNN): 20 (10.0%)
- Evaluation Metrics (EVAL): 19 (9.5%)
- Support Vector Machines (SVM): 16 (8.0%)
- Optimization (OPT): 16 (8.0%)
- And 8 more categories...

## MkDocs Configuration

The `mkdocs.yml` file includes:

- **Material Theme**: Modern, responsive design with dark mode
- **Navigation**: Tabs, sections, and instant loading
- **Search**: Full-text search with suggestions
- **Extensions**: Admonitions, code highlighting, math support, diagrams
- **Plugins**: Minify, search optimization

## Development

### Adding New Content

1. Create markdown files in the `docs/` directory
2. Update `mkdocs.yml` navigation structure
3. Run `mkdocs serve` to preview changes
4. Build with `mkdocs build` when ready

### Customization

- **Theme colors**: Edit `mkdocs.yml` → `theme.palette`
- **Custom CSS**: Edit `docs/stylesheets/extra.css`
- **Navigation**: Edit `mkdocs.yml` → `nav`

## License

CC BY-NC-SA 4.0 DEED

## Credits

Generated with [Claude Code](https://claude.com/claude-code) using the Learning Graph Generator Skill v0.03.

## Next Steps

- Run the `book-chapter-generator` skill to create chapter structure
- Review and refine concept dependencies
- Add interactive MicroSims and visualizations
- Generate quiz questions for each chapter
- Create glossary from concept list
