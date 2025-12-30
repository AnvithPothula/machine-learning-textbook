# Book Metrics

This file contains overall metrics for the intelligent textbook.

| Metric Name | Value | Link | Notes |
|-------------|-------|------|-------|
| Chapters | 12 | [Chapters](../chapters/index.md) | Number of chapter directories |
| Concepts | 200 | [Concept List](./concept-list.md) | Concepts from learning graph |
| Glossary Terms | 199 | [Glossary](../glossary.md) | Defined terms |
| FAQs | 86 | [FAQ](../faq.md) | Frequently asked questions |
| Quiz Questions | 120 | - | Questions across all chapters |
| Diagrams | 0 | - | Level 4 headers starting with '#### Diagram:' |
| Equations | 970 | - | LaTeX expressions (inline and display) |
| MicroSims | 16 | [Simulations](../sims/index.md) | Interactive MicroSims |
| Total Words | 90,312 | - | Words in all markdown files |
| Links | 93 | - | Hyperlinks in markdown format |
| Equivalent Pages | 369 | - | Estimated pages (250 words/page + visuals) |

## Metrics Explanation

- **Chapters**: Count of chapter directories containing index.md files
- **Concepts**: Number of rows in learning-graph.csv
- **Glossary Terms**: H4 headers in glossary.md
- **FAQs**: H3 headers in faq.md
- **Quiz Questions**: H4 headers with numbered questions (e.g., '#### 1.') or H2 headers in quiz.md files
- **Diagrams**: H4 headers starting with '#### Diagram:'
- **Equations**: LaTeX expressions using $ and $$ delimiters
- **MicroSims**: Directories in docs/sims/ with index.md files
- **Total Words**: All words in markdown files (excluding code blocks and URLs)
- **Links**: Markdown-formatted links `[text](url)`
- **Equivalent Pages**: Based on 250 words/page + 0.25 page/diagram + 0.5 page/MicroSim
