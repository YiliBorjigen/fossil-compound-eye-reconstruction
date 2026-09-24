# Contributing to SplitScope

Work inside `apps/splitscope/`; do not alter the parent repository's datasets or historical scientific results as part of an interface change.

1. Open an issue describing the user problem or a minimal numerical counterexample. Do not upload confidential data.
2. Keep the runtime dependency-free and the downloaded HTML self-contained.
3. Keep prediction, split assignment and metrics in `src/engine.mjs`, separate from the DOM.
4. Add a meaningful invariant or independent expected-value test for changes to numerical behaviour.
5. Run `npm test` and `npm run build`, then check the affected controls in a browser.
6. Commit the regenerated `dist/index.html` and `dist/splitscope.html` with source changes.

Scientific claims should distinguish interpolation, regional holdout and independent-specimen transfer. Do not label a score difference as confirmed leakage. Cite upstream ideas and obtain permission/licensing before adding external datasets.

Potential next contributions: independently cross-check the predictions against a standard library, improve keyboard inspection of individual points, or add documented repeated-holdout uncertainty without turning repeated points into independent specimens.
