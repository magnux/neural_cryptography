# Licensing and attribution

The original work in **Neural Cryptography** is licensed under the unmodified
[Apache License 2.0](../LICENSE). Original author:
**Alejandro Hernandez Ruiz (magnux)**. The project's attribution is recorded in
[NOTICE](../NOTICE), and citation metadata in [CITATION.cff](../CITATION.cff).

## Coverage

| Component | Original project material covered |
| --- | --- |
| Neural-QR | The optical transport specification, diagrams, and project-owned renderer, acquisition and transport implementations as they are published |
| Neural-Corrector | Model architectures, training, synthetic generation, evaluation, export and browser integration code; original model releases explicitly published under this license |
| Neural-ZKBoo | The project's circuit, proof implementation, application protocol and documentation |
| Repository | Other original source, scripts, tests and documentation, unless a file states a different license |

One license applies to these components; there are no extra component-specific
restrictions. Their names describe this project's work, not ownership of the
underlying ZKBoo research or other cited prior work. See
[research connections](RESEARCH_CONNECTIONS.md) for that attribution.

## Commercial use and credit

Apache-2.0 permits commercial and private use, modification, and redistribution,
including in proprietary products. When distributing this work or derivatives,
comply with section 4: provide the license, retain applicable source notices,
mark changed files, and reproduce the applicable attribution from `NOTICE` in
one of the permitted locations. Those locations include distributed notices,
source/documentation, or the place a product normally displays third-party notices.

The project notice identifies both the original author and
`https://github.com/magnux/neural_cryptography`. Keep it when redistributing the
applicable project material. No visible UI badge or advertising credit is added
as a condition. Citation in publications is requested through `CITATION.cff`;
this request does not add a restriction to Apache-2.0.

The license includes the section 7 warranty disclaimer and section 8 limitation
of liability, subject to their stated legal exceptions. Research functionality,
recovery results, and a successful authentication test are not guarantees of
security, fitness, or performance.

## Methods, patents and third-party material

The license grants rights in the covered implementations and written designs;
it does not establish exclusive copyright ownership of an abstract method,
algorithm, or idea. Apache-2.0 section 3 includes a patent grant for qualifying
contributor-controlled claims, not clearance of every possible third-party claim.
The project does not represent these experimental constructions as audited
cryptography or as a standardized QR format.

[Third-party software](../THIRD_PARTY.md) keeps its own licenses, provenance and
notices. Do not replace vendor copyright notices with the project author's name.
External checkpoints, recordings and datasets also retain their own terms.
Running the training or export scripts does not automatically license someone
else's weights or grant rights in their data. The license of any separately
published model/data release must be stated by its rights holder.

## Prepared apps and model exports

`npm run prepare:apps` includes the project `LICENSE`, `NOTICE`, citation metadata
and dependency inventory in each independently served app. Vendor notices remain
beside their dependencies. The corrector export and preparation workflows carry
the project license and attribution with model bundles; these notices cover the
project contribution and do not override separately licensed checkpoint content.
Local recordings and checkpoints are not published or modified by adopting this license.

## References

- [Apache License 2.0, sections 2–4 and 7–8](https://www.apache.org/licenses/LICENSE-2.0)
- [Apache guidance for applying the license](https://www.apache.org/legal/apply-license)
- [U.S. Copyright Office: ideas, systems and methods versus their expression](https://www.copyright.gov/help/faq/faq-protect.html)

The license text controls; this guide explains its application to the repository.
