# Security Policy

## Supported versions

Only the latest commit on `master` is actively maintained. There are no LTS branches.

## Reporting a vulnerability

If you find a security issue, please report it privately rather than opening a public issue.

**Email:** marcosashiglesias@gmail.com

Include:

- A description of the vulnerability
- Steps to reproduce or a proof of concept
- The affected component (simulation, website, Modal worker)

I'll acknowledge receipt within 48 hours and aim to have a fix or mitigation within a week, depending on severity.

## Scope

The main attack surface is the web frontend and the Modal render API. The C simulation binary runs server-side in a sandboxed container and does not accept untrusted network input directly.

Areas of particular concern:

- **OBJ file uploads** -- the parser handles untrusted mesh data. Malformed files should not cause crashes or memory corruption.
- **API rate limiting** -- the render endpoint has per-IP rate limits to prevent abuse.
- **Modal worker** -- runs with GPU access on shared infrastructure. Secrets (AWS keys, Modal tokens) must not leak through stdout or error responses.
- **Frontend** -- standard web risks (XSS via URL hash params, CSRF on the render endpoint).

Out of scope:

- Denial of service via submitting expensive simulations (this is an inherent cost of the tool, mitigated by duration caps and rate limits)
- Vulnerabilities in third-party dependencies that have no feasible exploit path in this project

## Disclosure

Once a fix is deployed, I'll document the issue in the commit message and, if warranted, in a GitHub security advisory. Credit will be given to the reporter unless they prefer to stay anonymous.
