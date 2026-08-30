# Security policy

## Supported versions

Security fixes are applied to the latest release and `main`. Older releases are
not maintained unless a release note explicitly says otherwise.

## Reporting a vulnerability

Do not open a public issue for a vulnerability, leaked credential, signed URL,
or private media.

Use GitHub's **Security → Report a vulnerability** form on this repository.
Include:

- the affected version or commit;
- the attack preconditions and impact;
- minimal reproduction steps or a proof of concept;
- any suggested mitigation;
- whether sensitive data or credentials may have been exposed.

You should receive an acknowledgement within seven days. Please allow time for
triage and a coordinated fix before public disclosure.

## Deployment boundary

media-analysis is designed as a private service-to-service worker. It is not a
public internet API.

- Keep port `5001`, `/ready`, `/docs`, and `/redoc` on a private network.
- Set a strong `MEDIA_ANALYSIS_KEY` and rotate it like any service credential.
- Restrict `MEDIA_ANALYSIS_ALLOWED_HOSTS` to exact signed-URL hosts.
- Never log or commit signed URLs, source media, frames, masks, or transcripts.
- Keep stub/reference model settings disabled in production.
- Treat `productionInferenceReady: false` as a deployment blocker, even when
  `ready: true`.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for model provenance and production
gate requirements.
