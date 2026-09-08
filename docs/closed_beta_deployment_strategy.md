# FIES Closed-Beta Deployment Strategy

## Objective

Deploy FIES to a small invited user cohort with production data, deterministic ranking, auditable reasoning traces, and fast rollback. Closed beta is for observing real query behavior and evidence gaps, not for personalized financial advice.

## Release gates before deployment

All of the following must be true:

- Unit + integration reasoning suites green.
- Production E2E finishes without unhandled exceptions.
- Hard evidence dimensions can complete in one investigation round for the flagship query.
- No mock capabilities are registered in the production composition root.
- Candidate universe passes mandate and data-quality gates.
- Direct/Regular share classes are deduplicated before ranking.
- Fund-specific documentary retrieval fails soft when documents are not indexed.
- Final answer is research/ranking language, not personalized advice.
- Secrets are sourced from AWS Secrets Manager or environment injection, never committed.
- Static AWS credentials are absent from the repository and runtime image.
- Production database migrations are applied and verified before traffic.

## AWS topology

Internet
  -> Route 53
  -> AWS WAF
  -> Application Load Balancer
  -> ECS Fargate / FastAPI service
       -> RDS PostgreSQL + pgvector
       -> S3 raw/processed document storage
       -> Secrets Manager
       -> Anthropic/OpenAI provider
       -> CloudWatch logs/metrics

Keep RDS in private subnets. ECS tasks run in private application subnets with controlled egress. The ALB is the only public application entry point.

## Closed-beta access

Use an invite-only authentication boundary. Preferred options:

1. Amazon Cognito user pool with manually invited users.
2. Existing organization SSO if already available.

Do not expose an unauthenticated public reasoning endpoint during closed beta.

Recommended initial cohort: 10-25 users.

## API release shape

Expose a versioned API:

- POST /api/v1/reason
- GET /api/v1/health
- GET /api/v1/ready

Return a request_id for every reasoning request.

For beta, apply:

- per-user rate limits;
- bounded request size;
- bounded reasoning rounds;
- bounded tool calls;
- bounded LLM calls;
- explicit provider timeout;
- database statement timeout;
- maximum candidate universe.

## Reasoning runtime limits

Recommended initial beta limits:

- max_investigation_rounds = 3
- max_replans = 2
- max_llm_calls = 2
- max_tool_calls = 30
- candidate discovery limit = 20
- final shortlist = 5-10

These remain harness-owned server-side controls and cannot be overridden by the client.

## Observability

Every request should emit:

- request_id
- user_id/hash
- investment_mandate
- eligible category count
- candidate count
- data-quality rejection count
- hard evidence pass count
- peer-comparable count
- ranked count
- planner LLM calls
- tool calls
- replans
- investigation rounds
- soft evidence gaps
- confidence score
- finalization vs abstention
- total latency
- planner latency
- retrieval latency
- database latency
- risk-computation latency
- ranking latency
- provider token usage/cost
- failure_domain
- retryable
- replannable

Create CloudWatch alarms for:

- API 5xx rate
- p95/p99 latency
- RDS CPU/connections
- ECS task restarts
- LLM provider errors
- evidence insufficiency spikes
- abstention spikes
- planner repair/replan spikes
- empty ranking results

## Beta dashboards

Dashboard 1: Runtime health
- request volume
- success/finalize rate
- p50/p95/p99 latency
- 5xx
- ECS/RDS health

Dashboard 2: Reasoning quality
- first-pass success rate
- replan rate
- evidence sufficiency rate
- candidate rejection rate
- soft evidence gaps
- average confidence
- abstention rate

Dashboard 3: Cost
- LLM calls/request
- prompt/completion tokens
- retrieval calls
- DB query latency
- estimated cost/request

## Deployment pipeline

Suggested CI/CD flow:

PR
 -> tests
 -> security/static checks
 -> Docker build
 -> image scan
 -> push to ECR
 -> deploy staging ECS service
 -> staging smoke E2E
 -> manual closed-beta approval
 -> deploy production beta ECS service
 -> production smoke test
 -> enable invited cohort

Use immutable image tags based on Git commit SHA.

## Rollout strategy

Start with one ECS task and a low invited-user limit.

Phase 1:
- internal users only
- verify traces, latency, ranking outputs

Phase 2:
- 10 beta users
- daily review of reasoning failures and evidence gaps

Phase 3:
- 25 beta users
- enable gradual traffic increase only if SLOs remain healthy

Do not automatically expand cohort based only on infrastructure health; reasoning-quality metrics must remain healthy as well.

## Initial beta SLOs

Targets for closed beta:

- availability >= 99%
- no fabricated financial evidence
- hard-evidence first-pass success >= 90%
- unhandled exception rate < 1%
- empty-ranking rate < 2%
- p95 reasoning latency target <= 30 seconds after latency optimization
- all final rankings include confidence and limitations

These are beta targets and should be recalibrated from real traffic.

## Failure policy

Execution failure:
- retry only if explicitly classified retryable
- apply bounded backoff
- do not automatically replan

Reasoning failure:
- replan if replannable and budget remains

Evidence insufficiency:
- hard evidence missing -> replan or abstain
- soft evidence missing -> finalize with limitation and confidence penalty

Configuration failure:
- fail fast
- no repeated reasoning rounds
- alarm operator

## Rollback criteria

Immediately disable or roll back the beta deployment if any of these occur:

- fabricated or untraceable financial evidence;
- user-visible cross-request data leakage;
- database credential/security exposure;
- repeated incorrect category eligibility;
- data-quality gate bypass;
- ranking produced without hard evidence;
- unexplained confidence inflation;
- sustained 5xx > 5%;
- sustained p95 latency above agreed beta ceiling;
- provider failure causes repeated runaway replans.

Rollback is an ECS task-definition rollback to the last known-good image.

## Beta feedback loop

For each problematic beta query, classify the issue as one of:

- planner semantics
- eligibility policy
- data quality
- missing structured data
- retrieval/document corpus
- peer comparison
- ranking
- confidence
- execution/infrastructure

Do not patch the LLM prompt first by default. Prefer deterministic policy, ontology, data, or capability fixes when the failure is structural.

## Go / No-Go

GO when:
- full test suite green;
- production smoke query succeeds;
- no mocks;
- confidence/limitations visible;
- hard evidence gates enforced;
- rollback tested;
- monitoring dashboards live.

NO-GO when:
- production ranking can bypass evidence gates;
- documents/data sources are silently substituted;
- provider or DB failures are not classified;
- there is no per-request trace;
- no rollback path exists.
