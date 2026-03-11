"""Deterministic corpus generator for A/B benchmark framework.

Generates entities across 6 types, relationships organized into topical
clusters, access patterns with hot/warm/cold/dormant tiers, and ground-truth
queries with graded relevance judgments across 8 categories.

Default scale is 1000 entities / 2500+ relationships / 80 queries / 12 clusters.
Pass ``total_entities`` to ``CorpusGenerator`` to scale up (e.g. 5000, 10000,
50000). All structure scales proportionally while preserving backward
compatibility at the default size.

All randomness flows through a seeded ``random.Random`` instance so that
``generate()`` is fully deterministic given the same seed.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import cast

import aiosqlite

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.relationship import Relationship
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex
from engram.storage.sqlite.hybrid_search import HybridSearchIndex

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

GROUP_ID = "benchmark"


@dataclass
class GroundTruthQuery:
    """A single benchmark query with graded relevance judgments."""

    query_id: str
    query_text: str
    relevant_entities: dict[str, int]  # entity_id -> grade 0-3
    category: str  # "direct", "recency", "frequency", "associative", "temporal_context", etc.
    relevant_episodes: dict[str, int] = field(default_factory=dict)  # episode_id -> grade


@dataclass
class ConversationScenario:
    """A multi-query scenario for testing working memory bridging."""

    name: str
    queries: list[str]
    expected_bridge: dict[int, set[str]]  # query_index -> entity_ids that should appear


@dataclass
class CorpusSpec:
    """Everything needed to reproduce a benchmark run."""

    entities: list[Entity]
    relationships: list[Relationship]
    access_events: list[tuple[str, float]]  # (entity_id, timestamp)
    ground_truth: list[GroundTruthQuery]
    metadata: dict = field(default_factory=dict)
    episodes: list[Episode] = field(default_factory=list)
    episode_entities: list[tuple[str, str]] = field(default_factory=list)  # (episode_id, entity_id)
    conversation_scenarios: list[ConversationScenario] = field(default_factory=list)


@dataclass
class CorpusScale:
    """Controls corpus size. All counts derived from total_entities.

    Default values reproduce the original 1000-entity corpus exactly.
    """

    total_entities: int = 1000

    # Entity type ratios (must sum to 1.0)
    person_ratio: float = 0.20
    technology_ratio: float = 0.20
    organization_ratio: float = 0.15
    location_ratio: float = 0.10
    project_ratio: float = 0.15
    concept_ratio: float = 0.20

    # Cluster sizing
    cluster_size: int = 80

    # Relationship density
    relationships_per_entity: float = 2.5
    intra_degree_range: tuple[int, int] = (2, 5)
    inter_bridges_range: tuple[int, int] = (3, 5)

    # Ground truth query counts per category
    direct_queries: int = 20
    recency_queries: int = 10
    frequency_queries: int = 10
    associative_queries: int = 10
    temporal_context_queries: int = 5
    semantic_queries: int = 10
    graph_traversal_queries: int = 10
    cross_cluster_queries: int = 5

    # Conversation scenarios
    max_scenarios: int = 10

    @property
    def num_clusters(self) -> int:
        """Number of topical clusters (min 12 for backward compat)."""
        return max(12, self.total_entities // self.cluster_size)

    @property
    def target_relationships(self) -> int:
        """Total relationships to generate."""
        return int(self.total_entities * self.relationships_per_entity)

    def compute_type_counts(self) -> dict[str, int]:
        """Compute entity counts per type from ratios.

        At default scale (1000): person=200, technology=200, organization=150,
        location=100, project=150, concept=200.
        """
        types = [
            ("person", self.person_ratio),
            ("technology", self.technology_ratio),
            ("organization", self.organization_ratio),
            ("location", self.location_ratio),
            ("project", self.project_ratio),
            ("concept", self.concept_ratio),
        ]
        counts = {t: int(self.total_entities * r) for t, r in types}
        remainder = self.total_entities - sum(counts.values())
        for i in range(remainder):
            t = types[i % len(types)][0]
            counts[t] += 1
        return counts


# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Hank",
    "Iris",
    "Jack",
    "Karen",
    "Leo",
    "Mia",
    "Noah",
    "Olivia",
    "Paul",
    "Quinn",
    "Rachel",
    "Sam",
    "Tara",
    "Uma",
    "Victor",
    "Wendy",
    "Xander",
    "Yuki",
    "Zara",
    "Adrian",
    "Beth",
    "Carlos",
    "Diana",
]

_LAST_NAMES = [
    "Smith",
    "Chen",
    "Patel",
    "Kim",
    "Garcia",
    "Mueller",
    "Santos",
    "Nakamura",
    "Williams",
    "Brown",
    "Lee",
    "Wang",
    "Anderson",
    "Taylor",
    "Thomas",
    "Jackson",
    "White",
    "Harris",
    "Martin",
    "Thompson",
    "Moore",
    "Clark",
    "Lewis",
    "Robinson",
    "Walker",
    "Young",
    "Allen",
    "King",
    "Wright",
    "Scott",
]

_TECHNOLOGIES = [
    "Python",
    "JavaScript",
    "TypeScript",
    "Rust",
    "Go",
    "Java",
    "C++",
    "Ruby",
    "Swift",
    "Kotlin",
    "React",
    "Vue",
    "Angular",
    "Svelte",
    "Next.js",
    "FastAPI",
    "Django",
    "Flask",
    "Express",
    "Spring Boot",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Redis",
    "Elasticsearch",
    "Kafka",
    "RabbitMQ",
    "GraphQL",
    "gRPC",
    "REST API",
    "Docker",
    "Kubernetes",
    "Terraform",
    "Ansible",
    "Jenkins",
    "GitHub Actions",
    "AWS Lambda",
    "Azure Functions",
    "GCP Cloud Run",
    "Vercel",
    "TensorFlow",
    "PyTorch",
    "scikit-learn",
    "pandas",
    "NumPy",
    "Jupyter",
    "MLflow",
    "Airflow",
    "Spark",
    "Flink",
    "Node.js",
    "Deno",
    "Bun",
    "Webpack",
    "Vite",
    "Tailwind CSS",
    "Bootstrap",
    "Material UI",
    "Three.js",
    "D3.js",
    "OpenAI API",
    "Claude API",
    "LangChain",
    "LlamaIndex",
    "Hugging Face",
    "CUDA",
    "WebAssembly",
    "WebRTC",
    "WebSocket",
    "HTTP/3",
    "SQLite",
    "DynamoDB",
    "Cassandra",
    "Neo4j",
    "FalkorDB",
    "Prometheus",
    "Grafana",
    "Datadog",
    "Sentry",
    "PagerDuty",
    "Linux",
    "Nginx",
    "HAProxy",
    "Envoy",
    "Istio",
    "ArgoCD",
    "Flux",
    "Helm",
    "Pulumi",
    "CDK",
    "Supabase",
    "Firebase",
    "Auth0",
    "Stripe API",
    "Twilio",
    "SendGrid",
    "Mapbox",
    "Figma API",
    "Notion API",
    "Slack API",
    # Pad to 200
    "Redux",
    "MobX",
    "Zustand",
    "Jotai",
    "Recoil",
    "SWR",
    "React Query",
    "tRPC",
    "Prisma",
    "Drizzle",
    "SQLAlchemy",
    "Alembic",
    "Celery",
    "Dramatiq",
    "FastStream",
    "Pydantic",
    "Zod",
    "io-ts",
    "Effect-TS",
    "RxJS",
    "Socket.io",
    "Actix",
    "Axum",
    "Tokio",
    "Warp",
    "Rocket",
    "Gin",
    "Echo",
    "Fiber",
    "Chi",
    "HTMX",
    "Alpine.js",
    "Solid.js",
    "Qwik",
    "Astro",
    "Remix",
    "Gatsby",
    "Nuxt",
    "SvelteKit",
    "Electron",
    "Tauri",
    "React Native",
    "Flutter",
    "Ionic",
    "Capacitor",
    "Expo",
    "SwiftUI",
    "Jetpack Compose",
    "Unity",
    "Unreal Engine",
    "Godot",
    "Blender API",
    "OpenCV",
    "FFmpeg",
    "ImageMagick",
    "Sharp",
    "Puppeteer",
    "Playwright",
    "Cypress",
    "Selenium",
    "Vitest",
    "Jest",
    "Mocha",
    "pytest",
    "JUnit",
    "Testcontainers",
    "k6",
    "Locust",
    "Artillery",
    "Gatling",
    "wrk",
    "Apache Bench",
    "hey",
    "Vegeta",
    "Chaos Monkey",
    "Litmus",
    "Gremlin",
    "LaunchDarkly",
    "Unleash",
    "Flagsmith",
    "PostHog",
    "Amplitude",
    "Segment",
    "Mixpanel",
    "FullStory",
    "LogRocket",
    "New Relic",
    "Dynatrace",
    "Splunk",
    "ELK Stack",
    "Loki",
    "Tempo",
    "Jaeger",
    "Zipkin",
    "OpenTelemetry",
    "Meilisearch",
    "Typesense",
    "Weaviate",
    "Qdrant",
    "Pinecone",
]

_ORGANIZATIONS = [
    "Acme Corp",
    "NovaTech",
    "Quantum Labs",
    "Stellar AI",
    "DataForge",
    "CloudNine",
    "ByteStream",
    "NeuralPath",
    "DeepMind",
    "OpenAI",
    "Anthropic",
    "Meta AI",
    "Google Brain",
    "Microsoft Research",
    "Apple ML",
    "Amazon Science",
    "Netflix Tech",
    "Spotify Engineering",
    "Uber Engineering",
    "Airbnb Tech",
    "Stripe Engineering",
    "Square",
    "Shopify",
    "Atlassian",
    "JetBrains",
    "HashiCorp",
    "Elastic",
    "Confluent",
    "Databricks",
    "Snowflake",
    "Palantir",
    "Scale AI",
    "Weights & Biases",
    "Cohere",
    "Stability AI",
    "Hugging Face Inc",
    "Lightning AI",
    "Modal",
    "Replicate",
    "Together AI",
    # Pad to 150
    "Cerebras",
    "Groq",
    "Inflection AI",
    "Mistral AI",
    "Adept AI",
    "Character AI",
    "Runway ML",
    "Midjourney",
    "Jasper AI",
    "Copy AI",
    "Notion Labs",
    "Figma Inc",
    "Canva",
    "Miro",
    "Linear",
    "Vercel Inc",
    "Netlify",
    "Cloudflare",
    "Fastly",
    "Akamai",
    "Twilio Inc",
    "SendGrid Inc",
    "Plaid",
    "Brex",
    "Ramp",
    "Gusto",
    "Rippling",
    "Lattice",
    "Culture Amp",
    "Lever",
    "Greenhouse",
    "Workday",
    "ServiceNow",
    "Salesforce AI",
    "HubSpot",
    "Zendesk",
    "Intercom",
    "Drift",
    "Gong",
    "Clari",
    "Outreach",
    "SalesLoft",
    "Apollo",
    "ZoomInfo",
    "Clearbit",
    "FullContact",
    "People Data Labs",
    "Snyk",
    "Sonar",
    "GitLab",
    "Bitbucket",
    "CircleCI",
    "Travis CI",
    "Buildkite",
    "Semaphore",
    "CodeClimate",
    "Codacy",
    "DeepSource",
    "SonarQube",
    "LaunchDarkly Inc",
    "Split",
    "Optimizely",
    "VWO",
    "Kameleoon",
    "Amplitude Inc",
    "Heap",
    "Pendo",
    "WalkMe",
    "Appcues",
    "Datadog Inc",
    "New Relic Inc",
    "Dynatrace Inc",
    "Splunk Inc",
    "Elastic Inc",
    "Grafana Labs",
    "Chronosphere",
    "Lightstep",
    "Honeycomb",
    "Monte Carlo",
    "Atlan",
    "dbt Labs",
    "Fivetran",
    "Airbyte",
    "Stitch",
    "Talend",
    "Informatica",
    "Matillion",
    "Hightouch",
    "Census",
    "Rudderstack",
    "mParticle",
    "Lytics",
    "ActionIQ",
    "Amperity",
    "Treasure Data",
    "Tealium",
    "OneTrust",
    "BigID",
    "Collibra",
    "Alation",
    "Immuta",
    "Privacera",
    "Securiti",
    "Transcend",
    "Ethyca",
    "DataGrail",
    "Anyscale",
    "Ray Labs",
    "Dask Labs",
    "Prefect",
    "Dagster",
    "Temporal Inc",
    "Conductor",
    "Camunda",
    "Zeebe",
    "Orkes",
    "PlanetScale",
    "CockroachDB",
    "TiDB",
    "YugabyteDB",
    "Neon",
    "Turso",
    "SingleStore",
    "ClickHouse Inc",
    "StarRocks",
    "Apache Doris",
]

_LOCATIONS = [
    "San Francisco",
    "New York",
    "London",
    "Tokyo",
    "Berlin",
    "Paris",
    "Singapore",
    "Toronto",
    "Sydney",
    "Seattle",
    "Austin",
    "Boston",
    "Chicago",
    "Los Angeles",
    "Denver",
    "Portland",
    "Atlanta",
    "Miami",
    "Vancouver",
    "Montreal",
    "Dublin",
    "Amsterdam",
    "Stockholm",
    "Helsinki",
    "Oslo",
    "Copenhagen",
    "Zurich",
    "Munich",
    "Barcelona",
    "Lisbon",
    "Tel Aviv",
    "Dubai",
    "Bangalore",
    "Mumbai",
    "Delhi",
    "Shanghai",
    "Beijing",
    "Shenzhen",
    "Seoul",
    "Taipei",
    "Hong Kong",
    "Melbourne",
    "Auckland",
    "Cape Town",
    "Nairobi",
    "Lagos",
    "Cairo",
    "São Paulo",
    "Mexico City",
    "Buenos Aires",
    "Santiago",
    "Lima",
    "Bogotá",
    "Medellín",
    "Remote",
    "Distributed",
    "Hybrid",
    "Vienna",
    "Prague",
    "Warsaw",
    "Budapest",
    "Bucharest",
    "Sofia",
    "Athens",
    "Istanbul",
    "Ankara",
    "Riyadh",
    "Doha",
    "Abu Dhabi",
    "Kuala Lumpur",
    "Jakarta",
    "Bangkok",
    "Ho Chi Minh City",
    "Manila",
    "Osaka",
    "Kyoto",
    "Nagoya",
    "Fukuoka",
    "Yokohama",
    "Sapporo",
    "Pittsburgh",
    "Philadelphia",
    "Washington DC",
    "Detroit",
    "Minneapolis",
    "Nashville",
    "Raleigh",
    "Salt Lake City",
    "Phoenix",
    "San Diego",
    "San Jose",
    "Sacramento",
    "Oakland",
    "Palo Alto",
    "Mountain View",
    "Cupertino",
    "Redmond",
    "Kirkland",
    "Bellevue",
    "Cambridge",
]

_PROJECT_PREFIXES = [
    "Project",
    "Operation",
    "Initiative",
    "Platform",
    "System",
]

_PROJECT_NAMES = [
    "Phoenix",
    "Atlas",
    "Nebula",
    "Horizon",
    "Catalyst",
    "Apex",
    "Prism",
    "Vertex",
    "Nexus",
    "Zenith",
    "Eclipse",
    "Aurora",
    "Titan",
    "Vanguard",
    "Odyssey",
    "Helios",
    "Artemis",
    "Orion",
    "Mercury",
    "Neptune",
    "Quantum",
    "Fusion",
    "Cosmos",
    "Stellar",
    "Nova",
    "Voyager",
    "Pioneer",
    "Sentinel",
    "Beacon",
    "Compass",
]

_PROJECT_SUFFIXES = [
    "Engine",
    "Platform",
    "Suite",
    "Hub",
    "Core",
    "SDK",
    "API",
    "Framework",
    "Toolkit",
    "Dashboard",
]

_CONCEPTS = [
    "Machine Learning",
    "Deep Learning",
    "Neural Networks",
    "Natural Language Processing",
    "Computer Vision",
    "Reinforcement Learning",
    "Transfer Learning",
    "Federated Learning",
    "Meta Learning",
    "Few-Shot Learning",
    "Zero-Shot Learning",
    "Prompt Engineering",
    "RAG",
    "Fine-Tuning",
    "RLHF",
    "DPO",
    "Constitutional AI",
    "AI Safety",
    "AI Alignment",
    "Interpretability",
    "Explainable AI",
    "Bias Mitigation",
    "Differential Privacy",
    "Homomorphic Encryption",
    "Secure Multi-Party Computation",
    "Knowledge Graphs",
    "Ontologies",
    "Semantic Web",
    "Graph Neural Networks",
    "Attention Mechanism",
    "Transformer Architecture",
    "Diffusion Models",
    "Generative AI",
    "Large Language Models",
    "Multimodal AI",
    "Embodied AI",
    "Autonomous Agents",
    "Tool Use",
    "Function Calling",
    "Chain of Thought",
    "Tree of Thought",
    "ReAct",
    "Plan and Execute",
    "Cognitive Architecture",
    "ACT-R",
    "Spreading Activation",
    "Memory Systems",
    "Episodic Memory",
    "Semantic Memory",
    "Working Memory",
    "Long-Term Memory",
    "Retrieval Augmented Generation",
    "Vector Databases",
    "Embedding Models",
    "Cosine Similarity",
    "HNSW",
    "IVF",
    "Product Quantization",
    "Dimensionality Reduction",
    "PCA",
    "t-SNE",
    "UMAP",
    # Pad to 200
    "Contrastive Learning",
    "Self-Supervised Learning",
    "Semi-Supervised Learning",
    "Active Learning",
    "Curriculum Learning",
    "Multi-Task Learning",
    "Continual Learning",
    "Online Learning",
    "Bayesian Optimization",
    "Hyperparameter Tuning",
    "Neural Architecture Search",
    "AutoML",
    "Model Compression",
    "Knowledge Distillation",
    "Pruning",
    "Quantization",
    "Mixed Precision Training",
    "Gradient Checkpointing",
    "Data Parallelism",
    "Model Parallelism",
    "Pipeline Parallelism",
    "Tensor Parallelism",
    "ZeRO",
    "DeepSpeed",
    "Megatron-LM",
    "Flash Attention",
    "Ring Attention",
    "Sliding Window Attention",
    "Multi-Head Attention",
    "Cross-Attention",
    "Self-Attention",
    "Positional Encoding",
    "Rotary Embeddings",
    "ALiBi",
    "Byte Pair Encoding",
    "SentencePiece",
    "Tokenization",
    "Vocabulary Selection",
    "Subword Segmentation",
    "Beam Search",
    "Nucleus Sampling",
    "Top-K Sampling",
    "Temperature Scaling",
    "Classifier-Free Guidance",
    "Score Distillation",
    "LoRA",
    "QLoRA",
    "Adapters",
    "Prefix Tuning",
    "Prompt Tuning",
    "In-Context Learning",
    "Instruction Tuning",
    "PEFT",
    "Model Merging",
    "Mixture of Experts",
    "Sparse Transformers",
    "State Space Models",
    "Mamba",
    "Linear Attention",
    "RWKV",
    "Hyena",
    "RetNet",
    "Retrieval Augmentation",
    "Memory Networks",
    "Hopfield Networks",
    "Associative Memory",
    "Content-Addressable Memory",
    "Neural Turing Machine",
    "Differentiable Programming",
    "Program Synthesis",
    "Neuro-Symbolic AI",
    "Causal Inference",
    "Counterfactual Reasoning",
    "Abductive Reasoning",
    "Inductive Logic Programming",
    "Abstract Reasoning",
    "Analogical Reasoning",
    "Common Sense Reasoning",
    "Spatial Reasoning",
    "Temporal Reasoning",
    "Object Detection",
    "Image Segmentation",
    "Optical Flow",
    "Depth Estimation",
    "3D Reconstruction",
    "NeRF",
    "Gaussian Splatting",
    "Point Clouds",
    "Mesh Generation",
    "Texture Synthesis",
    "Style Transfer",
    "Image Inpainting",
    "Super Resolution",
    "Image Generation",
    "Video Generation",
    "Audio Generation",
    "Speech Synthesis",
    "Voice Cloning",
    "Music Generation",
    "Sound Event Detection",
    "Automatic Speech Recognition",
    "Speaker Diarization",
    "Emotion Recognition",
    "Sentiment Analysis",
    "Named Entity Recognition",
    "Relation Extraction",
    "Coreference Resolution",
    "Dependency Parsing",
    "Constituency Parsing",
    "Semantic Role Labeling",
    "Word Sense Disambiguation",
    "Machine Translation",
    "Text Summarization",
    "Question Answering",
    "Dialog Systems",
    "Chatbots",
    "Information Retrieval",
    "Document Understanding",
    "Table Understanding",
    "Code Generation",
    "Code Understanding",
    "Code Review",
    "Bug Detection",
    "Test Generation",
    "Formal Verification",
    "Theorem Proving",
    "SAT Solving",
    "Constraint Satisfaction",
    "Optimization",
    "Evolutionary Algorithms",
    "Genetic Programming",
    "Swarm Intelligence",
    "Ant Colony Optimization",
    "Simulated Annealing",
    "Monte Carlo Methods",
    "Markov Decision Processes",
    "Partially Observable MDPs",
    "Multi-Agent Systems",
    "Game Theory",
    "Mechanism Design",
    "Auction Theory",
    "Social Choice Theory",
    "Voting Theory",
]

# ---------------------------------------------------------------------------
# Type-pair predicates
# ---------------------------------------------------------------------------

_TYPE_PREDICATES: dict[tuple[str, str], list[str]] = {
    ("person", "organization"): ["WORKS_AT", "FOUNDED", "ADVISES"],
    ("person", "technology"): ["USES", "CREATED", "EXPERT_IN"],
    ("person", "person"): ["KNOWS", "MENTORS", "COLLABORATES_WITH"],
    ("technology", "technology"): ["DEPENDS_ON", "INTEGRATES_WITH", "ALTERNATIVE_TO"],
    ("organization", "location"): ["HEADQUARTERED_IN", "HAS_OFFICE_IN"],
    ("organization", "technology"): ["USES", "DEVELOPS", "SPONSORS"],
    ("project", "technology"): ["BUILT_WITH", "INTEGRATES"],
    ("project", "organization"): ["OWNED_BY", "FUNDED_BY"],
    ("person", "project"): ["LEADS", "CONTRIBUTES_TO"],
    ("person", "location"): ["BASED_IN", "RELOCATED_TO"],
    ("concept", "technology"): ["IMPLEMENTED_IN", "ENABLES"],
    ("concept", "concept"): ["RELATED_TO", "PREREQUISITE_OF", "EXTENDS"],
    ("person", "concept"): ["RESEARCHES", "EXPERT_IN", "STUDIES"],
}

# ---------------------------------------------------------------------------
# Cluster definitions
# ---------------------------------------------------------------------------

_CLUSTER_DEFS: list[dict] = [
    {
        "name": "ML Team at BigCorp",
        "types": {"person": 15, "technology": 10, "organization": 5},
        "description": "Machine learning engineering team at a large corporation",
        "domains": [
            "ML infrastructure",
            "recommendation engines",
            "data pipelines",
            "model training",
            "feature engineering",
        ],
    },
    {
        "name": "Web Stack Cluster",
        "types": {"technology": 25, "project": 10},
        "description": "Modern web development technology stack",
        "domains": [
            "API design",
            "frontend frameworks",
            "server-side rendering",
            "developer tooling",
            "web performance",
        ],
    },
    {
        "name": "Bay Area Startups",
        "types": {"organization": 15, "location": 5, "person": 10},
        "description": "Startup ecosystem in the San Francisco Bay Area",
        "domains": [
            "scalability",
            "growth engineering",
            "platform engineering",
            "product development",
            "startup operations",
        ],
    },
    {
        "name": "Data Science Pipeline",
        "types": {"technology": 15, "concept": 15},
        "description": "End-to-end data science and analytics workflow",
        "domains": [
            "real-time analytics",
            "batch processing",
            "data governance",
            "statistical modeling",
            "data visualization",
        ],
    },
    {
        "name": "Cloud Infrastructure",
        "types": {"technology": 15, "organization": 5, "project": 10},
        "description": "Cloud computing and infrastructure automation",
        "domains": [
            "cloud-native architecture",
            "container orchestration",
            "infrastructure as code",
            "service mesh",
            "cloud migration",
        ],
    },
    {
        "name": "AI Safety Research",
        "types": {"concept": 15, "person": 10, "organization": 5},
        "description": "Research on AI alignment and safety",
        "domains": [
            "AI alignment",
            "safety evaluation",
            "interpretability",
            "robustness testing",
            "responsible AI",
        ],
    },
    {
        "name": "European Tech Hub",
        "types": {"location": 15, "organization": 10, "person": 5},
        "description": "Technology companies and workers across Europe",
        "domains": [
            "distributed teams",
            "regulatory compliance",
            "GDPR engineering",
            "cross-border operations",
            "remote collaboration",
        ],
    },
    {
        "name": "Mobile Development",
        "types": {"technology": 15, "project": 5, "person": 5},
        "description": "Mobile app development across iOS and Android",
        "domains": [
            "mobile UI frameworks",
            "cross-platform development",
            "app performance",
            "mobile testing",
            "push notification systems",
        ],
    },
    {
        "name": "DevOps Toolchain",
        "types": {"technology": 15, "concept": 5, "project": 5},
        "description": "CI/CD, infrastructure as code, and observability tools",
        "domains": [
            "CI/CD automation",
            "observability",
            "deployment pipelines",
            "incident management",
            "release engineering",
        ],
    },
    {
        "name": "NLP Research Group",
        "types": {"concept": 15, "person": 10, "technology": 5},
        "description": "Natural language processing research and applications",
        "domains": [
            "language modeling",
            "text classification",
            "named entity recognition",
            "semantic parsing",
            "dialogue systems",
        ],
    },
    {
        "name": "FinTech Ecosystem",
        "types": {"organization": 15, "technology": 5, "location": 5},
        "description": "Financial technology companies and platforms",
        "domains": [
            "payment processing",
            "regulatory technology",
            "fraud detection",
            "digital banking",
            "financial data analytics",
        ],
    },
    {
        "name": "Gaming & Graphics",
        "types": {"technology": 10, "concept": 5, "project": 10},
        "description": "Game engines, 3D graphics, and interactive media",
        "domains": [
            "game engine design",
            "real-time rendering",
            "shader programming",
            "physics simulation",
            "interactive media",
        ],
    },
]

# ---------------------------------------------------------------------------
# Summaries for searchability
# ---------------------------------------------------------------------------

_PERSON_SUMMARY_TEMPLATES = [
    "{name} is a software engineer specializing in {tech}.",
    "{name} is a researcher focusing on {concept} at {org}.",
    "{name} leads engineering efforts in {tech} development.",
    "{name} works as a senior developer with expertise in {tech}.",
    "{name} is a data scientist exploring {concept} applications.",
    "{name} is a founding engineer at {org}, building {tech} systems.",
    "{name} is a tech lead responsible for {tech} infrastructure.",
    "{name} contributes to open source {tech} projects regularly.",
]

_TECH_SUMMARY_TEMPLATES = [
    "{name} is a technology used for building modern applications.",
    "{name} is a popular framework in the software development ecosystem.",
    "{name} provides tools and libraries for efficient development.",
    "{name} is widely adopted for production workloads and development.",
]

_ORG_SUMMARY_TEMPLATES = [
    "{name} is a technology company focused on innovation.",
    "{name} develops cutting-edge software solutions.",
    "{name} is an organization working on AI and technology products.",
    "{name} is a company providing engineering tools and services.",
]

_LOC_SUMMARY_TEMPLATES = [
    "{name} is a major technology hub with a thriving startup ecosystem.",
    "{name} hosts numerous tech companies and research institutions.",
    "{name} is known for its vibrant technology and engineering community.",
]

_PROJECT_SUMMARY_TEMPLATES = [
    "{name} is an initiative to build next-generation infrastructure.",
    "{name} is a platform project focused on developer productivity.",
    "{name} aims to deliver scalable solutions for modern workloads.",
]

_CONCEPT_SUMMARY_TEMPLATES = [
    "{name} is a key concept in modern AI and machine learning research.",
    "{name} is a technique used extensively in data science and AI.",
    "{name} represents an important area of computer science research.",
    "{name} is a fundamental topic studied in AI and software engineering.",
]

# ---------------------------------------------------------------------------
# Word pools for enriched summaries
# ---------------------------------------------------------------------------

_ADJECTIVES = [
    "innovative",
    "experienced",
    "prolific",
    "passionate",
    "versatile",
    "meticulous",
    "pragmatic",
    "visionary",
    "dedicated",
    "resourceful",
    "analytical",
    "collaborative",
    "detail-oriented",
    "forward-thinking",
    "adaptable",
    "tenacious",
    "strategic",
    "inventive",
    "methodical",
    "ambitious",
    "driven",
    "insightful",
    "creative",
    "diligent",
    "accomplished",
    "seasoned",
    "proactive",
    "dynamic",
    "reliable",
    "thorough",
]

_DOMAINS = [
    "distributed systems",
    "real-time analytics",
    "API design",
    "cloud-native architecture",
    "data pipelines",
    "developer tooling",
    "platform engineering",
    "site reliability",
    "security engineering",
    "performance optimization",
    "scalability",
    "observability",
    "event-driven systems",
    "microservices",
    "edge computing",
    "data governance",
    "ML infrastructure",
    "search systems",
    "recommendation engines",
    "stream processing",
    "batch processing",
    "CI/CD automation",
    "infrastructure as code",
    "container orchestration",
    "service mesh",
    "API gateway design",
    "database internals",
    "compiler design",
    "language runtimes",
    "protocol design",
]

_ORG_TYPES = [
    "startup",
    "enterprise",
    "research lab",
    "open-source collective",
    "consultancy",
    "venture-backed company",
    "public company",
    "non-profit",
    "accelerator",
    "incubator",
    "industry consortium",
    "government agency",
    "academic spin-off",
    "remote-first company",
    "developer tools company",
    "platform provider",
    "SaaS vendor",
    "data company",
    "AI lab",
    "cloud provider",
]

# Suffixes for procedural technology names at >200 entities
_TECH_GEN_SUFFIXES = [
    "Engine",
    "Studio",
    "Accelerator",
    "Compiler",
    "Orchestrator",
    "Analyzer",
    "Connector",
    "Optimizer",
    "Processor",
    "Controller",
]


# ---------------------------------------------------------------------------
# Corpus Generator
# ---------------------------------------------------------------------------


class CorpusGenerator:
    """Generate a deterministic benchmark corpus.

    All randomness flows through ``self._rng`` (seeded ``random.Random``)
    so the output is fully reproducible.
    """

    def __init__(
        self,
        seed: int = 42,
        reference_time: float | None = None,
        total_entities: int = 1000,
    ) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        # Use a fixed reference time for deterministic access event timestamps.
        # Default: 2025-06-01T00:00:00 UTC (epoch 1748736000).
        self._now = reference_time if reference_time is not None else 1748736000.0
        self._scale = CorpusScale(total_entities=total_entities)

    # -- public API ----------------------------------------------------------

    def generate(self) -> CorpusSpec:
        """Build the full corpus. Pure, deterministic, no I/O."""
        entities = self._generate_entities()
        entity_map: dict[str, Entity] = {e.id: e for e in entities}
        type_index: dict[str, list[str]] = {}
        for e in entities:
            type_index.setdefault(e.entity_type, []).append(e.id)

        clusters = self._assign_clusters(entities, type_index)
        relationships, rel_idx = self._generate_relationships(
            entities,
            entity_map,
            type_index,
            clusters,
        )
        access_events, access_tiers = self._generate_access_events(
            entities,
            clusters,
        )
        ground_truth = self._generate_ground_truth(
            entities,
            entity_map,
            type_index,
            clusters,
            relationships,
            rel_idx,
            access_events,
            access_tiers,
        )

        # Enrich summaries last so RNG state doesn't affect prior generation
        self._enrich_summaries(entities, entity_map, clusters)

        # Generate episodes after summaries are enriched (so content references real text)
        adj = self._build_adjacency(relationships)
        episodes, episode_entities = self._generate_episodes(
            entities,
            entity_map,
            access_tiers,
            adj,
        )

        # Update recency ground truth with episode relevance
        self._add_episode_ground_truth(ground_truth, episodes, episode_entities)

        # Generate conversation scenarios for working memory benchmark
        conversation_scenarios = self._generate_conversation_scenarios(
            entity_map,
            clusters,
            adj,
        )

        return CorpusSpec(
            entities=entities,
            relationships=relationships,
            access_events=access_events,
            ground_truth=ground_truth,
            metadata={
                "seed": self._seed,
                "generated_at": self._now,
                "num_entities": len(entities),
                "num_relationships": len(relationships),
                "num_access_events": len(access_events),
                "num_queries": len(ground_truth),
                "num_episodes": len(episodes),
                "num_clusters": len(clusters),
                "entity_type_counts": {t: len(ids) for t, ids in type_index.items()},
                "clusters": [
                    {
                        "name": c["name"],
                        "members": c["members"],
                        "domains": c.get("domains", []),
                        "description": c.get("description", ""),
                    }
                    for c in clusters
                ],
            },
            episodes=episodes,
            episode_entities=episode_entities,
            conversation_scenarios=conversation_scenarios,
        )

    async def load(
        self,
        corpus: CorpusSpec,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        search_index: SearchIndex,
        structure_aware: bool = False,
        cfg: ActivationConfig | None = None,
    ) -> float:
        """Load a corpus into live stores. Returns elapsed seconds.

        Uses bulk SQL inserts with a single commit for SQLite stores,
        falling back to individual create calls for other backends.

        When ``structure_aware=True`` and the search index supports
        embeddings, entities are re-indexed with predicate-enriched text
        using the natural-language format from ``_index_entity_with_structure``
        (e.g. ``"Alice. person. Engineer. Relationships: Alice works at TechCorp"``).
        """
        start = time.time()
        if cfg is None:
            cfg = ActivationConfig()

        entity_map = {e.id: e for e in corpus.entities}

        # Try bulk SQLite path (single transaction instead of N commits).
        # Check for aiosqlite.Connection to avoid false positives with mocks.
        db = getattr(graph_store, "_db", None)
        if db is not None and type(db).__module__.startswith("aiosqlite"):
            await self._bulk_load_sqlite(
                corpus,
                graph_store,
                cast(aiosqlite.Connection, db),
                search_index,
                structure_aware,
                cfg,
                entity_map,
            )
        else:
            await self._load_sequential(
                corpus,
                graph_store,
                search_index,
                structure_aware,
                cfg,
                entity_map,
            )

        # Record all access events (in-memory store, already fast)
        access_group = corpus.entities[0].group_id if corpus.entities else GROUP_ID
        for entity_id, timestamp in corpus.access_events:
            await activation_store.record_access(
                entity_id,
                timestamp,
                group_id=access_group,
            )

        return time.time() - start

    async def _bulk_load_sqlite(
        self,
        corpus: CorpusSpec,
        graph_store: GraphStore,
        db: aiosqlite.Connection,
        search_index: SearchIndex,
        structure_aware: bool,
        cfg: ActivationConfig,
        entity_map: dict[str, Entity],
    ) -> None:
        """Bulk-insert entities, relationships, episodes via executemany."""
        now = datetime.utcnow().isoformat()

        # 1. Bulk insert entities
        entity_rows = []
        encrypt = getattr(graph_store, "_encrypt", None)
        for e in corpus.entities:
            summary = encrypt(e.group_id, e.summary) if encrypt else e.summary
            entity_rows.append(
                (
                    e.id,
                    e.name,
                    e.entity_type,
                    summary,
                    json.dumps(e.attributes) if e.attributes else None,
                    e.group_id,
                    e.created_at.isoformat() if e.created_at else now,
                    now,
                    e.access_count,
                    e.last_accessed.isoformat() if e.last_accessed else None,
                    1 if e.pii_detected else 0,
                    json.dumps(e.pii_categories) if e.pii_categories else None,
                )
            )
        await db.executemany(
            """INSERT INTO entities
               (id, name, entity_type, summary, attributes, group_id,
                created_at, updated_at, access_count, last_accessed,
                pii_detected, pii_categories)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            entity_rows,
        )

        # 2. Bulk insert relationships
        rel_rows = []
        for r in corpus.relationships:
            rel_rows.append(
                (
                    r.id,
                    r.source_id,
                    r.target_id,
                    r.predicate,
                    r.weight,
                    r.valid_from.isoformat() if r.valid_from else None,
                    r.valid_to.isoformat() if r.valid_to else None,
                    r.created_at.isoformat() if r.created_at else now,
                    r.source_episode,
                    r.group_id,
                    r.confidence,
                )
            )
        await db.executemany(
            """INSERT INTO relationships
               (id, source_id, target_id, predicate, weight,
                valid_from, valid_to, created_at, source_episode,
                group_id, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rel_rows,
        )

        # 3. Bulk insert episodes
        ep_rows = []
        for ep in corpus.episodes:
            content = encrypt(ep.group_id, ep.content) if encrypt else ep.content
            ep_rows.append(
                (
                    ep.id,
                    content,
                    ep.source,
                    ep.status.value if hasattr(ep.status, "value") else ep.status,
                    ep.group_id,
                    ep.session_id,
                    ep.created_at.isoformat() if ep.created_at else now,
                    ep.updated_at.isoformat() if ep.updated_at else now,
                    ep.error,
                    ep.retry_count,
                    ep.processing_duration_ms,
                )
            )
        await db.executemany(
            """INSERT INTO episodes
               (id, content, source, status, group_id, session_id,
                created_at, updated_at, error, retry_count,
                processing_duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ep_rows,
        )

        # 4. Bulk insert episode-entity links
        link_rows = [(ep_id, ent_id) for ep_id, ent_id in corpus.episode_entities]
        await db.executemany(
            "INSERT OR IGNORE INTO episode_entities (episode_id, entity_id) VALUES (?, ?)",
            link_rows,
        )

        # Single commit for all inserts
        await db.commit()

        # 5. Index entities and episodes in search index
        # For FTS5: index_entity/index_episode are no-ops (triggers handle it)
        # For HybridSearchIndex: bulk-embed and batch-upsert vectors
        if isinstance(search_index, HybridSearchIndex) and search_index._embeddings_enabled:
            await self._bulk_embed(corpus, search_index, structure_aware, cfg)
        else:
            # FTS5 no-ops, just call in case of custom SearchIndex
            for entity in corpus.entities:
                await search_index.index_entity(entity)
            for episode in corpus.episodes:
                await search_index.index_episode(episode)

    async def _load_sequential(
        self,
        corpus: CorpusSpec,
        graph_store: GraphStore,
        search_index: SearchIndex,
        structure_aware: bool,
        cfg: ActivationConfig,
        entity_map: dict[str, Entity],
    ) -> None:
        """Fallback: individual create calls for non-SQLite backends."""
        for entity in corpus.entities:
            await graph_store.create_entity(entity)
            await search_index.index_entity(entity)

        for rel in corpus.relationships:
            await graph_store.create_relationship(rel)

        if structure_aware and getattr(search_index, "_embeddings_enabled", False):
            await self._reindex_structure_aware(
                corpus,
                search_index,
                entity_map,
                cfg,
            )

        for episode in corpus.episodes:
            await graph_store.create_episode(episode)
            await search_index.index_episode(episode)

        for episode_id, entity_id in corpus.episode_entities:
            await graph_store.link_episode_entity(episode_id, entity_id)

    async def _bulk_embed(
        self,
        corpus: CorpusSpec,
        search_index: HybridSearchIndex,
        structure_aware: bool,
        cfg: ActivationConfig,
    ) -> None:
        """Bulk-embed entities and episodes, then batch-upsert vectors.

        Instead of embedding one entity at a time (1000 API calls for 1k
        entities), collects all texts and embeds in bulk batches (~8 calls
        at batch_size=128).
        """
        provider = search_index._provider
        vectors = search_index._vectors
        entity_map = {e.id: e for e in corpus.entities}

        # Build final text for each entity
        if structure_aware:
            entity_texts = self._build_structure_texts(
                corpus,
                entity_map,
                cfg,
            )
        else:
            entity_texts = {}
            for e in corpus.entities:
                if e.name:
                    text = e.name
                    if e.summary:
                        text = f"{e.name}: {e.summary}"
                    entity_texts[e.id] = text

        # Bulk embed all entity texts
        eid_list = list(entity_texts.keys())
        text_list = [entity_texts[eid] for eid in eid_list]

        if text_list:
            all_embeddings = await provider.embed(text_list)
            if all_embeddings and len(all_embeddings) == len(eid_list):
                items: list[tuple[str, str, str, str | None, list[float]]] = []
                for eid, text, emb in zip(
                    eid_list,
                    text_list,
                    all_embeddings,
                ):
                    entity = entity_map[eid]
                    items.append(
                        (
                            eid,
                            "entity",
                            entity.group_id,
                            text,
                            emb,
                        )
                    )
                await vectors.batch_upsert(items)
                await vectors.db.commit()

        # Bulk embed episodes
        ep_texts = {}
        for ep in corpus.episodes:
            if ep.content:
                ep_texts[ep.id] = ep.content

        ep_ids = list(ep_texts.keys())
        ep_text_list = [ep_texts[epid] for epid in ep_ids]

        if ep_text_list:
            ep_embeddings = await provider.embed(ep_text_list)
            if ep_embeddings and len(ep_embeddings) == len(ep_ids):
                episode_items: list[tuple[str, str, str, str | None, list[float]]] = []
                for epid, text, emb in zip(
                    ep_ids,
                    ep_text_list,
                    ep_embeddings,
                ):
                    ep = next(e for e in corpus.episodes if e.id == epid)
                    episode_items.append(
                        (
                            epid,
                            "episode",
                            ep.group_id,
                            text,
                            emb,
                        )
                    )
                await vectors.batch_upsert(episode_items)
                await vectors.db.commit()

    def _build_structure_texts(
        self,
        corpus: CorpusSpec,
        entity_map: dict[str, Entity],
        cfg: ActivationConfig,
    ) -> dict[str, str]:
        """Build predicate-enriched text for each entity."""
        pred_index: dict[str, list[tuple[str, str, bool]]] = {}
        for rel in corpus.relationships:
            tgt = entity_map.get(rel.target_id)
            target_name = tgt.name if tgt else rel.target_id
            pred_index.setdefault(rel.source_id, []).append(
                (rel.predicate, target_name, True),
            )
            src = entity_map.get(rel.source_id)
            source_name = src.name if src else rel.source_id
            pred_index.setdefault(rel.target_id, []).append(
                (rel.predicate, source_name, False),
            )

        predicate_weights = cfg.predicate_weights
        default_weight = cfg.predicate_weight_default
        natural_names = cfg.predicate_natural_names
        max_rels = cfg.structure_max_relationships

        result: dict[str, str] = {}
        for entity in corpus.entities:
            preds = pred_index.get(entity.id, [])

            parts = [entity.name]
            if entity.entity_type:
                parts.append(entity.entity_type)
            if entity.summary:
                parts.append(entity.summary)
            text = ". ".join(parts) + "."

            if preds:
                preds_sorted = sorted(
                    preds,
                    key=lambda p: predicate_weights.get(
                        p[0],
                        default_weight,
                    ),
                    reverse=True,
                )
                rel_parts: list[str] = []
                for pred, other_name, is_source in preds_sorted[:max_rels]:
                    pred_natural = natural_names.get(
                        pred,
                        pred.lower().replace("_", " "),
                    )
                    if is_source:
                        rel_parts.append(
                            f"{entity.name} {pred_natural} {other_name}",
                        )
                    else:
                        rel_parts.append(
                            f"{other_name} {pred_natural} {entity.name}",
                        )
                text += " Relationships: " + ", ".join(rel_parts)

            result[entity.id] = text

        return result

    async def _reindex_structure_aware(
        self,
        corpus: CorpusSpec,
        search_index: SearchIndex,
        entity_map: dict[str, Entity],
        cfg: ActivationConfig,
    ) -> None:
        """Re-index entities with predicate-enriched text."""
        pred_index: dict[str, list[tuple[str, str, bool]]] = {}
        for rel in corpus.relationships:
            tgt = entity_map.get(rel.target_id)
            target_name = tgt.name if tgt else rel.target_id
            pred_index.setdefault(rel.source_id, []).append(
                (rel.predicate, target_name, True),
            )
            src = entity_map.get(rel.source_id)
            source_name = src.name if src else rel.source_id
            pred_index.setdefault(rel.target_id, []).append(
                (rel.predicate, source_name, False),
            )

        predicate_weights = cfg.predicate_weights
        default_weight = cfg.predicate_weight_default
        natural_names = cfg.predicate_natural_names
        max_rels = cfg.structure_max_relationships

        for entity in corpus.entities:
            preds = pred_index.get(entity.id, [])
            if not preds:
                continue

            preds_sorted = sorted(
                preds,
                key=lambda p: predicate_weights.get(p[0], default_weight),
                reverse=True,
            )

            rel_parts: list[str] = []
            for pred, other_name, is_source in preds_sorted[:max_rels]:
                pred_natural = natural_names.get(
                    pred,
                    pred.lower().replace("_", " "),
                )
                if is_source:
                    rel_parts.append(f"{entity.name} {pred_natural} {other_name}")
                else:
                    rel_parts.append(f"{other_name} {pred_natural} {entity.name}")

            parts = [entity.name]
            if entity.entity_type:
                parts.append(entity.entity_type)
            if entity.summary:
                parts.append(entity.summary)
            text = ". ".join(parts) + "."
            if rel_parts:
                text += " Relationships: " + ", ".join(rel_parts)

            enriched = Entity(
                id=entity.id,
                name=text,
                entity_type=entity.entity_type,
                summary=None,
                group_id=entity.group_id,
            )
            await search_index.index_entity(enriched)

    # -- entity generation ---------------------------------------------------

    def _generate_entities(self) -> list[Entity]:
        entities: list[Entity] = []
        base_time = datetime.utcnow()
        type_counts = self._scale.compute_type_counts()

        # Person — first/last combos; beyond 900 add suffix letter
        person_pool = len(_FIRST_NAMES) * len(_LAST_NAMES)
        for i in range(type_counts["person"]):
            if i < person_pool:
                first = _FIRST_NAMES[i % len(_FIRST_NAMES)]
                last = _LAST_NAMES[(i // len(_FIRST_NAMES)) % len(_LAST_NAMES)]
                name = f"{first} {last}"
            else:
                base = i % person_pool
                suffix_num = i // person_pool
                first = _FIRST_NAMES[base % len(_FIRST_NAMES)]
                last = _LAST_NAMES[(base // len(_FIRST_NAMES)) % len(_LAST_NAMES)]
                name = f"{first} {last} {chr(64 + suffix_num)}"
            eid = f"ent_bench_per_{i:04d}"
            summary = self._rng.choice(_PERSON_SUMMARY_TEMPLATES).format(
                name=name,
                tech=self._rng.choice(_TECHNOLOGIES[:50]),
                concept=self._rng.choice(_CONCEPTS[:30]),
                org=self._rng.choice(_ORGANIZATIONS[:20]),
            )
            entities.append(
                Entity(
                    id=eid,
                    name=name,
                    entity_type="person",
                    summary=summary,
                    group_id=GROUP_ID,
                    created_at=base_time - timedelta(days=self._rng.randint(1, 365)),
                )
            )

        # Technology — pool then concept+suffix combos
        for i in range(type_counts["technology"]):
            if i < len(_TECHNOLOGIES):
                name = _TECHNOLOGIES[i]
            else:
                extra = i - len(_TECHNOLOGIES)
                concept = _CONCEPTS[extra % len(_CONCEPTS)]
                suffix = _TECH_GEN_SUFFIXES[(extra // len(_CONCEPTS)) % len(_TECH_GEN_SUFFIXES)]
                name = f"{concept} {suffix}"
                cycle = extra // (len(_CONCEPTS) * len(_TECH_GEN_SUFFIXES))
                if cycle > 0:
                    name = f"{name} v{cycle + 1}"
            eid = f"ent_bench_tech_{i:04d}"
            summary = self._rng.choice(_TECH_SUMMARY_TEMPLATES).format(name=name)
            entities.append(
                Entity(
                    id=eid,
                    name=name,
                    entity_type="technology",
                    summary=summary,
                    group_id=GROUP_ID,
                    created_at=base_time - timedelta(days=self._rng.randint(1, 365)),
                )
            )

        # Organization — pool then adj+domain+org_type combos
        for i in range(type_counts["organization"]):
            if i < len(_ORGANIZATIONS):
                name = _ORGANIZATIONS[i]
            else:
                extra = i - len(_ORGANIZATIONS)
                adj = _ADJECTIVES[extra % len(_ADJECTIVES)]
                domain = _DOMAINS[(extra // len(_ADJECTIVES)) % len(_DOMAINS)]
                org_type = _ORG_TYPES[
                    (extra // (len(_ADJECTIVES) * len(_DOMAINS))) % len(_ORG_TYPES)
                ]
                name = f"{adj.title()} {domain.title()} {org_type.title()}"
            eid = f"ent_bench_org_{i:04d}"
            summary = self._rng.choice(_ORG_SUMMARY_TEMPLATES).format(name=name)
            entities.append(
                Entity(
                    id=eid,
                    name=name,
                    entity_type="organization",
                    summary=summary,
                    group_id=GROUP_ID,
                    created_at=base_time - timedelta(days=self._rng.randint(1, 365)),
                )
            )

        # Location — pool then adj+domain district combos
        for i in range(type_counts["location"]):
            if i < len(_LOCATIONS):
                name = _LOCATIONS[i]
            else:
                extra = i - len(_LOCATIONS)
                adj = _ADJECTIVES[extra % len(_ADJECTIVES)]
                region = _DOMAINS[(extra // len(_ADJECTIVES)) % len(_DOMAINS)]
                name = f"{adj.title()} {region.title()} District"
            eid = f"ent_bench_loc_{i:04d}"
            summary = self._rng.choice(_LOC_SUMMARY_TEMPLATES).format(name=name)
            entities.append(
                Entity(
                    id=eid,
                    name=name,
                    entity_type="location",
                    summary=summary,
                    group_id=GROUP_ID,
                    created_at=base_time - timedelta(days=self._rng.randint(1, 365)),
                )
            )

        # Project — combinatorial prefix+name+suffix; numeric dedup beyond pool
        used_project_names: set[str] = set()
        for i in range(type_counts["project"]):
            prefix = _PROJECT_PREFIXES[i % len(_PROJECT_PREFIXES)]
            pname = _PROJECT_NAMES[(i // len(_PROJECT_PREFIXES)) % len(_PROJECT_NAMES)]
            suffix = _PROJECT_SUFFIXES[
                (i // (len(_PROJECT_PREFIXES) * len(_PROJECT_NAMES))) % len(_PROJECT_SUFFIXES)
            ]
            name = f"{prefix} {pname} {suffix}"
            if name in used_project_names:
                name = f"{prefix} {pname} {suffix} {i}"
            used_project_names.add(name)
            eid = f"ent_bench_proj_{i:04d}"
            summary = self._rng.choice(_PROJECT_SUMMARY_TEMPLATES).format(name=name)
            entities.append(
                Entity(
                    id=eid,
                    name=name,
                    entity_type="project",
                    summary=summary,
                    group_id=GROUP_ID,
                    created_at=base_time - timedelta(days=self._rng.randint(1, 365)),
                )
            )

        # Concept — pool then adj+concept combos
        for i in range(type_counts["concept"]):
            if i < len(_CONCEPTS):
                name = _CONCEPTS[i]
            else:
                extra = i - len(_CONCEPTS)
                adj = _ADJECTIVES[extra % len(_ADJECTIVES)]
                base_concept = _CONCEPTS[(extra // len(_ADJECTIVES)) % len(_CONCEPTS)]
                name = f"{adj.title()} {base_concept}"
            eid = f"ent_bench_con_{i:04d}"
            summary = self._rng.choice(_CONCEPT_SUMMARY_TEMPLATES).format(name=name)
            entities.append(
                Entity(
                    id=eid,
                    name=name,
                    entity_type="concept",
                    summary=summary,
                    group_id=GROUP_ID,
                    created_at=base_time - timedelta(days=self._rng.randint(1, 365)),
                )
            )

        return entities

    # -- cluster assignment --------------------------------------------------

    def _assign_clusters(
        self,
        entities: list[Entity],
        type_index: dict[str, list[str]],
    ) -> list[dict]:
        """Assign entities to topical clusters.

        At default scale (1000): 12 clusters from _CLUSTER_DEFS, unchanged.
        At larger scales: cluster member counts scale proportionally, and
        extra procedural clusters are added for unassigned entities.

        Returns a list of cluster dicts with 'members' and 'hub' keys added.
        """
        clusters: list[dict] = []
        assigned: set[str] = set()
        scale_factor = self._scale.total_entities / 1000.0

        for cluster_def in _CLUSTER_DEFS:
            members: list[str] = []
            for etype, base_count in cluster_def["types"].items():
                count = max(1, int(base_count * scale_factor))
                available = [eid for eid in type_index.get(etype, []) if eid not in assigned]
                chosen = available[:count] if len(available) >= count else available
                members.extend(chosen)
                assigned.update(chosen)

            hub = members[0] if members else None
            clusters.append(
                {
                    **cluster_def,
                    "members": members,
                    "hub": hub,
                }
            )

        # Extra clusters at larger scale
        num_extra = max(0, self._scale.num_clusters - len(_CLUSTER_DEFS))
        if num_extra > 0:
            unassigned = [e.id for e in entities if e.id not in assigned]
            per_cluster = max(1, len(unassigned) // (num_extra + 1))

            for ci in range(num_extra):
                domain = _DOMAINS[ci % len(_DOMAINS)]
                cluster_name = f"{domain.title()} Group {ci + 1}"
                desc = f"Working group focused on {domain}"

                # Derive cluster-specific domains from _DOMAINS pool
                start_idx = (ci * 3) % len(_DOMAINS)
                extra_domains = [_DOMAINS[(start_idx + j) % len(_DOMAINS)] for j in range(5)]

                chunk = unassigned[:per_cluster]
                unassigned = unassigned[per_cluster:]

                hub = chunk[0] if chunk else None
                clusters.append(
                    {
                        "name": cluster_name,
                        "types": {},
                        "description": desc,
                        "domains": extra_domains,
                        "members": chunk,
                        "hub": hub,
                    }
                )
                assigned.update(chunk)

        # Long-tail: unassigned entities get noted but no cluster
        long_tail = [e.id for e in entities if e.id not in assigned]
        clusters.append(
            {
                "name": "Long Tail",
                "types": {},
                "description": "Unassigned entities with sparse connections",
                "domains": list(_DOMAINS),  # fallback to global pool
                "members": long_tail,
                "hub": None,
            }
        )

        return clusters

    # -- summary enrichment --------------------------------------------------

    def _enrich_summaries(
        self,
        entities: list[Entity],
        entity_map: dict[str, Entity],
        clusters: list[dict],
    ) -> None:
        """Replace template summaries with richer, unique descriptions.

        Uses cluster membership, relationships context, and varied word pools
        to produce 30-80 word descriptions. Mutates entities in-place.
        All randomness via self._rng for determinism.

        Summaries are **cluster-aware**: domain terms, technology references,
        and concept references are drawn from the entity's own cluster to
        prevent keyword pollution across clusters (which degrades FTS5/embedding
        discrimination for cross-cluster queries).
        """
        # Build entity-to-cluster map
        eid_to_cluster: dict[str, str] = {}
        eid_to_cluster_members: dict[str, list[str]] = {}
        for cluster in clusters:
            cname = cluster["name"]
            members = cluster["members"]
            for m in members:
                eid_to_cluster[m] = cname
                eid_to_cluster_members[m] = members

        # Build per-cluster entity names by type for cluster-aware references
        cluster_names_by_type: dict[str, dict[str, list[str]]] = {}
        for cluster in clusters:
            cname = cluster["name"]
            by_type: dict[str, list[str]] = {}
            for m in cluster["members"]:
                if m in entity_map:
                    etype = entity_map[m].entity_type
                    by_type.setdefault(etype, []).append(entity_map[m].name)
            cluster_names_by_type[cname] = by_type

        # Build cluster domain lookup
        cluster_domains: dict[str, list[str]] = {}
        for cluster in clusters:
            cluster_domains[cluster["name"]] = cluster.get("domains", list(_DOMAINS))

        for entity in entities:
            etype = entity.entity_type
            name = entity.name
            cluster_name = eid_to_cluster.get(entity.id, "Long Tail")
            adj1 = self._rng.choice(_ADJECTIVES)
            adj2 = self._rng.choice(_ADJECTIVES)

            # Use cluster-specific domains instead of global pool
            domains = cluster_domains.get(cluster_name, list(_DOMAINS))
            domain = self._rng.choice(domains)

            # Pick a related entity name from same cluster for context
            cluster_members = eid_to_cluster_members.get(entity.id, [])
            peer_ids = [m for m in cluster_members if m != entity.id]
            peer_name = entity_map[self._rng.choice(peer_ids)].name if peer_ids else None

            # Cluster-local entity name lookups
            cnames = cluster_names_by_type.get(cluster_name, {})

            if etype == "person":
                tech = (
                    self._rng.choice(cnames["technology"])
                    if "technology" in cnames
                    else self._rng.choice(_TECHNOLOGIES[:80])
                )
                concept = (
                    self._rng.choice(cnames["concept"])
                    if "concept" in cnames
                    else self._rng.choice(_CONCEPTS[:60])
                )
                org = (
                    self._rng.choice(cnames["organization"])
                    if "organization" in cnames
                    else self._rng.choice(_ORGANIZATIONS[:40])
                )
                parts = [
                    f"{name} is a {adj1} engineer",
                    f"specializing in {domain} and {tech}.",
                ]
                if peer_name:
                    parts.append(
                        f"Within the {cluster_name} group, {name} collaborates"
                        f" closely with {peer_name}."
                    )
                parts.append(
                    f"Known for {adj2} problem-solving, they have"
                    f" contributed to work on {concept} at {org}."
                )
                entity.summary = " ".join(parts)

            elif etype == "technology":
                # Pick alt_tech from same cluster, excluding self
                tech_pool = [t for t in cnames.get("technology", []) if t != name]
                alt_tech = (
                    self._rng.choice(tech_pool)
                    if tech_pool
                    else self._rng.choice(_TECHNOLOGIES[:80])
                )
                parts = [
                    f"{name} is a {adj1} technology widely used in {domain}.",
                ]
                if peer_name:
                    parts.append(
                        f"In the {cluster_name} ecosystem, it is often paired"
                        f" with {peer_name} for production workloads."
                    )
                parts.append(
                    f"Developers value its {adj2} approach to building"
                    f" reliable, scalable systems alongside tools like {alt_tech}."
                )
                entity.summary = " ".join(parts)

            elif etype == "organization":
                org_type = self._rng.choice(_ORG_TYPES)
                tech = (
                    self._rng.choice(cnames["technology"])
                    if "technology" in cnames
                    else self._rng.choice(_TECHNOLOGIES[:80])
                )
                parts = [
                    f"{name} is a {adj1} {org_type} focused on {domain}.",
                ]
                if peer_name:
                    parts.append(
                        f"As part of the {cluster_name} network, {name}"
                        f" partners with {peer_name} on key initiatives."
                    )
                parts.append(
                    f"The organization is recognized for its {adj2}"
                    f" engineering culture and use of {tech}."
                )
                entity.summary = " ".join(parts)

            elif etype == "location":
                org = (
                    self._rng.choice(cnames["organization"])
                    if "organization" in cnames
                    else self._rng.choice(_ORGANIZATIONS[:40])
                )
                parts = [
                    f"{name} is a {adj1} technology hub known for {domain}.",
                ]
                if peer_name:
                    parts.append(
                        f"Within the {cluster_name} region, {name} has strong"
                        f" ties to {peer_name} through shared talent pools."
                    )
                parts.append(
                    f"The area hosts {adj2} companies like {org} and attracts"
                    f" global engineering talent year-round."
                )
                entity.summary = " ".join(parts)

            elif etype == "project":
                tech = (
                    self._rng.choice(cnames["technology"])
                    if "technology" in cnames
                    else self._rng.choice(_TECHNOLOGIES[:80])
                )
                concept = (
                    self._rng.choice(cnames["concept"])
                    if "concept" in cnames
                    else self._rng.choice(_CONCEPTS[:60])
                )
                parts = [
                    f"{name} is a {adj1} initiative targeting {domain}.",
                ]
                if peer_name:
                    parts.append(
                        f"Positioned within the {cluster_name} portfolio, it"
                        f" builds on foundations laid by {peer_name}."
                    )
                parts.append(
                    f"The project employs {adj2} methodologies using {tech}"
                    f" to advance {concept} capabilities."
                )
                entity.summary = " ".join(parts)

            elif etype == "concept":
                tech = (
                    self._rng.choice(cnames["technology"])
                    if "technology" in cnames
                    else self._rng.choice(_TECHNOLOGIES[:80])
                )
                # Pick alt_concept from same cluster, excluding self
                concept_pool = [c for c in cnames.get("concept", []) if c != name]
                alt_concept = (
                    self._rng.choice(concept_pool)
                    if concept_pool
                    else self._rng.choice(_CONCEPTS[:60])
                )
                parts = [
                    f"{name} is a {adj1} area of research in {domain}.",
                ]
                if peer_name:
                    parts.append(
                        f"In the {cluster_name} field, it is closely"
                        f" connected to {peer_name} through shared foundations."
                    )
                parts.append(
                    f"Practitioners apply {adj2} techniques from {alt_concept}"
                    f" and implement solutions using {tech}."
                )
                entity.summary = " ".join(parts)

    # -- relationship generation ---------------------------------------------

    def _get_predicate_for_pair(
        self,
        type_a: str,
        type_b: str,
    ) -> str | None:
        """Look up a valid predicate for a (type_a, type_b) pair."""
        preds = _TYPE_PREDICATES.get((type_a, type_b))
        if preds:
            return self._rng.choice(preds)
        # Try reverse
        preds = _TYPE_PREDICATES.get((type_b, type_a))
        if preds:
            return self._rng.choice(preds)
        return None

    def _generate_relationships(
        self,
        entities: list[Entity],
        entity_map: dict[str, Entity],
        type_index: dict[str, list[str]],
        clusters: list[dict],
    ) -> tuple[list[Relationship], int]:
        """Generate 2500+ relationships organized by cluster topology."""
        relationships: list[Relationship] = []
        rel_idx = 0
        existing_pairs: set[tuple[str, str]] = set()
        base_time = datetime.utcnow()

        def _add_rel(
            src_id: str,
            tgt_id: str,
            predicate: str | None = None,
            valid_from: datetime | None = None,
            valid_to: datetime | None = None,
        ) -> bool:
            nonlocal rel_idx
            if src_id == tgt_id:
                return False
            pair = (src_id, tgt_id)
            rev_pair = (tgt_id, src_id)
            if pair in existing_pairs or rev_pair in existing_pairs:
                return False

            src_type = entity_map[src_id].entity_type
            tgt_type = entity_map[tgt_id].entity_type

            if predicate is None:
                predicate = self._get_predicate_for_pair(src_type, tgt_type)
            if predicate is None:
                # Fallback for types without defined predicates
                predicate = "RELATED_TO"

            relationships.append(
                Relationship(
                    id=f"rel_bench_{rel_idx:04d}",
                    source_id=src_id,
                    target_id=tgt_id,
                    predicate=predicate,
                    weight=round(self._rng.uniform(0.5, 1.0), 2),
                    valid_from=valid_from
                    or (base_time - timedelta(days=self._rng.randint(1, 180))),
                    valid_to=valid_to,
                    group_id=GROUP_ID,
                    confidence=round(self._rng.uniform(0.7, 1.0), 2),
                )
            )
            existing_pairs.add(pair)
            rel_idx += 1
            return True

        # Intra-cluster edges: each member connects to 2-5 others
        for cluster in clusters:
            members = cluster["members"]
            if len(members) < 2:
                continue
            for member in members:
                num_connections = self._rng.randint(2, min(5, len(members) - 1))
                targets = [m for m in members if m != member]
                self._rng.shuffle(targets)
                connected = 0
                for target in targets:
                    if connected >= num_connections:
                        break
                    if _add_rel(member, target):
                        connected += 1

        # Inter-cluster bridge edges: 3-5 per cluster
        real_clusters = [c for c in clusters if c["name"] != "Long Tail"]
        for i, cluster in enumerate(real_clusters):
            num_bridges = self._rng.randint(3, 5)
            src_members = cluster["members"]
            if not src_members:
                continue
            for _ in range(num_bridges):
                other_idx = self._rng.randint(0, len(real_clusters) - 1)
                if other_idx == i:
                    other_idx = (other_idx + 1) % len(real_clusters)
                tgt_members = real_clusters[other_idx]["members"]
                if not tgt_members:
                    continue
                src = self._rng.choice(src_members)
                tgt = self._rng.choice(tgt_members)
                _add_rel(src, tgt)

        # Long-tail: 1-2 random connections each
        long_tail = clusters[-1]["members"] if clusters else []
        all_ids = [e.id for e in entities]
        for lt_id in long_tail:
            num_conns = self._rng.randint(1, 2)
            for _ in range(num_conns):
                target = self._rng.choice(all_ids)
                _add_rel(lt_id, target)

        # Fill to target relationship count with random cross-type edges
        all_entity_ids = [e.id for e in entities]
        while rel_idx < self._scale.target_relationships:
            src = self._rng.choice(all_entity_ids)
            tgt = self._rng.choice(all_entity_ids)
            _add_rel(src, tgt)

        return relationships, rel_idx

    # -- access events -------------------------------------------------------

    def _generate_access_events(
        self,
        entities: list[Entity],
        clusters: list[dict],
    ) -> tuple[list[tuple[str, float]], dict[str, str]]:
        """Generate access events with hot/warm/cold/dormant tiers.

        Returns (events, tier_map) where tier_map maps entity_id to tier.
        """
        events: list[tuple[str, float]] = []
        tier_map: dict[str, str] = {}

        # Gather hub entity IDs for priority hot assignment
        hub_ids: set[str] = set()
        for cluster in clusters:
            if cluster.get("hub"):
                hub_ids.add(cluster["hub"])

        # Shuffle entity IDs deterministically for tier assignment
        all_ids = [e.id for e in entities]
        shuffled = list(all_ids)
        self._rng.shuffle(shuffled)

        # Move hub entities to front so they become hot
        hub_list = [eid for eid in shuffled if eid in hub_ids]
        non_hub = [eid for eid in shuffled if eid not in hub_ids]
        ordered = hub_list + non_hub

        n = len(ordered)
        hot_end = int(n * 0.10)  # 10% -> ~100
        warm_end = hot_end + int(n * 0.25)  # 25% -> ~250
        cold_end = warm_end + int(n * 0.35)  # 35% -> ~350
        # Remaining ~30% are dormant

        for i, eid in enumerate(ordered):
            if i < hot_end:
                # Hot: 10-50 accesses in last 0-7 days
                tier_map[eid] = "hot"
                num_accesses = self._rng.randint(10, 50)
                for _ in range(num_accesses):
                    offset_secs = self._rng.uniform(0, 7 * 86400)
                    events.append((eid, self._now - offset_secs))
            elif i < warm_end:
                # Warm: 5-15 accesses, 1-30 days old
                tier_map[eid] = "warm"
                num_accesses = self._rng.randint(5, 15)
                for _ in range(num_accesses):
                    offset_secs = self._rng.uniform(1 * 86400, 30 * 86400)
                    events.append((eid, self._now - offset_secs))
            elif i < cold_end:
                # Cold: 1-3 accesses, 7-90 days old
                tier_map[eid] = "cold"
                num_accesses = self._rng.randint(1, 3)
                for _ in range(num_accesses):
                    offset_secs = self._rng.uniform(7 * 86400, 90 * 86400)
                    events.append((eid, self._now - offset_secs))
            else:
                # Dormant: no accesses
                tier_map[eid] = "dormant"

        # Sort events by timestamp for deterministic ordering
        events.sort(key=lambda x: x[1])

        return events, tier_map

    # -- episode generation --------------------------------------------------

    _EPISODE_TEMPLATES = [
        "Working on {entity_name} — {summary}. Involves {related}.",
        "Meeting about {entity_name}. Key topics: {summary}. Participants: {related}.",
        "Research session on {entity_name}. Notes: {summary}.",
        "Discussion about {entity_name} ({entity_type}). Context: {summary}.",
        "Deep dive into {entity_name}. {summary}. Connected to {related}.",
        "Review of {entity_name} progress. {summary}. Stakeholders: {related}.",
    ]

    def _generate_episodes(
        self,
        entities: list[Entity],
        entity_map: dict[str, Entity],
        access_tiers: dict[str, str],
        adj: dict[str, set[str]],
    ) -> tuple[list[Episode], list[tuple[str, str]]]:
        """Generate synthetic episodes tied to the temporal distribution.

        Returns (episodes, episode_entity_links) where links are (episode_id, entity_id).
        """
        episodes: list[Episode] = []
        episode_entities: list[tuple[str, str]] = []
        ep_idx = 0

        hot_ids = [e.id for e in entities if access_tiers.get(e.id) == "hot"]
        warm_ids = [e.id for e in entities if access_tiers.get(e.id) == "warm"]
        cold_ids = [e.id for e in entities if access_tiers.get(e.id) == "cold"]

        def _make_episode(
            entity_id: str,
            min_days: float,
            max_days: float,
        ) -> None:
            nonlocal ep_idx
            entity = entity_map[entity_id]
            neighbors = sorted(adj.get(entity_id, set()))
            related_names = [entity_map[n].name for n in neighbors[:3] if n in entity_map]
            related_str = ", ".join(related_names) if related_names else entity.name

            template = self._rng.choice(self._EPISODE_TEMPLATES)
            content = template.format(
                entity_name=entity.name,
                entity_type=entity.entity_type,
                summary=entity.summary[:120] if entity.summary else entity.name,
                related=related_str,
            )

            offset_secs = self._rng.uniform(min_days * 86400, max_days * 86400)
            created_at = datetime.utcfromtimestamp(self._now - offset_secs)
            ep_id = f"ep_bench_{ep_idx:04d}"

            episodes.append(
                Episode(
                    id=ep_id,
                    content=content,
                    source="benchmark",
                    status=EpisodeStatus.COMPLETED,
                    group_id=GROUP_ID,
                    created_at=created_at,
                )
            )

            # Link to main entity + up to 2 neighbors
            episode_entities.append((ep_id, entity_id))
            for n_id in neighbors[:2]:
                episode_entities.append((ep_id, n_id))

            ep_idx += 1

        # Hot entities (100): 1 episode each, 0-7 days ago
        for eid in hot_ids:
            _make_episode(eid, 0, 7)

        # Warm entities (250): ~40% get 1 episode, 1-30 days ago
        for eid in warm_ids:
            if self._rng.random() < 0.40:
                _make_episode(eid, 1, 30)

        # Cold entities (350): ~15% get 1 episode, 7-90 days ago
        for eid in cold_ids:
            if self._rng.random() < 0.15:
                _make_episode(eid, 7, 90)

        return episodes, episode_entities

    def _add_episode_ground_truth(
        self,
        ground_truth: list[GroundTruthQuery],
        episodes: list[Episode],
        episode_entities: list[tuple[str, str]],
    ) -> None:
        """Add episode relevance to recency queries based on created_at timestamps."""
        # Build episode -> entity links
        ep_entity_map: dict[str, set[str]] = {}
        for ep_id, ent_id in episode_entities:
            ep_entity_map.setdefault(ep_id, set()).add(ent_id)

        for query in ground_truth:
            if query.category != "recency":
                continue

            # Recency queries grade by time windows — match episodes the same way
            # Parse windows from the existing relevant entity grades to determine
            # what time range this query covers
            for episode in episodes:
                ep_age_days = (self._now - episode.created_at.timestamp()) / 86400.0
                # Simple heuristic: episodes within the query's implied window
                # get graded based on freshness
                if ep_age_days <= 1:
                    grade = 3
                elif ep_age_days <= 7:
                    grade = 2
                elif ep_age_days <= 30:
                    grade = 1
                else:
                    continue

                # Only include if episode mentions entities relevant to this query
                ep_entities = ep_entity_map.get(episode.id, set())
                if ep_entities & set(query.relevant_entities.keys()):
                    query.relevant_episodes[episode.id] = grade

    # -- conversation scenario generation ------------------------------------

    def _generate_conversation_scenarios(
        self,
        entity_map: dict[str, Entity],
        clusters: list[dict],
        adj: dict[str, set[str]],
    ) -> list[ConversationScenario]:
        """Generate multi-query conversation scenarios for working memory benchmarking.

        Each scenario picks two clusters with bridge entities between them
        and creates a 3-query conversation:
          Q1: ask about cluster A's hub entity
          Q2: ask about cluster B's hub entity
          Q3: ask a bridging question — expected to find bridge entities via WM
        """
        scenarios: list[ConversationScenario] = []
        real_clusters = [c for c in clusters if c["name"] != "Long Tail" and c.get("hub")]

        if len(real_clusters) < 2:
            return scenarios

        # Build entity->cluster index
        entity_cluster: dict[str, int] = {}
        for ci, cluster in enumerate(real_clusters):
            for member in cluster["members"]:
                entity_cluster[member] = ci

        # Find cluster pairs with bridge entities
        bridge_pairs: dict[tuple[int, int], list[str]] = {}
        for eid, cluster_idx in entity_cluster.items():
            neighbors = adj.get(eid, set())
            for neighbor in neighbors:
                n_cluster = entity_cluster.get(neighbor)
                if n_cluster is not None and n_cluster != cluster_idx:
                    key = (min(cluster_idx, n_cluster), max(cluster_idx, n_cluster))
                    bridge_pairs.setdefault(key, []).append(eid)

        # Generate scenarios from distinct cluster pairs
        pair_list = sorted(bridge_pairs.keys())
        self._rng.shuffle(pair_list)

        for pair_idx, (ci_a, ci_b) in enumerate(pair_list[: self._scale.max_scenarios]):
            cluster_a = real_clusters[ci_a]
            cluster_b = real_clusters[ci_b]
            hub_a = cluster_a["hub"]
            hub_b = cluster_b["hub"]

            if not hub_a or not hub_b:
                continue

            hub_a_name = entity_map[hub_a].name
            hub_b_name = entity_map[hub_b].name
            desc_a = cluster_a["description"]
            desc_b = cluster_b["description"]

            bridge_ids = set(bridge_pairs[(ci_a, ci_b)])

            queries = [
                f"Tell me about {hub_a_name}",
                f"What do you know about {hub_b_name}?",
                f"How does {desc_a} connect to {desc_b}?",
            ]

            # Expected bridge: query 2 (index 2) should find bridge entities
            expected_bridge: dict[int, set[str]] = {2: bridge_ids}

            scenarios.append(
                ConversationScenario(
                    name=f"bridge_{cluster_a['name']}_to_{cluster_b['name']}",
                    queries=queries,
                    expected_bridge=expected_bridge,
                )
            )

        return scenarios

    # -- ground truth queries ------------------------------------------------

    def _build_adjacency(
        self,
        relationships: list[Relationship],
    ) -> dict[str, set[str]]:
        """Build undirected adjacency map from relationships."""
        adj: dict[str, set[str]] = {}
        for rel in relationships:
            adj.setdefault(rel.source_id, set()).add(rel.target_id)
            adj.setdefault(rel.target_id, set()).add(rel.source_id)
        return adj

    def _bfs_2hop(
        self,
        start: str,
        adj: dict[str, set[str]],
    ) -> tuple[set[str], set[str]]:
        """Return (1-hop neighbors, 2-hop neighbors) via BFS."""
        hop1 = adj.get(start, set())
        hop2: set[str] = set()
        for n1 in hop1:
            for n2 in adj.get(n1, set()):
                if n2 != start and n2 not in hop1:
                    hop2.add(n2)
        return hop1, hop2

    def _find_shortest_path_entities(
        self,
        start: str,
        end: str,
        adj: dict[str, set[str]],
        max_depth: int = 5,
    ) -> list[str]:
        """BFS shortest path returning entity IDs on path (exclusive of endpoints)."""
        if start == end:
            return []
        from collections import deque

        visited: dict[str, str | None] = {start: None}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited[neighbor] = node
                    if neighbor == end:
                        # Reconstruct path
                        path: list[str] = []
                        cur: str | None = neighbor
                        while cur is not None:
                            path.append(cur)
                            cur = visited[cur]
                        path.reverse()
                        # Return intermediate nodes (exclude start and end)
                        return path[1:-1] if len(path) > 2 else []
                    queue.append((neighbor, depth + 1))
        return []

    def _build_predicate_index(
        self,
        relationships: list[Relationship],
    ) -> dict[str, dict[str, list[str]]]:
        """Build entity_id -> {predicate -> [target_ids]} index."""
        idx: dict[str, dict[str, list[str]]] = {}
        for rel in relationships:
            idx.setdefault(rel.source_id, {}).setdefault(rel.predicate, []).append(rel.target_id)
            idx.setdefault(rel.target_id, {}).setdefault(rel.predicate, []).append(rel.source_id)
        return idx

    def _bfs_3hop(
        self,
        start: str,
        adj: dict[str, set[str]],
    ) -> tuple[set[str], set[str], set[str]]:
        """Return (1-hop, 2-hop, 3-hop) neighbor sets via BFS."""
        hop1 = adj.get(start, set())
        hop2: set[str] = set()
        for n1 in hop1:
            for n2 in adj.get(n1, set()):
                if n2 != start and n2 not in hop1:
                    hop2.add(n2)
        hop3: set[str] = set()
        for n2 in hop2:
            for n3 in adj.get(n2, set()):
                if n3 != start and n3 not in hop1 and n3 not in hop2:
                    hop3.add(n3)
        return hop1, hop2, hop3

    def _find_bridge_entities(
        self,
        clusters: list[dict],
        adj: dict[str, set[str]],
    ) -> list[tuple[str, int, int]]:
        """Find entities that connect two clusters.

        Returns list of (entity_id, cluster_idx_a, cluster_idx_b).
        """
        # Build entity->cluster index
        entity_cluster: dict[str, int] = {}
        real_clusters = [c for c in clusters if c["name"] != "Long Tail"]
        for ci, cluster in enumerate(real_clusters):
            for member in cluster["members"]:
                entity_cluster[member] = ci

        bridges: list[tuple[str, int, int]] = []
        seen: set[str] = set()
        for eid, cluster_idx in entity_cluster.items():
            neighbors = adj.get(eid, set())
            for neighbor in neighbors:
                n_cluster = entity_cluster.get(neighbor)
                if n_cluster is not None and n_cluster != cluster_idx and eid not in seen:
                    bridges.append((eid, cluster_idx, n_cluster))
                    seen.add(eid)
                    break

        return bridges

    def _generate_ground_truth(
        self,
        entities: list[Entity],
        entity_map: dict[str, Entity],
        type_index: dict[str, list[str]],
        clusters: list[dict],
        relationships: list[Relationship],
        rel_count: int,
        access_events: list[tuple[str, float]],
        access_tiers: dict[str, str],
    ) -> list[GroundTruthQuery]:
        """Generate ground truth queries across 8 categories."""
        adj = self._build_adjacency(relationships)
        queries: list[GroundTruthQuery] = []
        query_idx = 0

        # Count accesses per entity
        access_counts: dict[str, int] = {}
        for eid, _ in access_events:
            access_counts[eid] = access_counts.get(eid, 0) + 1

        # Build last-access time per entity (used by recency + temporal_context)
        last_access_time: dict[str, float] = {}
        for eid, ts in access_events:
            if eid not in last_access_time or ts > last_access_time[eid]:
                last_access_time[eid] = ts

        # Identify entities by tier
        hot_entities = [eid for eid, tier in access_tiers.items() if tier == "hot"]
        # Dormant entities have no access events; they are implicitly grade 0

        # Real clusters (excluding long tail)
        real_clusters = [c for c in clusters if c["name"] != "Long Tail" and len(c["members"]) > 2]

        # ---- DIRECT (15) ----
        direct_templates = [
            "What do you know about {name}?",
            "Tell me about {name}",
            "What information do we have on {name}?",
        ]
        # Pick entities from different clusters for variety
        direct_candidates: list[str] = []
        for cluster in real_clusters:
            if cluster.get("hub"):
                direct_candidates.append(cluster["hub"])
        # Fill remaining from various types
        for etype in ["person", "technology", "concept"]:
            for eid in type_index.get(etype, [])[:5]:
                if eid not in direct_candidates:
                    direct_candidates.append(eid)
                if len(direct_candidates) >= 20:
                    break
        # Also include some warm/cold entities for variety
        for etype in ["organization", "project", "location"]:
            for eid in type_index.get(etype, [])[:3]:
                if eid not in direct_candidates:
                    direct_candidates.append(eid)
        self._rng.shuffle(direct_candidates)

        for i in range(self._scale.direct_queries):
            eid = direct_candidates[i % len(direct_candidates)]
            entity = entity_map[eid]
            template = direct_templates[i % len(direct_templates)]
            query_text = template.format(name=entity.name)

            relevant: dict[str, int] = {eid: 3}
            hop1, hop2 = self._bfs_2hop(eid, adj)
            # Grade 2 for direct neighbors (cap at 10 for tractability)
            hop1_list = sorted(hop1)[:10]
            for n_id in hop1_list:
                relevant[n_id] = 2
            # Grade 1 for 2-hop (cap at 5)
            hop2_list = sorted(hop2)[:5]
            for n_id in hop2_list:
                relevant[n_id] = 1

            queries.append(
                GroundTruthQuery(
                    query_id=f"q_direct_{query_idx:03d}",
                    query_text=query_text,
                    relevant_entities=relevant,
                    category="direct",
                )
            )
            query_idx += 1

        # ---- RECENCY (10) ----
        # Each query targets a different time window so relevance sets differ.
        # Window = (min_days_ago, max_days_ago) — entity matches if its
        # last_access_time falls within the window.
        recency_specs = [
            # (template, g3_window, g2_window, g1_window)
            ("What was I working on in the last few hours?", (0, 0.5), (0.5, 1), (1, 3)),
            ("What topics came up today?", (0, 1), (1, 3), (3, 7)),
            ("What was I looking at yesterday?", (1, 2), (0, 1), (2, 7)),
            ("What did I work on this week?", (0, 7), (7, 14), (14, 30)),
            ("What came up in the last few days?", (0, 3), (3, 7), (7, 14)),
            ("What have I been working on this month?", (0, 30), (30, 60), (60, 90)),
            ("What was I focused on last week?", (7, 14), (0, 7), (14, 30)),
            ("What did I deal with last month?", (30, 60), (7, 30), (60, 90)),
            ("What's fresh in my memory?", (0, 1), (1, 7), (7, 30)),
            ("What came up most recently?", (0, 3), (3, 14), (14, 30)),
        ]

        for spec_idx in range(self._scale.recency_queries):
            template, g3_win, g2_win, g1_win = recency_specs[spec_idx % len(recency_specs)]
            graded: dict[int, list[str]] = {3: [], 2: [], 1: []}
            for eid, ts in last_access_time.items():
                days_ago = (self._now - ts) / 86400.0
                if g3_win[0] <= days_ago < g3_win[1]:
                    graded[3].append(eid)
                elif g2_win[0] <= days_ago < g2_win[1]:
                    graded[2].append(eid)
                elif g1_win[0] <= days_ago < g1_win[1]:
                    graded[1].append(eid)

            # Cap per grade (sorted by entity ID for determinism)
            caps = {3: 30, 2: 20, 1: 15}
            capped: dict[str, int] = {}
            for grade in (3, 2, 1):
                for eid in sorted(graded[grade])[: caps[grade]]:
                    capped[eid] = grade

            # Safety: if no entities matched any window, take nearest 5
            if not capped:
                nearest = sorted(
                    last_access_time,
                    key=lambda e: last_access_time[e],
                    reverse=True,
                )[:5]
                for eid in nearest:
                    capped[eid] = 3

            queries.append(
                GroundTruthQuery(
                    query_id=f"q_recency_{query_idx:03d}",
                    query_text=template,
                    relevant_entities=capped,
                    category="recency",
                )
            )
            query_idx += 1

        # ---- FREQUENCY (10) ----
        frequency_templates = [
            "What are my most important topics?",
            "What do I focus on the most?",
            "What entities have I interacted with most?",
            "Show me my most frequently accessed items",
            "What do I keep coming back to?",
            "My top focus areas",
            "Most referenced topics",
            "What am I most engaged with?",
            "Key areas of interest",
            "My primary focus entities",
        ]

        for i in range(self._scale.frequency_queries):
            relevant = {}
            for eid, count in access_counts.items():
                if count >= 20:
                    relevant[eid] = 3
                elif count >= 10:
                    relevant[eid] = 2
                elif count >= 5:
                    relevant[eid] = 1

            queries.append(
                GroundTruthQuery(
                    query_id=f"q_frequency_{query_idx:03d}",
                    query_text=frequency_templates[i % len(frequency_templates)],
                    relevant_entities=relevant,
                    category="frequency",
                )
            )
            query_idx += 1

        # ---- ASSOCIATIVE (10) ----
        associative_templates = [
            "How does {a} connect to {b}?",
            "What links {a} and {b}?",
            "What is the relationship between {a} and {b}?",
            "How are {a} and {b} related?",
            "Tell me about the connection between {a} and {b}",
        ]

        # Pick pairs from different clusters that have a path between them
        assoc_pairs: list[tuple[str, str]] = []
        assoc_target = self._scale.associative_queries
        for ci in range(len(real_clusters)):
            for cj in range(ci + 1, len(real_clusters)):
                if len(assoc_pairs) >= assoc_target:
                    break
                members_i = real_clusters[ci]["members"]
                members_j = real_clusters[cj]["members"]
                if not members_i or not members_j:
                    continue
                a = real_clusters[ci].get("hub") or members_i[0]
                b = real_clusters[cj].get("hub") or members_j[0]
                assoc_pairs.append((a, b))
            if len(assoc_pairs) >= assoc_target:
                break

        for i in range(self._scale.associative_queries):
            if i < len(assoc_pairs):
                a_id, b_id = assoc_pairs[i]
            else:
                # Fallback: random pair from hot entities
                a_id = hot_entities[i % len(hot_entities)] if hot_entities else entities[i].id
                idx_b = (i + 1) % len(hot_entities) if hot_entities else i + 1
                b_id = hot_entities[idx_b] if hot_entities else entities[idx_b].id

            a_name = entity_map[a_id].name
            b_name = entity_map[b_id].name
            template = associative_templates[i % len(associative_templates)]
            query_text = template.format(a=a_name, b=b_name)

            relevant = {a_id: 3, b_id: 3}
            # Find path entities
            path_entities = self._find_shortest_path_entities(a_id, b_id, adj)
            for pe in path_entities:
                relevant[pe] = 3
            # 1-hop from path gets grade 2
            path_set = {a_id, b_id} | set(path_entities)
            for pe in path_set:
                for neighbor in sorted(adj.get(pe, set()))[:3]:
                    if neighbor not in relevant:
                        relevant[neighbor] = 2
            # Same cluster members get grade 1
            for cluster in real_clusters:
                if a_id in cluster["members"] or b_id in cluster["members"]:
                    for m in cluster["members"][:5]:
                        if m not in relevant:
                            relevant[m] = 1

            queries.append(
                GroundTruthQuery(
                    query_id=f"q_associative_{query_idx:03d}",
                    query_text=query_text,
                    relevant_entities=relevant,
                    category="associative",
                )
            )
            query_idx += 1

        # ---- TEMPORAL_CONTEXT (5) ----
        # Tests retrieval of entities matching a time window AND a cluster domain.
        # Router classifies as TEMPORAL → activation-heavy weights (0.55 act).
        # The cluster domain in the query text provides semantic discrimination.
        temporal_context_templates = [
            "What {domain} topics came up recently?",
            "Who was I working with on {domain} lately?",
            "What {domain} work happened this week?",
            "Recent activity in {domain}",
            "What's new in {domain}?",
        ]

        for i in range(self._scale.temporal_context_queries):
            cluster = real_clusters[i % len(real_clusters)]
            domains = cluster.get("domains", ["technology"])
            # Pick a domain deterministically (round-robin across domains)
            domain_phrase = domains[i % len(domains)]
            query_text = temporal_context_templates[i % len(temporal_context_templates)].format(
                domain=domain_phrase
            )

            # Grade by recency within cluster. After summary enrichment,
            # >80% of cluster members will contain their cluster's domain
            # keywords, making them FTS5-discoverable. Grade purely by
            # temporal signal here.
            temporal_relevant: dict[str, int] = {}
            for mid in cluster["members"]:
                if mid in last_access_time:
                    days_ago = (self._now - last_access_time[mid]) / 86400.0
                    if days_ago < 7:
                        temporal_relevant[mid] = 3
                    elif days_ago < 30:
                        temporal_relevant[mid] = 2
            # Cluster members without recent access get grade 1 (cap 5)
            for mid in sorted(cluster["members"])[:5]:
                if mid not in temporal_relevant:
                    temporal_relevant[mid] = 1

            # Safety: if no grade-3 entities, use top-accessed cluster members
            if not any(g == 3 for g in temporal_relevant.values()):
                cluster_accessed = sorted(
                    [(mid, access_counts.get(mid, 0)) for mid in cluster["members"]],
                    key=lambda x: -x[1],
                )
                for mid, _ in cluster_accessed[:3]:
                    temporal_relevant[mid] = 3

            queries.append(
                GroundTruthQuery(
                    query_id=f"q_temporal_context_{query_idx:03d}",
                    query_text=query_text,
                    relevant_entities=temporal_relevant,
                    category="temporal_context",
                )
            )
            query_idx += 1

        # ---- SEMANTIC (10) — graded by relationship predicates ----
        pred_index = self._build_predicate_index(relationships)
        semantic_templates = [
            "What does {person} work on at {org}?",
            "How does {person} use {tech}?",
            "Tell me about {person}'s work with {tech} at {org}",
            "What is {person}'s role involving {concept}?",
            "{person}'s connection to {org} and {tech}",
            "What does {person} research in {concept}?",
            "How does {person} contribute to {tech}?",
            "What teams does {person} work with at {org}?",
            "What is {person} known for at {org}?",
            "{person}'s expertise in {concept} and {tech}",
        ]

        person_ids_all = type_index.get("person", [])
        org_ids_all = type_index.get("organization", [])
        tech_ids_all = type_index.get("technology", [])
        concept_ids_all = type_index.get("concept", [])

        for i in range(self._scale.semantic_queries):
            semantic_relevant: dict[str, int] = {}
            # Pick a focal entity (person) and grade by predicate richness
            focal_person = (
                person_ids_all[(i * 7 + 3) % len(person_ids_all)]
                if person_ids_all
                else entities[i].id
            )
            # Find entities connected by specific predicates
            person_preds = pred_index.get(focal_person, {})

            # Grade 3: connected via WORKS_AT + USES (multiple predicates)
            works_targets = set(person_preds.get("WORKS_AT", []))
            uses_targets = set(person_preds.get("USES", []))
            multi_pred = works_targets | uses_targets
            for eid in sorted(multi_pred)[:5]:
                semantic_relevant[eid] = 3
            semantic_relevant[focal_person] = 3

            # Grade 2: connected via any single predicate
            for pred, targets in person_preds.items():
                for eid in sorted(targets)[:3]:
                    if eid not in semantic_relevant:
                        semantic_relevant[eid] = 2

            # Grade 1: 2-hop neighbors not already graded
            hop1, hop2 = self._bfs_2hop(focal_person, adj)
            for eid in sorted(hop2)[:5]:
                if eid not in semantic_relevant:
                    semantic_relevant[eid] = 1

            # Build query text using focal person's actual connections
            works_at_list = person_preds.get("WORKS_AT", [])
            uses_list = person_preds.get("USES", [])
            concept_list = person_preds.get("EXPERT_IN", []) or person_preds.get("RESEARCHES", [])

            org_name = (
                entity_map[works_at_list[0]].name
                if works_at_list
                else (
                    entity_map[org_ids_all[i % len(org_ids_all)]].name if org_ids_all else "Unknown"
                )
            )
            tech_name = (
                entity_map[uses_list[0]].name
                if uses_list
                else (
                    entity_map[tech_ids_all[i % len(tech_ids_all)]].name
                    if tech_ids_all
                    else "Unknown"
                )
            )
            concept_name = (
                entity_map[concept_list[0]].name
                if concept_list
                else (
                    entity_map[concept_ids_all[i % len(concept_ids_all)]].name
                    if concept_ids_all
                    else "Unknown"
                )
            )
            focal_person_name = entity_map[focal_person].name
            query_text = semantic_templates[i % len(semantic_templates)].format(
                person=focal_person_name,
                org=org_name,
                tech=tech_name,
                concept=concept_name,
            )

            queries.append(
                GroundTruthQuery(
                    query_id=f"q_semantic_{query_idx:03d}",
                    query_text=query_text,
                    relevant_entities=semantic_relevant,
                    category="semantic",
                )
            )
            query_idx += 1

        # ---- GRAPH_TRAVERSAL (10) — graded by hop distance ----
        traversal_templates = [
            "What entities are closely connected to {name}?",
            "Show me the neighborhood of {name}",
            "What is directly related to {name}?",
            "Find entities near {name} in the graph",
            "Who and what are linked to {name}?",
            "Map the connections around {name}",
            "What clusters with {name}?",
            "Show me {name}'s network",
            "What entities orbit {name}?",
            "Explore the vicinity of {name}",
        ]

        # Pick hub entities for traversal queries (well-connected)
        traversal_targets: list[str] = []
        for cluster in real_clusters:
            if cluster.get("hub"):
                traversal_targets.append(cluster["hub"])
        # Pad if needed
        while len(traversal_targets) < self._scale.graph_traversal_queries:
            idx_pad = len(traversal_targets) % len(entities)
            eid = entities[idx_pad].id
            if eid not in traversal_targets:
                traversal_targets.append(eid)

        for i in range(self._scale.graph_traversal_queries):
            target_id = traversal_targets[i % len(traversal_targets)]
            target_name = entity_map[target_id].name
            query_text = traversal_templates[i % len(traversal_templates)].format(name=target_name)

            hop1, hop2, hop3 = self._bfs_3hop(target_id, adj)

            relevant = {target_id: 3}
            # Grade 3 for 1-hop
            for eid in sorted(hop1)[:10]:
                relevant[eid] = 3
            # Grade 2 for 2-hop
            for eid in sorted(hop2)[:8]:
                if eid not in relevant:
                    relevant[eid] = 2
            # Grade 1 for 3-hop
            for eid in sorted(hop3)[:5]:
                if eid not in relevant:
                    relevant[eid] = 1

            queries.append(
                GroundTruthQuery(
                    query_id=f"q_traversal_{query_idx:03d}",
                    query_text=query_text,
                    relevant_entities=relevant,
                    category="graph_traversal",
                )
            )
            query_idx += 1

        # ---- CROSS_CLUSTER (5) — graded by bridge role ----
        cross_templates = [
            "What connects the {a} and {b} domains?",
            "How do {a} and {b} relate to each other?",
            "What bridges {a} and {b}?",
            "Find the link between {a} and {b}",
            "What ties {a} to {b}?",
        ]

        bridges = self._find_bridge_entities(clusters, adj)
        self._rng.shuffle(bridges)

        # Deduplicate by cluster pair (unordered)
        seen_pairs: set[frozenset[int]] = set()
        deduped: list[tuple[str, int, int]] = []
        for bridge_id, ci_a, ci_b in bridges:
            pair = frozenset({ci_a, ci_b})
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                deduped.append((bridge_id, ci_a, ci_b))
        bridges = deduped

        for i in range(self._scale.cross_cluster_queries):
            if i < len(bridges):
                bridge_id, ci_a, ci_b = bridges[i]
                cluster_a_phrase = (
                    real_clusters[ci_a]["description"]
                    if ci_a < len(real_clusters)
                    else "unknown domain"
                )
                cluster_b_phrase = (
                    real_clusters[ci_b]["description"]
                    if ci_b < len(real_clusters)
                    else "unknown domain"
                )
            else:
                # Fallback: pick two clusters and find any shared neighbors
                ci_a = i % len(real_clusters)
                ci_b = (i + 1) % len(real_clusters)
                cluster_a_phrase = real_clusters[ci_a]["description"]
                cluster_b_phrase = real_clusters[ci_b]["description"]
                members_ci = real_clusters[ci_a]["members"]
                bridge_id = members_ci[0] if members_ci else entities[0].id

            query_text = cross_templates[i % len(cross_templates)].format(
                a=cluster_a_phrase,
                b=cluster_b_phrase,
            )

            relevant = {bridge_id: 3}

            # Other bridge entities between same clusters get grade 3
            for bid, ca, cb in bridges:
                if bid != bridge_id and {ca, cb} == {ci_a, ci_b}:
                    relevant[bid] = 3

            # Path entities between clusters get grade 2
            members_a = set(real_clusters[ci_a]["members"]) if ci_a < len(real_clusters) else set()
            members_b = set(real_clusters[ci_b]["members"]) if ci_b < len(real_clusters) else set()
            bridge_neighbors = adj.get(bridge_id, set())
            for n in sorted(bridge_neighbors)[:5]:
                if n not in relevant:
                    relevant[n] = 2

            # Cluster members get grade 1
            for m in sorted(members_a)[:3]:
                if m not in relevant:
                    relevant[m] = 1
            for m in sorted(members_b)[:3]:
                if m not in relevant:
                    relevant[m] = 1

            queries.append(
                GroundTruthQuery(
                    query_id=f"q_cross_cluster_{query_idx:03d}",
                    query_text=query_text,
                    relevant_entities=relevant,
                    category="cross_cluster",
                )
            )
            query_idx += 1

        return queries


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def generate_corpus(seed: int = 42, total_entities: int = 1000) -> CorpusSpec:
    """Convenience wrapper: generate a benchmark corpus with default settings."""
    return CorpusGenerator(seed=seed, total_entities=total_entities).generate()
