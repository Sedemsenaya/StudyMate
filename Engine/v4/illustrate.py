"""
StudyMate v4 - AI illustration mode (fast, CPU friendly).

Why v4's Ollama path was slow: the model wrote the whole lesson (every scene's
40-140 word narration + every visual as JSON), with no output cap, and was asked
again whenever coverage failed. On a CPU that is 2,000-4,000 generated tokens per
section, so 400-2,000 s each.

This mode splits the work:
  * rules (instant)   read the question, options, answer and explanation word for
                      word, and build the scenes and narration
  * AI (short calls)  only DRAWS: one call per question returns a small JSON
                      sketch - the scenario diagram, the answer diagram, one line
                      per wrong option, a takeaway (~250-400 tokens, capped)
  * vision (short)    a vision model reads each picture in the explanation
                      (one or two sentences, capped) and the text model uses it
Every call has a time limit; a slow or broken answer falls back to the rules
sketch for that question only, so a run never stalls. Every answer is cached
(Engine/Output/.ai_cache), so re-runs skip the AI.

Visuals drawn here (registered with v3 visuals.py):
  arch     AWS-style architecture: big service icons, labels under them, arrows,
           the answer's service circled in green
  choices  option cards with an icon each; on the answer scene the right one
           gets a green tick, the wrong ones a red cross and the reason
"""
import base64
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
sys.path.insert(0, str(ENGINE / "v3"))
sys.path.insert(0, str(ENGINE / "visualization"))

import visuals                                                     # noqa: E402
from canvas import GREEN, INK, MUTED, RED, fit_text, text_size     # noqa: E402

ICON_FILE = HERE / "icons_aws.json"     # AWS logos (CC0) + Fluent Emoji (MIT), see "_licenses"
try:
    BUNDLED = json.loads(ICON_FILE.read_text(encoding="utf-8"))["icons"]
except Exception:
    BUNDLED = {}

# ---------------------------------------------------------------- AWS services -> icon

SERVICES = [  # (regex, short label, icon file stem)
    (r"Kinesis Data Firehose|Data Firehose|Firehose", "Kinesis Data Firehose", "aws-kinesis"),
    (r"Kinesis Data Streams|Kinesis Data Analytics|Managed Service for Apache Flink|Kinesis", "Kinesis Data Streams", "aws-kinesis"),
    (r"Simple Queue Service|\bSQS\b", "Amazon SQS", "aws-sqs"),
    (r"Simple Notification Service|\bSNS\b", "Amazon SNS", "aws-sns"),
    (r"EventBridge|CloudWatch Events", "EventBridge", "aws-eventbridge"),
    (r"Step Functions", "Step Functions", "aws-step-functions"),
    (r"\bLambda\b", "AWS Lambda", "aws-lambda"),
    (r"\bFargate\b", "AWS Fargate", "aws-fargate"),
    (r"\bECS\b|Elastic Container Service", "Amazon ECS", "aws-ecs"),
    (r"\bEKS\b|Elastic Kubernetes", "Amazon EKS", "aws-eks"),
    (r"Elastic Beanstalk", "Elastic Beanstalk", "aws-elastic-beanstalk"),
    (r"Lightsail", "Lightsail", "aws-lightsail"),
    (r"Auto Scaling", "Auto Scaling", "aws-ec2"),
    (r"\bEC2\b|Elastic Compute Cloud|instances?\b", "Amazon EC2", "aws-ec2"),
    (r"Application Load Balancer|\bALB\b|Network Load Balancer|\bNLB\b|Gateway Load Balancer|\bGWLB\b|Elastic Load Balanc|\bELB\b|load balancer", "Load Balancer", "aws-elb"),
    (r"API Gateway", "API Gateway", "aws-api-gateway"),
    (r"CloudFront", "CloudFront", "aws-cloudfront"),
    (r"Route 53", "Route 53", "aws-route53"),
    (r"Global Accelerator", "Global Accelerator", "generic-network"),
    (r"Transit Gateway", "Transit Gateway", "aws-vpc"),
    (r"Direct Connect", "Direct Connect", "generic-network"),
    (r"Site-to-Site VPN|\bVPN\b", "Site-to-Site VPN", "generic-lock"),
    (r"VPC endpoint|gateway endpoint|interface endpoint|PrivateLink", "VPC Endpoint", "aws-vpc"),
    (r"NAT gateway|NAT instance", "NAT Gateway", "aws-vpc"),
    (r"internet gateway", "Internet Gateway", "generic-internet"),
    (r"security group", "Security Group", "generic-shield"),
    (r"\bNACL|network ACL", "Network ACL", "generic-shield"),
    (r"\bVPC\b|subnet", "Amazon VPC", "aws-vpc"),
    (r"S3 Glacier|Glacier", "S3 Glacier", "aws-glacier"),
    (r"\bS3\b|Simple Storage Service", "Amazon S3", "aws-s3"),
    (r"\bEFS\b|Elastic File System", "Amazon EFS", "generic-folder"),
    (r"\bFSx\b", "Amazon FSx", "generic-folder"),
    (r"\bEBS\b|Elastic Block Store", "Amazon EBS", "generic-backup"),
    (r"Storage Gateway|Volume Gateway|File Gateway|Tape Gateway", "Storage Gateway", "generic-backup"),
    (r"Snowball|Snowmobile|Snowcone", "AWS Snowball", "generic-backup"),
    (r"DataSync", "AWS DataSync", "generic-backup"),
    (r"AWS Backup", "AWS Backup", "aws-backup"),
    (r"Database Migration Service|\bDMS\b", "AWS DMS", "generic-database"),
    (r"Aurora", "Amazon Aurora", "aws-aurora"),
    (r"DynamoDB", "DynamoDB", "aws-dynamodb"),
    (r"ElastiCache|Redis|Memcached", "ElastiCache", "aws-elasticache"),
    (r"Redshift", "Amazon Redshift", "aws-redshift"),
    (r"DocumentDB", "DocumentDB", "aws-documentdb"),
    (r"Neptune", "Amazon Neptune", "aws-neptune"),
    (r"Keyspaces", "Keyspaces", "aws-keyspaces"),
    (r"Timestream", "Timestream", "aws-timestream"),
    (r"\bRDS\b|Relational Database", "Amazon RDS", "aws-rds"),
    (r"Athena", "Amazon Athena", "aws-athena"),
    (r"\bGlue\b", "AWS Glue", "aws-glue"),
    (r"\bEMR\b", "Amazon EMR", "generic-chart"),
    (r"QuickSight", "QuickSight", "aws-quicksight"),
    (r"OpenSearch|Elasticsearch", "OpenSearch", "aws-open-search"),
    (r"Lake Formation", "Lake Formation", "aws-lake-formation"),
    (r"\bMSK\b|Managed Streaming for Apache Kafka|Kafka", "Amazon MSK", "aws-msk"),
    (r"Amazon MQ", "Amazon MQ", "aws-mq"),
    (r"AppSync", "AppSync", "aws-appsync"),
    (r"Cognito", "Cognito", "aws-cognito"),
    (r"\bIAM\b|role|policy|policies", "IAM", "aws-iam"),
    (r"Organizations|\bSCP\b|service control polic", "Organizations", "aws-iam"),
    (r"\bKMS\b|Key Management", "AWS KMS", "aws-kms"),
    (r"Secrets Manager", "Secrets Manager", "aws-secrets-manager"),
    (r"Systems Manager|\bSSM\b|Parameter Store", "Systems Manager", "aws-systems-manager"),
    (r"Certificate Manager|\bACM\b", "Certificate Manager", "aws-certificate-manager"),
    (r"\bWAF\b", "AWS WAF", "aws-waf"),
    (r"Shield", "AWS Shield", "aws-shield"),
    (r"GuardDuty", "GuardDuty", "generic-search"),
    (r"Macie", "Amazon Macie", "generic-search"),
    (r"Inspector", "Inspector", "generic-search"),
    (r"CloudTrail", "CloudTrail", "aws-cloudtrail"),
    (r"CloudWatch", "CloudWatch", "aws-cloudwatch"),
    (r"AWS Config|\bConfig rules?", "AWS Config", "aws-config"),
    (r"X-Ray", "X-Ray", "aws-xray"),
    (r"StackSets?", "CloudFormation StackSets", "aws-cloudformation"),
    (r"CloudFormation", "CloudFormation", "aws-cloudformation"),
    (r"Resource Access Manager|\bRAM\b", "AWS RAM", "generic-users"),
    (r"origin access (identity|control)|\bOAI\b|\bOAC\b", "Origin Access Identity", "generic-key"),
    (r"Elastic IP", "Elastic IP", "generic-network"),
    (r"CodePipeline", "CodePipeline", "aws-codepipeline"),
    (r"CodeDeploy", "CodeDeploy", "aws-codedeploy"),
    (r"CodeBuild", "CodeBuild", "aws-codebuild"),
    (r"Directory Service|Active Directory|AD Connector|Simple AD", "Directory Service", "generic-users"),
    (r"\bSES\b|Simple Email", "Amazon SES", "aws-ses"),
    (r"Amplify", "Amplify", "aws-amplify"),
]
SERVICES = [(re.compile(rx, re.I), label, icon) for rx, label, icon in SERVICES]

TECH = [(re.compile(rx, re.I), stem) for rx, stem in [
    (r"\bdocker", "logo-docker"), (r"kubernetes|\bk8s\b|kubectl|\bpods?\b", "logo-kubernetes"),
    (r"\bhelm\b", "logo-helm"), (r"terraform", "logo-terraform"), (r"\bredis\b", "logo-redis"),
    (r"\breact\b", "logo-react"), (r"azure", "logo-microsoft-azure"), (r"postgres", "logo-postgresql"),
    (r"mysql", "logo-mysql"), (r"mongo", "logo-mongodb"), (r"\bpython\b", "logo-python"),
    (r"javascript|\bjs\b", "logo-javascript"), (r"typescript", "logo-typescript"), (r"node\.?js", "logo-nodejs"),
    (r"\bgithub\b", "logo-github"), (r"\bgit\b", "logo-git"), (r"nginx", "logo-nginx"), (r"\blinux\b", "logo-linux-tux"),
    (r"ubuntu", "logo-ubuntu"), (r"kafka", "logo-kafka"), (r"elasticsearch", "logo-elasticsearch"),
    (r"grafana", "logo-grafana"), (r"prometheus", "logo-prometheus"), (r"jenkins", "logo-jenkins"),
    (r"ansible", "logo-ansible"), (r"google cloud|\bgcp\b", "logo-google-cloud"), (r"tensorflow", "logo-tensorflow"),
    (r"pytorch", "logo-pytorch"), (r"numpy", "logo-numpy"), (r"pandas", "logo-pandas"), (r"jupyter", "logo-jupyter"),
    (r"\bhtml", "logo-html-5"), (r"\bcss\b", "logo-css-3"), (r"\bjava\b", "logo-java"), (r"c\+\+", "logo-c-plusplus"),
    (r"\bgolang\b|\bgo\b(?= (?:code|program|language))", "logo-go"), (r"\brust\b", "logo-rust"),
    (r"sqlite", "logo-sqlite"), (r"graphql", "logo-graphql"), (r"spark", "logo-apache-spark"), (r"hadoop", "logo-hadoop"),
    (r"airflow", "logo-airflow"), (r"databricks", "logo-databricks"), (r"snowflake", "logo-snowflake"),
    (r"openai|chatgpt|\bgpt", "logo-openai"), (r"nvidia|cuda", "logo-nvidia"), (r"raspberry", "logo-raspberry-pi"),
    (r"arduino", "logo-arduino"), (r"firebase", "logo-firebase"), (r"flask", "logo-flask"), (r"django", "logo-django"),
    (r"fastapi", "logo-fastapi"), (r"\bvue\b", "logo-vue"), (r"angular", "logo-angular"), (r"next\.?js", "logo-nextjs"),
    (r"rabbitmq", "logo-rabbitmq"), (r"memcached", "logo-memcached"), (r"cassandra", "logo-cassandra"),
    (r"neo4j", "logo-neo4j"), (r"oracle", "logo-oracle"), (r"windows", "logo-microsoft-windows"),
    (r"android", "logo-android"), (r"\bbash\b|shell script", "logo-bash"), (r"vs ?code|visual studio code",
                                                                           "logo-visual-studio-code")]]

GENERIC = {  # word in a label -> generic icon
    "user": "generic-user", "customer": "generic-user", "client": "generic-user", "admin": "generic-user",
    "users": "generic-users", "team": "generic-users", "developers": "generic-users", "people": "generic-users",
    "internet": "generic-internet", "web": "generic-internet", "public": "generic-internet",
    "on-premises": "generic-office", "on-premise": "generic-office", "data center": "generic-office",
    "office": "generic-office", "company": "generic-office", "corporate": "generic-office",
    "mobile": "generic-mobile", "phone": "generic-mobile", "app": "generic-mobile", "device": "generic-sensor",
    "iot": "generic-sensor", "sensor": "generic-sensor", "laptop": "generic-laptop",
    "server": "generic-computer", "application": "generic-computer", "website": "generic-internet",
    "database": "generic-database", "db": "generic-database", "data": "generic-database",
    "file": "generic-file", "files": "generic-file", "document": "generic-file", "report": "generic-file",
    "logs": "generic-log", "log": "generic-log", "video": "generic-video", "image": "generic-image",
    "game": "generic-game", "gaming": "generic-game", "lock": "generic-lock", "encrypt": "generic-lock",
    "key": "generic-key", "secret": "generic-key", "clock": "generic-clock", "schedule": "generic-clock",
    "metrics": "generic-chart", "analytics": "generic-chart", "dashboard": "generic-chart",
    "region": "generic-region", "global": "generic-region", "queue": "generic-queue",
    "email": "generic-email", "alert": "generic-alert", "notification": "generic-alert",
    "cost": "generic-money", "money": "generic-money", "backup": "generic-backup",
    "storage": "generic-bucket", "producer": "generic-sensor", "producers": "generic-sensor",
    "consumer": "generic-computer", "consumers": "generic-computer", "shop": "generic-shopping",
    "gear": "generic-gear", "apps": "generic-computer", "everyone": "generic-users",
    "attacker": "generic-warning", "supplier": "generic-users", "suppliers": "generic-users", "players": "generic-users",
    "account": "generic-office", "accounts": "generic-office", "department": "generic-office",
    "departments": "generic-office", "firewall": "generic-shield", "edge": "generic-region",
    "template": "generic-file", "bucket policy": "generic-file", "allow list": "generic-file",
    "hospital": "generic-hospital", "bank": "generic-bank", "security": "generic-shield",
    "attack": "generic-warning", "ddos": "generic-warning", "error": "generic-warning",
}


def services_in(text, limit=6):
    """AWS services named in text, in the order they appear (deduplicated)."""
    hits = []
    for rx, label, icon in SERVICES:
        for m in rx.finditer(text or ""):
            hits.append((m.start(), label, icon))
            break
    hits.sort()
    labels = {l for _, l, _ in hits}
    out, seen = [], set()
    for pos, label, icon in hits:
        if label in seen or any(label in o and label != o for o in labels):
            continue                                   # "CloudFormation" inside "CloudFormation StackSets"
        seen.add(label)
        out.append((label, icon))
    # "Kinesis Data Firehose" also matches the generic Kinesis pattern
    names = [l for l, _ in out]
    if "Kinesis Data Firehose" in names and "Kinesis Data Streams" in names and \
            not re.search(r"Data Streams|Data Analytics", text or "", re.I):
        out = [x for x in out if x[0] != "Kinesis Data Streams"]
    return out[:limit]


def icon_for(name):
    """Icon file stem for an AWS service or a plain noun; None if nothing fits."""
    s = str(name or "")
    for rx, stem in TECH:                       # Docker, Kubernetes, Redis... before AWS words like "instance"
        if rx.search(s):
            return stem
    for rx, _, icon in SERVICES:
        if rx.search(s):
            return icon
    low = s.lower()
    for w in sorted(GENERIC, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", low):
            return GENERIC[w]
    return None


class AwsIcons:
    """IconResolver look-alike: bundled AWS + generic icons first, then the v3 resolver."""

    def __init__(self, fallback=None):
        self.fallback = fallback
        self.cache = {}
        try:
            from icons import IconHit
            self.Hit = IconHit
        except Exception:
            self.Hit = None

    def find(self, concept, subject=None):
        stem = icon_for(concept)
        if stem and stem in BUNDLED and self.Hit:
            svg = BUNDLED[stem].encode("utf-8")
            src = "logos set (CC0)" if stem.startswith(("aws", "logo")) else "Fluent Emoji (MIT)"
            return self.Hit("bundled", stem, 1.0, svg=svg, credit=src)
        if self.fallback:
            return self.fallback.find(concept, subject)
        return None


# ---------------------------------------------------------------- the two visuals

def _graph(v):
    nodes = []
    for i, n in enumerate(v.get("nodes") or []):
        if isinstance(n, str):
            n = {"id": n, "label": n}
        nid = str(n.get("id") or n.get("label") or i)
        nodes.append({"id": nid, "label": str(n.get("label") or nid)[:40],
                      "icon": n.get("icon") or n.get("label"),
                      "highlight": bool(n.get("highlight"))})
    ids = {n["id"] for n in nodes}
    by_label = {n["label"].lower(): n["id"] for n in nodes}
    edges = []
    for e in v.get("edges") or []:
        if isinstance(e, dict):
            a, b, lab = str(e.get("from", "")), str(e.get("to", "")), str(e.get("label") or "")
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            a, b, lab = str(e[0]), str(e[1]), str(e[2]) if len(e) > 2 else ""
        else:
            continue
        a, b = by_label.get(a.lower(), a), by_label.get(b.lower(), b)
        if a in ids and b in ids and a != b:
            edges.append((a, b, lab[:24]))
    return nodes, edges


def _columns(nodes, edges):
    """Left-to-right layers (longest path from a source)."""
    ids = [n["id"] for n in nodes]
    col = {i: 0 for i in ids}
    for _ in range(len(ids)):
        changed = False
        for a, b, _ in edges:
            if col[b] < col[a] + 1 and col[a] + 1 < len(ids):
                col[b] = col[a] + 1
                changed = True
        if not changed:
            break
    cols = {}
    for n in nodes:
        cols.setdefault(col[n["id"]], []).append(n)
    return [cols[k] for k in sorted(cols)]


def build_arch(cv, v, box, step):
    nodes, edges = _graph(v)
    if not nodes:
        return step
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cols = _columns(nodes, edges)
    nc = len(cols)
    rows = max(len(c) for c in cols)
    cell_w = W / nc
    cell_h = min(H / rows, 300)
    icon = max(64, min(140, cell_w * 0.45, cell_h * 0.5))
    lab_h = 52
    boxes = {}
    order = []
    for c, colnodes in enumerate(cols):
        top = y0 + (H - len(colnodes) * cell_h) / 2
        for r, n in enumerate(colnodes):
            cx = x0 + cell_w * (c + 0.5)
            cy = top + cell_h * (r + 0.5) - lab_h / 2
            order.append((n, cx, cy))
    for n, cx, cy in order:                         # one node per animation step
        ix, iy = cx - icon / 2, cy - icon / 2
        bw = max(icon + 40, min(cell_w - 30, 230))
        frame = (cx - bw / 2, iy - 14, bw, icon + lab_h + 30)
        if n["highlight"]:
            cv.rect(*frame, step, "#ebfbee", "#2f9e44", 3)
        import drawlib
        aws = (icon_for(n["label"]) or "").startswith(("aws", "logo")) or \
            (icon_for(n["icon"]) or "").startswith(("aws", "logo"))
        dev = None if aws else drawlib.device_for(n["label"])
        placed = False
        if dev:                                     # servers, databases, laptops...: drawn by the engine
            drawlib.draw_device(cv, dev, cx, iy + icon / 2, icon, step)
            placed = True
            dev = "drawn"
        elif cv.icons:
            placed = cv.icon(n["icon"], ix, iy, icon, step)
        if placed:
            lab, sz = fit_text(n["label"], bw - 10, lab_h, 22, 14)
            cv.text(cx, iy + icon + 8, lab, step, sz, INK, "center", "top", math=False)
        else:                                       # no fitting icon: a hand-drawn box with the label
            from canvas import fill_for
            bh = icon * 0.8
            by = iy + (icon + lab_h - bh) / 2
            cv.rect(cx - (bw - 16) / 2, by, bw - 16, bh, step,
                    "#b2f2bb" if n["highlight"] else fill_for(len(boxes)))
            cv.label_in(cx - (bw - 16) / 2, by, bw - 16, bh, n["label"], step, 22)
        boxes[n["id"]] = frame
        if n["highlight"] and v.get("highlight_text", "✔ answer"):
            cv.text(frame[0] + frame[2] - 8, frame[1] - 4, v.get("highlight_text", "✔ answer"), step, 18,
                    "#2f9e44", "right", "bottom", math=False)
        step += 1
    for a, b, lab in edges:                         # then the arrows, one by one
        cv.arrow_between(boxes[a], boxes[b], step, lab)
        step += 1
    return step


def build_choices(cv, v, box, step):
    items = v.get("items") or []
    if not items:
        return step
    reveal = bool(v.get("reveal"))
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    n = len(items)
    per_row = 2 if n > 3 else n
    nrows = (n + per_row - 1) // per_row
    gap = 26
    cw = (W - gap * (per_row - 1)) / per_row
    ch = min(220, (H - gap * (nrows - 1)) / nrows)
    for i, it in enumerate(items):
        r, c = divmod(i, per_row)
        x = x0 + c * (cw + gap)
        y = y0 + (H - nrows * ch - (nrows - 1) * gap) / 2 + r * (ch + gap)
        ok = it.get("correct")
        fill = ("#ebfbee" if ok else "#fff5f5") if reveal else "#f8f9fa"
        stroke = ("#2f9e44" if ok else "#e03131") if reveal else INK
        cv.rect(x, y, cw, ch, step, fill, stroke, 2.5 if reveal else 2)
        isz = min(72, ch * 0.45)
        cv.text(x + 18, y + 14, it.get("letter", ""), step, 26, MUTED, math=False)
        cv.icon(it.get("icon") or it.get("name"), x + 18, y + 50, isz, step)
        tx = x + 18 + isz + 18
        tw = cw - (tx - x) - 16
        name, nsz = fit_text(it.get("name", ""), tw, ch * 0.42, 23, 14)
        cv.text(tx, y + 16, name, step, nsz, INK, math=False)
        _, nh = text_size(name, nsz)
        sub = it.get("why") if reveal else it.get("detail")
        if sub:
            s, ssz = fit_text(sub, tw, ch - nh - 34, 18, 12)
            cv.text(tx, y + 24 + nh, s, step, ssz, "#2f9e44" if (reveal and ok) else
                    ("#c92a2a" if reveal else MUTED), math=False)
        if reveal:
            mark = "✔" if ok else "✘"
            cv.text(x + cw - 16, y + 10, mark, step + 1, 40, "#2f9e44" if ok else "#e03131",
                    "right", "top", math=False)
        step += 1
    return step + 1


visuals.BUILDERS["arch"] = build_arch
visuals.BUILDERS["choices"] = build_choices
visuals.ALIASES.update({"aws": "arch", "architecture_icons": "arch", "options": "choices"})


# ---------------------------------------------------------------- Ollama (short, capped calls)

SYSTEM = """You are StudyMate's illustrator. You turn ONE AWS exam question into small
hand-drawn diagrams. You do not write the lesson; you only sketch it.
Rules: every label is 1-4 words. Use real AWS service names ("Amazon S3",
"AWS Lambda", "Kinesis Data Streams") or one of these actors: users, internet,
on-premises, mobile app, application, database, files, logs, region.
scenario = the situation in the question BEFORE it is solved (3-6 nodes).
solution = the architecture of the correct answer (3-7 nodes); set highlight
true on the node(s) that ARE the answer.
explain = 2-3 short plain sentences: why the answer works (max 60 words).
wrong = one entry per wrong option: a 2-5 word name and why it fails (max 14 words).
takeaway = one exam tip, max 12 words. title = the concept, max 6 words.
Answer with JSON only."""

_NODE = {"type": "object", "properties": {"id": {"type": "string"}, "label": {"type": "string"},
                                          "highlight": {"type": "boolean"}},
         "required": ["id", "label"]}
_EDGE = {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"},
                                          "label": {"type": "string"}}, "required": ["from", "to"]}
_DIAG = {"type": "object", "properties": {"nodes": {"type": "array", "items": _NODE},
                                          "edges": {"type": "array", "items": _EDGE}},
         "required": ["nodes", "edges"]}
SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"}, "scenario": _DIAG, "solution": _DIAG,
    "explain": {"type": "string"},
    "wrong": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "why": {"type": "string"}}, "required": ["name", "why"]}},
    "takeaway": {"type": "string"}},
    "required": ["title", "scenario", "solution", "explain", "wrong", "takeaway"]}

VISION_PROMPT = ("This picture is from an AWS exam explanation. In at most two sentences, say what it "
                 "shows (diagram, table or screenshot) and its key labels or numbers. No preamble.")
VISION_MODELS = ["qwen2.5vl:3b", "qwen2.5vl", "gemma3:4b", "gemma3", "moondream", "llava-phi3",
                 "llava:7b", "llava", "minicpm-v", "llama3.2-vision"]


class FastAI:
    def __init__(self, model="qwen3:8b", vision="auto", host=None, timeout=150,
                 num_predict=520, cache_dir=None, log=print):
        from ollama import Client
        self.client = Client(host=host, timeout=timeout) if host else Client(timeout=timeout)
        self.log = log
        names = [m.get("model") or m.get("name") for m in self._models()]
        self.model = self._pick(model, names)
        if not self.model:
            raise RuntimeError(f"model {model!r} is not pulled (ollama pull {model})")
        self.vision = None
        if vision not in ("off", "none", "ocr"):
            want = [vision] if vision != "auto" else VISION_MODELS
            self.vision = next((self._pick(w, names) for w in want if self._pick(w, names)), None)
        self.num_predict = num_predict
        self.cache = Path(cache_dir or ENGINE / "Output" / ".ai_cache")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.stats = {"calls": 0, "cached": 0, "fallback": 0, "seconds": 0.0, "tokens": 0}

    def warm_up(self, timeout=600):
        """Load the model once (so the first real call isn't charged for loading). Returns (secs, on_gpu)."""
        from ollama import Client
        t = time.perf_counter()
        host = getattr(self.client, "_client", None)
        c = Client(host=str(host.base_url)) if host is not None else Client()
        try:
            c._client.timeout = timeout
        except Exception:
            pass
        try:
            c.generate(model=self.model, prompt="ok", keep_alive="30m", options={"num_predict": 1})
        except Exception as ex:
            self.log(f"[ai] warm-up failed ({type(ex).__name__}: {str(ex)[:80]})")
        gpu = None
        try:
            ps = c.ps()
            for m in (ps.get("models") if isinstance(ps, dict) else ps.models):
                m = dict(m) if not isinstance(m, dict) else m
                if (m.get("model") or m.get("name", "")).startswith(self.model.split(":")[0]):
                    gpu = int(m.get("size_vram") or 0) > 0
        except Exception:
            pass
        return time.perf_counter() - t, gpu

    def start_budget(self, seconds):
        self._budget_end = time.perf_counter() + seconds if seconds else None
        self._budget_noted = False

    def out_of_budget(self):
        end = getattr(self, "_budget_end", None)
        if end and time.perf_counter() > end:
            if not self._budget_noted:
                self._budget_noted = True
                self.log("[ai] AI time budget used up: remaining scenes keep the engine's drawing and the "
                         "slide's own words (raise --ai-budget to tutor more)")
            return True
        return False

    def unload(self):
        """Free the model's RAM/CPU before rendering."""
        for m in {self.model, self.vision} - {None}:
            try:
                self.client.generate(model=m, prompt="", keep_alive=0)
            except Exception:
                pass

    def _models(self):
        r = self.client.list()
        return [dict(m) if not isinstance(m, dict) else m for m in (r.get("models") if isinstance(r, dict) else r.models)]

    @staticmethod
    def _pick(want, names):
        if want in names:
            return want
        base = want.split(":")[0]
        for n in names:
            if n == want + ":latest" or (":" not in want and n.split(":")[0] == base):
                return n
        return None

    def _cached(self, kind, key, fn):
        p = self.cache / f"{kind}_{hashlib.sha1(key.encode()).hexdigest()[:20]}.json"
        if p.exists():
            self.stats["cached"] += 1
            return json.loads(p.read_text(encoding="utf-8"))
        out = fn()
        if out and (not isinstance(out, dict) or any(v for v in out.values())):   # never cache empty answers
            p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out

    def see(self, png_b64, max_side=672):
        """Two-sentence description of a picture, or None."""
        if not self.vision:
            return None
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")
        if max(img.size) > max_side:
            k = max_side / max(img.size)
            img = img.resize((int(img.width * k), int(img.height * k)))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        jpg = base64.b64encode(buf.getvalue()).decode()

        def call():
            t = time.perf_counter()
            try:
                r = self.client.chat(model=self.vision, messages=[
                    {"role": "user", "content": VISION_PROMPT, "images": [jpg]}],
                    options={"temperature": 0.1, "num_predict": 90, "num_ctx": 4096},
                    keep_alive="30m")
                self._count(r, t)
                cap = re.sub(r"\s+", " ", r["message"]["content"]).strip()
                return {"caption": cap} if cap else None
            except Exception as ex:
                self.log(f"[vision] failed ({type(ex).__name__}: {str(ex)[:80]})")
                return None
        out = self._cached("see_" + self.vision.replace(":", "_"), jpg[:20000] + jpg[-2000:], call)
        return out["caption"] if out else None

    def _count(self, r, t):
        self.stats["calls"] += 1
        self.stats["seconds"] += time.perf_counter() - t
        try:
            self.stats["tokens"] += int(r.get("eval_count") or 0)
        except Exception:
            pass

    def sketch(self, prompt):
        def call():
            t = time.perf_counter()
            try:
                r = self.client.chat(model=self.model, messages=[
                    {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    format=SCHEMA, think=False, keep_alive="30m",
                    options={"temperature": 0.2, "num_predict": self.num_predict, "num_ctx": 4096})
                self._count(r, t)
                txt = re.sub(r"<think>.*?</think>", "", r["message"]["content"], flags=re.S)
                return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
            except Exception as ex:
                self.log(f"[ai] sketch failed after {time.perf_counter() - t:.0f}s "
                         f"({type(ex).__name__}: {str(ex)[:80]}); rules sketch used")
                self.stats["fallback"] += 1
                return None
        return self._cached("sketch_" + self.model.replace(":", "_"), SYSTEM + prompt, call)


def ocr_caption(png_b64):
    """No vision model: read the words in the picture with RapidOCR (pip install rapidocr_onnxruntime)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return None
    global _OCR
    try:
        _OCR
    except NameError:
        _OCR = RapidOCR()
    res, _ = _OCR(base64.b64decode(png_b64))
    words = [r[1] for r in (res or []) if r[2] > 0.6]
    return ("The picture shows: " + ", ".join(words[:14]) + ".") if words else None


# ---------------------------------------------------------------- rules sketch (no AI / fallback)

ACTOR = [(r"on-premises|data center|datacenter", "On-premises"), (r"\bIoT\b|sensor|device", "Devices"),
         (r"mobile", "Mobile app"), (r"gam(e|ing)", "Players"), (r"users?\b|customers?|clients?", "Users"),
         (r"producers?", "Producers")]


def _actor(text):
    for rx, lab in ACTOR:
        if re.search(rx, text, re.I):
            return lab
    return None


def _sentences(text, n):
    s = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "").strip())
    return " ".join(s[:n]).strip()


def short_option(text):
    sv = services_in(text, 3)
    if sv:
        return " + ".join(l for l, _ in sv[:2])
    return " ".join(text.split()[:6]) + ("…" if len(text.split()) > 6 else "")


def rules_sketch(q):
    stem, opts = q["stem"], q["options"]
    right = [o for o in opts if o["correct"]]
    rtxt = " ".join(o["text"] for o in right)
    actor = _actor(stem) or "Users"
    sv_stem = [s for s in services_in(stem, 4)]
    nodes = [{"id": "a", "label": actor}]
    for i, (lab, _) in enumerate(sv_stem or [("Application", None)]):
        nodes.append({"id": f"s{i}", "label": lab})
    scen = {"nodes": nodes, "edges": [{"from": nodes[i]["id"], "to": nodes[i + 1]["id"]}
                                      for i in range(len(nodes) - 1)]}
    sv = services_in(rtxt, 5) or [(short_option(rtxt), None)]
    sn = [{"id": "a", "label": actor}] + [{"id": f"r{i}", "label": l, "highlight": i == 0}
                                          for i, (l, _) in enumerate(sv)]
    sol = {"nodes": sn, "edges": [{"from": sn[i]["id"], "to": sn[i + 1]["id"]} for i in range(len(sn) - 1)]}
    expl = q["correct_explanation"]
    for o in right:
        expl = expl.replace(o["text"], "")
    return {"title": " + ".join(l for l, _ in sv[:2])[:60] or f"Question {q['number']}",
            "scenario": scen, "solution": sol, "explain": _sentences(expl, 2),
            "wrong": [{"name": short_option(o["text"]),
                       "why": _sentences(o.get("why") or "", 1)[:160] or "Does not meet the requirement."}
                      for o in opts if not o["correct"]],
            "takeaway": "", "_by": "rules"}


# ---------------------------------------------------------------- scenes

LETTERS = "ABCDEFGH"


def _clean_sketch(sk, q):
    """Trust but verify: keep the AI sketch inside what the renderer can draw well."""
    base = rules_sketch(q)
    if not isinstance(sk, dict) or not any(sk.get(k) for k in ("scenario", "solution", "explain")):
        return base
    out = dict(base, _by="ai")
    for key in ("scenario", "solution"):
        d = sk.get(key)
        if isinstance(d, dict) and 2 <= len(d.get("nodes") or []) <= 9:
            d["nodes"] = d["nodes"][:7]
            out[key] = d
    if not any(n.get("highlight") for n in out["solution"]["nodes"] if isinstance(n, dict)):
        ans = {l for l, _ in services_in(" ".join(o["text"] for o in q["options"] if o["correct"]))}
        for n in out["solution"]["nodes"]:
            if isinstance(n, dict) and (services_in(n.get("label", ""), 1) or [("", "")])[0][0] in ans:
                n["highlight"] = True
    for key, n in (("explain", 420), ("takeaway", 110), ("title", 60)):
        if isinstance(sk.get(key), str) and sk[key].strip():
            out[key] = sk[key].strip()[:n]
    if isinstance(sk.get("wrong"), list) and sk["wrong"]:
        w = [x for x in sk["wrong"] if isinstance(x, dict) and x.get("name")]
        n_wrong = sum(not o["correct"] for o in q["options"])
        if len(w) == n_wrong:
            out["wrong"] = [{"name": x["name"][:50], "why": str(x.get("why", ""))[:140]} for x in w]
    return out


def question_prompt(q, captions):
    opts = "\n".join(f"{LETTERS[i]}. {o['text']}{'   <- CORRECT' if o['correct'] else ''}"
                     for i, o in enumerate(q["options"]))
    ex = re.sub(r"\s+", " ", q["correct_explanation"])[:900]
    pics = "".join(f"\nPicture {i + 1}: {c}" for i, c in enumerate(captions) if c)
    return f"Question: {q['stem']}\nOptions:\n{opts}\nExplanation: {ex}{pics}"


def question_scenes(q, sk, captions, subject):
    n = q["number"]
    opts = q["options"]
    right = [o for o in opts if o["correct"]]
    multi = len(right) > 1
    ask = re.split(r"(?<=[.!?])\s+", re.sub(r"\s*\(Select \w+\)\s*$", "", q["stem"]))[-1]
    units = [q["stem"]] + [o["text"] for o in opts]
    sc = []
    # 1 the scenario, drawn
    sc.append({"title": f"Q{n}: {sk['title']}", "idea_kind": "scenario", "visual_type": "arch",
               "visuals": [dict(sk["scenario"], type="arch")],
               "note": (ask if len(ask) < 120 else ask[:117] + "…"),
               "narration": f"Question {n}. {q['stem']}", "units": [q["stem"]]})
    # 2 the choices
    wnames = iter(w["name"] for w in sk["wrong"])
    names = [short_option(o["text"]) if o["correct"] else next(wnames, short_option(o["text"])) for o in opts]
    letters = [{"letter": LETTERS[i], "name": names[i],
                "detail": o["text"] if len(o["text"]) < 170 else o["text"][:167] + "…",
                "icon": (services_in(o["text"], 1) or [("", None)])[0][0] or short_option(o["text"])}
               for i, o in enumerate(opts)]
    pick = "Pick two." if multi and len(right) == 2 else ("Pick three." if len(right) == 3 else "")
    sc.append({"title": f"Q{n}: The choices", "idea_kind": "choices", "visual_type": "choices",
               "visuals": [{"type": "choices", "items": letters}],
               "note": pick or "Pause and choose before the answer.",
               "narration": " ".join(f"Option {LETTERS[i]}: {o['text']}." for i, o in enumerate(opts)),
               "units": [o["text"] for o in opts]})
    # 3 the answer, drawn
    ans = " and ".join(f"option {LETTERS[opts.index(o)]}, {o['text']}" for o in right)
    sc.append({"title": f"Q{n}: Answer – " + ", ".join(LETTERS[opts.index(o)] for o in right),
               "idea_kind": "solution", "visual_type": "arch",
               "visuals": [dict(sk["solution"], type="arch")],
               "note": sk.get("takeaway") or "",
               "narration": f"The answer is {ans}. {sk.get('explain', '')}", "units": []})
    # 4 why the others are wrong
    wrong = iter(sk["wrong"])
    cards = []
    for i, o in enumerate(opts):
        w = {"name": short_option(o["text"]), "why": "Correct: " + (sk.get("takeaway") or "meets every requirement.")} \
            if o["correct"] else next(wrong, {"name": short_option(o["text"]), "why": ""})
        cards.append({"letter": LETTERS[i], "name": w["name"], "why": w["why"], "correct": o["correct"],
                      "icon": (services_in(o["text"], 1) or [("", None)])[0][0] or w["name"]})
    sc.append({"title": f"Q{n}: Why not the others", "idea_kind": "comparison", "visual_type": "choices",
               "visuals": [{"type": "choices", "items": cards, "reveal": True}], "note": "",
               "narration": " ".join(f"Option {c['letter']}, {c['name']}: {c['why']}" for c in cards if not c["correct"]),
               "units": []})
    # 5 the pictures from the explanation
    for k, im in enumerate(q["images"][:2]):
        cap = captions[k] if k < len(captions) else None
        sc.append({"title": f"Q{n}: From the explanation" + (f" ({k + 1})" if len(q["images"]) > 1 else ""),
                   "idea_kind": "figure", "visual_type": "figure",
                   "visuals": [{"type": "figure", "png": im["png"], "w": im["w"], "h": im["h"],
                                "caption": f"page {im['page']}"}],
                   "note": "", "narration": cap or "Here is the picture from the explanation.", "units": []})
    for s in sc:
        s["source"] = {"pages": q["pages"], "question": n}
        s["concept"] = sk["title"]
    return sc, units


def plan_quiz(questions, pdf_name, first_lesson=1, ai=None, per_lesson=10, vision_mode="auto",
              log=print, parallel=1, given=None):
    """Questions -> lessons (one per `per_lesson` questions) of IR scenes.
    given: {"sketches": {num: sketch}, "captions": {num: [..]}} written by hand (testing / editing)."""
    given = given or {}
    g_sk = {int(k): v for k, v in (given.get("sketches") or {}).items()}
    g_cap = {int(k): v for k, v in (given.get("captions") or {}).items()}
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.perf_counter()
    dropped = [q["number"] for q in questions if len(q["options"]) < 2]
    questions = [q for q in questions if len(q["options"]) >= 2]
    if dropped:
        log(f"[plan] skipped question(s) {dropped}: options not on these pages (widen --pages)")
    # pictures first (the text model reads their captions)
    caps = {}
    imgs = [(q["number"], k, im) for q in questions for k, im in enumerate(q["images"][:2])]
    if imgs and ai and ai.vision:
        log(f"[vision] {ai.vision}: {len(imgs)} pictures")
    for j, (num, k, im) in enumerate(imgs, 1):
        t = time.perf_counter()
        if num in g_cap and k < len(g_cap[num]):
            caps.setdefault(num, []).append(g_cap[num][k])
            continue
        c = ai.see(im["png"]) if ai and ai.vision else (ocr_caption(im["png"]) if vision_mode != "off" else None)
        caps.setdefault(num, []).append(c)
        if ai and ai.vision and (j % 5 == 0 or j == len(imgs)):
            log(f"[vision] {j}/{len(imgs)} pictures, last {time.perf_counter() - t:.1f}s")
    t_vis = time.perf_counter() - t0

    t1 = time.perf_counter()
    sketches = {}

    def one(q):
        a = time.perf_counter()
        if q["number"] in g_sk:
            return q["number"], dict(_clean_sketch(g_sk[q["number"]], q), _by="file"), 0.0
        sk = ai.sketch(question_prompt(q, caps.get(q["number"], []))) if ai else None
        return q["number"], _clean_sketch(sk, q), time.perf_counter() - a

    with ThreadPoolExecutor(max(1, parallel)) as ex:
        for j, (num, sk, secs) in enumerate(ex.map(one, questions), 1):
            sketches[num] = sk
            if ai or g_sk:
                log(f"[ai] Q{num}: {sk['_by']} sketch, {secs:.1f}s  ({j}/{len(questions)})")
    t_ai = time.perf_counter() - t1

    lessons = []
    for g in range(0, len(questions), per_lesson):
        grp = questions[g:g + per_lesson]
        li = first_lesson + len(lessons)
        scenes, units = [], []
        for q in grp:
            s, u = question_scenes(q, sketches[q["number"]], caps.get(q["number"], []), "AWS")
            scenes += s
            units += u
        for i, s in enumerate(scenes, 1):
            s["id"] = f"L{li:02d}-S{i:03d}"
            s["source"]["pdf"] = pdf_name
        title = f"Questions {grp[0]['number']}-{grp[-1]['number']}"
        lessons.append({"title": title, "subject": "AWS practice test", "source_pdf": pdf_name,
                        "scenes": scenes, "all_units": units})
    timing = {"vision_s": round(t_vis, 1), "ai_s": round(t_ai, 1),
              "ai_sketches": sum(s["_by"] == "ai" for s in sketches.values()),
              "rules_sketches": sum(s["_by"] == "rules" for s in sketches.values()),
              "file_sketches": sum(s["_by"] == "file" for s in sketches.values())}
    return lessons, timing
