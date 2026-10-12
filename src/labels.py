"""Label metadata for every dataset in this repo (AAPD, RMSC-V2, PHATE).

Each dataset contributes:
  - <DATASET>_LABELS:       ordered list of label names (index == label id in the
                            multi-hot vector, so the order is load-bearing)
  - <DATASET>_DESCRIPTIONS: {label_name: natural-language description}

The descriptions are *not* cosmetic: they are the text fed to the encoder in
`scripts/02_encode_labels.py` to initialise the LSA label embeddings and the
label nodes of the Phase-3 graph. Changing a description changes the model
input, so treat these tables as part of the experimental configuration.

Use `get_labels(dataset_name)` to obtain (names, descriptions) in label order.
"""

from __future__ import annotations

# ============================================================================
# AAPD — 54 arXiv categories (English)
#
# The ORDER IS THE DATASET'S OWN, not alphabetical: index i here must be bit i
# of the multi-hot string in the TSV files. Do not sort this list.
#
# Descriptions are deliberately 1-2 clauses each and spell out what separates a
# label from its nearest siblings. A bare "Machine Learning" gives the encoder
# almost nothing to tell cs.LG from stat.ML from cs.AI; the longer text gives
# the label encoder actual discriminating signal to embed.
# ============================================================================

AAPD_LABELS = [
    "cs.IT",              # 0
    "math.IT",            # 1
    "cs.LG",              # 2
    "cs.AI",              # 3
    "stat.ML",            # 4
    "cs.DS",              # 5
    "cs.DM",              # 6
    "cs.SI",              # 7
    "cs.LO",              # 8
    "math.CO",            # 9
    "physics.soc-ph",     # 10
    "cs.CC",              # 11
    "cs.NI",              # 12
    "math.OC",            # 13
    "cs.CL",              # 14
    "cs.CV",              # 15
    "cs.CR",              # 16
    "cs.DC",              # 17
    "cs.SY",              # 18
    "cs.NE",              # 19
    "cs.IR",              # 20
    "cs.GT",              # 21
    "quant-ph",           # 22
    "cs.CY",              # 23
    "cs.PL",              # 24
    "cs.DB",              # 25
    "cs.SE",              # 26
    "math.PR",            # 27
    "cs.CG",              # 28
    "cs.NA",              # 29
    "cs.HC",              # 30
    "cs.MA",              # 31
    "math.NA",            # 32
    "cs.RO",              # 33
    "cs.CE",              # 34
    "cs.FL",              # 35
    "math.ST",            # 36
    "stat.TH",            # 37
    "cs.DL",              # 38
    "cmp-lg",             # 39
    "cs.MM",              # 40
    "cs.PF",              # 41
    "math.LO",            # 42
    "cond-mat.stat-mech", # 43
    "stat.AP",            # 44
    "stat.ME",            # 45
    "cs.MS",              # 46
    "cond-mat.dis-nn",    # 47
    "q-bio.NC",           # 48
    "cs.SC",              # 49
    "physics.data-an",    # 50
    "math.NT",            # 51
    "q-bio.QM",           # 52
    "nlin.AO",            # 53
]

AAPD_DESCRIPTIONS = {
    "cs.IT":           "Information Theory: coding theory, channel capacity, source and channel coding, communication limits.",
    "math.IT":         "Information Theory, mathematics side: the same core subject as cs.IT but classified under mathematics, emphasizing rigorous proofs of entropy and coding bounds.",
    "cs.LG":           "Machine Learning from a computer-science and systems angle: algorithms for supervised, unsupervised and reinforcement learning, deep learning architectures, and their computational properties.",
    "cs.AI":           "Artificial Intelligence broadly: reasoning, knowledge representation, planning, search and multi-agent intelligence, wider in scope than machine learning alone.",
    "stat.ML":         "Machine Learning from a statistics angle: probabilistic models, generalization bounds, and the statistical theory underlying learning algorithms, as opposed to the systems/algorithms angle of cs.LG.",
    "cs.DS":           "Data Structures and Algorithms: design, analysis and complexity of algorithms and the data structures that support them.",
    "cs.DM":           "Discrete Mathematics as applied in computer science: combinatorial structures, graph theory and discrete methods used in algorithm design.",
    "cs.SI":           "Social and Information Networks: structure and dynamics of social networks, information diffusion, community detection and network science.",
    "cs.LO":           "Logic in Computer Science: formal logic, proof theory, type theory and model checking as foundations of computation.",
    "math.CO":         "Combinatorics: enumerative and structural combinatorics, graph theory and combinatorial optimization from a pure-mathematics perspective.",
    "physics.soc-ph":  "Physics and Society: physics-inspired methods (statistical mechanics, network theory) applied to social systems and collective human behavior.",
    "cs.CC":           "Computational Complexity: classification of problems by the resources (time, space) needed to solve them, complexity classes and hardness/reduction results.",
    "cs.NI":           "Networking and Internet Architecture: network protocols, internet architecture, routing and network performance.",
    "math.OC":         "Optimization and Control, mathematics side: convex and non-convex optimization theory and mathematical control systems.",
    "cs.CL":           "Computation and Language / Natural Language Processing: understanding, processing and generating human language, including language models, parsing and machine translation.",
    "cs.CV":           "Computer Vision and Pattern Recognition: image and video understanding, object detection/recognition, and visual pattern analysis.",
    "cs.CR":           "Cryptography and Security: cryptographic protocols, network and system security, privacy and adversarial threats.",
    "cs.DC":           "Distributed, Parallel, and Cluster Computing: algorithms and systems for computation spread across multiple machines or processors.",
    "cs.SY":           "Systems and Control (legacy CS category, largely superseded by eess.SY): engineering systems modeling and control-theoretic methods.",
    "cs.NE":           "Neural and Evolutionary Computing: neural network architectures and evolutionary/genetic algorithms used for optimization and learning.",
    "cs.IR":           "Information Retrieval: search engines, document indexing, ranking and retrieval from large collections.",
    "cs.GT":           "Computer Science and Game Theory: algorithmic game theory, mechanism design and strategic multi-agent interaction.",
    "quant-ph":        "Quantum Physics: quantum computation, quantum information theory and quantum-mechanical systems.",
    "cs.CY":           "Computers and Society: social, ethical, legal and policy implications of computing technology.",
    "cs.PL":           "Programming Languages: language design, formal semantics, type systems and compiler implementation.",
    "cs.DB":           "Databases: database systems, query processing, transaction management and data storage.",
    "cs.SE":           "Software Engineering: software design methodology, development processes, testing and maintenance.",
    "math.PR":         "Probability: probability theory and stochastic processes from a pure-mathematics perspective.",
    "cs.CG":           "Computational Geometry: algorithms for geometric problems, spatial data structures and geometric computation.",
    "cs.NA":           "Numerical Analysis (legacy CS category, mostly superseded by math.NA): numerical algorithms for continuous mathematical problems.",
    "cs.HC":           "Human-Computer Interaction: design and evaluation of interactive systems, interfaces and user experience.",
    "cs.MA":           "Multiagent Systems: coordination, communication, negotiation and emergent behavior among multiple autonomous agents.",
    "math.NA":         "Numerical Analysis, mathematics side: convergence and error analysis of numerical algorithms.",
    "cs.RO":           "Robotics: robot perception, motion planning, control and autonomous physical systems.",
    "cs.CE":           "Computational Engineering, Finance, and Science: computational methods applied to engineering, financial and scientific problems.",
    "cs.FL":           "Formal Languages and Automata Theory: automata, formal grammars and the theory of formal languages.",
    "math.ST":         "Statistics Theory, mathematics side: rigorous mathematical statistics, estimation theory and inference (overlaps heavily with stat.TH).",
    "stat.TH":         "Statistics Theory: theoretical foundations of statistical inference methods, cross-listed with math.ST.",
    "cs.DL":           "Digital Libraries: organization, storage, retrieval, curation and preservation of digital document collections.",
    "cmp-lg":          "Computational Linguistics (legacy pre-1998 arXiv category, predecessor of cs.CL): early computational approaches to linguistic analysis.",
    "cs.MM":           "Multimedia: encoding, processing, indexing and retrieval of audio, video, image and multimedia content.",
    "cs.PF":           "Performance: measurement, benchmarking and modeling of the performance of computing systems.",
    "math.LO":         "Logic, mathematics side: mathematical logic, set theory, model theory and proof theory.",
    "cond-mat.stat-mech": "Statistical Mechanics: statistical physics of many-particle systems, phase transitions and thermodynamic behavior.",
    "stat.AP":         "Applications (Statistics): applied statistical methods used in domains such as biology, finance, medicine and social science.",
    "stat.ME":         "Methodology (Statistics): development and validation of new statistical methods and techniques, as opposed to applying existing ones (stat.AP).",
    "cs.MS":           "Mathematical Software: design and implementation of software libraries and tools for mathematical computation.",
    "cond-mat.dis-nn": "Disordered Systems and Neural Networks: statistical-physics approaches to disordered materials and to neural-network models.",
    "q-bio.NC":        "Neurons and Cognition: computational and quantitative approaches to neuroscience, brain function and cognition.",
    "cs.SC":           "Symbolic Computation: algorithms for exact symbolic and algebraic manipulation of mathematical expressions (computer algebra).",
    "physics.data-an": "Data Analysis, Statistics and Probability, physics side: statistical and data-analysis techniques used in experimental physics.",
    "math.NT":         "Number Theory: properties of integers, primes, Diophantine equations and related number-theoretic structures.",
    "q-bio.QM":        "Quantitative Methods: mathematical, statistical and computational techniques applied to biological questions, distinct from q-bio.NC's neuroscience focus.",
    "nlin.AO":         "Adaptation and Self-Organizing Systems: nonlinear dynamics of adaptive, self-organizing and complex systems.",
}


# ============================================================================
# RMSC-V2 — 22 music-style categories (Chinese music reviews)
#
# Descriptions are bilingual (English + Chinese) so the Chinese WordPiece
# encoder gets signal from both scripts.
# ============================================================================

RMSC_LABELS = [
    "classical", "indie", "alternative", "hiphop", "folk",
    "r&b", "metal", "postpunk", "ost", "country",
    "britpop", "electronic", "newage", "postrock", "piano",
    "soul", "jazz", "pop", "darkwave", "punk",
    "jpop", "rock",
]

RMSC_DESCRIPTIONS = {
    "classical":   "Classical Music 古典音乐",
    "indie":       "Independent Music 独立音乐",
    "alternative": "Alternative Music 另类音乐",
    "hiphop":      "Hip Hop and Rap Music 嘻哈说唱",
    "folk":        "Folk Music 民谣音乐",
    "r&b":         "Rhythm and Blues Music 节奏布鲁斯",
    "metal":       "Heavy Metal Music 金属音乐",
    "postpunk":    "Post-Punk Music 后朋克",
    "ost":         "Original Soundtrack Music 原声音乐",
    "country":     "Country Music 乡村音乐",
    "britpop":     "British Pop Music 英伦摇滚",
    "electronic":  "Electronic Music 电子音乐",
    "newage":      "New Age Music 新世纪音乐",
    "postrock":    "Post-Rock Music 后摇",
    "piano":       "Piano Music 钢琴音乐",
    "soul":        "Soul Music 灵魂乐",
    "jazz":        "Jazz Music 爵士乐",
    "pop":         "Pop Music 流行音乐",
    "darkwave":    "Darkwave Music 暗潮音乐",
    "punk":        "Punk Rock Music 朋克摇滚",
    "jpop":        "Japanese Pop Music 日本流行音乐",
    "rock":        "Rock Music 摇滚乐",
}


# ============================================================================
# PHATE — 4 Persian hate-speech categories (Persian text)
#
# Descriptions are bilingual (English + Persian) and deliberately spell out how
# each label differs from its closest sibling, since 'Hate' vs 'HateSpeech' and
# 'Vulgar' vs 'Hate' are otherwise near-indistinguishable to an encoder.
# ============================================================================

PHATE_LABELS = [
    "Violenc",      # 0
    "Hate",         # 1
    "Vulgar",       # 2
    "HateSpeech",   # 3
]

PHATE_DESCRIPTIONS = {
    "Violenc":    "Violence خشونت: content that threatens, incites, or glorifies physical violence against a person or group.",
    "Hate":       "Hate نفرت: general hateful or hostile sentiment/insults directed at a person or group, without necessarily meeting the formal criteria of targeted hate speech below.",
    "Vulgar":     "Vulgar Language سخنان زشت: crude, obscene or profane language that is offensive in tone but not necessarily hateful or targeted at a protected group.",
    "HateSpeech": "Hate Speech محتوای نفرت‌پراکنی: umbrella label: positive whenever Violence, Hate or Vulgar is present",
}


# ============================================================================
# Dataset registry
# ============================================================================

DATASETS = {
    "aapd": {
        "labels": AAPD_LABELS,
        "descriptions": AAPD_DESCRIPTIONS,
        "num_labels": len(AAPD_LABELS),
        "language": "english",
    },
    "rmsc": {
        "labels": RMSC_LABELS,
        "descriptions": RMSC_DESCRIPTIONS,
        "num_labels": len(RMSC_LABELS),
        "language": "chinese",
    },
    "phate": {
        "labels": PHATE_LABELS,
        "descriptions": PHATE_DESCRIPTIONS,
        "num_labels": len(PHATE_LABELS),
        "language": "persian",
    },
}


def get_labels(dataset_name: str) -> tuple[list[str], list[str]]:
    """Return (label_names, label_descriptions) for a dataset, in label order."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Supported: {list(DATASETS.keys())}"
        )
    info = DATASETS[dataset_name]
    names = info["labels"]
    descs = [info["descriptions"].get(name, name) for name in names]
    return names, descs


def get_num_labels(dataset_name: str) -> int:
    """Return the number of labels for a dataset."""
    return DATASETS[dataset_name.lower()]["num_labels"]


def get_language(dataset_name: str) -> str:
    """Return the dataset language ('english' | 'chinese' | 'persian')."""
    return DATASETS[dataset_name.lower()]["language"]


# Fail fast on an inconsistent table rather than silently embedding a bare
# label name as its own "description".
for _name, _info in DATASETS.items():
    assert len(_info["labels"]) == _info["num_labels"]
    _missing = [l for l in _info["labels"] if l not in _info["descriptions"]]
    assert not _missing, f"{_name}: missing description(s) for {_missing}"
