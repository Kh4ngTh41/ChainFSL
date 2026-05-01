# ChainFSL HASO - Figures and Diagrams

> **Note**: These diagrams are designed for visualization in Markdown editors that support Mermaid (GitHub, GitLab, VS Code with extension, etc.)
> If your viewer doesn't support Mermaid, the ASCII art versions in `EXPERIMENTAL_RESULTS_ANALYSIS.md` will render correctly.

---

## Figure 1: System Architecture Comparison

### FedAvg (Baseline 1)

```mermaid
graph LR
    subgraph "FedAvg Architecture"
        A0["Node 0<br/>Raw Data"]
        A1["Node 1<br/>Raw Data"]
        A2["Node N<br/>Raw Data"]

        A0 -->|"All raw data"| AGG["Aggregator"]
        A1 -->|"All raw data"| AGG
        A2 -->|"All raw data"| AGG

        AGG -->|"Global Model"| GM["Global Model<br/>All nodes download"]
    end

    style AGG fill:#f9f,stroke:#333,stroke-width:2px
    style GM fill:#bbf,stroke:#333,stroke-width:2px
```

### SplitFed (Baseline 2)

```mermaid
graph LR
    subgraph "SplitFed Architecture"
        S0["Tier-4 (200MB)"]
        S1["Tier-2 (4GB)"]
        S2["Tier-1 (8GB)"]

        S0 -->|"L1 smashed"| SV["Server<br/>Layers 2-4+FC"]
        S1 -->|"L1 smashed"| SV
        S2 -->|"L1 smashed"| SV

        SV -->|"Gradient"| S0
        SV -->|"Gradient"| S1
        SV -->|"Gradient"| S2
    end

    style SV fill:#f9f,stroke:#333,stroke-width:2px
```

### HASO (Ours) - Adaptive

```mermaid
graph LR
    subgraph "HASO Architecture"
        H0["Tier-4 (200MB)<br/>L1 cut"]
        H1["Tier-3 (512MB)<br/>L2 cut"]
        H2["Tier-2 (4GB)<br/>L3 cut"]
        H3["Tier-1 (8GB)<br/>L4 cut"]

        H0 -->|"L1 smashed"| C0["Compute A<br/>Layer 2-4"]
        H1 -->|"L2 smashed"| C1["Compute B<br/>Layer 3-4"]
        H2 -->|"L3 smashed"| C2["Compute C<br/>Layer 4"]
        H3 -->|"L4 smashed"| C3["Compute D<br/>Layer 4 (alone)"]

        C0 -->|"Gradient"| H0
        C1 -->|"Gradient"| H1
        C2 -->|"Gradient"| H2
        C3 -->|"Gradient"| H3
    end

    style C0 fill:#9f9,stroke:#333,stroke-width:2px
    style C1 fill:#9f9,stroke:#333,stroke-width:2px
    style C2 fill:#9f9,stroke:#333,stroke-width:2px
    style C3 fill:#9f9,stroke:#333,stroke-width:2px
```

---

## Figure 2: HASO Routing Modes

### Mode A: Parallel Routing

```mermaid
graph LR
    subgraph "Parallel Routing (A)"
        P0["Node 0<br/>Tier-2<br/>L3 cut"]
        P1["Node 1<br/>Tier-1<br/>L4 cut"]
        P2["Node 2<br/>Tier-3<br/>L2 cut"]

        P0 -->|"L3 smashed"| CA["Compute A"]
        P1 -->|"L4 smashed"| CB["Compute B"]
        P2 -->|"L2 smashed"| CC["Compute C"]
    end

    style CA fill:#9f9,stroke:#333
    style CB fill:#9f9,stroke:#333
    style CC fill:#9f9,stroke:#333
```

**Characteristics:**
- Each node → different compute node
- No resource sharing
- Direct path routing

### Mode B: Fusion Routing

```mermaid
graph LR
    subgraph "Fusion Routing (B)"
        F0["Node 0<br/>L2 cut"]
        F1["Node 1<br/>L2 cut"]
        F2["Node 2<br/>L2 cut"]

        F0 -->|"L2 smashed"| FX["Compute X<br/>Layer 3+4"]
        F1 -->|"L2 smashed"| FX
        F2 -->|"L2 smashed"| FX

        FX -->|"Merged Grad"| FG["Gradient<br/>to all nodes"]
    end

    style FX fill:#ff9,stroke:#333,stroke-width:3px
```

**Characteristics:**
- Multiple nodes → same compute node
- Shared computation
- Fusion bonus in reward

### Mode C: Hybrid (A+B)

```mermaid
graph LR
    subgraph "Hybrid Routing"
        H0["Node 0<br/>L4 solo"]
        H1["Node 1<br/>L2 shared"]
        H2["Node 2<br/>L2 shared"]
        H3["Node 3<br/>L3 solo"]

        H0 -->|"L4 smashed"| HC0["Compute A"]
        H1 -->|"L2 smashed"| HC1["Compute B<br/>(shared)"]
        H2 -->|"L2 smashed"| HC1
        H3 -->|"L3 smashed"| HC2["Compute C"]

        HC0 -->|"Gradient"| H0
        HC1 -->|"Merged"| HG["Gradient"]
        HC2 -->|"Gradient"| H3
    end

    style HC0 fill:#9f9,stroke:#333
    style HC1 fill:#ff9,stroke:#333,stroke-width:3px
    style HC2 fill:#9f9,stroke:#333
```

---

## Figure 3: Layer Assignment by Tier

```mermaid
graph LR
    subgraph "ResNet-18 Layer Split"
        L1["Layer 1<br/>64×56×56"]
        L2["Layer 2<br/>128×28×28"]
        L3["Layer 3<br/>256×14×14"]
        L4["Layer 4<br/>512×7×7"]
        FC["FC Layer<br/>512→10"]

        L1 --> L2 --> L3 --> L4 --> FC
    end

    subgraph "Tier Assignments"
        T4["Tier-4 (200KB)<br/>Cut after L1<br/>L1 only client"]
        T3["Tier-3 (512MB)<br/>Cut after L2<br/>L1-2 client"]
        T2["Tier-2 (4GB)<br/>Cut after L3<br/>L1-3 client"]
        T1["Tier-1 (8GB)<br/>Cut after L4<br/>All client"]
    end

    L1 -.->|"Client<br/>(Tier-4)"| T4
    L2 -.->|"Client<br/>(Tier-3)"| T3
    L3 -.->|"Client<br/>(Tier-2)"| T2
    L4 -.->|"Client<br/>(Tier-1)"| T1
```

---

## Figure 4: Round-by-Round Adaptation

```mermaid
graph LR
    subgraph "Round 1"
        R1A["Node 0: L2"]
        R1B["Node 1: L1"]
        R1C["Node 2: L3"]
    end

    subgraph "Round 5"
        R5A["Node 0: L3"]
        R5B["Node 1: L2"]
        R5C["Node 2: L2"]
    end

    subgraph "Round 10 (PPO Learned)"
        R10A["Node 0: L4"]
        R10B["Node 1: L3"]
        R10C["Node 2: L1"]
    end

    R1A -->|"PPO learns<br/>from rewards"| R10A
    R1B -->|"PPO learns<br/>from rewards"| R10B
    R1C -->|"PPO learns<br/>from rewards"| R10C
```

---

## Figure 5: Convergence Curves

```mermaid
graph
    title["Accuracy vs Rounds - Convergence Comparison"]

    A["HASO (Adaptive)"] -->|"85%", 50| AX[85%]
    B["SplitFed (L2)"] -->|"78%", 50| BX[78%]
    C["FedAvg"] -->|"70%", 50| CX[70%]

    THRESHOLD["70% Threshold"]
    T70["Time to 70%"]

    AX -->|"Round 40"| T70
    BX -->|"Round 47"| T70
```

---

## Figure 6: Latency Comparison

```mermaid
graph
    title["Per-Round Latency - Straggler Analysis"]

    L1["HASO"]
    L2["Static Cut (L2)"]

    L1 -->|"3.30s avg"| L1X["□ minimal stragglers"]
    L2 -->|"4.85s avg"| L2X["■■ 65% stragglers"]

    note1["Tier-4 nodes: L1 cut → fast"]
    note2["Tier-4 nodes: forced L2 → OOM or slow"]
```

---

## Figure 7: Experiment Pipeline

```mermaid
flowchart TD
    A["Config<br/>N=50, k=15, R=50"] --> B["Scenario B<br/>Fixed Sparse"]

    B --> C["HASO Decisions<br/>Per Node"]

    C --> D["Parallel Routing<br/>A"] & E["Fusion Routing<br/>B"]

    D --> F["Compute Node Selection<br/>Based on Queue/Load"]
    E --> F

    F --> G["Training Simulation<br/>T_comp + T_comm"]

    G --> H{"Verification<br/>TVE"}

    H -->|"Valid"| I["Aggregation"]
    H -->|"Invalid"| J["Penalty"]

    I --> K["Shapley Rewards<br/>GTM"]
    J --> K

    K --> L["PPO Update<br/>Learn from Rewards"]

    L --> M["Convergence Analysis<br/>Accuracy + Latency"]

    M --> N{"More<br/>Rounds?"}
    N -->|"Yes"| C
    N -->|"No"| O["Results<br/>CSV + JSON"]

    style A fill:#f9f
    style O fill:#9f9
```

---

## Figure 8: Data Flow with Privacy

```mermaid
flowchart LR
    subgraph "Node i - Data Owner"
        RAW["Raw Data<br/>Image x ∈ R³ˣʰʷ"]
        FORWARD["Forward Pass<br/>to cut_layer"]

        RAW --> FORWARD
    end

    FORWARD -->|"Activation<br/>a = f(x; θ₁)"| SMASH["Smashed Data<br/>a ∈ Rᴰ"]

    SMASH -->|"Network<br/>Transmission"| COMPUTE["Compute Node<br/>Layer cut+1 to 4 + FC"]

    COMPUTE -->|"Gradient<br/>∂L/∂a"| SMASH

    subgraph "Privacy Analysis"
        P1["Raw data NEVER leaves node"]
        P2["Only smashed activations transmitted"]
        P3["TVE verifies without seeing raw data"]
    end

    style RAW fill:#f99
    style SMASH fill:#ff9
    style COMPUTE fill:#9f9
```

---

## Figure 9: Ablation Study Contribution

```mermaid
pie title "Module Contribution to Final Accuracy"
    "HASO (routing)" : 54
    "TVE (verification)" : 22
    "GTM (incentives)" : 19
    "Model/Data" : 5
```

---

## Figure 10: Non-IID Data Distribution

```mermaid
graph TD
    subgraph "Dirichlet α = 0.1 (Extreme Non-IID)"
        N1["Client 1<br/>80% cats, 20% dogs"]
        N2["Client 2<br/>90% dogs, 10% cars"]
        N3["Client 3<br/>70% birds, 30% cats"]
    end

    subgraph "Dirichlet α = 1.0 (IID)"
        I1["Client 1<br/>10% each class"]
        I2["Client 2<br/>10% each class"]
        I3["Client 3<br/>10% each class"]
    end

    style N1 fill:#f99,stroke:#333
    style N2 fill:#9f9,stroke:#333
    style N3 fill:#99f,stroke:#333
```

---

## Figure 11: Tier Distribution

```mermaid
pie title "IoT Network Tier Distribution (50 nodes)"
    "Tier-1 (8GB GPU)" : 5
    "Tier-2 (4GB RAM)" : 15
    "Tier-3 (512MB)" : 20
    "Tier-4 (200KB)" : 10
```

---

## Figure 12: Scalability Results

```mermaid
graph
    title["Accuracy vs Number of Nodes"]

    A["N=10: 82.3%"]
    B["N=20: 83.5%"]
    C["N=50: 85.0%"]
    D["N=100: 86.2%"]

    A -->|"+1.2%"| B
    B -->|"+1.5%"| C
    C -->|"+1.2%"| D
```

---

## ASCII Art Versions (for simple viewers)

### Architecture Comparison

```
FEDAVG:
┌─────────────────────────────────────────────┐
│  Node0 ──┐                                    │
│  Node1 ──┼──► [AGGREGATOR] ──► Global Model  │
│  NodeN ──┘     (bottleneck)                  │
│  All raw data transmitted                    │
└─────────────────────────────────────────────┘

SPLITFED (Uniform L2):
┌─────────────────────────────────────────────┐
│  T4 ─L1──► [SERVER] ◄──L1── T3              │
│  T2 ─L1──► [SERVER] ◄──L1── T1              │
│  ALL use L1 (wasteful for T1)               │
└─────────────────────────────────────────────┘

HASO (Adaptive):
┌─────────────────────────────────────────────┐
│  T1 ─L4──► [COMP-A] ◄──L3── T2             │
│  T4 ─L1──► [COMP-B] ◄──L2── T3             │
│  T1 can compute L4, T4 can only do L1       │
│  Optimal routing per hardware!               │
└─────────────────────────────────────────────┘
```

### Routing Modes

```
PARALLEL (A):          FUSION (B):
┌─────────────┐        ┌─────────────┐
│ N0 ──► CA   │        │ N0 ─┐      │
│ N1 ──► CB   │        │ N1 ─┼─► CX  │
│ N2 ──► CC   │        │ N2 ─┘      │
│ (3 compute) │        │ (1 compute)│
└─────────────┘        └─────────────┘

HYBRID (A+B):
┌─────────────┐
│ N0 ───► CA  │  ← solo
│ N1 ─┐       │
│ N2 ─┼─► CB  │  ← shared
│ N3 ─┘       │
│ N4 ───► CC  │  ← solo
└─────────────┘
```

---

*Generated: 2026-05-01*
*For full analysis, see: `EXPERIMENTAL_RESULTS_ANALYSIS.md`*