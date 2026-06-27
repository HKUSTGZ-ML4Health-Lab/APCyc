import { createFileRoute } from "@tanstack/react-router";
import { useState, useRef, useEffect, type ReactNode } from "react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "APCyc | Property-Informed Design of Cyclic Peptides via Automated Cyclization" },
      {
        name: "description",
        content:
          "APCyc is a target-aware latent diffusion framework for automated cyclization and property-informed cyclic peptide design. KDD 2026.",
      },
      { property: "og:title", content: "APCyc — Property-Informed Cyclic Peptide Design" },
      {
        property: "og:description",
        content:
          "Target-aware latent diffusion with automated cyclization and multi-property guidance. KDD 2026.",
      },
      { property: "og:url", content: "/" },
    ],
    links: [{ rel: "canonical", href: "/" }],
  }),
  component: Index,
});

const assetUrl = (path: string) => `${import.meta.env.BASE_URL}${path}`;

const PAPER_URL = assetUrl("assets/apcyc/paper.pdf");
const CODE_URL = "https://github.com/HKUSTGZ-ML4Health-Lab/APCyc";
const TEASER_URL = assetUrl("assets/apcyc/teaser.png");
const OVERVIEW_URL = assetUrl("assets/apcyc/overview.png");
const INJECTION_URL = assetUrl("assets/apcyc/injection.png");
const RADAR_URL = assetUrl("assets/apcyc/radar.png");
const DENOISING_2XYI_URL = assetUrl("assets/apcyc/denoising_2xyi.mp4");
const DENOISING_5Y59_URL = assetUrl("assets/apcyc/denoising_5y59.mp4");
const DENOISING_1T4F_URL = assetUrl("assets/apcyc/denoising_1t4f.mp4");

type SectionProps = {
  id?: string;
  eyebrow?: string;
  title?: string;
  sub?: string;
  children: ReactNode;
};

type ResultRow = {
  m: string;
  v: number[];
  h?: boolean[];
};

type CaseRow = [
  method: string,
  vina: number,
  immunogenicity: number,
  permeability: number,
  protease: number,
];
type AblationRow = [
  setting: string,
  protease: number,
  immunogenicity: number,
  highlighted: boolean,
];
type PermeabilityRow = [setting: string, permeability: number, highlighted: boolean];

const NAV = [
  ["Overview", "overview"],
  ["Method", "method"],
  ["Results", "results"],
  ["Case Study", "case"],
  ["Citation", "citation"],
];

function Nav() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-[color:var(--paper)]/75 border-b border-[color:var(--border)]">
      <div className="mx-auto max-w-7xl px-6 h-16 flex items-center justify-between">
        <a href="#top" className="flex items-center gap-2 font-display font-bold tracking-tight">
          <span className="inline-block w-7 h-7 rounded-lg bg-gradient-to-br from-bio-cyan via-bio-blue to-bio-violet ring-glow" />
          <span className="text-lg">APCyc</span>
        </a>
        <div className="hidden md:flex items-center gap-7 text-sm text-ink-soft">
          {NAV.map(([l, h]) => (
            <a key={h} href={`#${h}`} className="hover:text-ink transition-colors">
              {l}
            </a>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <a
            href={PAPER_URL}
            target="_blank"
            rel="noreferrer"
            className="hidden sm:inline-flex items-center px-3 py-1.5 rounded-lg text-sm border border-[color:var(--border)] text-ink hover:bg-[color:var(--muted)] transition"
          >
            Paper
          </a>
          <a
            href={CODE_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center px-3 py-1.5 rounded-lg text-sm bg-gradient-to-r from-bio-blue to-bio-violet text-white shadow-[0_8px_30px_-8px_rgba(43,91,255,0.5)] hover:opacity-90 transition"
          >
            Code
          </a>
        </div>
      </div>
    </nav>
  );
}

function Section({ id, eyebrow, title, sub, children }: SectionProps) {
  return (
    <section id={id} className="relative py-20 md:py-24 px-6">
      <div className="mx-auto max-w-7xl">
        {(eyebrow || title) && (
          <div className="mb-10 md:mb-14 max-w-3xl">
            {eyebrow && (
              <div className="text-[11px] uppercase tracking-[0.22em] text-bio-blue mb-3 font-mono">
                {eyebrow}
              </div>
            )}
            {title && (
              <h2 className="text-4xl md:text-5xl font-semibold leading-[1.1] text-ink">{title}</h2>
            )}
            {sub && <p className="mt-4 text-lg text-ink-soft leading-relaxed">{sub}</p>}
          </div>
        )}
        {children}
      </div>
    </section>
  );
}

function Figure({
  src,
  alt,
  label,
  caption,
  maxWidth = "max-w-5xl",
}: {
  src: string;
  alt: string;
  label: string;
  caption: string;
  maxWidth?: string;
}) {
  return (
    <figure className={`mx-auto ${maxWidth}`}>
      <div className="card-surface p-3 md:p-4">
        <img
          src={src}
          alt={alt}
          loading="lazy"
          className="block w-full h-auto rounded-xl bg-white"
        />
      </div>
      <figcaption className="mt-4 text-center text-lg text-ink-soft max-w-3xl mx-auto leading-relaxed">
        <span className="font-display font-semibold text-ink mr-1.5">{label}.</span>
        <span className="font-serif italic text-ink/85">{caption}</span>
      </figcaption>
    </figure>
  );
}

function Hero() {
  return (
    <header id="top" className="relative pt-32 pb-16 px-6 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-50 pointer-events-none [mask-image:radial-gradient(60%_55%_at_50%_30%,black,transparent)]" />
      <div className="absolute -top-40 right-[-10%] w-[640px] h-[640px] rounded-full bg-bio-violet/15 blur-[120px] pointer-events-none" />
      <div className="absolute -bottom-40 -left-20 w-[560px] h-[560px] rounded-full bg-bio-cyan/15 blur-[120px] pointer-events-none" />
      <div className="absolute top-1/3 left-1/3 w-[420px] h-[420px] rounded-full bg-bio-rose/10 blur-[120px] pointer-events-none" />
      <div className="relative mx-auto max-w-4xl text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full soft-chip text-[11px] font-mono text-bio-blue tracking-wider">
          <span className="w-1.5 h-1.5 rounded-full bg-bio-blue animate-pulse" />
          KDD 2026 · Jeju Island, Republic of Korea
        </div>
        <h1 className="mt-8 font-display text-7xl md:text-[8.5rem] font-medium leading-[1] tracking-[-0.04em]">
          <span className="text-gradient">APCyc</span>
        </h1>
        <h2 className="mt-8 font-serif text-[1.75rem] md:text-[2.6rem] text-ink leading-[1.25] tracking-[-0.01em] max-w-3xl mx-auto font-normal">
          Property-Informed Design of{" "}
          <em className="text-bio-blue not-italic font-medium" style={{ fontStyle: "italic" }}>
            Cyclic Peptides
          </em>{" "}
          via Automated Cyclization
        </h2>
        <p className="mt-7 text-[1.05rem] md:text-lg text-ink-soft max-w-2xl mx-auto leading-[1.7]">
          A target-aware latent diffusion framework that automatically selects cyclization linkages
          and sites while optimizing drug-relevant peptide properties.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-2 text-xs">
          {[
            ["Automated Cyclization", "bio-blue"],
            ["Property Guidance", "bio-rose"],
            ["Cyclic Peptide Design", "bio-green"],
            ["Latent Diffusion", "bio-violet"],
          ].map(([t, c]) => (
            <span
              key={t}
              className={`px-3 py-1.5 rounded-full soft-chip text-ink-soft tracking-wide`}
            >
              <span
                className={`inline-block w-1.5 h-1.5 rounded-full bg-${c} mr-1.5 align-middle`}
              />
              {t}
            </span>
          ))}
        </div>
        <div className="mt-10 text-sm text-ink-soft">
          <div className="text-ink font-medium tracking-tight">
            Yifan Zhao<sup className="text-bio-blue">*</sup>, Lang Qin
            <sup className="text-bio-blue">*</sup>, Jintai Chen
            <sup className="text-bio-rose">†</sup>
          </div>
          <div className="mt-1.5">
            The Hong Kong University of Science and Technology (Guangzhou)
          </div>
          <div className="mt-1.5 text-xs text-ink-soft/80">
            * Equal contribution &nbsp;·&nbsp; † Corresponding author
          </div>
        </div>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <a
            href={PAPER_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-bio-blue to-bio-violet text-white font-medium shadow-[0_18px_50px_-14px_rgba(43,91,255,0.55)] hover:opacity-90 transition"
          >
            Read Paper →
          </a>
          <a
            href={CODE_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-ink text-white hover:opacity-90 transition"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 .5a11.5 11.5 0 0 0-3.63 22.42c.57.1.78-.25.78-.55v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.34-1.28-1.7-1.28-1.7-1.04-.72.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.3-5.24-1.28-5.24-5.7 0-1.26.45-2.3 1.18-3.1-.12-.3-.52-1.48.1-3.08 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.78 0c2.21-1.49 3.18-1.18 3.18-1.18.62 1.6.22 2.78.1 3.08.74.8 1.18 1.84 1.18 3.1 0 4.43-2.7 5.4-5.27 5.69.42.36.78 1.07.78 2.16v3.2c0 .3.21.66.79.55A11.5 11.5 0 0 0 12 .5Z" />
            </svg>
            Code
          </a>
        </div>
      </div>
      <div className="relative mx-auto max-w-6xl mt-16">
        <Figure
          src={TEASER_URL}
          alt="APCyc concept figure showing receptor pocket context, automated cyclization mechanism with linkage matrix and property guidance, and generated cyclic peptide candidates."
          label="Figure 1"
          caption="APCyc takes a receptor binding-pocket context, automatically infers cyclization sites and linkage types, and generates property-optimized cyclic peptide candidates with the help of Bayesian posterior guidance."
          maxWidth="max-w-6xl"
        />
      </div>
    </header>
  );
}

function TLDR() {
  const cards = [
    [
      "What it designs",
      "Target-conditioned cyclic peptides with generated sequence, structure, and cyclization topology.",
      "bio-blue",
    ],
    [
      "What it automates",
      "Cyclization linkage type and residue-level linkage site selection from receptor pocket context.",
      "bio-green",
    ],
    [
      "What it optimizes",
      "Affinity, permeability, protease resistance, solubility, and immunogenicity.",
      "bio-rose",
    ],
  ] as const;
  return (
    <Section
      id="overview"
      eyebrow="TL;DR"
      title="A diffusion framework that designs cyclic peptides — and decides how they cyclize."
    >
      <div className="card-tinted p-8 md:p-10">
        <p className="font-serif text-xl md:text-2xl text-ink leading-[1.55]">
          APCyc is a target-aware de novo cyclic peptide generation framework that explicitly models
          cyclization and jointly optimizes essential physicochemical properties. It expands the
          residue vocabulary, encodes cyclization-site and linkage-type information, learns
          cyclization-aware representations, and uses Bayesian posterior guidance to steer sampling
          toward cyclic peptides satisfying multiple property objectives.
        </p>
        <div className="mt-8 grid md:grid-cols-3 gap-4">
          {cards.map(([t, d, c]) => (
            <div key={t} className="p-5 rounded-xl bg-white border border-[color:var(--border)]">
              <div className={`text-xs font-mono uppercase tracking-wider text-${c}`}>{t}</div>
              <div className="mt-2 text-ink leading-relaxed text-sm">{d}</div>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}

function Problem() {
  const items = [
    {
      t: "Cyclization is not post-processing",
      d: "Existing target-aware peptide models often generate linear peptides and treat cyclization as a post-hoc modification. This misses cyclization-specific constraints and pocket-specific topology requirements.",
      c: "bio-blue",
      icon: "◯",
    },
    {
      t: "Linkage choices are target-dependent",
      d: "Cyclization type and linkage site should depend on receptor pocket geometry. Fixed heuristics or manually predefined cyclization rules can fail across diverse binding pockets.",
      c: "bio-cyan",
      icon: "⌬",
    },
    {
      t: "Drug-like properties are coupled",
      d: "Permeability, solubility, protease resistance, immunogenicity, and binding affinity interact with cyclization topology. Cyclic peptide design is a multi-objective optimization problem.",
      c: "bio-amber",
      icon: "≋",
    },
  ];
  return (
    <Section eyebrow="Motivation" title="Why cyclic peptide design is hard">
      <div className="grid md:grid-cols-3 gap-5">
        {items.map((it) => (
          <div key={it.t} className="card-surface p-6 hover:ring-glow transition">
            <div className={`text-3xl text-${it.c} mb-3`}>{it.icon}</div>
            <h3 className="text-lg font-semibold mb-2">{it.t}</h3>
            <p className="text-ink-soft leading-relaxed text-sm">{it.d}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function Contributions() {
  const items = [
    {
      n: "01",
      t: "Automated pocket-conditioned cyclization",
      c: "bio-blue",
      d: "APCyc learns to select cyclization types and residue-level linkage sites directly from the receptor binding-pocket context, rather than relying on predefined cyclization constraints.",
      stats: [] as Array<[string, string]>,
    },
    {
      n: "02",
      t: "Property-informed controllable generation",
      c: "bio-rose",
      d: "APCyc uses Bayesian posterior guidance to steer cyclic peptide generation across affinity, permeability, protease resistance, solubility, and immunogenicity.",
      stats: [
        ["0.107", "Membrane permeability proxy"],
        ["−1.474", "Protease resistance"],
      ] as Array<[string, string]>,
    },
    {
      n: "03",
      t: "End-to-end cyclic peptide design",
      c: "bio-violet",
      d: "APCyc unifies cyclization selection, all-atom generation, and property-informed guidance in a single latent diffusion framework.",
      stats: [
        ["−758.545", "Rosetta total score"],
        ["0.971", "Consistency"],
      ] as Array<[string, string]>,
    },
  ];
  return (
    <Section eyebrow="Key contributions" title="Three pieces, one unified framework">
      <div className="grid md:grid-cols-3 gap-5">
        {items.map((it) => (
          <div key={it.n} className="card-surface p-7 relative overflow-hidden min-w-0">
            <div
              aria-hidden
              className={`pointer-events-none absolute -top-4 -right-2 text-[88px] md:text-[104px] font-display font-semibold leading-none select-none bg-${it.c} bg-clip-text text-transparent opacity-20`}
            >
              {it.n}
            </div>
            <div className={`relative text-sm font-mono tracking-wider text-${it.c}`}>{it.n}</div>
            <h3 className="relative mt-2 text-xl font-semibold leading-snug pr-10">{it.t}</h3>
            <p className="relative mt-3 text-ink-soft text-sm leading-relaxed">{it.d}</p>
            {it.stats.length > 0 && (
              <div className="relative mt-5 grid grid-cols-2 gap-3">
                {it.stats.map(([v, l]) => (
                  <div
                    key={l}
                    className="p-3 rounded-lg bg-[color:var(--muted)] border border-[color:var(--border)]"
                  >
                    <div className="text-2xl font-display font-semibold text-gradient">{v}</div>
                    <div className="text-[11px] text-ink-soft mt-0.5">{l}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </Section>
  );
}

function Concept() {
  const parts = [
    {
      k: "A",
      c: "bio-cyan",
      t: "Receptor Pocket Context",
      d: "The receptor pocket provides structural context for target-conditioned generation. APCyc treats the local pocket as a fixed conditioning signal throughout sampling.",
    },
    {
      k: "B",
      c: "bio-green",
      t: "APCyc Mechanism",
      d: "APCyc explicitly predicts cyclization sites and linkage types. A cyclization modeling matrix encodes pairwise site probabilities, while amide, disulfide, and isopeptide linkages are jointly considered.",
    },
    {
      k: "C",
      c: "bio-rose",
      t: "Cyclic Candidates",
      d: "Property guidance steers generation toward different therapeutic profiles — affinity-optimized, membrane-permeable, and multi-property balanced candidates.",
    },
  ];
  return (
    <Section eyebrow="Concept" title="APCyc at a glance">
      <div className="grid md:grid-cols-3 gap-5">
        {parts.map((p) => (
          <div key={p.k} className="card-surface p-6">
            <div className="flex items-center gap-2">
              <span
                className={`inline-flex w-7 h-7 items-center justify-center rounded-md bg-${p.c}/10 text-${p.c} font-mono text-sm`}
              >
                {p.k}
              </span>
              <div className={`text-xs font-mono uppercase tracking-wider text-${p.c}`}>
                Part {p.k}
              </div>
            </div>
            <h3 className="mt-3 text-xl font-semibold">{p.t}</h3>
            <p className="mt-2 text-ink-soft leading-relaxed text-sm">{p.d}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function MethodOverview() {
  const steps = [
    [
      "Pocket extraction",
      "Extract a local receptor pocket and treat it as a fixed conditioning context.",
      "bio-cyan",
    ],
    [
      "Latent encoding",
      "Encode residue identities, 3D coordinates, and cyclization topology into residue and coordinate latents.",
      "bio-blue",
    ],
    [
      "Latent diffusion",
      "Reverse sampling in latent space via an AM-EGNN denoiser under the fixed pocket condition.",
      "bio-violet",
    ],
    [
      "Topology injection",
      "Inject predicted cyclization topology into message passing so linkage sites and types guide denoising.",
      "bio-green",
    ],
    [
      "Posterior guidance",
      "Property energy surrogates and Bayesian posterior guidance steer sampling toward desired profiles.",
      "bio-lime",
    ],
    [
      "Decoding",
      "Decode the final latent representation into a receptor–cyclic peptide complex.",
      "bio-amber",
    ],
  ] as const;
  return (
    <Section
      id="method"
      eyebrow="Method"
      title="Sampling workflow"
      sub="From receptor pocket to property-optimized cyclic peptide — end-to-end inside a single latent diffusion model."
    >
      <div className="mb-12">
        <Figure
          src={OVERVIEW_URL}
          alt="Sampling workflow of APCyc, including pocket extraction, latent encoding, latent diffusion, topology injection, property energy surrogates, posterior guidance, and decoding into receptor-cyclic peptide complex."
          label="Figure 2"
          caption="Overview of the APCyc sampling pipeline. A receptor pocket conditions a latent diffusion process where an AM-EGNN denoiser is steered by both injected cyclization topology and Bayesian posterior gradients from property energy surrogates, and finally decoded into a receptor–cyclic peptide complex."
          maxWidth="max-w-6xl"
        />
      </div>
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {steps.map(([t, d, c], i) => (
          <div key={t} className="card-surface p-5 relative">
            <div className={`text-xs font-mono text-${c}`}>
              STEP {String(i + 1).padStart(2, "0")}
            </div>
            <h4 className="mt-1 text-lg font-semibold">{t}</h4>
            <p className="mt-2 text-sm text-ink-soft leading-relaxed">{d}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function Injection() {
  return (
    <Section
      eyebrow="Topology Injection"
      title="Automated cyclization pair injection"
      sub="APCyc infers pocket-adaptive cyclization topology rather than fixing it in advance."
    >
      <div className="mb-12">
        <Figure
          src={INJECTION_URL}
          alt="Cyclization topology injection module showing pairwise representation, structural and cyclization features, linkage matrix prediction, cyclization type distribution, and AM-EGNN message modulation."
          label="Figure 3"
          caption="Cyclization-topology injection. APCyc builds pairwise residue representations, splits them into structural and cyclization features, predicts a linkage matrix and a cyclization-type distribution, and injects both signals into AM-EGNN through edge features, biases, and gate-based message modulation."
          maxWidth="max-w-6xl"
        />
      </div>
      <div className="mx-auto max-w-3xl">
        <div className="card-surface p-8 md:p-10 text-center">
          <div className="text-2xl md:text-3xl font-display font-semibold text-bio-blue mb-4">
            How it works
          </div>
          <p className="text-lg md:text-xl text-ink-soft leading-relaxed">
            APCyc constructs pairwise residue representations and separates them into structural and
            cyclization features. It predicts both a cyclization linkage matrix and a cyclization
            type distribution, then injects these topology signals into AM-EGNN through edge feature
            augmentation, bias, and gate-based message modulation.
          </p>
          <div className="mt-6 font-mono text-base md:text-lg text-ink leading-relaxed">
            Topology signal <span className="text-bio-cyan">→</span> Edge modulation{" "}
            <span className="text-bio-green">→</span> Geometry-aware denoising
          </div>
        </div>
      </div>
    </Section>
  );
}

function Guidance() {
  const props = [
    ["Affinity", "↑", "bio-blue"],
    ["Permeability", "↑", "bio-green"],
    ["Protease resistance", "↑", "bio-lime"],
    ["Solubility", "↑", "bio-cyan"],
    ["Immunogenicity", "↓", "bio-violet"],
  ] as const;
  const cfg = [
    ["Base", 0, 0, 0, 0, 0],
    ["Affinity", 10, 0, 0, 0, 0],
    ["Protease", 8, 8, 0, 2, 0],
    ["Permeable", 8, 2, 10, 2, 0],
    ["Multi-prop", 6, 4, 4, 3, 5],
  ] as const;
  return (
    <Section eyebrow="Property guidance" title="Bayesian posterior steering">
      <div className="grid lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 card-surface p-7">
          <p className="text-ink-soft leading-relaxed">
            APCyc samples from a property-conditioned posterior rather than only from a
            receptor-conditioned prior. Differentiable property surrogates provide gradients that
            guide the diffusion trajectory toward desired peptide profiles.
          </p>
          <div className="mt-6 space-y-3">
            {props.map(([n, dir, c]) => (
              <div key={n} className="flex items-center gap-3">
                <div className="w-44 text-sm">{n}</div>
                <div className="flex-1 h-2 rounded-full bg-[color:var(--muted)] overflow-hidden">
                  <div
                    className={`h-full bg-${c}`}
                    style={{ width: `${55 + Math.random() * 35}%` }}
                  />
                </div>
                <div className={`text-${c} font-mono text-sm w-6 text-right`}>{dir}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="lg:col-span-7 card-tinted p-7">
          <div className="text-xs font-mono uppercase tracking-wider text-bio-blue mb-3">
            Guidance weights
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-ink-soft">
                <tr className="border-b border-[color:var(--border)]">
                  <th className="text-left py-2 pr-3 font-medium">Objective</th>
                  <th className="text-right px-2 font-medium">Aff.</th>
                  <th className="text-right px-2 font-medium">Prot.</th>
                  <th className="text-right px-2 font-medium">Perm.</th>
                  <th className="text-right px-2 font-medium">Sol.</th>
                  <th className="text-right px-2 font-medium">Imm.</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {cfg.map((row) => (
                  <tr
                    key={row[0]}
                    className="border-b border-[color:var(--border)]/60 hover:bg-white"
                  >
                    <td className="py-2 pr-3 font-sans">{row[0]}</td>
                    {row.slice(1).map((v, i) => (
                      <td
                        key={i}
                        className={`text-right px-2 ${v === 0 ? "text-ink-soft/40" : "text-ink"}`}
                      >
                        {v}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Section>
  );
}

const propRows: ResultRow[] = [
  { m: "PepGLAD", v: [1.593, -1.555, -0.01, -9.097] },
  { m: "CPComposer", v: [1.52, -1.604, -0.307, -7.576] },
  { m: "PepFlow", v: [1.624, -1.735, -0.093, -9.654] },
  {
    m: "APCyc Membrane-permeable",
    v: [1.374, -1.499, 0.107, -8.96],
    h: [false, false, true, false],
  },
  {
    m: "APCyc Multi-property joint",
    v: [1.428, -1.474, -0.051, -9.102],
    h: [false, true, false, false],
  },
  { m: "APCyc Protease res.-optimized", v: [1.377, -1.492, 0.045, -9.062] },
  { m: "APCyc Affinity-optimized", v: [1.404, -1.496, 0.046, -8.977] },
];
const mainRows: ResultRow[] = [
  { m: "PepGLAD", v: [-735.528, 0.973, 0.883, 0.669, 0.948, -4.544, -12.14] },
  { m: "CPComposer", v: [-743.422, 0.809, 0.7, 0.898, 0.953, -3.686, -10.716] },
  { m: "PepFlow", v: [-713.165, 0.943, 0.711, 0.702, 0.828, -4.772, -11.711] },
  {
    m: "APCyc Membrane-permeable",
    v: [-747.898, 0.907, 0.671, 0.787, 0.971, -4.738, -10.872],
    h: [false, false, false, false, true, false, false],
  },
  {
    m: "APCyc Multi-property joint",
    v: [-758.545, 0.904, 0.664, 0.761, 0.967, -4.726, -11.328],
    h: [true, false, false, false, false, false, false],
  },
  {
    m: "APCyc Protease res.-optimized",
    v: [-747.176, 0.907, 0.689, 0.799, 0.968, -4.793, -12.178],
    h: [false, false, false, false, false, true, true],
  },
  { m: "APCyc Affinity-optimized", v: [-741.56, 0.898, 0.648, 0.79, 0.934, -4.653, -10.981] },
];

function ResultsTable({
  headers,
  rows,
  format,
}: {
  headers: string[];
  rows: ResultRow[];
  format: (v: number) => string;
}) {
  return (
    <div className="overflow-x-auto -mx-2 px-2">
      <table className="w-full text-sm min-w-[640px]">
        <thead>
          <tr className="text-ink-soft text-xs uppercase tracking-wider border-b border-[color:var(--border)]">
            <th className="text-left py-3 pr-4 font-medium">Method</th>
            {headers.map((h) => (
              <th key={h} className="text-right px-3 py-3 font-medium whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isAPCyc = r.m.startsWith("APCyc");
            return (
              <tr
                key={r.m}
                className={`border-b border-[color:var(--border)]/60 hover:bg-[color:var(--muted)] transition ${isAPCyc ? "bg-bio-blue/[0.06]" : ""}`}
              >
                <td className={`py-3 pr-4 ${isAPCyc ? "text-ink font-medium" : "text-ink-soft"}`}>
                  {r.m}
                </td>
                {r.v.map((v, i) => {
                  const hi = r.h?.[i];
                  return (
                    <td
                      key={i}
                      className={`text-right px-3 py-3 font-mono ${hi ? "text-bio-blue font-semibold" : "text-ink"}`}
                    >
                      {hi && <span className="mr-1 text-bio-amber">★</span>}
                      {format(v)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function DenoisingDemos() {
  const demos = [
    {
      pdb: "2xyi",
      src: DENOISING_2XYI_URL,
      caption:
        "Latent noise progressively denoised into a target-bound cyclic peptide for pocket 2xyi.",
    },
    {
      pdb: "5y59",
      src: DENOISING_5Y59_URL,
      caption: "Generation trajectory from Gaussian latent to a cyclized binder for pocket 5y59.",
    },
    {
      pdb: "1t4f",
      src: DENOISING_1T4F_URL,
      caption: "Denoising process for pocket 1t4f.",
    },
  ];
  const [active, setActive] = useState(0);
  const [ended, setEnded] = useState<boolean[]>([false, false, false]);
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const current = demos[active];

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (ended[active]) {
      v.currentTime = v.duration || 0;
    } else {
      v.currentTime = 0;
    }
    setPlaying(false);
  }, [active, ended]);

  const handleTagClick = (i: number) => {
    setActive(i);
  };

  const handleVideoClick = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused || v.ended) {
      v.currentTime = 0;
      v.play();
      setPlaying(true);
    } else {
      v.pause();
      setPlaying(false);
    }
  };

  const handleEnded = () => {
    setPlaying(false);
    setEnded((prev) => {
      const next = [...prev];
      next[active] = true;
      return next;
    });
  };

  return (
    <Section
      id="demos"
      eyebrow="Demo"
      title="Denoising in action: from noise to cyclic peptide"
      sub="Pick a target pocket. Each clip shows APCyc denoising a latent sample into a target-aware cyclic peptide."
    >
      <div className="mx-auto max-w-5xl">
        <div className="text-center mb-3">
          <span className="text-sm font-medium text-ink-soft">Example Target-aware Generation</span>
        </div>
        <div className="flex flex-wrap items-center justify-center gap-3 mb-6">
          {demos.map((d, i) => (
            <button
              key={d.pdb}
              onClick={() => handleTagClick(i)}
              className={`px-4 py-2 rounded-full text-sm font-mono uppercase tracking-wider border transition ${
                i === active
                  ? "bg-gradient-to-r from-bio-blue to-bio-violet text-white border-transparent shadow-[0_8px_30px_-8px_rgba(43,91,255,0.5)]"
                  : "border-black/20 text-ink-soft hover:text-ink hover:bg-[color:var(--muted)]"
              }`}
            >
              PDB {d.pdb}
            </button>
          ))}
        </div>
        <figure className="card-surface p-3 md:p-4">
          <div
            className="relative rounded-xl overflow-hidden bg-black cursor-pointer group"
            onClick={handleVideoClick}
          >
            <video
              ref={videoRef}
              key={current.src}
              src={current.src}
              muted
              playsInline
              onEnded={handleEnded}
              className="block w-full h-auto pointer-events-none"
            />
            {!playing && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-14 h-14 rounded-full bg-white/90 text-ink flex items-center justify-center shadow-lg opacity-80 group-hover:opacity-100 transition">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                </div>
              </div>
            )}
            <span className="absolute top-3 left-3 px-2.5 py-1 rounded-md text-xs font-mono uppercase tracking-wider bg-black/60 text-white backdrop-blur-sm">
              PDB {current.pdb}
            </span>
          </div>
          <figcaption className="mt-4 text-center text-lg text-ink-soft max-w-3xl mx-auto leading-relaxed">
            <span className="font-display font-semibold text-ink mr-1.5">{current.pdb}.</span>
            <span className="font-serif italic text-ink/85">{current.caption}</span>
          </figcaption>
        </figure>
      </div>
    </Section>
  );
}

function Results() {
  const highlights = [
    ["0.107", "Best membrane permeability proxy", "bio-green"],
    ["−1.474", "Best protease resistance", "bio-lime"],
    ["−758.545", "Best Rosetta total score", "bio-blue"],
    ["0.971", "Best consistency (Cramér's V)", "bio-cyan"],
    ["−4.793", "Strongest Vina (Protease-opt)", "bio-violet"],
    ["−12.178", "Strongest dG (Protease-opt)", "bio-amber"],
  ] as const;
  return (
    <Section
      id="results"
      eyebrow="Experiments"
      title="Results across therapeutic & generative metrics"
    >
      <div className="mb-12">
        <Figure
          src={RADAR_URL}
          alt="Radar chart comparing PepGLAD, CPComposer, PepFlow, and APCyc variants across therapeutic properties and generative metrics."
          label="Figure 4"
          caption="Radar comparison across therapeutic and generative metrics. APCyc variants expand the Pareto frontier relative to PepGLAD, CPComposer, and PepFlow, with each guidance profile specializing in a different property direction."
          maxWidth="max-w-4xl"
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-10">
        {highlights.map(([v, l, c]) => (
          <div key={l} className="card-surface p-4">
            <div className={`text-2xl font-display font-bold text-${c}`}>{v}</div>
            <div className="text-[11px] text-ink-soft mt-1 leading-tight">{l}</div>
          </div>
        ))}
      </div>

      <div className="card-surface p-6 md:p-8 mb-6">
        <div className="flex items-baseline justify-between mb-4 flex-wrap gap-2">
          <h3 className="text-xl font-semibold">Property metrics</h3>
          <div className="text-xs text-ink-soft font-mono">
            ↑ higher is better for Solubility, Protease, Permeability · ↓ lower is better for
            Immunogenicity
          </div>
        </div>
        <ResultsTable
          headers={["Solubility ↑", "Protease res. ↑", "Membrane perm. ↑", "Immunogenicity ↓"]}
          rows={propRows}
          format={(v) => v.toFixed(3).replace("-", "−")}
        />
      </div>

      <div className="card-surface p-6 md:p-8 mb-6">
        <h3 className="text-xl font-semibold mb-4">Main results</h3>
        <ResultsTable
          headers={[
            "Rosetta ↓",
            "Cyc. ↑",
            "Energy ↑",
            "Diversity ↑",
            "Cramér's V ↑",
            "Vina ↓",
            "dG ↓",
          ]}
          rows={mainRows}
          format={(v) => (Math.abs(v) >= 100 ? v.toFixed(3) : v.toFixed(3)).replace("-", "−")}
        />
      </div>

      <div className="card-tinted p-6 md:p-8">
        <p className="font-serif text-lg text-ink leading-relaxed">
          APCyc variants occupy complementary regions of the therapeutic–generative trade-off space.
          The <span className="text-bio-green font-semibold not-italic">membrane-permeable</span>{" "}
          setting achieves the highest permeability proxy, the{" "}
          <span className="text-bio-lime font-semibold not-italic">multi-property</span> setting
          achieves the best protease resistance and Rosetta stability, and the{" "}
          <span className="text-bio-violet font-semibold not-italic">
            protease-resistance optimized
          </span>{" "}
          setting yields the strongest binding affinity among the evaluated APCyc variants.
        </p>
      </div>
    </Section>
  );
}

function CaseStudy() {
  const cases: Array<{ id: string; rows: CaseRow[] }> = [
    {
      id: "3rc4",
      rows: [
        ["APCyc", -4.96, -12.615, 2.408, -1.637],
        ["Baseline mean", -1.149, -10.636, 0.806, -1.639],
      ],
    },
    {
      id: "4xal",
      rows: [
        ["APCyc", -6.82, -12.36, 3.015, -1.256],
        ["Baseline mean", -4.023, -10.688, 0.917, -1.768],
      ],
    },
  ];
  return (
    <Section id="case" eyebrow="Case study" title="Target-level analysis">
      <div className="grid md:grid-cols-2 gap-6">
        {cases.map((c) => (
          <div key={c.id} className="card-surface p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-2xl font-bold font-mono text-gradient">{c.id}</h3>
              <span className="text-xs text-ink-soft font-mono">PDB target</span>
            </div>
            <table className="w-full text-sm">
              <thead className="text-ink-soft text-xs uppercase">
                <tr className="border-b border-[color:var(--border)]">
                  <th className="text-left py-2 font-medium">Method</th>
                  <th className="text-right px-2 font-medium">Vina ↓</th>
                  <th className="text-right px-2 font-medium">Imm. ↓</th>
                  <th className="text-right px-2 font-medium">Perm. ↑</th>
                  <th className="text-right px-2 font-medium">Prot. ↑</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {c.rows.map((r, i) => (
                  <tr
                    key={i}
                    className={`border-b border-[color:var(--border)]/60 ${i === 0 ? "text-bio-blue font-semibold" : "text-ink-soft"}`}
                  >
                    <td className="py-2 font-sans">{r[0]}</td>
                    {r.slice(1).map((v, j) => (
                      <td key={j} className="text-right px-2">
                        {Number(v).toFixed(3).replace("-", "−")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-4 text-sm text-ink-soft">
              APCyc improves membrane permeability from{" "}
              <span className="text-ink">{c.rows[1][3]}</span> to{" "}
              <span className="text-bio-green font-medium">{c.rows[0][3]}</span> and strengthens
              Vina binding while reducing immunogenicity.
            </p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function Ablation() {
  const proteaseRows: AblationRow[] = [
    ["Multi-property guided", -1.474, -9.102, true],
    ["Protease-optimized", -1.492, -9.062, false],
    ["Base no guidance", -1.519, -9.011, false],
  ];
  const permeabilityRows: PermeabilityRow[] = [
    ["Membrane-permeable guided", 0.107, true],
    ["Multi-property guided", -0.051, false],
    ["Base no guidance", 0.042, false],
  ];

  return (
    <Section eyebrow="Ablation" title="Does guidance matter?">
      <div className="grid md:grid-cols-2 gap-6">
        <div className="card-surface p-6">
          <h4 className="font-semibold mb-3">Protease resistance & immunogenicity</h4>
          <table className="w-full text-sm">
            <thead className="text-ink-soft text-xs uppercase">
              <tr className="border-b border-[color:var(--border)]">
                <th className="text-left py-2 font-medium">Setting</th>
                <th className="text-right px-2 font-medium">Protease ↑</th>
                <th className="text-right px-2 font-medium">Imm. ↓</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {proteaseRows.map(([setting, protease, immunogenicity, highlighted], i) => (
                <tr
                  key={i}
                  className={`border-b border-[color:var(--border)]/60 ${highlighted ? "bg-bio-lime/[0.08]" : ""}`}
                >
                  <td className="py-2 font-sans">{setting}</td>
                  <td
                    className={`text-right px-2 ${highlighted ? "text-bio-lime font-semibold" : "text-ink"}`}
                  >
                    {protease.toFixed(3).replace("-", "−")}
                  </td>
                  <td className="text-right px-2 text-ink">
                    {immunogenicity.toFixed(3).replace("-", "−")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card-surface p-6">
          <h4 className="font-semibold mb-3">Membrane permeability proxy</h4>
          <table className="w-full text-sm">
            <thead className="text-ink-soft text-xs uppercase">
              <tr className="border-b border-[color:var(--border)]">
                <th className="text-left py-2 font-medium">Setting</th>
                <th className="text-right px-2 font-medium">Permeability ↑</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {permeabilityRows.map(([setting, permeability, highlighted], i) => (
                <tr
                  key={i}
                  className={`border-b border-[color:var(--border)]/60 ${highlighted ? "bg-bio-green/[0.08]" : ""}`}
                >
                  <td className="py-2 font-sans">{setting}</td>
                  <td
                    className={`text-right px-2 ${highlighted ? "text-bio-green font-semibold" : "text-ink"}`}
                  >
                    {permeability.toFixed(3).replace("-", "−")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-4 text-sm text-ink-soft">
            Explicit guidance is essential for steering cyclic peptide generation toward desired
            physicochemical objectives.
          </p>
        </div>
      </div>
    </Section>
  );
}

function Dataset() {
  const stats = [
    ["2.71M", "CPSea complexes"],
    ["71,867", "CPCore complexes"],
    ["70,284", "Training set"],
    ["1,583", "Validation set"],
    ["56", "Test targets"],
  ];
  return (
    <Section eyebrow="Setup" title="Dataset & evaluation">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-8">
        {stats.map(([v, l]) => (
          <div key={l} className="card-surface p-5 text-center">
            <div className="text-3xl font-display font-bold text-gradient">{v}</div>
            <div className="text-xs text-ink-soft mt-1">{l}</div>
          </div>
        ))}
      </div>
      <div className="grid md:grid-cols-3 gap-5">
        <div className="card-surface p-6">
          <div className="text-xs uppercase font-mono tracking-wider text-bio-cyan mb-2">
            Cyclization topologies
          </div>
          <ul className="space-y-2 text-sm text-ink-soft">
            <li>• Mainchain amide bonds</li>
            <li>• Disulfide bonds</li>
            <li>• Side-chain isopeptide linkages (Lys–Asp/Asn)</li>
          </ul>
        </div>
        <div className="card-surface p-6">
          <div className="text-xs uppercase font-mono tracking-wider text-bio-green mb-2">
            Baselines
          </div>
          <div className="flex flex-wrap gap-2">
            {["PepGLAD", "CPComposer", "PepFlow"].map((b) => (
              <span
                key={b}
                className="px-3 py-1 rounded-full bg-[color:var(--muted)] border border-[color:var(--border)] text-sm"
              >
                {b}
              </span>
            ))}
          </div>
          <div className="mt-4 text-xs text-ink-soft">
            CPCore is a refined subset of CPSea, curated from the AlphaFold Database.
          </div>
        </div>
        <div className="card-surface p-6">
          <div className="text-xs uppercase font-mono tracking-wider text-bio-amber mb-2">
            Evaluation
          </div>
          <div className="flex flex-wrap gap-1.5 text-[11px]">
            {[
              "GRAVY",
              "logP",
              "TPSA",
              "rTPSA",
              "CamSol",
              "ProsperousPlus",
              "BigMHC",
              "AutoDock Vina",
              "Rosetta",
              "Diversity",
              "Cramér's V",
            ].map((m) => (
              <span
                key={m}
                className="px-2 py-1 rounded-md bg-[color:var(--muted)] border border-[color:var(--border)] font-mono text-ink-soft"
              >
                {m}
              </span>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
}

const BIBTEX = `@inproceedings{zhao2026apcyc,
  title={APCyc: Property-Informed Design of Cyclic Peptides via Automated Cyclization},
  author={Zhao, Yifan and Qin, Lang and Chen, Jintai},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year={2026},
  address={Jeju Island, Republic of Korea},
  publisher={ACM},
  doi={10.1145/3770855.3818908}
}`;

function Citation() {
  const [copied, setCopied] = useState(false);
  return (
    <Section id="citation" eyebrow="Cite" title="Citation">
      <div className="card-surface p-7 mb-6">
        <div className="text-xs uppercase font-mono tracking-wider text-bio-blue mb-3">
          ACM Reference Format
        </div>
        <p className="font-serif text-lg text-ink leading-relaxed">
          Yifan Zhao, Lang Qin, and Jintai Chen. 2026. APCyc: Property-Informed Design of Cyclic
          Peptides via Automated Cyclization. In{" "}
          <em>
            Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2
            (KDD 2026)
          </em>
          , August 9–13, 2026, Jeju Island, Republic of Korea. ACM, New York, NY, USA, 12 pages.{" "}
          <a
            href="https://doi.org/10.1145/3770855.3818908"
            className="text-bio-blue hover:underline"
          >
            https://doi.org/10.1145/3770855.3818908
          </a>
        </p>
      </div>
      <div className="card-surface overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-[color:var(--border)] bg-[color:var(--muted)]">
          <div className="text-xs font-mono uppercase tracking-wider text-bio-blue">BibTeX</div>
          <button
            onClick={() => {
              navigator.clipboard.writeText(BIBTEX);
              setCopied(true);
              setTimeout(() => setCopied(false), 1800);
            }}
            className="text-xs px-3 py-1.5 rounded-md bg-white hover:bg-[color:var(--accent)] border border-[color:var(--border)] transition"
          >
            {copied ? "Copied ✓" : "Copy BibTeX"}
          </button>
        </div>
        <pre className="p-5 text-xs md:text-sm font-mono text-ink overflow-x-auto leading-relaxed bg-white">
          {BIBTEX}
        </pre>
      </div>
    </Section>
  );
}

function Footer() {
  return (
    <footer className="relative mt-10 border-t border-[color:var(--border)] px-6 py-12 bg-white/40">
      <div className="mx-auto max-w-7xl grid md:grid-cols-3 gap-8">
        <div>
          <div className="flex items-center gap-2 font-display font-bold text-lg">
            <span className="inline-block w-7 h-7 rounded-lg bg-gradient-to-br from-bio-cyan via-bio-blue to-bio-violet" />
            APCyc
          </div>
          <p className="mt-3 text-sm text-ink-soft">
            Property-Informed Design of Cyclic Peptides via Automated Cyclization. KDD 2026.
          </p>
        </div>
        <div className="text-sm text-ink-soft">
          <div className="text-ink font-medium mb-2">Lab</div>
          HKUST(GZ) ML4Health Lab
          <br />
          AI-Peptide Drug Design Joint Laboratory
        </div>
        <div className="text-sm">
          <div className="text-ink font-medium mb-2">Links</div>
          <div className="flex flex-col gap-1.5">
            <a href={PAPER_URL} className="text-ink-soft hover:text-bio-blue transition">
              Paper (PDF)
            </a>
            <a href={CODE_URL} className="text-ink-soft hover:text-bio-blue transition">
              GitHub
            </a>
          </div>
        </div>
      </div>
      <div className="mx-auto max-w-7xl mt-10 pt-6 border-t border-[color:var(--border)]/60 text-xs text-ink-soft/80 flex justify-between flex-wrap gap-2">
        <span>© 2026 APCyc authors · CC BY 4.0</span>
        <span className="font-mono">target-aware · cyclization · diffusion</span>
      </div>
    </footer>
  );
}

export function Index() {
  return (
    <div className="min-h-screen text-ink">
      <Nav />
      <Hero />
      <TLDR />
      <Problem />
      <Contributions />
      <Concept />
      <MethodOverview />
      <Injection />
      <DenoisingDemos />
      <Results />
      <CaseStudy />
      <Ablation />
      <Guidance />
      <Dataset />
      <Citation />
      <Footer />
    </div>
  );
}
