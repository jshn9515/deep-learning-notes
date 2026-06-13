#set document(
  title: "Deep Learning Notes",
  author: "jshn9515",
  description: "A collection of notes and code examples for deep learning concepts and techniques.",
  keywords: (
        "Deep Learning Tutorial",
        "PyTorch",
        "Quarto",
    ),
)
#set page(paper: "a4")

#set text(
    font: "Libertinus Serif",
    size: 12pt,
    fallback: false,
    top-edge: "bounds",
    bottom-edge: "bounds",
    lang: "en",
    region: "us",
)

#set par(
    justify: true,
    spacing: 1em,
)

#show math.equation: set block(breakable: true)
#show figure: set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)
#show cite: it => { super(it) }
