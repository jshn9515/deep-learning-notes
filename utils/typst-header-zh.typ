#set document(
  title: "深度学习笔记",
  author: "橘生淮南",
  description: "A collection of notes and code examples for deep learning concepts and techniques.",
  keywords: (
        "Deep Learning Tutorial",
        "PyTorch",
        "Quarto",
    ),
)
#set page(paper: "a4")

#let SimSun = ((name: "Times New Roman", covers: "latin-in-cjk"), "SimSun")
#set text(
    font: SimSun,
    size: 12pt,
    fallback: false,
    top-edge: "bounds",
    bottom-edge: "bounds",
    lang: "zh",
    region: "cn",
)

#set par(
    justify: true,
    spacing: 1em,
)

#show math.equation: set block(breakable: true)
#show figure: set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)
#show cite: it => { super(it) }
