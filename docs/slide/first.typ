#import "@preview/slydst:0.1.4": *

#set text(font: ("Hiragino Kaku Gothic ProN", "New Computer Modern"))
#show figure.caption: it => it.body
#show link: underline

#show: slides.with(
  title: "研究進捗報告",
  subtitle: "Biophysical ニューロンモデルに対する代理モデル",
  authors: ("Haruki Munechika",),
  ratio: 16/9
)

#show figure.caption: set text(size: 9pt)
#set text(size: 12pt)


#include "shared/intro.typ"