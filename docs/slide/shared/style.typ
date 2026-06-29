
// 数値文字列を小数第2位まで丸める（非数値はそのまま返す）
#let fmt(v) = {
  // 前後の空白を除去
  let trimmed = v.trim()
  // 整数・小数・科学記法を判定する正規表現
  if trimmed.match(regex("^-?[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?$")) != none {
    // 数値なら小数第2位に丸めて文字列に変換
    str(calc.round(float(trimmed), digits: 2))
  } else {
    // 非数値（ヘッダー等）はそのまま返す
    trimmed
  }
}

// CSV読み込み＋数値フォーマット＋booktabs描画
// 指定した行インデックス（0始まり、ヘッダー除く）だけを抜き出して描画
#let booktabs(path, rows: none, offset: -2,size: 0.78em) = {
  let data = csv(path)
  let header = data.at(0)
  // rows指定があればその行だけ、なければ全行
  let all_rows = data.slice(1)
  let selected = if rows == none {
    all_rows
  } else {
    rows.map(i => all_rows.at(i + offset))
  }
  table(
    columns: header.len(),
    stroke: none,
    inset: (x: 6pt, y: 5pt),
    align: (col, _) => if col == 0 { left } else { right },
    table.hline(stroke: 1.2pt),
    ..header.map(h => text(weight: "bold", size: size, h)),
    table.hline(stroke: 0.6pt),
    ..selected.flatten().map(v => text(size: size, fmt(v))),
    table.hline(stroke: 1.2pt),
  )
}
