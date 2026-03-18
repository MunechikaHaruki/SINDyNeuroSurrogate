import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import matplotlib.pyplot as plt

    def export_tex_to_png(tex_dict, output_dir="figures/formulas/"):
        """
        文字列の辞書を受け取り、一括でPNG出力してmarimoで表示する
        """

        # ディレクトリ作成（exist_ok=Trueでスッキリ書けます）
        os.makedirs(output_dir, exist_ok=True)

        generated_files = []

        for filename, tex in tex_dict.items():
            path = os.path.join(output_dir, f"{filename}.png")

            # 描画設定（dpi=300で保存用は高画質に）
            fig = plt.figure(figsize=(8, 1.5), dpi=300)
            fig.text(
                0.5, 0.5, f"${tex}$", size=24, va="center", ha="center", color="black"
            )

            # 保存（透過PNG）
            plt.savefig(path, bbox_inches="tight", pad_inches=0.1, transparent=True)
            plt.close(fig)
            generated_files.append(path)

        # marimoのUI要素を返す
        return mo.vstack(
            [
                mo.md(f"### ✅ {len(generated_files)} 個の数式を生成しました"),
                mo.md(f"保存先: `{os.path.abspath(output_dir)}`"),
                # ここで height を指定して表示サイズをシュッとさせる
                *[mo.image(src=f, height=40, width=300) for f in generated_files],
            ]
        )

    return (export_tex_to_png,)


@app.cell
def _(export_tex_to_png):
    # current
    export_tex_to_png(
        {
            "current_inter_compartment": r"I_{i,j} =  g_{i,j} (V_j - V_i)",
            "current_axial": r"I_{i(axial)} = \sum_{j \in \text{neighbors}} g_{i,j} (V_j - V_i) +I_{inj}",
        }
    )

    return


@app.cell
def _(export_tex_to_png):
    export_tex_to_png(
        {
            "hh_main": r"C_m \frac{dV}{dt} = -g_{leak}(V-E_{rest}) - I_{ion}(m,h,n)+I_{ext}",
            "hh_gate": r"\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x \quad (x=m,h,n)",
            "passive_comp": r"C_m \frac{dV}{dt} = -g_{leak}(V-E_{rest}) +I_{ext}",
        }
    )
    return


@app.cell
def _(export_tex_to_png):
    # sindy_eq
    export_tex_to_png(
        {
            "sindy_eq": r"\frac{dV}{dt} = \sum_i a_i \theta_i(V, g', I_{ext})",
            "sindy_eq2": r"\frac{dg'}{dt} = \sum_i b_i \vartheta_i(V, g', I_{ext})",
        }
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
