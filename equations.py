import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import os
    import marimo as mo

    def export_tex_to_png(tex_dict, output_dir="figures/formulas/"):
        """
        文字列の辞書を受け取り、一括でPNG出力する
        tex_dict: {"ファイル名": "TeX数式"}
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        generated_files = []
    
        for filename, tex in tex_dict.items():
            # ファイルパスの設定
            path = os.path.join(output_dir, f"{filename}.png")
        
            # 描画設定
            fig = plt.figure(figsize=(6, 1.5), dpi=300)
            # 数式を中央に配置（$で囲む）
            fig.text(0.5, 0.5, f"${tex}$", 
                     size=24, va='center', ha='center', color='black')
        
            # 保存（透過PNG）
            plt.savefig(path, bbox_inches='tight', pad_inches=0.1, transparent=True)
            plt.close(fig)
            generated_files.append(path)
        
        return generated_files



    return export_tex_to_png, mo, os


@app.cell
def _(export_tex_to_png, mo, os):
    # --- ここに書きたい数式を文字列で並べる ---
    formulas_to_generate = {
        "current_inter_compartment": r"I_{i,j} =  g_{i,j} (V_j - V_i)",
        "current_axial":r"I_{i(axial)} = \sum_{j \in \text{neighbors}} g_{i,j} (V_j - V_i) +I_{inj}",
    "hh_main": r"C_m \frac{dV}{dt} = -g_{leak}(V-E_{rest}) - I_{ion}(m,h,n)+I_{ext}",
        "hh_gate": r"\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x \quad (x=m,h,n)",
        "passive_comp": r"C_m \frac{dV}{dt} = -g_{leak}(V-E_{rest}) +I_{ext}",
        "sindy_eq":r"\frac{dV}{dt} = \sum_i a_i \theta_i(V, g', I_{ext})",
        "sindy_eq2":r"\frac{dg'}{dt} = \sum_i b_i \vartheta_i(V, g', I_{ext})"
    }

    # 実行
    files = export_tex_to_png(formulas_to_generate)

    # marimo上での確認
    mo.vstack([
        mo.md(f"### ✅ {len(files)} 個の数式を生成しました"),
        mo.md(f"保存先ディレクトリ: `{os.path.abspath('output_formulas')}`"),
        *[mo.image(src=f) for f in files]
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
