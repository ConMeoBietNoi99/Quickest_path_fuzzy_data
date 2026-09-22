import os
import time
import random
import matplotlib.pyplot as plt
from typing import Dict, List

# Import trực tiếp các cấu trúc Dataclass và thuật toán từ CS_self.py
from qpp_fuzzy_solver import (
    Network,
    Arc,
    TriangularFuzzy,
    parametric_algorithm
)


def generate_random_parametric_network(n: int, density: float = 0.25) -> Network:
    """Tạo mạng ngẫu nhiên với n đỉnh và mật độ cung 0.25 sử dụng class Network."""
    net = Network(n, source=0, sink=n - 1)

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() <= density:
                # Sinh số mờ tam giác cho lead_time
                t1 = random.uniform(1.0, 5.0)
                t2 = t1 + random.uniform(1.0, 8.0)
                t3 = t2 + random.uniform(1.0, 8.0)
                lead_time = TriangularFuzzy(t1, t2, t3)

                # Sinh số mờ tam giác cho capacity
                r1 = random.uniform(2.0, 10.0)
                r2 = r1 + random.uniform(1.0, 10.0)
                r3 = r2 + random.uniform(1.0, 10.0)
                capacity = TriangularFuzzy(r1, r2, r3)

                net.add_arc(Arc(i, j, lead_time, capacity))

    return net


def run_parametric_scaling_experiment():
    n_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    density = 0.25
    V = 100.0
    instances_per_size = 10  # Tổng = 10 * 10 = 120 instances

    results = []

    print("Bắt đầu chạy thực nghiệm Parametric Algorithm Scaling (Cấu trúc Dataclass)...")
    print("-" * 80)
    print(f"{'n':<5} | {'h_L':<8} | {'h_R':<8} | {'Breakpoints M':<15} | {'BSP solves':<12} | {'Time (s)':<10}")
    print("-" * 80)

    for n in n_values:
        total_hL = 0.0
        total_hR = 0.0
        total_M = 0.0
        total_bsp = 0.0
        total_time = 0.0

        for _ in range(instances_per_size):
            network = generate_random_parametric_network(n, density)

            start_time = time.time()
            # Gọi parametric_algorithm trả về đối tượng ParametricResult (truy xuất qua dấu chấm .)
            res = parametric_algorithm(network, V)
            elapsed = time.time() - start_time

            total_hL += res.h_L
            total_hR += res.h_R
            total_M += res.breakpoints
            total_bsp += res.bsp_solves
            total_time += elapsed

        avg_hL = total_hL / instances_per_size
        avg_hR = total_hR / instances_per_size
        avg_M = total_M / instances_per_size
        avg_bsp = total_bsp / instances_per_size
        avg_time = total_time / instances_per_size

        results.append({
            "n": n,
            "h_L": avg_hL,
            "h_R": avg_hR,
            "M": avg_M,
            "bsp_solves": avg_bsp,
            "time": avg_time
        })

        print(f"{n:<5} | {avg_hL:<8.1f} | {avg_hR:<8.1f} | {avg_M:<15.1f} | {avg_bsp:<12.1f} | {avg_time:<10.2f}")

    print("-" * 80)

    # Tạo thư mục đầu ra đúng theo yêu cầu
    output_dir = "Resul_parametric_random"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Xuất file mã LaTeX của bảng kết quả
    latex_path = os.path.join(output_dir, "table_scaling_parametric.tex")
    latex_content = r"""\begin{table}[h]
\centering
\caption{Scaling of the parametric algorithm}
\label{tab:scaling_parametric}
\begin{tabular}{rrrrrr}
\toprule
$n$ & $h_{\mathrm{L}}$ & $h_{\mathrm{R}}$ & Breakpoints $M$ & BSP solves & Time (s) \\
\midrule
"""
    for r in results:
        latex_content += f"{r['n']} & {r['h_L']:.1f} & {r['h_R']:.1f} & {r['M']:.1f} & {r['bsp_solves']:.1f} & {r['time']:.2f} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    # 2. Vẽ biểu đồ đường theo phong cách chuẩn LaTeX (Serif font, lưới rõ nét, 2 trục y)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300
    })

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    ns = [r["n"] for r in results]
    times = [r["time"] for r in results]
    bsps = [r["bsp_solves"] for r in results]

    # Trục y thứ nhất: Thời gian chạy (Time)
    color = 'tab:blue'
    ax1.set_xlabel('Network size ($n$ nodes)')
    ax1.set_ylabel('Wall-clock Time (s)', color=color)
    line1 = ax1.plot(ns, times, color=color, marker='o', linewidth=1.5, label='Time (s)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Trục y thứ hai: Số lần giải bài toán BSP (BSP solves)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('BSP Solves', color=color)
    line2 = ax2.plot(ns, bsps, color=color, marker='s', linestyle='--', linewidth=1.5, label='BSP solves')
    ax2.tick_params(axis='y', labelcolor=color)

    # Gộp chung chú thích (legend)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Scaling of the Parametric Algorithm ($V=100$, density=0.25)')
    plt.tight_layout()

    pdf_path = os.path.join(output_dir, "scaling_parametric_plot.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"\n[Thông báo] Đã xuất file LaTeX vào: {latex_path}")
    print(f"[Thông báo] Đã xuất biểu đồ PDF vào: {pdf_path}")


if __name__ == "__main__":
    random.seed(42)
    run_parametric_scaling_experiment()