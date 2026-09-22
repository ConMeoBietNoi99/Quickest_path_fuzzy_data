import os
import time
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# Import trực tiếp các cấu trúc chuẩn từ file qpp_fuzzy_solver
from qpp_fuzzy_solver import TriangularFuzzy, Network, Arc, level_solver

def generate_random_network(n: int, density: float = 0.4) -> Network:
    """Tạo mạng ngẫu nhiên (hoặc DAG) với n đỉnh và mật độ cung khoảng density sử dụng class Network."""
    net = Network(n, source=0, sink=n - 1)

    for i in range(n):
        for j in range(i + 1, n):  # Đảm bảo i < j để tuân thủ thứ tự đỉnh/mạng phân cấp
            if random.random() <= density:
                # Sinh số mờ tam giác cho lead_time t_ij = (t1, t2, t3)
                t1 = random.uniform(1.0, 5.0)
                t2 = t1 + random.uniform(1.0, 8.0)
                t3 = t2 + random.uniform(1.0, 8.0)
                lead_time = TriangularFuzzy(t1, t2, t3)

                # Sinh số mờ tam giác cho capacity r_ij = (r1, r2, r3)
                r1 = random.uniform(2.0, 10.0)
                r2 = r1 + random.uniform(1.0, 10.0)
                r3 = r2 + random.uniform(1.0, 10.0)
                capacity = TriangularFuzzy(r1, r2, r3)

                net.add_arc(Arc(i, j, lead_time, capacity))

    return net


def run_level_solver_verification_and_export(alpha: float = 0.5):
    """Chạy thực nghiệm instances và xuất file LaTeX vào thư mục Result_level_solver_random."""
    n_values = [5, 6, 7, 8, 9, 10]
    V_values = [6, 12, 24, 48]
    instances_per_config = 20  # Tổng số cấu hình nhân instances

    table_rows = []

    print("Bắt đầu chạy thực nghiệm xác thực LevelSolver (Cấu trúc Dataclass)...")
    print("-" * 65)
    print(f"{'n':<5} | {'Demand V':<10} | {'Avg Paths':<12} | {'Avg Cells':<12} | {'Time (s)':<10}")
    print("-" * 65)

    for n in n_values:
        for V in V_values:
            total_paths = 0.0
            total_cells = 0.0
            total_time = 0.0

            for _ in range(instances_per_config):
                # Khởi tạo mạng ngẫu nhiên với class Network
                network = generate_random_network(n, density=0.4)

                start_time = time.time()
                # Gọi level_solver trả về đối tượng LevelResult (truy cập bằng dấu chấm .)
                res = level_solver(network, alpha, V)
                elapsed = time.time() - start_time

                total_paths += len(res.paths)
                total_cells += res.cells_solved
                total_time += elapsed

            avg_paths = total_paths / instances_per_config
            avg_cells = total_cells / instances_per_config
            avg_time = total_time / instances_per_config

            table_rows.append({
                "n": n,
                "V": V,
                "avg_paths": avg_paths,
                "avg_cells": avg_cells,
                "time": avg_time
            })

            print(f"{n:<5} | {V:<10} | {avg_paths:<12.2f} | {avg_cells:<12.2f} | {avg_time:<10.4f}")

    print("-" * 65)

    # Tạo thư mục đầu ra nếu chưa tồn tại
    output_dir = "Result_level_solver_random"
    os.makedirs(output_dir, exist_ok=True)

    # Nội dung mã LaTeX chuẩn cho bảng kết quả
    latex_content = r"""\begin{table}[h]
\centering
\caption{Verification and performance of \textsc{LevelSolver} via capacity threshold lattice sweeping.}
\label{kq1}
\begin{tabular}{crrrr}
\toprule
$n$ & Demand $V$ & Avg Paths & Avg Cells & Time (s) \\
\midrule
"""
    for row in table_rows:
        latex_content += f"{row['n']} & {row['V']} & {row['avg_paths']:.2f} & {row['avg_cells']:.2f} & {row['time']:.4f} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Ghi file LaTeX vào thư mục vừa tạo
    file_path = os.path.join(output_dir, "table_kq1.tex")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"\n[Thông báo] Đã xuất thành công mã LaTeX vào: {file_path}")


if __name__ == "__main__":
    random.seed(42)
    run_level_solver_verification_and_export(alpha=0.5)
