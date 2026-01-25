"""Scan depolarizing noise for multiple radii of the max-rate HaPPY code."""

from __future__ import annotations

import numpy as np

from LEGO_HQEC.OperatorPush.Presets.HaPPY_code import setup_max_rate_happy
from LEGO_HQEC.OperatorPush.PushingToolbox import batch_push
from LEGO_HQEC.QuDec.InputProcessor import extract_logicals_from_result_dict, extract_stabilizers_from_result_dict
from LEGO_HQEC.QuDec.TN_decoder import tn_quantum_error_correction_decoder_multiprocess


def scan_success_rates(radii: list[int], p_start: float, p_end: float, p_step: float, trials: int = 20,
                       rx: float = 1 / 3, rz: float = 1 / 3, n_process: int = 1) -> None:
    """Print success-rate curves for multiple code radii under depolarizing noise."""

    ry = 1 - rx - rz
    p_values = np.arange(p_start, p_end + p_step, p_step)

    for radius in radii:
        tensor_list = setup_max_rate_happy(R=radius)
        push_results = batch_push(tensor_list)
        stabilizers = extract_stabilizers_from_result_dict(push_results)
        logical_zs, logical_xs = extract_logicals_from_result_dict(push_results)

        print(f"\nScanning max-rate HaPPY (R={radius}) with {len(logical_xs)} logical qubits:")
        for p_depo in p_values:
            success_rate = tn_quantum_error_correction_decoder_multiprocess(
                tensor_list=tensor_list,
                p=float(p_depo),
                rx=rx,
                ry=ry,
                rz=rz,
                N=trials,
                n_process=n_process,
                stabilizers=stabilizers,
                logical_xs=logical_xs,
                logical_zs=logical_zs,
            )
            print(f"  p={p_depo:.2f} -> success rate {success_rate:.3f}")


if __name__ == "__main__":
    scan_success_rates(radii=[0, 1], p_start=0.00, p_end=0.30, p_step=0.05, trials=100, n_process=8)
