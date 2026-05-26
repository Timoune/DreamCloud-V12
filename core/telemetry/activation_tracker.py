# core/telemetry/activation_tracker.py
"""
ActivationTracker — samples the activation values of all memories
and computes aggregate statistics (mean, max, variance).  A high mean
activation indicates that the same memories are being retrieved
repeatedly, which correlates 0.0

        if starved:
            logger.info(
                f"[StarvationDetector] {len(starved)}/{len(all_ids)} "
                f"memories starved ({ratio:.1%})"
            )

        self._last_check = now
        return {
            "starved_ids": starved,
            "starved_ratio": ratio,
            "total_memories": len(all_ids),
        }