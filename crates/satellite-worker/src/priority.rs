//! Priority calculation for job scheduling.

use satellite_lockfree::Priority;

/// Calculates job priority based on branch metrics.
pub struct PriorityCalculator {
    /// Thresholds for priority levels.
    thresholds: [f64; 3],
}

impl PriorityCalculator {
    /// Creates a new priority calculator.
    pub fn new() -> Self {
        Self {
            thresholds: [0.8, 0.5, 0.2], // High, MediumHigh, MediumLow
        }
    }

    /// Calculates priority from confidence score.
    pub fn from_confidence(&self, confidence: f64) -> Priority {
        if confidence >= self.thresholds[0] {
            Priority::High
        } else if confidence >= self.thresholds[1] {
            Priority::MediumHigh
        } else if confidence >= self.thresholds[2] {
            Priority::MediumLow
        } else {
            Priority::Low
        }
    }

    /// Calculates confidence from branch metrics.
    ///
    /// confidence = f(jobs_completed, success_rate, time_elapsed)
    pub fn calculate_confidence(
        &self,
        jobs_completed: u64,
        success_rate: f64,
        time_elapsed_ms: u64,
    ) -> f64 {
        // Decay older branches
        let time_factor = 1.0 / (1.0 + (time_elapsed_ms as f64 / 10000.0));

        // Reward high success rates
        let success_factor = success_rate.powi(2);

        // Small bonus for more completed jobs (exploration)
        let completion_factor = (jobs_completed as f64).ln_1p() / 10.0;

        (time_factor * 0.4 + success_factor * 0.5 + completion_factor * 0.1).clamp(0.0, 1.0)
    }
}

impl Default for PriorityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_from_confidence() {
        let calc = PriorityCalculator::new();
        assert_eq!(calc.from_confidence(0.9), Priority::High);
        assert_eq!(calc.from_confidence(0.6), Priority::MediumHigh);
        assert_eq!(calc.from_confidence(0.3), Priority::MediumLow);
        assert_eq!(calc.from_confidence(0.1), Priority::Low);
    }
}
