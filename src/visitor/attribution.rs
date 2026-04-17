//! Per-dimension attribution visitor producing a [`DiVector`].
//!
//! At every internal node the visitor splits the dampened contribution
//! across the dimensions that contributed to the cut probability, then
//! routes each per-dim share into either `high[d]` (when the queried
//! point lies above the subtree's bounding box on dim `d`) or
//! `low[d]` (when below). Dimensions for which the point already lies
//! inside the subtree's box receive zero contribution at this depth.
//!
//! Summing `high[d] + low[d]` per dimension at the end identifies the
//! dimensions that drove the anomaly score. The argmax of that sum
//! is the most-anomalous feature.

use crate::domain::{BoundingBox, Cut, DiVector, ensure_finite};
use crate::error::{RcfError, RcfResult};
use crate::visitor::Visitor;
use crate::visitor::scoring::{damp, normalizer, score_seen, score_unseen};

/// Visitor that produces a per-dimension [`DiVector`] attribution.
///
/// Construction validates the queried `point` (rejects `NaN` / `±∞`)
/// and pre-allocates a [`DiVector`] of matching dimensionality.
#[derive(Debug, Clone)]
pub struct AttributionVisitor {
    /// Per-dimension `(high, low)` accumulator.
    di: DiVector,
    /// Queried point, kept so we can decide which side of each
    /// internal bounding box it lies on.
    point: Vec<f64>,
    /// Tree-wide leaf-mass total used for damping and normalisation.
    total_mass: u64,
}

impl AttributionVisitor {
    /// Build a fresh attribution visitor for the queried `point`.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    /// - [`RcfError::InvalidConfig`] when `point.is_empty()`.
    pub fn new(point: Vec<f64>, total_mass: u64) -> RcfResult<Self> {
        if point.is_empty() {
            return Err(RcfError::InvalidConfig(
                "AttributionVisitor: point must not be empty".into(),
            ));
        }
        ensure_finite(&point)?;
        let dim = point.len();
        Ok(Self {
            di: DiVector::zeros(dim),
            point,
            total_mass,
        })
    }

    /// Snapshot of the in-progress attribution before the traversal
    /// completes. Useful for diagnostics; production callers consume
    /// the visitor via [`Visitor::result`].
    #[must_use]
    pub fn current(&self) -> &DiVector {
        &self.di
    }

    /// Tree-wide leaf-mass total this visitor was built with.
    #[must_use]
    pub fn total_mass(&self) -> u64 {
        self.total_mass
    }
}

impl Visitor for AttributionVisitor {
    type Output = DiVector;

    fn accept_internal(
        &mut self,
        depth: usize,
        mass: u64,
        _cut: &Cut,
        bbox: &BoundingBox,
        prob_cut: f64,
        per_dim_prob: &[f64],
    ) {
        let p = prob_cut.clamp(0.0, 1.0);
        if p <= 0.0 {
            return;
        }
        let blend = (1.0 - p) * score_seen(depth, mass) + p * score_unseen(depth, mass);
        let dampened = blend * damp(mass, self.total_mass);

        let dim = self.di.dim().min(per_dim_prob.len()).min(bbox.dim());
        for (d, &dim_prob) in per_dim_prob.iter().take(dim).enumerate() {
            if dim_prob <= 0.0 {
                continue;
            }
            // Share of the dampened contribution attributable to dim d.
            let share = dim_prob / p;
            let contribution = dampened * share;
            // Route to high vs low based on which side of the
            // subtree's bounding box the queried point sits on for
            // this dimension. Inside the box → no contribution
            // (per_dim_prob[d] would already be 0 in that case, so
            // this branch is just defensive).
            if self.point[d] > bbox.max()[d] {
                let _ = self.di.add_high(d, contribution);
            } else if self.point[d] < bbox.min()[d] {
                let _ = self.di.add_low(d, contribution);
            }
        }
    }

    fn accept_leaf(&mut self, _depth: usize, _mass: u64, _point_idx: usize) {
        // No per-dimension cut at the leaf — nothing to attribute.
    }

    fn result(self) -> DiVector {
        let norm = normalizer(self.total_mass);
        let mut di = self.di;
        if norm > 0.0 {
            // Divide each component by the normaliser so the magnitude
            // tracks the scalar anomaly score.
            let _ = di.scale(norm);
        }
        di
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests compare against intentional closed-form constants.
mod tests {
    use super::*;
    use crate::domain::BoundingBox;

    fn unit_bbox_2d() -> BoundingBox {
        let mut b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[1.0, 1.0]).unwrap();
        b
    }

    #[test]
    fn new_rejects_empty_point() {
        let err = AttributionVisitor::new(vec![], 4).unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn new_rejects_non_finite() {
        assert!(matches!(
            AttributionVisitor::new(vec![1.0, f64::NAN], 4).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn fresh_visitor_starts_zeroed() {
        let v = AttributionVisitor::new(vec![1.0, 2.0, 3.0], 4).unwrap();
        assert_eq!(v.current().total(), 0.0);
        assert_eq!(v.total_mass(), 4);
    }

    #[test]
    fn zero_prob_cut_contributes_nothing() {
        let mut v = AttributionVisitor::new(vec![0.5, 0.5], 4).unwrap();
        v.accept_internal(1, 2, &Cut::new(0, 0.5), &unit_bbox_2d(), 0.0, &[0.0, 0.0]);
        assert_eq!(v.current().total(), 0.0);
    }

    #[test]
    fn point_above_bbox_routes_to_high() {
        // Point above bbox on dim 0 only.
        let mut v = AttributionVisitor::new(vec![100.0, 0.5], 8).unwrap();
        let bbox = unit_bbox_2d();
        // Synthesise prob_cut + per_dim_prob: all of the cut
        // probability concentrated on dim 0.
        v.accept_internal(1, 2, &Cut::new(0, 0.5), &bbox, 0.5, &[0.5, 0.0]);
        let cur = v.current();
        assert!(cur.high()[0] > 0.0, "dim 0 high should accumulate");
        assert_eq!(cur.high()[1], 0.0);
        assert_eq!(cur.low()[0], 0.0);
        assert_eq!(cur.low()[1], 0.0);
    }

    #[test]
    fn point_below_bbox_routes_to_low() {
        let mut v = AttributionVisitor::new(vec![0.5, -100.0], 8).unwrap();
        let bbox = unit_bbox_2d();
        v.accept_internal(1, 2, &Cut::new(1, 0.5), &bbox, 0.5, &[0.0, 0.5]);
        let cur = v.current();
        assert!(cur.low()[1] > 0.0, "dim 1 low should accumulate");
        assert_eq!(cur.high()[1], 0.0);
        assert_eq!(cur.high()[0], 0.0);
        assert_eq!(cur.low()[0], 0.0);
    }

    #[test]
    fn argmax_identifies_anomalous_dim() {
        // Point anomalous on dim 2 (above bbox), normal otherwise.
        let mut v = AttributionVisitor::new(vec![0.5, 0.5, 100.0, 0.5], 16).unwrap();
        let mut bbox = BoundingBox::from_point(&[0.0; 4]).unwrap();
        bbox.extend(&[1.0; 4]).unwrap();
        v.accept_internal(2, 8, &Cut::new(2, 0.5), &bbox, 0.6, &[0.0, 0.0, 0.6, 0.0]);
        let di = v.result();
        assert_eq!(di.argmax(), Some(2));
    }

    #[test]
    fn result_scales_by_normalizer() {
        let mut v = AttributionVisitor::new(vec![100.0, 0.5], 4).unwrap();
        let bbox = unit_bbox_2d();
        v.accept_internal(1, 2, &Cut::new(0, 0.5), &bbox, 0.5, &[0.5, 0.0]);
        let raw_high0 = v.current().high()[0];
        let di = v.result();
        // result divides by normalizer(4) = log2(4) = 2.
        assert!((di.high()[0] - raw_high0 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn point_inside_bbox_no_contribution() {
        // Point inside bbox on every dim → no attribution even with
        // a non-zero prob_cut (which itself wouldn't happen in
        // practice but the visitor must stay defensive).
        let mut v = AttributionVisitor::new(vec![0.5, 0.5], 8).unwrap();
        let bbox = unit_bbox_2d();
        v.accept_internal(1, 2, &Cut::new(0, 0.5), &bbox, 0.4, &[0.2, 0.2]);
        assert_eq!(v.current().total(), 0.0);
    }
}
