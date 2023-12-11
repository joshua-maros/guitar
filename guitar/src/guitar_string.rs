use std::f64::consts::PI;

use scratchpad::{Canvas, Vec2, FloatExt};

use crate::constants::{
    DELTA_TIME, GUITAR_STRING_DENSITY, GUITAR_STRING_LENGTH, GUITAR_STRING_POINTS,
    GUITAR_STRING_TENSION, INTERNAL_DISSIPATIVE_TERM_OF_VISCOELASTIC_TYPE,
};

#[derive(Clone, Copy, Default)]
struct StringPoint {
    displacement: f64,
    velocity: f64,
    current_curvature: f64,
    previous_curvature: f64,
}

pub struct GuitarString {
    points: [StringPoint; GUITAR_STRING_POINTS],
}

impl GuitarString {
    pub fn new() -> Self {
        Self {
            points: [StringPoint::default(); GUITAR_STRING_POINTS],
        }
    }

    /// Returns a value proportional to how much current a guitar pickup at the
    /// specified location would give back. 0.0 indicates the top of the string
    /// while 1.0 indicates the bottom.
    pub fn pickup(&self, x: f64) -> f64 {
        let i = (x * (GUITAR_STRING_POINTS - 1) as f64).round() as usize;
        self.points[i].velocity
    }

    fn update_curvature(&mut self) {
        for i in 1..GUITAR_STRING_POINTS - 1 {
            self.points[i].current_curvature = (self.points[i + 1].displacement
                + self.points[i - 1].displacement
                - 2.0 * self.points[i].displacement)
                / (GUITAR_STRING_LENGTH / (GUITAR_STRING_POINTS - 1) as f64).powi(2);
        }
        self.points[0].current_curvature = self.points[1].current_curvature;
        self.points[GUITAR_STRING_POINTS - 1].current_curvature =
            self.points[GUITAR_STRING_POINTS - 2].current_curvature;
    }

    fn update_velocity(&mut self, current_time: f64) {
        for i in 0..GUITAR_STRING_POINTS {
            let x = i as f64 / (GUITAR_STRING_POINTS - 1) as f64;
            let finger_force = pluck_force(x, current_time);
            let tension_force = GUITAR_STRING_TENSION * self.points[i].current_curvature;
            let curvature_dt =
                (self.points[i].current_curvature - self.points[i].previous_curvature) / DELTA_TIME;
            let damping_force = INTERNAL_DISSIPATIVE_TERM_OF_VISCOELASTIC_TYPE
                * GUITAR_STRING_TENSION
                * curvature_dt;
            self.points[i].velocity +=
                (finger_force + tension_force + damping_force) / GUITAR_STRING_DENSITY * DELTA_TIME;
        }
    }

    fn update_displacement(&mut self) {
        for p in self.points.iter_mut() {
            p.displacement += p.velocity * DELTA_TIME;
        }
    }

    pub fn update(&mut self, current_time: f64, end_height: f64) {
        self.update_curvature();
        self.update_velocity(current_time);
        self.update_displacement();
        self.points[0] = StringPoint::default();
        self.points[GUITAR_STRING_POINTS - 1] = StringPoint::default();
        self.points[GUITAR_STRING_POINTS - 1].displacement = end_height;
        for i in 0..GUITAR_STRING_POINTS {
            self.points[i].previous_curvature = self.points[i].current_curvature;
        }
    }

    pub fn end_tension(&self) -> f64 {
        let diff = self.points[GUITAR_STRING_POINTS - 1].displacement
            - self.points[GUITAR_STRING_POINTS - 2].displacement;
        GUITAR_STRING_TENSION * diff / (GUITAR_STRING_LENGTH / (GUITAR_STRING_POINTS - 1) as f64)
    }

    pub fn draw(&self, onto: &mut Canvas<f32>) {
        let mut points = Vec::new();
        // Converts the guitar string's points into a series of coordinates to be displayed on the screen.
        for (i, &p) in self.points.iter().enumerate() {
            let x = i as f64 / (GUITAR_STRING_POINTS - 1) as f64;
            let y = p.displacement;
            points.push(Vec2::new(
                x as f32,
                y.map_range(-0.01..0.01, 1.0..0.0) as f32,
            ));
        }
        onto.draw_path(&points, 1.0);
    }
}

fn pluck_force(x: f64, t: f64) -> f64 {
    let time_component = if t < 0.0 || t > 0.015 {
        0.0
    } else {
        (1.0 - (t * PI / 0.015).cos()) / 2.0
    };
    let space_component = (-(50.0 * (x - 0.2)).powi(2)).exp();
    time_component * space_component * 100.0
}
