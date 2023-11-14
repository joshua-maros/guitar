use std::f64::consts::PI;

use scratchpad::{Canvas, FloatExt, Vec2, WaitUntilClick};

const GUITAR_STRING_POINTS: usize = 4096;
const GUITAR_STRING_LENGTH: f64 = 0.65;
const GUITAR_STRING_DENSITY: f64 = 5.25e-3;
const GUITAR_STRING_TENSION: f64 = 60.0;

#[derive(Clone, Copy, Default)]
struct StringPoint {
    displacement: f64,
    velocity: f64,
}

struct GuitarString {
    points: [StringPoint; GUITAR_STRING_POINTS],
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

fn main() {
    let mut string = GuitarString {
        points: [StringPoint {
            displacement: 0.0,
            velocity: 0.0,
        }; GUITAR_STRING_POINTS],
    };

    // for i in 0..GUITAR_STRING_POINTS {
    //     let x = i as f64 / (GUITAR_STRING_POINTS - 1) as f64;
    //     string.points[i].displacement = 0.8 * (-((x - 0.5) * 50.0).powi(2)).exp();
    // }

    let mut canvas = Canvas::new(512, 512);
    for step in 0..5_000_000 {
        let dt = 0.000_02;
        let t = step as f64 / 100.0 * dt;

        if step % 1000 == 0 {
            canvas.clear(0.0);
            let mut points = Vec::new();
            // Converts the guitar string's points into a series of coordinates to be displayed on the screen.
            for (i, &p) in string.points.iter().enumerate() {
                let x = i as f64 / (GUITAR_STRING_POINTS - 1) as f64;
                let y = p.displacement;
                // let y = displacement_derivative[i];
                // let y = displacement_curvature[i];
                points.push(Vec2::new(x as f32, y.map_range(-1.0..1.0, 1.0..0.0) as f32));
            }
            canvas.draw_path(&points, 1.0);
            canvas.show();
            if canvas.is_mouse_down() {
                canvas.wait_until_click();
            }
        }

        // Computes the first derivative of displacement according to position
        let mut displacement_derivative = [0.0; GUITAR_STRING_POINTS];
        for i in 1..GUITAR_STRING_POINTS - 1 {
            displacement_derivative[i] = 0.5
                * (string.points[i + 1].displacement - string.points[i - 1].displacement)
                / (GUITAR_STRING_LENGTH / (GUITAR_STRING_POINTS - 1) as f64);
        }
        let mut filtered = [0.0; GUITAR_STRING_POINTS];
        for i in 1..GUITAR_STRING_POINTS - 1 {
            filtered[i] = 0.25
                * (displacement_derivative[i + 1]
                    + 2.0 * displacement_derivative[i]
                    + displacement_derivative[i - 1]);
        }
        displacement_derivative = filtered;

        // Computes the curvature of displacement according to position
        let mut displacement_curvature = [0.0; GUITAR_STRING_POINTS];
        for i in 1..GUITAR_STRING_POINTS - 1 {
            displacement_curvature[i] = 0.5
                * (displacement_derivative[i + 1] - displacement_derivative[i - 1])
                / (GUITAR_STRING_LENGTH / (GUITAR_STRING_POINTS - 1) as f64);
        }
        displacement_curvature[0] = displacement_curvature[2];
        displacement_curvature[1] = displacement_curvature[2];
        displacement_curvature[GUITAR_STRING_POINTS - 1] =
            displacement_curvature[GUITAR_STRING_POINTS - 3];
        displacement_curvature[GUITAR_STRING_POINTS - 2] =
            displacement_curvature[GUITAR_STRING_POINTS - 3];

        // Applies various forces to the string.
        for (i, p) in string.points.iter_mut().enumerate() {
            let x = i as f64 / (GUITAR_STRING_POINTS - 1) as f64;
            let finger_force = pluck_force(x, t) / GUITAR_STRING_DENSITY;
            // let finger_force = 0.0;
            let tension_force = GUITAR_STRING_TENSION * displacement_curvature[i];
            p.velocity += (finger_force + tension_force) * dt;
        }

        // Applies the velocity of the string to itself.
        for p in string.points.iter_mut() {
            p.displacement += p.velocity * dt;
        }

        string.points[0] = StringPoint::default();
        string.points[GUITAR_STRING_POINTS - 1] = StringPoint::default();
    }
    canvas.wait_until_click();
}
