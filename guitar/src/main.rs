mod constants;
mod guitar_string;

use std::{f32::consts::TAU, f64::consts::PI, fs::File};

use constants::{MAX_SIMULATION_TIME, SCREEN_REFRESH_INTERVAL};
use guitar_string::GuitarString;
use itertools::Itertools;
use scratchpad::{Canvas, FloatExt, Vec2, WaitUntilClick};
use wav::{BitDepth, Header};

use crate::constants::DELTA_TIME;

fn save_audio_to_file(data: &[f64]) {
    let max = data.iter().fold(0.0f64, |a, &b| a.max(b));
    let normalized = data.iter().map(|&x| (x / max) as f32).collect_vec();

    let rate = 1.0 / DELTA_TIME;
    let rate = rate as u32;
    println!("{rate}Hz");
    let header = Header::new(wav::WAV_FORMAT_IEEE_FLOAT, 1, rate, 32);
    let mut file = File::create("audio.wav").unwrap();
    let data = BitDepth::ThirtyTwoFloat(normalized);
    wav::write(header, &data, &mut file).unwrap();
    println!("Audio saved to audio.wav");
}

fn update_window(canvas: &mut Canvas<f32>, string: &GuitarString) {
    canvas.clear(0.0);
    string.draw(canvas);
    canvas.show();
    // Pause if the mouse is pressed.
    if canvas.is_mouse_down() {
        canvas.wait_until_click();
    }
}

fn curvature(points: [f32; 3], d: f32) -> f32 {
    (points[0] - 2.0 * points[1] + points[2]) / (d * d)
}

fn fourth_derivative(points: [f32; 5], d: f32) -> f32 {
    (points[0] - 4.0 * points[1] + 6.0 * points[2] - 4.0 * points[3] + points[4]) / (d * d * d * d)
}

fn get_points<const N: usize>(
    canvas: &Canvas<f32>,
    center: [usize; 2],
    offsets: [[isize; 2]; N],
) -> [f32; N] {
    let mut points = [0.0; N];
    for (i, offset) in offsets.iter().enumerate() {
        let x = (center[0] as isize + offset[0]).rem_euclid(canvas.width() as isize);
        let y = (center[1] as isize + offset[1]).rem_euclid(canvas.height() as isize);
        points[i] = canvas.get_pixel(x as usize, y as usize);
    }
    points
}

fn main2() {
    const SIZE: usize = 256;
    let mut displacement = Canvas::new(SIZE, SIZE);
    let mut velocity = Canvas::new(SIZE, SIZE);
    let mut debug = Canvas::new(SIZE, SIZE);

    displacement.clear(0.0);
    velocity.clear(0.0);

    displacement.shade(|point| {
        // let len = (point - Vec2::new(0.5, 0.5)).length();
        // (-len.powi(2) / 0.01).exp()
        ((point.x - 0.5) * TAU).sin() * ((point.y - 0.5) * TAU).sin() * 0.3
    });

    let d = 1.0 / SIZE as f32;
    let dt = 1e-3;

    for frame in 0..1_000_000 {
        for y in 0..SIZE {
            for x in 0..SIZE {
                let x_points = get_points(
                    &displacement,
                    [x, y],
                    [[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]],
                );
                let x_partial = fourth_derivative(x_points, d);
                let y_points = get_points(
                    &displacement,
                    [x, y],
                    [[0, -2], [0, -1], [0, 0], [0, 1], [0, 2]],
                );
                let y_partial = fourth_derivative(y_points, d);

                let mixed_a_points =
                    get_points(&displacement, [x, y], [[-1, -1], [0, -1], [1, -1]]);
                let mixed_a = curvature(mixed_a_points, d);
                let mixed_b_points = get_points(&displacement, [x, y], [[-1, 0], [0, 0], [1, 0]]);
                let mixed_b = curvature(mixed_b_points, d);
                let mixed_c_points = get_points(&displacement, [x, y], [[-1, 1], [0, 1], [1, 1]]);
                let mixed_c = curvature(mixed_c_points, d);

                let mixed = curvature([mixed_a, mixed_b, mixed_c], d);

                let acceleration = -1e-6 * (x_partial + 2.0 * mixed + y_partial);
                let new_velocity = velocity.get_pixel(x, y) + acceleration * dt;
                velocity.set_pixel(x, y, new_velocity);
                let new_displacement =
                    displacement.get_pixel(x, y) + (new_velocity - 0.5 * acceleration * dt) * dt;
                displacement.set_pixel(x, y, new_displacement);

                debug.set_pixel(x, y, x_partial / 10_000.0);
            }
        }
        if frame % 10 == 0 {
            println!("Frame {}", frame);
            displacement.show();
            // debug.show();
            // (&mut displacement, &mut debug).wait_until_click();
        }
    }
}

fn main() {
    main2();

    let mut string = GuitarString::new();

    let mut window = Canvas::new(512, 512);
    let mut step = 0;
    let mut audio_data = Vec::new();

    loop {
        let t = step as f64 * DELTA_TIME;
        if t > MAX_SIMULATION_TIME {
            break;
        }

        step += 1;
        if step % SCREEN_REFRESH_INTERVAL == 0 {
            println!("{t:.03}s");
            update_window(&mut window, &string);
        }

        string.update(t);
        audio_data.push(string.pickup(0.5));
    }

    save_audio_to_file(&audio_data);
}
