mod constants;
mod guitar_string;

use std::{f32::consts::TAU, f64::consts::PI, fs::File};

use constants::{MAX_SIMULATION_TIME, SCREEN_REFRESH_INTERVAL};
use guitar_string::GuitarString;
use itertools::Itertools;
use scratchpad::{Canvas, FloatExt, Vec2, WaitUntilClick};
use sprs::{CsMat, TriMat};
use sprs_ldl::Ldl;
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

fn shape_fn(diff: f32) -> f32 {
    (1.0 - diff.abs()).max(0.0)
}

fn shape_fn_d1(diff: f32) -> f32 {
    if diff.abs() <= 1.0 {
        -diff.signum()
    } else {
        0.0
    }
}

type FLOAT = f64;

fn int_two_shape_fns(center1: f64, center2: f64) -> f64 {
    let difference = (center1 - center2).abs();
    if difference < 1.0 {
        0.5 * difference.powi(3) - difference.powi(2) + 2.0 / 3.0
    } else if difference < 2.0 {
        -1.0 / 6.0 * (difference - 2.0).powi(3)
    } else {
        0.0
    }
}

fn int_two_shape_fn_d1s(center1: f64, center2: f64) -> f64 {
    let difference = (center1 - center2).abs();
    if difference < 1.0 {
        difference.map_range(0.0..1.0, 2.0..-1.0)
    } else if difference < 2.0 {
        difference.map_range(1.0..2.0, -1.0..0.0)
    } else {
        0.0
    }
}

fn main2() {
    let mut window = Canvas::new(512, 512);

    const NUM_NODES: usize = 200;
    let mut a = [0.0; NUM_NODES];
    let mut b = [0.0; NUM_NODES];
    for i in 0..NUM_NODES {
        let x = i as FLOAT / (NUM_NODES - 1) as FLOAT - 0.4;
        let v = (-x * x / 0.02).exp();
        a[i] = v;
        b[i] = v;
    }
    let dt = 1.0;
    let dt2 = dt * dt;
    let mut frame = 0;
    loop {
        let mut matrix = [[0.0; NUM_NODES]; NUM_NODES];
        let mut vector = [0.0; NUM_NODES];
        for test_fn in 0..NUM_NODES {
            let test_fn_center = test_fn as FLOAT;
            for shape_fn in 0..NUM_NODES {
                let shape_fn_center = shape_fn as FLOAT;
                matrix[test_fn][shape_fn] +=
                    int_two_shape_fns(test_fn_center, shape_fn_center) / dt2;
                matrix[test_fn][shape_fn] += int_two_shape_fn_d1s(test_fn_center, shape_fn_center);
                vector[test_fn] +=
                    b[shape_fn] * 2.0 / dt2 * int_two_shape_fns(test_fn_center, shape_fn_center);
                vector[test_fn] -=
                    a[shape_fn] / dt2 * int_two_shape_fns(test_fn_center, shape_fn_center);
            }
        }
        for index in 0..NUM_NODES {
            matrix[0][index] = 0.0;
            matrix[NUM_NODES - 1][index] = 0.0;
            matrix[index][0] = 0.0;
            matrix[index][NUM_NODES - 1] = 0.0;
        }
        matrix[0][0] = 1.0;
        matrix[NUM_NODES - 1][NUM_NODES - 1] = 1.0;
        vector[0] = b[0];
        vector[NUM_NODES - 1] = b[NUM_NODES - 1];

        let mut smat = TriMat::new((NUM_NODES, NUM_NODES));
        for row in 0..NUM_NODES {
            for col in 0..NUM_NODES {
                let val = matrix[row][col];
                if val != 0.0 {
                    smat.add_triplet(row, col, val);
                }
            }
        }
        let smat = smat.to_csc::<usize>();
        let solver = Ldl::new().numeric(smat.view()).unwrap();
        let vector = solver.solve(&vector[..]);

        if frame % 100 == 0 {
            println!("frame {}", frame);
            window.clear(0.0);
            let mut nodes = Vec::new();
            for i in 0..NUM_NODES {
                let x = i as f32 / (NUM_NODES - 1) as f32;
                let y = b[i].map_range(-1.0..1.0, 1.0..0.0) as f32;
                nodes.push(Vec2::new(x, y));
            }
            window.draw_path(&nodes, 1.0);
            window.show();
        }
        frame += 1;

        a = b;
        b = vector.try_into().unwrap();
    }
}

fn main3() {
    let mut window = Canvas::new(512, 512);

    const NUM_NODES: usize = 200;
    let mut disp = [0.0; NUM_NODES];
    let mut vel = [0.0; NUM_NODES];
    for i in 0..NUM_NODES {
        let x = i as FLOAT / (NUM_NODES - 1) as FLOAT - 0.4;
        let v = (-x * x / 0.02).exp();
        disp[i] = v;
    }
    let dt = 1.0;
    let mut frame = 0;
    loop {
        for i in 1..NUM_NODES - 1 {
            let left = disp[i - 1];
            let center = disp[i];
            let right = disp[i + 1];
            let force = left - 2.0 * center + right;
            vel[i] += force * dt;
        }
        for i in 1..NUM_NODES - 1 {
            disp[i] += vel[i] * dt;
        }

        if frame % (if frame > 1_000_000 { 5 } else { 1000 }) == 0 {
            println!("frame {}", frame);
            window.clear(0.0);
            let mut nodes = Vec::new();
            for i in 0..NUM_NODES {
                let x = i as f32 / (NUM_NODES - 1) as f32;
                let y = disp[i].map_range(-1.0..1.0, 1.0..0.0) as f32;
                nodes.push(Vec2::new(x, y));
            }
            window.draw_path(&nodes, 1.0);
            window.show();
        }
        frame += 1;
    }
}

fn main() {
    main2();
    return;

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
