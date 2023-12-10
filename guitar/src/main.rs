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

// fn int_two_shape_fns(center1: f64, center2: f64) -> f64 {
//     let difference = (center1 - center2).abs();
//     if difference < 1.0 {
//         0.5 * difference.powi(3) - difference.powi(2) + 2.0 / 3.0
//     } else if difference < 2.0 {
//         -1.0 / 6.0 * (difference - 2.0).powi(3)
//     } else {
//         0.0
//     }
// }

// fn int_two_shape_fn_d1s(center1: f64, center2: f64) -> f64 {
//     let difference = (center1 - center2).abs();
//     if difference < 1.0 {
//         difference.map_range(0.0..1.0, 2.0..-1.0)
//     } else if difference < 2.0 {
//         difference.map_range(1.0..2.0, -1.0..0.0)
//     } else {
//         0.0
//     }
// }

fn int_two_shape_fns(center1: FLOAT, center2: FLOAT) -> FLOAT {
    let difference = (center1 - center2).abs();
    if difference < 1.0 {
        0.5 * difference.powi(3) - difference.powi(2) + 2.0 / 3.0
    } else if difference < 2.0 {
        -1.0 / 6.0 * (difference - 2.0).powi(3)
    } else {
        0.0
    }
}

fn int_two_shape_fn_d1s(center1: FLOAT, center2: FLOAT) -> FLOAT {
    let difference = (center1 - center2).abs();
    if difference < 1.0 {
        difference.map_range(0.0..1.0, 2.0..-1.0)
    } else if difference < 2.0 {
        difference.map_range(1.0..2.0, -1.0..0.0)
    } else {
        0.0
    }
}

fn int_two_hermite_polys(poly_type_1: usize, poly_type_2: usize) -> FLOAT {
    const RESULTS: [[FLOAT; 4]; 4] = [
        [26.0 / 35.0, 22.0 / 105.0, 9.0 / 35.0, -13.0 / 105.0],
        [22.0 / 105.0, 8.0 / 105.0, 13.0 / 105.0, -2.0 / 35.0],
        [9.0 / 35.0, 13.0 / 105.0, 26.0 / 35.0, -22.0 / 105.0],
        [-13.0 / 105.0, -2.0 / 35.0, -22.0 / 105.0, 8.0 / 105.0],
    ];
    0.5 * RESULTS[poly_type_1][poly_type_2]
}

fn int_two_hermite_poly_d1s(poly_type_1: usize, poly_type_2: usize) -> FLOAT {
    const RESULTS: [[FLOAT; 4]; 4] = [
        [3.0 / 5.0, 1.0 / 10.0, -3.0 / 5.0, 1.0 / 10.0],
        [1.0 / 10.0, 4.0 / 15.0, -1.0 / 10.0, -1.0 / 15.0],
        [-3.0 / 5.0, -1.0 / 10.0, 3.0 / 5.0, -1.0 / 10.0],
        [1.0 / 10.0, -1.0 / 15.0, -1.0 / 10.0, 4.0 / 15.0],
    ];
    RESULTS[poly_type_1][poly_type_2]
}

fn main2() {
    let mut window = Canvas::new(512, 512);

    const NUM_NODES: usize = 101;
    const NUM_PARAMS: usize = NUM_NODES * 2;
    let mut a = [0.0; NUM_PARAMS];
    for i in 0..NUM_NODES {
        let x = i as FLOAT / (NUM_NODES - 1) as FLOAT - 0.5;
        let v = (-x * x / 0.05).exp();
        let d = -40.0 * x * (-x * x / 0.05).exp() / (NUM_NODES as FLOAT);
        a[i * 2] = v;
        a[i * 2 + 1] = d;
    }
    let mut b = a.clone();
    let mut c = a.clone();
    let dt = 0.1;
    let dt2 = dt * dt;
    let mut frame = 0;

    // window.draw_graph(Vec2::ZERO, Vec2::ONE, 1.0, |x| {
    //     let x = x as FLOAT;
    //     let x = x.map_range(0.0..1.0, -1.0..1.0);
    //     let i = 0;
    //     let y1 = a[i] * 0.25 * (2.0 - 3.0 * x + x.powi(3));
    //     let y2 = a[i + 1] * 0.25 * (1.0 - x - x.powi(2) + x.powi(3));
    //     let y3 = a[i + 2] * 0.25 * (2.0 + 3.0 * x - x.powi(3));
    //     let y4 = a[i + 3] * 0.25 * (-1.0 - x + x.powi(2) + x.powi(3));
    //     (y1 + y2 + y3 + y4) as f32
    // });
    // window.show();
    // window.wait_until_click();

    let mut matrix = [[0.0; NUM_PARAMS]; NUM_PARAMS];
    for element in 0..NUM_NODES - 1 {
        for (test_fn_kind, shape_fn_kind) in (0..4).cartesian_product(0..4) {
            let test_fn_node = element * 2 + test_fn_kind;
            let shape_fn_node = element * 2 + shape_fn_kind;
            matrix[test_fn_node][shape_fn_node] +=
                int_two_hermite_polys(test_fn_kind, shape_fn_kind) / dt2;
            // matrix[test_fn_node][shape_fn_node] +=
            //     int_two_hermite_poly_d1s(test_fn_kind, shape_fn_kind);
        }
    }
    // println!("{:#?}", matrix);
    // return;
    // for index in 0..NUM_PARAMS {
    //     for p in 0..1 {
    //         matrix[p][index] = 0.0;
    //         matrix[NUM_PARAMS - 2 + p][index] = 0.0;
    //         matrix[index][p] = 0.0;
    //         matrix[index][NUM_PARAMS - 2 + p] = 0.0;
    //     }
    // }
    // for p in 0..1 {
    //     matrix[p][p] = 1.0;
    //     matrix[NUM_PARAMS - 2 + p][NUM_PARAMS - 2 + p] = 1.0;
    // }

    let mut smat = TriMat::new((NUM_PARAMS, NUM_PARAMS));
    for row in 0..NUM_PARAMS {
        for col in 0..NUM_PARAMS {
            let val = matrix[row][col];
            if val != 0.0 {
                smat.add_triplet(row, col, val);
            }
        }
    }
    let smat = smat.to_csc::<usize>();
    let solver = Ldl::new().numeric(smat.view()).unwrap();

    loop {
        let mut vector = [0.0; NUM_PARAMS];
        for element in 0..NUM_NODES - 1 {
            for (test_fn_kind, shape_fn_kind) in (0..4).cartesian_product(0..4) {
                let test_fn_node = element * 2 + test_fn_kind;
                let shape_fn_node = element * 2 + shape_fn_kind;
                vector[test_fn_node] += c[shape_fn_node] * 2.0 / dt2
                    * int_two_hermite_polys(test_fn_kind, shape_fn_kind);
                vector[test_fn_node] -= b[shape_fn_node] * 1.0 / dt2
                    * int_two_hermite_polys(test_fn_kind, shape_fn_kind);
                vector[test_fn_node] -=
                    c[shape_fn_node] * int_two_hermite_poly_d1s(test_fn_kind, shape_fn_kind);
            }
        }
        // for p in 0..1 {
        //     vector[p] = b[p];
        //     vector[NUM_PARAMS - 2 + p] = b[NUM_PARAMS - 2 + p];
        // }

        let vector = solver.solve(&vector[..]);
        // println!("{:.4?}", c);
        // println!("{:.4?}", vector);

        if frame % 1_000 == 0 {
            println!("frame {}", frame);
            window.clear(0.0);
            let mut nodes = Vec::new();
            for i in 0..NUM_NODES {
                let x = i as f32 / (NUM_NODES - 1) as f32;
                let y = c[i * 2].map_range(-1.0..1.0, 1.0..0.0) as f32;
                nodes.push(Vec2::new(x, y));
            }
            window.draw_path(&nodes, 1.0);
            window.show();
            // window.wait_until_click();
        }
        frame += 1;

        b = c;
        c = vector.try_into().unwrap();
    }
}

fn main3() {
    let mut window = Canvas::new(512, 512);

    const NUM_NODES: usize = 200;
    let mut disp = [0.0; NUM_NODES];
    let mut vel = [0.0; NUM_NODES];
    for i in 0..NUM_NODES {
        let x = i as FLOAT / (NUM_NODES - 1) as FLOAT - 0.5;
        let v = (-x * x / 0.05).exp();
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

struct LabeledShapeFn {
    x_kind: usize,
    y_kind: usize,
    node_offset: usize,
}

const fn lsf(x_kind: usize, y_kind: usize, node_offset: usize) -> LabeledShapeFn {
    LabeledShapeFn {
        x_kind,
        y_kind,
        node_offset,
    }
}

fn compute_plate_matrix() {
    const SIZE: usize = 8;
    const NUM_PARAMS: usize = SIZE * SIZE * 3;
    let dt = 0.1;
    let dt2 = dt * dt;

    const FUNCTIONS: [LabeledShapeFn; 12] = [
        lsf(0, 0, 0),
        lsf(1, 0, 1),
        lsf(0, 1, 2),
        lsf(2, 0, 3),
        lsf(3, 0, 4),
        lsf(2, 1, 5),
        lsf(0, 2, SIZE * 3),
        lsf(1, 2, SIZE * 3 + 1),
        lsf(0, 3, SIZE * 3 + 2),
        lsf(2, 2, SIZE * 3 + 3),
        lsf(3, 2, SIZE * 3 + 4),
        lsf(2, 3, SIZE * 3 + 5),
    ];

    let mut matrix = vec![0.0; NUM_PARAMS * NUM_PARAMS];
    for (ely, elx) in (0..SIZE - 1).cartesian_product(0..SIZE - 1) {
        let index = (ely * SIZE + elx) * 3;
        for (test_fn, shape_fn) in (FUNCTIONS.iter()).cartesian_product(FUNCTIONS.iter()) {
            let test_fn_node = index + test_fn.node_offset;
            let shape_fn_node = index + shape_fn.node_offset;
            matrix[test_fn_node * NUM_PARAMS + shape_fn_node] +=
                int_two_hermite_polys(test_fn.x_kind, shape_fn.x_kind)
                    * int_two_hermite_polys(test_fn.y_kind, shape_fn.y_kind)
                    / dt2;
            // matrix[test_fn_node][shape_fn_node] +=
            //     int_two_hermite_poly_d1s(test_fn_kind, shape_fn_kind);
        }
    }
    fn param_index(x: usize, y: usize, param: usize) -> usize {
        (y * SIZE + x) * 3 + param
    }
    fn matrix_index(test_param: usize, shape_param: usize) -> usize {
        test_param * NUM_PARAMS + shape_param
    }
    for node_pos in 0..SIZE {
        let param_a = param_index(node_pos, 0, 0);
        let param_b = param_index(0, node_pos, 0);
        let param_c = param_index(node_pos, SIZE - 1, 0);
        let param_d = param_index(SIZE - 1, node_pos, 0);
        for param in [param_a, param_b, param_c, param_d] {
            for other_param in 0..NUM_PARAMS {
                matrix[matrix_index(param, other_param)] = 0.0;
                matrix[matrix_index(other_param, param)] = 0.0;
                matrix[matrix_index(param, param)] = 1.0;
            }
        }
    }
    let mut smat = TriMat::new((NUM_PARAMS, NUM_PARAMS));
    for row in 0..NUM_PARAMS {
        for col in 0..NUM_PARAMS {
            let val = matrix[row * NUM_PARAMS + col];
            if val != 0.0 {
                smat.add_triplet(row, col, val);
            }
        }
    }
    let smat = smat.to_csc::<usize>();
    let solver = Ldl::new().numeric(smat.view()).unwrap();

    let mut inv_image = Canvas::new(NUM_PARAMS, NUM_PARAMS);

    for param_index in 0..NUM_PARAMS {
        let mut vector = [0.0; NUM_PARAMS];
        vector[param_index] = 1.0;
        let vector = solver.solve(&vector[..]);
        for x in 0..NUM_PARAMS {
            inv_image.set_pixel(x, param_index, vector[x] as f32);
        }
    }

    inv_image.upscale_nearest(800 / inv_image.width());
    inv_image.show();
    inv_image.wait_until_click();
}

fn main4() {
    let mut window = Canvas::new(512, 512);

    const SIZE: usize = 30;
    const NUM_PARAMS: usize = SIZE * SIZE * 3;
    let mut a = vec![0.0; NUM_PARAMS];
    let transient_size = 0.03;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let i = y * SIZE + x;
            let x = x as FLOAT / (SIZE - 1) as FLOAT - 0.2;
            let y = y as FLOAT / (SIZE - 1) as FLOAT - 0.5;
            let xv = (-x * x / transient_size).exp();
            let yv = (-y * y / transient_size).exp();
            let xd =
                -(1.0 / transient_size) * x * (-x * x / transient_size).exp() / (SIZE as FLOAT);
            let yd =
                -(1.0 / transient_size) * y * (-y * y / transient_size).exp() / (SIZE as FLOAT);
            a[i * 3] = xv * yv;
            a[i * 3 + 1] = xd * yv;
            a[i * 3 + 2] = xv * yd;
        }
    }
    let mut b = a.clone();
    let dt = 0.1;
    let dt2 = dt * dt;
    let mut frame = 0;

    // window.draw_graph(Vec2::ZERO, Vec2::ONE, 1.0, |x| {
    //     let x = x as FLOAT;
    //     let x = x.map_range(0.0..1.0, -1.0..1.0);
    //     let i = 0;
    //     let y1 = a[i] * 0.25 * (2.0 - 3.0 * x + x.powi(3));
    //     let y2 = a[i + 1] * 0.25 * (1.0 - x - x.powi(2) + x.powi(3));
    //     let y3 = a[i + 2] * 0.25 * (2.0 + 3.0 * x - x.powi(3));
    //     let y4 = a[i + 3] * 0.25 * (-1.0 - x + x.powi(2) + x.powi(3));
    //     (y1 + y2 + y3 + y4) as f32
    // });
    // window.show();
    // window.wait_until_click();

    const FUNCTIONS: [LabeledShapeFn; 12] = [
        lsf(0, 0, 0),
        lsf(1, 0, 1),
        lsf(0, 1, 2),
        lsf(2, 0, 3),
        lsf(3, 0, 4),
        lsf(2, 1, 5),
        lsf(0, 2, SIZE * 3),
        lsf(1, 2, SIZE * 3 + 1),
        lsf(0, 3, SIZE * 3 + 2),
        lsf(2, 2, SIZE * 3 + 3),
        lsf(3, 2, SIZE * 3 + 4),
        lsf(2, 3, SIZE * 3 + 5),
    ];

    let mut matrix = vec![0.0; NUM_PARAMS * NUM_PARAMS];
    for (ely, elx) in (0..SIZE - 1).cartesian_product(0..SIZE - 1) {
        let index = (ely * SIZE + elx) * 3;
        for (test_fn, shape_fn) in (FUNCTIONS.iter()).cartesian_product(FUNCTIONS.iter()) {
            let test_fn_node = index + test_fn.node_offset;
            let shape_fn_node = index + shape_fn.node_offset;
            matrix[test_fn_node * NUM_PARAMS + shape_fn_node] +=
                int_two_hermite_polys(test_fn.x_kind, shape_fn.x_kind)
                    * int_two_hermite_polys(test_fn.y_kind, shape_fn.y_kind)
                    / dt2;
            // matrix[test_fn_node][shape_fn_node] +=
            //     int_two_hermite_poly_d1s(test_fn_kind, shape_fn_kind);
        }
    }
    fn param_index(x: usize, y: usize, param: usize) -> usize {
        (y * SIZE + x) * 3 + param
    }
    fn matrix_index(test_param: usize, shape_param: usize) -> usize {
        test_param * NUM_PARAMS + shape_param
    }
    for node_pos in 0..SIZE {
        let param_a = param_index(node_pos, 0, 0);
        let param_b = param_index(0, node_pos, 0);
        let param_c = param_index(node_pos, SIZE - 1, 0);
        let param_d = param_index(SIZE - 1, node_pos, 0);
        for param in [param_a, param_b, param_c, param_d] {
            for other_param in 0..NUM_PARAMS {
                matrix[matrix_index(param, other_param)] = 0.0;
                matrix[matrix_index(other_param, param)] = 0.0;
                matrix[matrix_index(param, param)] = 1.0;
            }
        }
    }
    // println!("{:#?}", matrix);
    // return;
    let mut smat = TriMat::new((NUM_PARAMS, NUM_PARAMS));
    for row in 0..NUM_PARAMS {
        for col in 0..NUM_PARAMS {
            let val = matrix[row * NUM_PARAMS + col];
            if val != 0.0 {
                smat.add_triplet(row, col, val);
            }
        }
    }
    let smat = smat.to_csc::<usize>();
    let solver = Ldl::new().numeric(smat.view()).unwrap();
    let mut audio = Vec::new();

    for _ in 0..100_000 {
        let mut vector = [0.0; NUM_PARAMS];
        for (ely, elx) in (0..SIZE - 1).cartesian_product(0..SIZE - 1) {
            let index = (ely * SIZE + elx) * 3;
            for (test_fn, shape_fn) in (FUNCTIONS.iter()).cartesian_product(FUNCTIONS.iter()) {
                let test_fn_node = index + test_fn.node_offset;
                let shape_fn_node = index + shape_fn.node_offset;
                vector[test_fn_node] += b[shape_fn_node] * 2.0 / dt2
                    * int_two_hermite_polys(test_fn.x_kind, shape_fn.x_kind)
                    * int_two_hermite_polys(test_fn.y_kind, shape_fn.y_kind);
                vector[test_fn_node] -= a[shape_fn_node] * 1.0 / dt2
                    * int_two_hermite_polys(test_fn.x_kind, shape_fn.x_kind)
                    * int_two_hermite_polys(test_fn.y_kind, shape_fn.y_kind);
                vector[test_fn_node] -= b[shape_fn_node]
                    * int_two_hermite_poly_d1s(test_fn.x_kind, shape_fn.x_kind)
                    * int_two_hermite_poly_d1s(test_fn.y_kind, shape_fn.y_kind);
            }
        }
        for node_pos in 0..SIZE {
            for index in [
                param_index(node_pos, 0, 0),
                param_index(0, node_pos, 0),
                param_index(node_pos, SIZE - 1, 0),
                param_index(SIZE - 1, node_pos, 0),
            ] {
                vector[index] = 0.0;
            }
        }

        let vector = solver.solve(&vector[..]);
        // println!("{:.4?}", c);
        // println!("{:.4?}", vector);

        if frame % 100 == 0 {
            println!("frame {}", frame);
            window.clear(0.0);
            window.shade(|pos| {
                let x = (pos.x * 0.999 * SIZE as f32) as usize;
                let y = (pos.y * 0.999 * SIZE as f32) as usize;
                b[(y * SIZE + x) * 3] as f32
            });
            window.show();
            // window.wait_until_click();
        }
        frame += 1;

        a = b;
        b = vector.try_into().unwrap();

        audio.push(b[param_index(SIZE / 2, SIZE / 4, 0)]);
    }

    save_audio_to_file(&audio);
}

fn main() {
    // compute_plate_matrix();
    main4();
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
