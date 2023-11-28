mod constants;
mod guitar_string;

use std::{f64::consts::PI, fs::File};

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

fn main() {
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
