#![allow(incomplete_features)]
#![feature(specialization)]

use std::{
    collections::HashSet,
    hash::Hash,
    marker::PhantomData,
    ops::{Add, Mul, Range},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

pub use bevy_math::{Vec2, Vec3, Vec4};
use image::ImageBuffer;
use itertools::Itertools;
use lazy_static::lazy_static;
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use noise::{Fbm, NoiseFn, Perlin, Seedable, SuperSimplex};
use ordered_float::NotNan;
use palette::{FromColor, Hsv, Hsva, LinSrgb, LinSrgba, Srgba};
pub use random_context::*;

lazy_static! {
    static ref WIN_X: Arc<Mutex<isize>> = Arc::new(Mutex::new(0));
}

fn position_window(width: isize) -> isize {
    let mut win_x = WIN_X.lock().unwrap();
    let x = *win_x;
    *win_x += width;
    if *win_x > 1920 {
        *win_x = 0;
    }
    x
}

fn perlin(octaves: usize, seed: u32) -> Fbm {
    let mut p = Fbm::new();
    p.frequency = 1.0;
    p.lacunarity = 2.0;
    p.octaves = octaves;
    p.persistence = 0.6;
    p.set_seed(seed)
}

pub fn value2(point: Vec2, seed: u32) -> f32 {
    let x_cell = point.x.floor() as i32;
    let y_cell = point.x.floor() as i32;
    let aa = RandomContext::new(&(seed, x_cell, y_cell)).g01f32();
    let ab = RandomContext::new(&(seed, x_cell, y_cell + 1)).g01f32();
    let ba = RandomContext::new(&(seed, x_cell + 1, y_cell)).g01f32();
    let bb = RandomContext::new(&(seed, x_cell + 1, y_cell + 1)).g01f32();
    let x = point.x.fract();
    let y = point.y.fract();
    let a = aa + (ab - aa) * y;
    let b = ba + (bb - ba) * y;
    a + (b - a) * x
}

pub fn perlin2(point: Vec2, octaves: usize) -> f32 {
    perlin(octaves, 0).get([point.x as f64, point.y as f64]) as f32
}

pub fn perlin2x2(point: Vec2, octaves: usize) -> Vec2 {
    Vec2::new(
        perlin(octaves, 0).get([point.x as f64, point.y as f64]) as f32,
        perlin(octaves, 1).get([point.x as f64, point.y as f64]) as f32,
    )
}

pub fn perlin3(point: Vec3, octaves: usize) -> f32 {
    perlin(octaves, 0).get([point.x as f64, point.y as f64, point.z as f64]) as f32
}

pub fn perlin4(point: Vec4, octaves: usize) -> f32 {
    perlin(octaves, 0).get([
        point.x as f64,
        point.y as f64,
        point.z as f64,
        point.w as f64,
    ]) as f32
}

pub fn simplex2(mut point: Vec2, octaves: usize) -> f32 {
    let mut result = 0.0;
    let s = SuperSimplex::new();
    let mut strength = 0.6;
    for layer in 0..octaves as u32 {
        s.set_seed(layer);
        result += strength * s.get([point.x as f64, point.y as f64]);
        point *= 2.0;
        strength *= 0.6;
    }
    result as f32
}

pub fn simplex2x2(point: Vec2, octaves: usize) -> Vec2 {
    Vec2::new(simplex2(point, octaves), simplex2(point + 5000.0, octaves))
}

pub fn simplex2c(mut point: Vec2, octaves: usize, lacunarity: f32, persistence: f32) -> f32 {
    let mut result = 0.0;
    let s = SuperSimplex::new();
    let mut strength = persistence;
    for layer in 0..octaves as u32 {
        result += strength * s.set_seed(layer).get([point.x as f64, point.y as f64]) as f32;
        point *= lacunarity;
        strength *= persistence;
    }
    result
}

pub type Color = LinSrgba;
pub const RED: Color = Color {
    color: palette::rgb::Rgb {
        red: 1.0,
        green: 0.0,
        blue: 0.0,
        standard: PhantomData,
    },
    alpha: 1.0,
};
pub const GREEN: Color = Color {
    color: palette::rgb::Rgb {
        red: 0.0,
        green: 1.0,
        blue: 0.0,
        standard: PhantomData,
    },
    alpha: 1.0,
};
pub const BLUE: Color = Color {
    color: palette::rgb::Rgb {
        red: 0.0,
        green: 0.0,
        blue: 1.0,
        standard: PhantomData,
    },
    alpha: 1.0,
};
pub const WHITE: Color = Color {
    color: palette::rgb::Rgb {
        red: 1.0,
        green: 1.0,
        blue: 1.0,
        standard: PhantomData,
    },
    alpha: 1.0,
};
pub const BLACK: Color = Color {
    color: palette::rgb::Rgb {
        red: 0.0,
        green: 0.0,
        blue: 0.0,
        standard: PhantomData,
    },
    alpha: 1.0,
};

pub trait Displayable {
    fn display(self) -> LinSrgba;
}

pub trait PixelData:
    Clone + Copy + Default + Mul<f32, Output = Self> + Add<Self, Output = Self> + Displayable
{
    fn magnitude(self) -> f32;
    fn lerp(self, other: Self, factor: f32) -> Self {
        self * (1.0 - factor) + other * factor
    }
}

impl Displayable for LinSrgba {
    fn display(self) -> LinSrgba {
        (*self).into()
    }
}

impl PixelData for LinSrgba {
    fn magnitude(self) -> f32 {
        self.alpha * self.blue.max(self.green).max(self.red)
    }
}

impl Displayable for f32 {
    fn display(self) -> LinSrgba {
        LinSrgba::new(self, self, self, 1.0)
    }
}

impl PixelData for f32 {
    fn magnitude(self) -> f32 {
        self
    }
}

impl Displayable for bool {
    fn display(self) -> LinSrgba {
        let luma = if self { 1.0 } else { 0.0 };
        LinSrgba::new(luma, luma, luma, 1.0)
    }
}

impl Displayable for Vec2 {
    fn display(self) -> LinSrgba {
        LinSrgba::new(self.x, self.y, 0.0, 1.0)
    }
}

impl PixelData for Vec2 {
    fn magnitude(self) -> f32 {
        self.max_element()
    }
}

impl Displayable for Vec3 {
    fn display(self) -> LinSrgba {
        LinSrgba::new(self.x, self.y, self.z, 1.0)
    }
}

impl PixelData for Vec3 {
    fn magnitude(self) -> f32 {
        self.max_element()
    }
}

impl Displayable for usize {
    fn display(self) -> LinSrgba {
        unique_color(self as _)
    }
}
pub struct Canvas<P> {
    data: Vec<P>,
    mask: Vec<bool>,
    width: usize,
    height: usize,
    window: Option<Window>,
    framebuffer: Vec<u32>,
}

impl<P: Clone> Clone for Canvas<P> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            mask: self.mask.clone(),
            width: self.width,
            height: self.height,
            window: None,
            framebuffer: vec![],
        }
    }
}

impl Canvas<Color> {
    pub fn gamma(&mut self, gamma: f32) {
        for pixel in &mut self.data {
            pixel.red = pixel.red.powf(gamma);
            pixel.green = pixel.green.powf(gamma);
            pixel.blue = pixel.blue.powf(gamma);
        }
    }

    pub fn set_r(&mut self, data: &Canvas<f32>) {
        assert_eq!(self.width, data.width);
        assert_eq!(self.height, data.height);
        for index in 0..self.data.len() {
            self.data[index].red = data.data[index];
        }
    }

    pub fn set_g(&mut self, data: &Canvas<f32>) {
        assert_eq!(self.width, data.width);
        assert_eq!(self.height, data.height);
        for index in 0..self.data.len() {
            self.data[index].green = data.data[index];
        }
    }

    pub fn set_b(&mut self, data: &Canvas<f32>) {
        assert_eq!(self.width, data.width);
        assert_eq!(self.height, data.height);
        for index in 0..self.data.len() {
            self.data[index].blue = data.data[index];
        }
    }
}

impl Canvas<Vec3> {
    pub fn load_v3(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let image = image::open(path).unwrap();
        let image = image.as_rgb8().unwrap();
        let mut this = Self::new(image.width() as _, image.height() as _);
        let m = |c| c as f32 / 255.0;
        this.data = image
            .pixels()
            .map(|p| Vec3::new(m(p.0[0]), m(p.0[1]), m(p.0[2])))
            .collect();
        this
    }
}

impl Canvas<f32> {
    pub fn noise<const D: u8>(width: usize, height: usize, seed: &(impl Hash + ?Sized)) -> Self {
        let mut this = Self::new(width, height);
        let mut r = RandomContext::new(seed);
        this.shade(|_| r.g01f32());
        this
    }

    pub fn save_f32(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        let image = ImageBuffer::from_fn(self.width as u32, self.height as u32, |x, y| {
            let v = self.data[y as usize * self.width + x as usize];
            image::Rgb([v, v, v])
        });
        image.save(path).unwrap();
    }

    /// Only considers full-brightness pixels.
    pub fn fast_toroidal_distance_field(&mut self) {
        let mut processed = self.map_het(|x| x >= 1.0);
        let mut edge = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let processed_px =
                    processed.get_pixel(if x == self.width - 1 { 0 } else { x + 1 }, y);
                let processed_nx =
                    processed.get_pixel(if x == 0 { self.width - 1 } else { x - 1 }, y);
                let processed_py =
                    processed.get_pixel(x, if y == self.width - 1 { 0 } else { y + 1 });
                let processed_ny =
                    processed.get_pixel(x, if y == 0 { self.height - 1 } else { y - 1 });
                if !processed.get_pixel(x, y)
                    && (processed_px || processed_py || processed_nx || processed_ny)
                {
                    edge.push((x, y));
                }
            }
        }
        let d = 1.0 / self.width as f32;
        let mut dist = d;
        while edge.len() > 0 {
            for &(x, y) in &edge {
                processed.set_pixel(x, y, true);
            }
            let mut next_edge = HashSet::new();
            for (x, y) in edge {
                self.set_pixel(x, y, 1.0 - dist);
                let xx = if x == self.width - 1 { 0 } else { x + 1 };
                if !processed.get_pixel(xx, y) {
                    next_edge.insert((xx, y));
                }
                let xx = if x == 0 { self.width - 1 } else { x - 1 };
                if !processed.get_pixel(xx, y) {
                    next_edge.insert((xx, y));
                }
                let yy = if y == self.height - 1 { 0 } else { y + 1 };
                if !processed.get_pixel(x, yy) {
                    next_edge.insert((x, yy));
                }
                let yy = if y == 0 { self.height - 1 } else { y - 1 };
                if !processed.get_pixel(x, yy) {
                    next_edge.insert((x, yy));
                }
            }
            dist += d;
            edge = next_edge.into_iter().collect();
        }
    }

    /// Only considers full-brightness pixels.
    pub fn fast_distance_field(&mut self) {
        let mut processed = self.map_het(|x| x >= 1.0);
        let mut edge = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let processed_px = x >= self.width - 1 || processed.get_pixel(x + 1, y);
                let processed_nx = x < 1 || processed.get_pixel(x - 1, y);
                let processed_py = y >= self.width - 1 || processed.get_pixel(x, y + 1);
                let processed_ny = y < 1 || processed.get_pixel(x, y - 1);
                if !processed.get_pixel(x, y)
                    && (processed_px || processed_py || processed_nx || processed_ny)
                {
                    edge.push((x, y));
                }
            }
        }
        let d = 1.0 / self.width as f32;
        let mut dist = d;
        while edge.len() > 0 {
            for &(x, y) in &edge {
                processed.set_pixel(x, y, true);
            }
            let mut next_edge = HashSet::new();
            for (x, y) in edge {
                self.set_pixel(x, y, 1.0 - dist);
                if x < self.width - 1 && !processed.get_pixel(x + 1, y) {
                    next_edge.insert((x + 1, y));
                }
                if x > 0 && !processed.get_pixel(x - 1, y) {
                    next_edge.insert((x - 1, y));
                }
                if y < self.width - 1 && !processed.get_pixel(x, y + 1) {
                    next_edge.insert((x, y + 1));
                }
                if y > 0 && !processed.get_pixel(x, y - 1) {
                    next_edge.insert((x, y - 1));
                }
            }
            dist += d;
            edge = next_edge.into_iter().collect();
        }
    }
}

impl<P> Canvas<P> {
    pub fn from_raw_data(data: Vec<P>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            data,
            mask: (0..(width * height)).map(|_| true).collect(),
            width,
            height,
            window: None,
            framebuffer: vec![],
        }
    }

    pub fn raw_data(&self) -> &[P] {
        &self.data
    }
}

impl<P: Default> Canvas<P> {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: (0..(width * height)).map(|_| P::default()).collect(),
            mask: (0..(width * height)).map(|_| true).collect(),
            width,
            height,
            window: None,
            framebuffer: vec![],
        }
    }
}

impl<P: Default + Clone> Canvas<P> {
    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        self.data.resize(width * height, Default::default());
        self.mask.resize(width * height, true);
    }

    pub fn upscale_nearest(&mut self, factor: usize) {
        let old_data = std::mem::take(&mut self.data);
        self.resize(self.width * factor, self.height * factor);
        for y in 0..self.height {
            for x in 0..self.width {
                self.data[y * self.width + x] =
                    old_data[y / factor * self.width / factor + x / factor].clone();
            }
        }
    }
}

impl<P> Canvas<P> {
    pub unsafe fn new_garbage(width: usize, height: usize) -> Self {
        let mut data = Vec::with_capacity(width * height);
        data.set_len(width * height);
        let mask = vec![true; width * height];
        Self {
            data,
            mask,
            width,
            height,
            window: None,
            framebuffer: vec![],
        }
    }

    fn make_window(width: usize, height: usize) -> Window {
        let mut win = Window::new("display", width, height, WindowOptions::default()).unwrap();
        win.set_position(position_window(width as isize), 0);
        win
    }

    fn get_window(window: &mut Option<Window>, width: usize, height: usize) -> &mut Window {
        let win = window.get_or_insert_with(|| Self::make_window(width, height));
        if win.get_size() != (width, height) {
            *win = Self::make_window(width, height);
        }
        win
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn replace_mask(&mut self, mask: Canvas<bool>) -> Canvas<bool> {
        assert_eq!(self.width, mask.width);
        assert_eq!(self.height, mask.height);
        let mut mask = mask;
        std::mem::swap(&mut self.mask, &mut mask.data);
        Canvas {
            data: mask.data,
            mask: vec![true; self.width * self.height],
            width: self.width,
            height: self.height,
            window: None,
            framebuffer: vec![],
        }
    }

    pub fn clear_mask(&mut self) -> Canvas<bool> {
        self.replace_mask(Canvas::new(self.width, self.height))
    }

    pub fn slice_mask(&mut self, center: Vec2, angle: f32) -> Canvas<bool> {
        let mut mask = Canvas::new(self.width, self.height);
        mask.draw_slice_mask(center, angle);
        self.replace_mask(mask)
    }
}

impl Canvas<bool> {
    pub fn draw_slice_mask(&mut self, center: Vec2, angle: f32) {
        let normal = Vec2::new(angle.cos(), angle.sin());
        self.shade(|p| {
            let d = p - center;
            d.dot(normal) > 0.0
        })
    }
}

impl<P: Copy> Canvas<P> {
    pub fn draw_canvas(&mut self, canvas: &Self, top_left: Vec2) {
        let x = (top_left.x.rem_euclid(1.0) * (self.width as f32)) as usize;
        let y = (top_left.y.rem_euclid(1.0) * (self.height as f32)) as usize;
        let offset = y * self.width + x;
        for x in 0..canvas.width {
            for y in 0..canvas.height {
                let index = y * self.width + x + offset;
                if self.mask[index] {
                    self.data[index] = canvas.data[y * canvas.width + x];
                }
            }
        }
    }

    pub fn blend_canvas(
        &mut self,
        canvas: &Self,
        top_left: Vec2,
        mut blend: impl FnMut(P, P) -> P,
    ) {
        let x = (top_left.x.rem_euclid(1.0) * (self.width as f32)) as usize;
        let y = (top_left.y.rem_euclid(1.0) * (self.height as f32)) as usize;
        let offset = y * self.width + x;
        for x in 0..canvas.width {
            for y in 0..canvas.height {
                let index = y * self.width + x + offset;
                let old = self.data[index];
                let new = canvas.data[y * canvas.width + x];
                if self.mask[index] {
                    self.data[index] = blend(old, new);
                }
            }
        }
    }

    pub fn draw_point(&mut self, point: Vec2, color: P) {
        self.shade_point(point, |_| color)
    }

    pub fn shade_point(&mut self, point: Vec2, shader: impl FnOnce(P) -> P) {
        let x = (point.x.rem_euclid(1.0) * (self.width as f32)) as usize;
        let y = (point.y.rem_euclid(1.0) * (self.height as f32)) as usize;
        let index = y * self.width + x;
        if self.mask[index] {
            let data = &mut self.data[index];
            let new = shader(*data);
            *data = new;
        }
    }

    pub fn draw_line(&mut self, start: Vec2, end: Vec2, color: P) {
        let xl = end.x - start.x;
        let yl = end.y - start.y;
        let iterate_over_x = xl.abs() > yl.abs();
        let dx = 1.0 / self.width as f32;
        let dy = 1.0 / self.height as f32;
        if xl.abs() < dx && yl.abs() < dy {
            self.draw_point(start, color);
            return;
        }
        let dir = if if iterate_over_x { xl > 0.0 } else { yl > 0.0 } {
            1.0
        } else {
            -1.0
        };
        let (dx, dy) = if iterate_over_x {
            (dx, yl / xl * dy)
        } else {
            (xl / yl * dx, dy)
        };
        let (dx, dy) = (dir * dx, dir * dy);
        let mut x = start.x;
        let mut y = start.y;
        for _ in 0..=(if iterate_over_x {
            xl * self.width as f32
        } else {
            yl * self.height as f32
        }
        .abs()) as u32
        {
            self.draw_point(Vec2::new(x, y), color);
            x += dx;
            y += dy;
        }
    }

    pub fn draw_path(&mut self, nodes: &[Vec2], color: P) {
        for start_index in 0..nodes.len() - 1 {
            self.draw_line(nodes[start_index], nodes[start_index + 1], color);
        }
    }

    pub fn draw_graph(
        &mut self,
        start: Vec2,
        end: Vec2,
        color: P,
        mut func: impl FnMut(f32) -> f32,
    ) {
        let mut prev_y = func(0.0);
        let start_pixel_x = (start.x * self.width as f32) as i32;
        let end_pixel_x = (end.x * self.width as f32) as i32;
        let width = self.width;
        let pixel_x_to_draw_x = move |x| x as f32 / width as f32;
        let func_y_to_draw_y = move |y| (y * 0.5 + 0.5) * (start.y - end.y) + end.y;
        for pixel_x in start_pixel_x + 1..end_pixel_x {
            let x = (pixel_x - start_pixel_x) as f32 / (end_pixel_x - start_pixel_x) as f32;
            let next_y = func(x);
            self.draw_line(
                Vec2::new(pixel_x_to_draw_x(pixel_x - 1), func_y_to_draw_y(prev_y)),
                Vec2::new(pixel_x_to_draw_x(pixel_x), func_y_to_draw_y(next_y)),
                color,
            );
            prev_y = next_y;
        }
    }

    pub fn draw_rect(&mut self, start: Vec2, end: Vec2, color: P) {
        let w = end.x - start.x;
        let h = end.y - start.y;
        let dx = 1.0 / self.width as f32;
        let dy = 1.0 / self.height as f32;
        let mut y = start.y;
        for _y in 0..(h / dy) as _ {
            let mut x = start.x;
            for _x in 0..(w / dx) as _ {
                self.draw_point(Vec2::new(x, y), color);
                x += dx;
            }
            y += dy;
        }
    }

    pub fn shade_path(&mut self, nodes: &[Vec2], mut shader: impl FnMut(usize) -> P) {
        for start_index in 0..nodes.len() - 1 {
            self.draw_line(
                nodes[start_index],
                nodes[start_index + 1],
                shader(start_index),
            );
        }
    }

    pub fn shade(&mut self, mut shader: impl FnMut(Vec2) -> P) {
        let dx = 1.0 / (self.width as f32);
        let dy = 1.0 / (self.height as f32);
        let mut point = Vec2::new(dx / 2.0, dy / 2.0);
        let mut index = 0;
        let mut total_shaded = 0;
        let mut last_update = Instant::now();
        let start = Instant::now();
        for _y in 0..self.height {
            for _x in 0..self.width {
                if self.mask[index] {
                    self.data[index] = shader(point);
                }
                point.x += dx;
                index += 1;
            }
            point.y += dy;
            point.x = dx / 2.0;
            total_shaded += 1;
            if last_update.elapsed() > Duration::from_secs(1) {
                last_update = Instant::now();
                let factor = total_shaded as f32 / self.height as f32;
                println!(
                    "{:.2}% done, ETC {:.2?}",
                    factor * 100.0,
                    Duration::from_secs_f32(
                        (1.0 - factor) * start.elapsed().as_secs_f32() / factor
                    )
                );
            }
        }
        if start.elapsed() > Duration::from_secs(1) {
            println!("Took {:.2?}", start.elapsed());
        }
    }

    pub fn blend_shade(&mut self, mut shader: impl FnMut(P, Vec2) -> P) {
        let dx = 1.0 / (self.width as f32);
        let dy = 1.0 / (self.height as f32);
        let mut point = Vec2::new(dx / 2.0, dy / 2.0);
        let mut index = 0;
        for _y in 0..self.height {
            for _x in 0..self.width {
                if self.mask[index] {
                    self.data[index] = shader(self.data[index], point);
                }
                point.x += dx;
                index += 1;
            }
            point.y += dy;
            point.x = dx / 2.0;
        }
    }

    pub fn iter_all_masked_pixels(&self) -> impl Iterator<Item = &P> {
        (0..self.width * self.height)
            .into_iter()
            .filter_map(move |i| {
                if self.mask[i] {
                    Some(&self.data[i])
                } else {
                    None
                }
            })
    }

    pub fn count(&self, mut predicate: impl FnMut(&P) -> bool) -> usize {
        self.iter_all_masked_pixels()
            .filter(|p| predicate(p))
            .count()
    }

    pub fn percent_matching(&self, mut predicate: impl FnMut(&P) -> bool) -> f32 {
        self.iter_all_masked_pixels()
            .filter(|p| predicate(p))
            .count() as f32
            / self.data.len() as f32
    }

    pub fn min<O: PartialOrd>(&self, mut predicate: impl FnMut(P, Vec2) -> O) -> (P, Vec2) {
        let pos = Vec2::new(0.5 / self.width() as f32, 0.5 / self.height() as f32);
        let mut min = (self.data[0], pos);
        let mut min_value = predicate(min.0, min.1);
        for x in 0..self.width {
            for y in 0..self.height {
                let pixel = self.get_pixel(x, y);
                let pos = Vec2::new(
                    (x as f32 + 0.5) / self.width() as f32,
                    (y as f32 + 0.5) / self.height() as f32,
                );
                let predicate = predicate(pixel, pos);
                if predicate < min_value {
                    min = (pixel, pos);
                    min_value = predicate;
                }
            }
        }
        min
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> P {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        self.data[y * self.width + x]
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, p: P) {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let index = y * self.width + x;
        if self.mask[index] {
            self.data[index] = p;
        }
    }

    pub fn get_nearest(&self, point: Vec2) -> P {
        assert!(1.0 - f32::EPSILON < 1.0);
        let x = (point.x.clamp(0.0, 1.0 - f32::EPSILON) * (self.width as f32)).floor() as usize;
        let y = (point.y.clamp(0.0, 1.0 - f32::EPSILON) * (self.height as f32)).floor() as usize;
        self.data[y * self.width + x]
    }

    pub fn clear(&mut self, color: P) {
        for i in 0..self.data.len() {
            if self.mask[i] {
                self.data[i] = color;
            }
        }
    }

    pub fn map(&mut self, mut shader: impl FnMut(P) -> P) {
        for x in 0..self.width {
            for y in 0..self.height {
                let index = y * self.width + x;
                if !self.mask[index] {
                    continue;
                }
                let p = shader(self.data[index]);
                self.data[index] = p;
            }
        }
    }

    pub fn map_het<P2>(&self, shader: impl FnMut(P) -> P2) -> Canvas<P2> {
        let data = self.data.iter().copied().map(shader).collect();
        Canvas {
            data,
            mask: self.mask.clone(),
            width: self.width,
            height: self.height,
            window: None,
            framebuffer: vec![],
        }
    }

    pub fn mouse(&self) -> Vec2 {
        self.window
            .as_ref()
            .map(|w| {
                let (x, y) = w.get_mouse_pos(MouseMode::Clamp).unwrap();
                Vec2::new(x / self.width as f32, y / self.height as f32)
            })
            .unwrap_or(Vec2::new(0.0, 0.0))
    }

    pub fn is_mouse_down(&self) -> bool {
        self.window
            .as_ref()
            .map(|w| w.get_mouse_down(MouseButton::Left))
            .unwrap_or(false)
    }
}

impl<P: Copy + Displayable> Canvas<P> {
    #[track_caller]
    pub fn map_show(&mut self, mut shader: impl FnMut(P) -> P) {
        self.framebuffer = vec![0; self.width * self.height];
        for index in 0..self.width * self.height {
            let value = shader(self.data[index]);
            let color = value.display().into_format::<u8, u8>();
            let color_bits =
                (color.red as u32) << 16 | (color.green as u32) << 8 | color.blue as u32;
            self.framebuffer[index] = color_bits;
        }
        self.update();
    }

    #[track_caller]
    pub fn show(&mut self) {
        self.map_show(|x| x)
    }
}

pub trait WaitUntilClick {
    fn is_mouse_down(&self) -> bool;
    fn update(&mut self);
    fn wait_until_click(&mut self) {
        while self.is_mouse_down() {
            self.update()
        }
        while !self.is_mouse_down() {
            self.update()
        }
    }
}

impl<P: Copy + Displayable> WaitUntilClick for Canvas<P> {
    fn is_mouse_down(&self) -> bool {
        Canvas::is_mouse_down(self)
    }

    fn update(&mut self) {
        let win = Self::get_window(&mut self.window, self.width, self.height);
        win.update_with_buffer(&self.framebuffer, self.width, self.height)
            .unwrap();
        if win.is_key_down(Key::Escape) || !win.is_open() {
            panic!("Exit requested");
        }
    }
}

impl<A: WaitUntilClick, B: WaitUntilClick> WaitUntilClick for (&mut A, &mut B) {
    fn is_mouse_down(&self) -> bool {
        self.0.is_mouse_down() || self.1.is_mouse_down()
    }

    fn update(&mut self) {
        self.0.update();
        self.1.update();
    }
}

impl<A: WaitUntilClick, B: WaitUntilClick, C: WaitUntilClick> WaitUntilClick
    for (&mut A, &mut B, &mut C)
{
    fn is_mouse_down(&self) -> bool {
        self.0.is_mouse_down() || self.1.is_mouse_down() || self.2.is_mouse_down()
    }

    fn update(&mut self) {
        self.0.update();
        self.1.update();
        self.2.update();
    }
}

impl<P: PixelData> Canvas<P> {
    pub fn add_point(&mut self, point: Vec2, color: P) {
        self.shade_point(point, |old| old + color)
    }

    pub fn add_canvas(&mut self, canvas: &Self, top_left: Vec2, factor: f32) {
        let x = (top_left.x.rem_euclid(1.0) * (self.width as f32)) as usize;
        let y = (top_left.y.rem_euclid(1.0) * (self.height as f32)) as usize;
        let offset = y * self.width + x;
        for x in 0..canvas.width {
            for y in 0..canvas.height {
                self.data[y * self.width + x + offset] = self.data[y * self.width + x + offset]
                    + canvas.data[y * canvas.width + x] * factor;
            }
        }
    }

    pub fn overlay_point(&mut self, point: Vec2, color: P, alpha: f32) {
        self.shade_point(point, |old| old.lerp(color, alpha))
    }

    pub fn draw_graph_hq(
        &mut self,
        start: Vec2,
        end: Vec2,
        color: P,
        mut func: impl FnMut(f32) -> f32,
        super_resolution: usize,
    ) {
        let start_pixel_x = (start.x * self.width as f32) as i32;
        let end_pixel_x = (end.x * self.width as f32) as i32;
        let width = self.width;
        let pixel_x_to_draw_x = move |x| x as f32 / width as f32;
        let func_y_to_draw_y = move |y| (y * 0.5 + 0.5) * (start.y - end.y) + end.y;
        for pixel_x in start_pixel_x..end_pixel_x {
            let x = (pixel_x - start_pixel_x) as f32 / (end_pixel_x - start_pixel_x) as f32;
            for dx in 0..super_resolution {
                let x = x
                    + (dx as f32 + random::<f32>())
                        / super_resolution as f32
                        / (end_pixel_x - start_pixel_x) as f32;
                let next_y = func(x);
                self.add_point(
                    Vec2::new(pixel_x_to_draw_x(pixel_x), func_y_to_draw_y(next_y)),
                    color,
                );
            }
        }
    }

    pub fn shade_hq(&mut self, mut shader: impl FnMut(Vec2) -> P, super_resolution: usize) {
        let dx = 1.0 / ((self.width * super_resolution) as f32);
        let dy = 1.0 / ((self.height * super_resolution) as f32);
        let mut index = 0;
        for y in 0..self.height {
            for x in 0..self.width {
                let mut pixel = P::default();
                for sy in 0..super_resolution {
                    for sx in 0..super_resolution {
                        let x = ((x * super_resolution + sx) as f32 + 0.5) * dx;
                        let y = ((y * super_resolution + sy) as f32 + 0.5) * dy;
                        pixel = pixel + shader(Vec2::new(x, y));
                    }
                }
                self.data[index] = pixel * (1.0 / (super_resolution * super_resolution) as f32);
                index += 1;
            }
        }
    }

    pub fn shade_path_in_out(&mut self, nodes: &[Vec2], color: P, power: f32) {
        self.shade_path(nodes, |index| {
            color
                * (((index as f32 + 0.5) * 2.0 / (nodes.len() as f32))
                    .ping_pong()
                    .powf(power))
        })
    }

    pub fn shade_path_out(&mut self, nodes: &[Vec2], color: P, power: f32) {
        self.shade_path(nodes, |index| {
            color * ((1.0 - (index as f32 + 0.5) / (nodes.len() as f32)).powf(power))
        })
    }

    pub fn shade_weighted_path(&mut self, nodes: &[(Vec2, f32)], color: P) {
        for start_index in 0..nodes.len() - 1 {
            self.draw_line(
                nodes[start_index].0,
                nodes[start_index + 1].0,
                color * (0.5 * (nodes[start_index].1 + nodes[start_index + 1].1)),
            );
        }
    }

    pub fn get_lerp(&self, point: Vec2) -> P {
        let clamp = |x: f32| x.clamp(0.0, 1.0 - f32::EPSILON);
        let clampw = |x: f32| x.clamp(0.0, (1.0 - f32::EPSILON) * self.width as f32);
        let clamph = |x: f32| x.clamp(0.0, (1.0 - f32::EPSILON) * self.height as f32);
        let point = point.clamp(Vec2::ZERO, Vec2::ONE - f32::EPSILON);
        assert!(1.0 - f32::EPSILON < 1.0);
        let fx = clampw(clamp(point.x) * (self.width as f32) - 0.5);
        let fy = clamph(clamp(point.y) * (self.height as f32) - 0.5);
        let x = fx.floor() as usize;
        let y = fy.floor() as usize;
        let index = y * self.width + x;
        let y1 = self.data[index].lerp(
            self.data[(x + 1).min(self.width - 1) + y * self.width],
            fx % 1.0,
        );
        let y = (y + 1).min(self.height - 1);
        let index = y * self.width + x;
        let y2 = self.data[index].lerp(
            self.data[(x + 1).min(self.width - 1) + y * self.width],
            fx % 1.0,
        );
        y1.lerp(y2, fy % 1.0)
    }

    pub fn get_gradient(&self, point: Vec2) -> Vec2 {
        let here = self.get_nearest(point).magnitude();
        let px = self
            .get_nearest(point + Vec2::X / self.width as f32)
            .magnitude();
        let py = self
            .get_nearest(point + Vec2::Y / self.height as f32)
            .magnitude();
        let dx = (px - here) * self.width as f32;
        let dy = (py - here) * self.height as f32;
        Vec2::new(dx, dy)
    }

    pub fn min_pixel(&self) -> P {
        *self
            .data
            .iter()
            .min_by_key(|p| NotNan::new(p.magnitude()).unwrap())
            .unwrap()
    }

    pub fn max_pixel(&self) -> P {
        *self
            .data
            .iter()
            .max_by_key(|p| NotNan::new(p.magnitude()).unwrap())
            .unwrap()
    }

    pub fn normalize(&mut self) {
        let max = self.max_pixel().magnitude();
        for pixel in &mut self.data {
            *pixel = *pixel * (1.0 / max);
        }
    }

    pub fn double_normalize(&mut self) {
        let min = self.min_pixel();
        let max = self.max_pixel().magnitude();
        for pixel in &mut self.data {
            *pixel = (*pixel + min * -1.0) * (1.0 / (max - min.magnitude()));
        }
    }

    pub fn normalize_columns(&mut self) {
        for column in 0..self.width {
            let mut max = 1e-10f32;
            for row in 0..self.height {
                let index = row * self.width + column;
                max = max.max(self.data[index].magnitude());
            }
            for row in 0..self.height {
                let index = row * self.width + column;
                let pixel = &mut self.data[index];
                *pixel = *pixel * (1.0 / max);
            }
        }
    }

    pub fn sum(&mut self, other: &Self) {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        for x in 0..self.width {
            for y in 0..self.height {
                let index = y * self.width + x;
                self.data[index] = self.data[index] + other.data[index];
            }
        }
    }

    pub fn max(&mut self, other: &Self) {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        for x in 0..self.width {
            for y in 0..self.height {
                let index = y * self.width + x;
                if other.data[index].magnitude() > self.data[index].magnitude() {
                    self.data[index] = other.data[index];
                }
            }
        }
    }

    pub fn distance_field(&mut self, max_dist: f32) {
        if self.width != self.height {
            todo!();
        }
        let mut new = Self::new(self.width, self.height);
        let max_dist_pixels = max_dist * self.width as f32;
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.data[y * self.width + x];
                let target = &mut new.data[y * self.width + x];
                if pixel.magnitude() > target.magnitude() {
                    *target = pixel;
                }
                let spread = pixel.magnitude() * max_dist_pixels + 1.0;
                let start = -spread as isize + 1;
                let end = spread as isize;
                for dy in start..end {
                    for dx in start..end {
                        let magnitude = 1.0 - ((dy * dy + dx * dx) as f32).sqrt() / spread;
                        if magnitude <= 0.0 {
                            continue;
                        }
                        let tx = x.wrapping_add_signed(dx).rem_euclid(self.width);
                        let ty = y.wrapping_add_signed(dy).rem_euclid(self.width);
                        let target = &mut new.data[ty * self.width + tx];
                        let new = pixel * magnitude;
                        if new.magnitude() > target.magnitude() {
                            *target = new;
                        }
                    }
                }
            }
        }
        self.data = new.data;
    }

    pub fn downscale_sum(&mut self) {
        assert!(self.width % 2 == 0);
        assert!(self.height % 2 == 0);
        let mut old_self = Self {
            window: None,
            width: self.width / 2,
            height: self.height / 2,
            data: vec![],
            mask: vec![],
            framebuffer: vec![],
        };
        std::mem::swap(self, &mut old_self);

        for y in 0..self.height {
            for x in 0..self.width {
                let base = (y * 2) * old_self.width + (x * 2);
                self.data.push(
                    old_self.data[base]
                        + old_self.data[base + 1]
                        + old_self.data[base + old_self.width]
                        + old_self.data[base + old_self.width + 1],
                );
                self.mask.push(
                    old_self.mask[base]
                        || old_self.mask[base + 1]
                        || old_self.mask[base + old_self.width]
                        || old_self.mask[base + old_self.width + 1],
                );
            }
        }
    }

    pub fn downscale_avg(&mut self) {
        self.downscale_sum();
        self.map(|p| p * 0.25);
    }

    pub fn downscale_blurry(&mut self) {
        assert!(
            self.width % 2 == 0 && self.height % 2 == 0,
            "Size must be a multiple of 2"
        );
        let old_data = self.clone();
        self.resize(self.width / 2, self.height / 2);
        let d = 1.0 / self.width as f32;
        self.shade(|p| {
            old_data.get_lerp(p) * (1.0 / 2.0)
                + (old_data.get_lerp(p + Vec2::new(d, d))
                    + old_data.get_lerp(p + Vec2::new(d, -d))
                    + old_data.get_lerp(p + Vec2::new(d, d))
                    + old_data.get_lerp(p + Vec2::new(-d, -d)))
                    * (1.0 / 8.0)
        });
    }

    pub fn upscale_linear(&mut self, factor: usize) {
        let old_data = std::mem::take(&mut self.data);
        self.resize(self.width * factor, self.height * factor);
        for y in 0..self.height / factor {
            let y2 = (y + 1) % (self.height / factor);
            for x in 0..self.width / factor {
                let x2 = (x + 1) % (self.width / factor);
                let p00 = old_data[y * self.width / factor + x];
                let p01 = old_data[y2 * self.width / factor + x];
                let p10 = old_data[y * self.width / factor + x2];
                let p11 = old_data[y2 * self.width / factor + x2];
                for dy in 0..factor {
                    let p0 = p00.lerp(p01, dy as f32 / factor as f32);
                    let p1 = p10.lerp(p11, dy as f32 / factor as f32);
                    for dx in 0..factor {
                        self.data[(y * factor + dy) * self.width + (x * factor + dx)] =
                            p0.lerp(p1, dx as f32 / factor as f32)
                    }
                }
            }
        }
    }

    pub fn upscale_blurry(&mut self, factor: usize) {
        let mut factor = factor;
        while factor > 1 {
            assert!(factor % 2 == 0, "Factor must be a power of 2");
            let old_data = self.clone();
            self.resize(self.width * 2, self.height * 2);
            factor /= 2;
            let d = 1.0 / self.width as f32;
            let d2 = d * 2.0;
            self.shade(|p| {
                (old_data.get_lerp(p + Vec2::new(d, d))
                    + old_data.get_lerp(p + Vec2::new(d, -d))
                    + old_data.get_lerp(p + Vec2::new(d, d))
                    + old_data.get_lerp(p + Vec2::new(-d, -d)))
                    * (1.0 / 6.0)
                    + (old_data.get_lerp(p + Vec2::new(d2, 0.0))
                        + old_data.get_lerp(p + Vec2::new(-d2, 0.0))
                        + old_data.get_lerp(p + Vec2::new(0.0, d2))
                        + old_data.get_lerp(p + Vec2::new(0.0, d2)))
                        * (1.0 / 12.0)
            });
        }
    }

    pub fn kawase_iteration(&mut self, iteration: i32) {
        let offset = 1.4f32.powi(iteration) / self.width as f32 / 2.0;
        let mut ping_pong = self.clone();
        ping_pong.shade(|point| {
            (self.get_lerp(point + offset)
                + self.get_lerp(point + Vec2::new(offset, -offset))
                + self.get_lerp(point + Vec2::new(-offset, offset))
                + self.get_lerp(point - offset))
                * 0.25
        });
        *self = ping_pong;
    }

    pub fn kawase_single(&mut self, iterations: i32) {
        for iteration in 0..iterations {
            self.kawase_iteration(iteration);
        }
    }
}

pub fn rgb(r: f32, g: f32, b: f32) -> LinSrgba {
    LinSrgba::new(r, g, b, 1.0)
}

pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> LinSrgba {
    LinSrgba::new(r, g, b, a)
}

pub fn hsv(hue: f32, saturation: f32, value: f32) -> LinSrgba {
    LinSrgba::from_color(Hsv::from_components((hue * 360.0, saturation, value)))
}

pub fn hsva(hue: f32, saturation: f32, value: f32, alpha: f32) -> LinSrgba {
    LinSrgba::from_color(Hsva::from_components((
        hue * 360.0,
        saturation,
        value,
        alpha,
    )))
}

pub fn unique_color(mut index: u32) -> LinSrgba {
    let shades = 3;
    let shade = index % (shades * shades);
    index /= shades * shades;
    let mut divisions = 1;
    while index > divisions {
        index -= divisions;
        divisions *= 2;
    }
    let hue = (2 * index + 1) as f32 / (divisions * 2) as f32;
    hsv(
        hue,
        (shade % shades + 1) as f32 / shades as f32,
        (shade / shades + 1) as f32 / shades as f32,
    )
}

pub trait FloatExt {
    fn lerp(self, other: Self, factor: Self) -> Self;
    fn ping_pong(self) -> Self;
    fn map_range(self, from: Range<Self>, to: Range<Self>) -> Self
    where
        Self: Sized;
    fn snap(self, interval: Self) -> Self;
}

impl FloatExt for f32 {
    fn lerp(self, other: Self, factor: Self) -> Self {
        self * (1.0 - factor) + other * factor
    }

    fn ping_pong(self) -> Self {
        let base = self.rem_euclid(2.0);
        if base > 1.0 {
            2.0 - base
        } else {
            base
        }
    }

    fn map_range(self, from: Range<Self>, to: Range<Self>) -> Self {
        let from_range = from.end - from.start;
        let to_range = to.end - to.start;
        (self - from.start) * to_range / from_range + to.start
    }

    fn snap(self, interval: Self) -> Self {
        (self / interval).round() * interval
    }
}

impl FloatExt for f64 {
    fn lerp(self, other: Self, factor: Self) -> Self {
        self * (1.0 - factor) + other * factor
    }

    fn ping_pong(self) -> Self {
        let base = self.rem_euclid(2.0);
        if base > 1.0 {
            2.0 - base
        } else {
            base
        }
    }

    fn map_range(self, from: Range<Self>, to: Range<Self>) -> Self {
        let from_range = from.end - from.start;
        let to_range = to.end - to.start;
        (self - from.start) * to_range / from_range + to.start
    }

    fn snap(self, interval: Self) -> Self {
        (self / interval).round() * interval
    }
}

pub trait VecExt {
    fn torus_length(self) -> f32;
    fn torus_distance(self, other: Self) -> f32;
    fn reflect(self, normal: Self) -> Self;
    fn rem_euclid(self, other: Self) -> Self;
}

impl VecExt for Vec2 {
    fn torus_length(self) -> f32 {
        let x = self.x.rem_euclid(1.0);
        let y = self.y.rem_euclid(1.0);
        let x = x.min(1.0 - x);
        let y = y.min(1.0 - y);
        (x * x + y * y).sqrt()
    }

    fn torus_distance(self, other: Self) -> f32 {
        (other - self).torus_length()
    }

    fn reflect(self, normal: Self) -> Self {
        self - normal * 2.0 * self.dot(normal)
    }

    fn rem_euclid(self, other: Self) -> Self {
        Vec2::new(self.x.rem_euclid(other.x), self.y.rem_euclid(other.y))
    }
}

pub struct CanvasStack<P> {
    canvases: Vec<Canvas<P>>,
    display_canvas: Canvas<P>,
}

impl<P: Default> CanvasStack<P> {
    pub fn new() -> Self {
        Self {
            canvases: Vec::new(),
            display_canvas: Canvas::new(1, 1),
        }
    }

    pub fn push(&mut self, canvas: Canvas<P>) {
        self.canvases.push(canvas);
    }
}

impl<P: Copy + Default> CanvasStack<P> {
    pub fn generate_by_filtering(
        base: Canvas<P>,
        iterations: usize,
        filter: impl Fn(&Canvas<P>, usize, Vec2) -> P,
    ) -> Self {
        let mut stack = Self::new();
        stack.push(base);
        for iteration in 0..iterations {
            let previous = stack.canvases.last().unwrap();
            let mut next = previous.clone();
            next.shade(|point| filter(previous, iteration, point));
            stack.push(next);
        }
        stack
    }

    pub fn generate_by_map(
        base: Canvas<P>,
        iterations: usize,
        mapper: impl Fn(&Canvas<P>) -> Canvas<P>,
    ) -> Self {
        let mut stack = Self::new();
        stack.push(base);
        for _ in 0..iterations {
            let previous = stack.canvases.last().unwrap();
            stack.push(mapper(previous));
        }
        stack
    }
}

impl<P: PixelData> CanvasStack<P> {
    pub fn generate_by_kawase_blur(base: Canvas<P>) -> Self {
        let mut stack = Self::new();
        let mut previous = base.clone();
        let iterations = base.width().ilog2();
        stack.push(base);
        for iteration in 0..iterations {
            let mut next = previous.clone();
            next.downscale_blurry();

            let mut upscaled = next.clone();
            upscaled.upscale_blurry(2 << iteration);
            stack.push(upscaled);

            previous = next;
        }
        stack
    }
}

impl<P: Copy + Default> CanvasStack<P> {
    fn update_display_canvas(&mut self) {
        let c0w = self.canvases[0].width();
        let c0h = self.canvases[0].height();
        if self
            .canvases
            .iter()
            .all(|c| (c.width(), c.height()) == (c0w, c0h))
        {
            let grid_size = (self.canvases.len() as f32).sqrt().ceil() as usize;
            self.display_canvas.resize(grid_size * c0w, grid_size * c0h);
            for (index, canvas) in self.canvases.iter().enumerate() {
                self.display_canvas.draw_canvas(
                    canvas,
                    Vec2::new(
                        (index % grid_size) as f32 / grid_size as f32,
                        (index / grid_size) as f32 / grid_size as f32,
                    ),
                );
            }
        } else {
            todo!()
        }
    }
}

impl<P: Copy + Default + Displayable> CanvasStack<P> {
    pub fn show(&mut self) {
        self.update_display_canvas();
        self.display_canvas.show();
    }
}

impl<P: Copy + Default + Displayable> WaitUntilClick for CanvasStack<P> {
    fn is_mouse_down(&self) -> bool {
        self.display_canvas.is_mouse_down()
    }

    fn update(&mut self) {
        self.update_display_canvas();
        self.display_canvas.update();
    }
}
