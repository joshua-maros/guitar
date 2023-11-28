use std::{
    hash::{Hash, Hasher},
    ops::Neg,
};

use bevy_math::{Vec2, Vec3};
use num::{Float, One, Zero};
use rand::{distributions::uniform::SampleUniform, RngCore};
pub use rand::{distributions::*, prelude::*, Rng};
use rand_distr::{Exp1, StandardNormal};

pub const LONG_TAIL_DIST: &'static [f32] = &[0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
pub const NORMALISH_DIST: &'static [f32] = &[2.0, 0.5, 0.2, 0.1, 0.1, 0.2, 0.5, 2.0];

impl RngCore for RandomContext {
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() & 0xFFFF_FFFF) as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.next();
        self.state
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let rand_data = self.next_u64().to_ne_bytes();
            for index in 0..chunk.len() {
                chunk[index] = rand_data[index];
            }
        }
        let remainder_len = dest.len() % 8;
        let remainder_start = dest.len() - remainder_len;
        for offset in 0..remainder_len {
            let rand_data = self.next_u64().to_ne_bytes();
            dest[remainder_start + offset] = rand_data[offset];
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

pub trait RngExt: Rng {
    fn gcs<T: Float>(&mut self, dof: T) -> T
    where
        StandardNormal: Distribution<T>,
        Exp1: Distribution<T>,
        Open01: Distribution<T>,
    {
        rand_distr::ChiSquared::new(dof).unwrap().sample(self)
    }

    fn g01<T: Zero + One + PartialOrd>(&mut self) -> T
    where
        T: SampleUniform,
    {
        self.gen_range(T::zero()..T::one())
    }

    fn g01f32(&mut self) -> f32 {
        self.g01()
    }

    fn g01v2(&mut self) -> Vec2 {
        Vec2::new(self.g01(), self.g01())
    }

    fn g01v3(&mut self) -> Vec3 {
        Vec3::new(self.g01(), self.g01(), self.g01())
    }

    fn g11<T: One + Neg<Output = T> + PartialOrd>(&mut self) -> T
    where
        T: SampleUniform,
    {
        self.gen_range(-T::one()..T::one())
    }

    fn g11f32(&mut self) -> f32 {
        self.g11()
    }

    fn g11v2(&mut self) -> Vec2 {
        Vec2::new(self.g11(), self.g11())
    }

    fn gballv2(&mut self) -> Vec2 {
        loop {
            let candidate = self.g11v2();
            if candidate.length_squared() <= 1.0 {
                return candidate;
            }
        }
    }

    fn g11v3(&mut self) -> Vec3 {
        Vec3::new(self.g11(), self.g11(), self.g11())
    }
}

impl<T: Rng> RngExt for T {}

#[derive(Clone, Copy, Debug)]
pub struct RandomContext {
    state: u64,
}

pub type Rr = RandomContext;

impl RandomContext {
    pub fn new(seed: &(impl Hash + ?Sized)) -> Self {
        let mut hasher = wyhash::WyHash::with_seed(0);
        seed.hash(&mut hasher);
        let state = hasher.finish();
        Self { state }
    }

    pub fn one_shot<T>(seed: u64, data: &(impl Hash + ?Sized)) -> T
    where
        Standard: Distribution<T>,
    {
        let mut hasher = wyhash::WyHash::with_seed(seed);
        data.hash(&mut hasher);
        Self {
            state: hasher.finish(),
        }
        .gen()
    }

    pub fn add_seed(&mut self, seed: &(impl Hash + ?Sized)) -> &mut Self {
        let mut hasher = wyhash::WyHash::with_seed(self.state);
        seed.hash(&mut hasher);
        self.state = hasher.finish();
        self
    }

    pub fn next(&mut self) {
        self.add_seed(&self.state.wrapping_add(1));
    }
}
